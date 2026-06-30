#!/usr/bin/env python3
"""
Vérifie que le thought_signature survit au cycle complet de traduction :

    Gemini natif  ──(sortie)──▶  OpenAI  ──(entrée)──▶  Gemini natif

Le test échoue (exit code 1) si le signature est perdu ou altéré à une
étape — c'est exactement le bug Kong qu'on cherche à éviter (sinon 400
sur Gemini 3.x au 2e tour).

Usage:
    python verify_signature.py input.json
    cat input.json | python verify_signature.py
    python verify_signature.py input.json --tool-suffix _tool
"""

import argparse
import json
import sys
import uuid


# ---------------------------------------------------------------------------
# SORTIE : Gemini natif -> OpenAI  (ce que Kong DEVRAIT faire)
# ---------------------------------------------------------------------------
def gemini_to_openai(gemini: dict, tool_suffix: str = "") -> dict:
    choices = []
    for cand in gemini.get("candidates", []):
        parts = cand.get("content", {}).get("parts", [])
        tool_calls, text_chunks = [], []

        for part in parts:
            if "text" in part:
                text_chunks.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                name = fc.get("name", "")
                if tool_suffix and not name.endswith(tool_suffix):
                    name += tool_suffix
                tc = {
                    "id": "call_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(fc.get("args", {}),
                                                ensure_ascii=False),
                    },
                }
                # *** préservation du signature ***
                if "thoughtSignature" in part:
                    tc["thought_signature"] = part["thoughtSignature"]
                tool_calls.append(tc)

        msg = {"role": "assistant"}
        if tool_calls:
            msg["content"] = None
            msg["tool_calls"] = tool_calls
        else:
            msg["content"] = "".join(text_chunks)

        choices.append({
            "index": cand.get("index", 0),
            "message": msg,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        })

    return {
        "model": gemini.get("modelVersion", ""),
        "object": "chat.completion",
        "id": gemini.get("responseId", "chatcmpl-" + uuid.uuid4().hex),
        "choices": choices,
        "created": 0,
    }


# ---------------------------------------------------------------------------
# ENTRÉE : OpenAI -> Gemini natif  (le tour suivant, ce que Kong renvoie à Gemini)
# ---------------------------------------------------------------------------
def openai_to_gemini(openai_msg: dict, tool_suffix: str = "") -> dict:
    parts = []
    for tc in openai_msg.get("tool_calls", []):
        fn = tc.get("function", {})
        name = fn.get("name", "")
        if tool_suffix and name.endswith(tool_suffix):
            name = name[: -len(tool_suffix)]
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        part = {"functionCall": {"name": name, "args": args}}
        # *** on remappe le signature INCHANGÉ ***
        if "thought_signature" in tc:
            part["thoughtSignature"] = tc["thought_signature"]
        parts.append(part)
    return {"role": "model", "parts": parts}


# ---------------------------------------------------------------------------
# Helpers d'extraction
# ---------------------------------------------------------------------------
def sigs_from_gemini(gemini: dict):
    out = []
    for cand in gemini.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "thoughtSignature" in part:
                out.append(part["thoughtSignature"])
    return out


def sigs_from_openai(openai: dict):
    out = []
    for ch in openai.get("choices", []):
        for tc in ch.get("message", {}).get("tool_calls", []):
            if "thought_signature" in tc:
                out.append(tc["thought_signature"])
    return out


def sigs_from_gemini_message(msg: dict):
    return [p["thoughtSignature"] for p in msg.get("parts", [])
            if "thoughtSignature" in p]


# ---------------------------------------------------------------------------
def run_checks(gemini_in: dict, tool_suffix: str) -> bool:
    ok = True

    def check(label, cond):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if not cond:
            ok = False

    print("Vérification du round-trip thought_signature")
    print("-" * 52)

    # 0. signatures présentes dans l'entrée
    src = sigs_from_gemini(gemini_in)
    print(f"\n0. Entrée Gemini native")
    check(f"au moins un thoughtSignature présent ({len(src)} trouvé(s))",
          len(src) > 0)
    if not src:
        print("\n  → Rien à vérifier : l'entrée ne contient pas de signature.")
        return False

    # 1. sortie OpenAI conserve le signature
    openai = gemini_to_openai(gemini_in, tool_suffix)
    out1 = sigs_from_openai(openai)
    print(f"\n1. Sortie OpenAI (Gemini → OpenAI)")
    check("le(s) signature(s) sont présents dans tool_calls",
          len(out1) == len(src))
    check("valeurs identiques (non altérées)", out1 == src)

    # 2. simulation d'un client qui sérialise/désérialise le message
    #    (round-trip JSON : reproduit ce que fait une lib client)
    assistant_msg = openai["choices"][0]["message"]
    reserialized = json.loads(json.dumps(assistant_msg))
    out2 = [tc["thought_signature"]
            for tc in reserialized.get("tool_calls", [])
            if "thought_signature" in tc]
    print(f"\n2. Round-trip JSON côté client (sérialise puis relit)")
    check("le signature survit à la (dé)sérialisation", out2 == src)

    # 3. retour vers Gemini (tour suivant)
    gemini_next = openai_to_gemini(reserialized, tool_suffix)
    out3 = sigs_from_gemini_message(gemini_next)
    print(f"\n3. Entrée Gemini reconstruite (OpenAI → Gemini, tour 2)")
    check("le signature est remappé vers parts[].thoughtSignature",
          len(out3) == len(src))
    check("valeur strictement inchangée vs origine", out3 == src)

    # 4. le nom d'outil fait aussi l'aller-retour proprement
    orig_names = [p["functionCall"]["name"]
                  for c in gemini_in.get("candidates", [])
                  for p in c.get("content", {}).get("parts", [])
                  if "functionCall" in p]
    back_names = [p["functionCall"]["name"] for p in gemini_next["parts"]]
    print(f"\n4. Cohérence du nom d'outil (suffixe ajouté puis retiré)")
    check(f"nom restauré à l'identique {orig_names} == {back_names}",
          orig_names == back_names)

    print("\n" + "-" * 52)
    print("RÉSULTAT :", "✓ round-trip OK — pas de 400 attendu"
          if ok else "✗ signature perdu/altéré — 400 probable sur Gemini 3.x")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Vérifie la préservation du thought_signature dans le "
                    "cycle Gemini → OpenAI → Gemini.")
    ap.add_argument("input", nargs="?",
                    help="Fichier JSON Gemini natif (défaut: stdin)")
    ap.add_argument("--tool-suffix", default="",
                    help="Suffixe outil (ex. '_tool')")
    args = ap.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    ok = run_checks(data, args.tool_suffix)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()