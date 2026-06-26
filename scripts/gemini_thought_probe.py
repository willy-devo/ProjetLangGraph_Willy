"""
Appel direct Gemini REST API (sans Kong) pour inspecter thought_signature.
Le thought_signature n'apparait QUE quand :
  1. Modele thinking (gemini-2.5-flash ou gemini-2.5-pro)
  2. Au moins un tool defini
  3. Le LLM decide d'appeler ce tool

Kong le supprime parce qu'il traduit la reponse en format OpenAI (qui n'a pas ce champ).
"""

import json
import httpx

GEMINI_API_KEY = "AIzaSyDmu1LnNKJ-wtkFPRVfJu0rIT-V_84YPnM"
MODEL = "gemini-2.5-flash"  # doit etre un modele thinking

url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

payload = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Quelle API utiliser pour creer une commande client ?"}]
        }
    ],
    "tools": [
        {
            "function_declarations": [
                {
                    "name": "search_apis",
                    "description": "Recherche des APIs dans un catalogue semantique",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
    ],
    "generationConfig": {
        "thinkingConfig": {"thinkingBudget": 512}
    }
}

with httpx.Client(timeout=60) as client:
    resp = client.post(url, json=payload, params={"key": GEMINI_API_KEY})
    resp.raise_for_status()
    data = resp.json()

parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])

# print("=== PARTS DE LA REPONSE ===")
# for i, part in enumerate(parts):
#     if part.get("thought"):
#         print(f"[{i}] THOUGHT (tronque) : {part.get('text', '')[:300]}")
#     elif "functionCall" in part:
#         fc = part["functionCall"]
#         print(f"[{i}] FUNCTION CALL : {fc.get('name')} | args={fc.get('args')}")
#         sig = fc.get("thoughtSignature") or part.get("thoughtSignature")
#         print(f"     thought_signature : {sig}")
#     else:
#         print(f"[{i}] TEXT : {part.get('text', '')[:300]}")

# print("\n=== USAGE ===")
# print(json.dumps(data.get("usageMetadata", {}), indent=2))

print("\n=== JSON BRUT COMPLET ===")
print(json.dumps(data, indent=2, ensure_ascii=False))
