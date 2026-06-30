"""
Nœuds du graphe agentic : agent_node → tools_node* → agent_node → END

  Le LLM décide de chercher en écrivant : SEARCH: <requête>
  (text-based tool calling — évite bind_tools + thought_signature Gemini)
  tools_node exécute la recherche Pinecone et injecte les résultats
  comme HumanMessage dans l'historique.
"""

from __future__ import annotations

import re
from functools import lru_cache

import httpx
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from agentic4api.config.settings import settings
from agentic4api.graph.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_BT
from agentic4api.graph.retriever import search
from agentic4api.graph.state import AgentState
from agentic4api.graph.tools import _format_candidate, search_apis_tool
from agentic4api.graph.transports import AsyncKongChatTransport, KongChatTransport


_RECO_RE   = re.compile(r"RECOMM[AE]N?DED_APIS\s*:\s*\[([^\]]*)\]", re.IGNORECASE)
# Le LLM déclenche une recherche en écrivant "SEARCH: <requête>" sur une ligne
_SEARCH_RE = re.compile(r"(?:^|\n)SEARCH:\s*(\S[^\n]*)", re.MULTILINE)


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    verify = settings.kong_verify_ssl
    return ChatOpenAI(
        base_url="http://kong-placeholder/v1",
        api_key=settings.kong_api_key,
        model=settings.gemini_model,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        http_client=httpx.Client(
            transport=KongChatTransport(settings.kong_chat_url, verify=verify),
            timeout=40.0,
        ),
        async_client=httpx.AsyncClient(
            transport=AsyncKongChatTransport(settings.kong_chat_url, verify=verify),
            timeout=40.0,
        ),
    )


def _usage_delta(response) -> dict:
    u = getattr(response, "usage_metadata", None) or {}
    t_in    = u.get("input_tokens", 0)
    t_out   = u.get("output_tokens", 0)
    t_total = u.get("total_tokens", 0)
    t_think = max(0, t_total - t_in - t_out)
    return {
        "tokens_in":    t_in,
        "tokens_out":   t_out,
        "tokens_think": t_think,
        "tokens_total": t_total,
        "tokens_detail": {
            "tokens_in":    [t_in],
            "tokens_out":   [t_out],
            "tokens_think": [t_think],
        },
    }


def _parse_apis(text: str) -> list[str]:
    m = _RECO_RE.search(text or "")
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    return [s.strip().strip("`*\"' ") for s in inner.split(",") if s.strip()]


def _trim_context(messages: list, max_chars: int) -> list:
    """Tronque les messages Pinecone si le contexte total dépasse max_chars.
    SystemMessage et HumanMessage originaux (question) sont toujours préservés."""
    total = sum(len(getattr(m, "content", "") or "") for m in messages)
    if total <= max_chars:
        return messages

    result = []
    budget = max_chars
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(m, HumanMessage) and content.startswith("[Résultats Pinecone"):
            allowed = max(0, budget - sum(len(getattr(x, "content", "") or "") for x in result))
            if allowed <= 0:
                continue
            m = HumanMessage(content=content[:allowed] + "…[tronqué]")
        result.append(m)
    return result


# ── Nœuds mode agentic ─────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """Nœud LLM : raisonne, décide de chercher (SEARCH:) ou de répondre."""
    messages = list(state.get("messages") or [])
    if settings.debug_mode:
        print("I AM IN AGENT NODE")
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    if not state.get("is_chat"):
        messages = _trim_context(messages, 11_000)
        if state.get("llm_call_count", 0) + 1 >= _MAX_LLM_CALLS:
            messages = messages + [HumanMessage(content=(
                "[Système] Tu as atteint la limite de recherches. "
                "Donne maintenant ta réponse finale avec RECOMMANDED_APIS. "
                "N'utilise plus SEARCH:."
            ))]

    if settings.debug_mode:
        call_n = state.get("llm_call_count", 0) + 1
        total_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
        print(f"  [agent call {call_n}] context={total_chars} chars (~{total_chars // 4} tokens)")
        print("  waiting for response...")
    response = _llm().invoke(messages)
    if settings.debug_mode:
        print("LLM Response received")
    text = response.content if isinstance(response.content, str) else str(response.content)

    out = {
        "messages":       [response],
        "llm_call_count": 1,
        **_usage_delta(response),
    }

    queries = _SEARCH_RE.findall(text)
    if queries:
        out["tool_call_inputs"] = [q.strip() for q in queries]
        out["tool_call_count"]  = len(queries)
    else:
        out["answer_text"] = text
        out["final_apis"]  = _parse_apis(text)

    return out


def tools_node(state: AgentState) -> dict:
    """Nœud outil : parse SEARCH: dans le dernier message AI, exécute Pinecone."""
    messages = state.get("messages", [])
    if settings.debug_mode:
        print("I AM IN TOOLS NODE")
    last_ai  = messages[-1]
    content  = getattr(last_ai, "content", "") or ""

    result_messages = []
    retrieved_slugs = {}

    for query in _SEARCH_RE.findall(content):
        query = query.strip()
        if not query:
            continue
        results = search(query, top_k=settings.top_k)

        for r in results:
            slug = r.get("slug", "")
            if slug:
                retrieved_slugs[slug] = retrieved_slugs.get(slug, 0) + 1

        body = "\n".join(_format_candidate(r) for r in results) if results else "Aucun résultat trouvé."
        result_messages.append(
            HumanMessage(content=f'[Résultats Pinecone pour: "{query}"]\n{body}')
        )

    return {
        "messages":        result_messages,
        "retrieved_slugs": retrieved_slugs,
    }


_MAX_LLM_CALLS = 6


def should_continue(state: AgentState) -> str:
    if settings.debug_mode:
        print("I AM IN SHOULD CONTINUE")
    if not state.get("is_chat") and state.get("llm_call_count", 0) >= _MAX_LLM_CALLS:
        return END
    messages = state.get("messages") or []
    last     = messages[-1]
    content  = getattr(last, "content", "") or ""
    if _SEARCH_RE.search(content):
        return "tools"
    return END


# ── Nœuds mode agentic — bind_tools (nécessite thought_signature) ──────────
# Activer via TOOL_MODE=bind_tools dans le .env.
# Ne fonctionne PAS avec Kong tant que Kong ne transmet pas la thought_signature Gemini.

@lru_cache(maxsize=1)
def _llm_with_tools():
    return _llm().bind_tools([search_apis_tool])


def agent_node_bt(state: AgentState) -> dict:
    """Mode bind_tools : le LLM appelle search_apis_tool via OpenAI structured tool calls."""
    messages = list(state.get("messages") or [])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT_BT)] + messages

    response = _llm_with_tools().invoke(messages)
    out = {
        "messages":       [response],
        "llm_call_count": 1,
        **_usage_delta(response),
    }

    if hasattr(response, "tool_calls") and response.tool_calls:
        queries = [
            tc["args"].get("query", "")
            for tc in response.tool_calls
            if tc["name"] == "search_apis_tool"
        ]
        out["tool_call_inputs"] = queries
        out["tool_call_count"]  = len(queries)
    else:
        text = response.content if isinstance(response.content, str) else str(response.content)
        out["answer_text"] = text
        out["final_apis"]  = _parse_apis(text)

    return out


def tools_node_bt(state: AgentState) -> dict:
    """Mode bind_tools : exécute les tool_calls structurés, trace retrieved_slugs."""
    messages = state.get("messages", [])
    last_ai  = messages[-1]

    tool_messages   = []
    retrieved_slugs = {}

    for tc in (getattr(last_ai, "tool_calls", None) or []):
        if tc["name"] != "search_apis_tool":
            continue
        query   = tc["args"].get("query", "")
        results = search(query, top_k=settings.top_k)

        for r in results:
            slug = r.get("slug", "")
            if slug:
                retrieved_slugs[slug] = retrieved_slugs.get(slug, 0) + 1

        content = "\n".join(_format_candidate(r) for r in results) if results else "Aucun résultat trouvé."
        tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    return {
        "messages":        tool_messages,
        "retrieved_slugs": retrieved_slugs,
    }


def should_continue_bt(state: AgentState) -> str:
    messages = state.get("messages") or []
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


