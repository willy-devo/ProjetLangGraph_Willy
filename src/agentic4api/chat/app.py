"""
Chat Chainlit — expose le graphe en UI conversationnelle avec mémoire.

Lancer en local :
    chainlit run src/agentic4api/chat/app.py -w

Le chat NE touche PAS le Google Sheet : il répond en direct. Le Sheet est réservé
au batch d'éval.

Mode stateful : la mémoire est conservée par session (thread_id unique par onglet).
"""

from __future__ import annotations

import uuid

import chainlit as cl

from agentic4api.config.settings import settings
from agentic4api.graph.build import build_graph

_graph = build_graph(use_memory=settings.chat_memory)


@cl.on_chat_start
async def start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(
        content="Bonjour ! Décris ton besoin et je te trouve l'API du catalogue."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    from agentic4api.observability.langfuse_helper import callbacks as lf_callbacks, get_handler
    thread_id  = cl.user_session.get("thread_id")
    lf_handler = get_handler(session_id=thread_id, trace_name="chat")
    config     = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [lf_handler] if lf_handler else [],
    }
    inputs    = {"messages": [("human", message.content)], "is_chat": True}

    msg         = cl.Message(content="")
    line_buffer = ""

    async for event in _graph.astream_events(inputs, config=config, version="v2"):
        kind = event["event"]

        # Affiche un Step discret pendant la recherche Pinecone
        if kind == "on_chain_start" and event.get("name") == "tools":
            async with cl.Step(name="Recherche dans le catalogue", type="tool"):
                pass

        elif kind == "on_chat_model_stream":
            chunk   = event["data"]["chunk"]
            content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if not content:
                continue

            # Accumule par ligne pour filtrer les appels SEARCH:
            line_buffer += content
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                if not line.strip().startswith("SEARCH:"):
                    await msg.stream_token(line + "\n")

    # Vide le buffer restant (dernière ligne sans \n)
    if line_buffer and not line_buffer.strip().startswith("SEARCH:"):
        await msg.stream_token(line_buffer)

    # Fallback si le LLM a boucle sans conclure (toutes les lignes etaient SEARCH:)
    if not msg.content.strip():
        msg.content = "Je n'ai pas trouvé d'API correspondante dans le catalogue pour cette demande."

    await msg.update()

    # Flush LangFuse pour s'assurer que la trace est envoyée
    if lf_handler:
        lf_handler.flush()
