"""
Chat Chainlit — expose le graphe en UI conversationnelle.

Lancer en local :
    chainlit run src/agentic4api/chat/app.py -w

Déployer : conteneuriser (Dockerfile) → Cloud Run. C'est CE fichier qui tourne
en permanence pour que devs / Marc puissent discuter avec l'agent.

Le chat NE touche PAS le Google Sheet : il répond en direct. Le Sheet est réservé
au batch d'éval.

Mode stateless : chaque question repart de zéro, pas d'historique entre messages.
"""

from __future__ import annotations

import chainlit as cl

from agentic4api.graph.build import graph


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Bonjour ! Décris ton besoin et je te trouve l'API du catalogue."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    inputs = {"messages": [("human", message.content)]}

    async for event in graph.astream_events(inputs, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                await msg.stream_token(
                    chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                )

    await msg.update()
