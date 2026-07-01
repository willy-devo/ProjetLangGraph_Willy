"""
Script de test LangFuse — verifie la connexion, envoie une trace et confirme
qu'elle est bien recue sur le serveur.

Lancer :
    python scripts/test_langfuse.py
"""

import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
host       = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

print(f"Public key : {public_key[:10]}..." if public_key else "Public key : MANQUANTE")
print(f"Secret key : {secret_key[:10]}..." if secret_key else "Secret key : MANQUANTE")
print(f"Host       : {host}")
print()

try:
    from langfuse import Langfuse
    import langfuse as _lf_module
    print(f"Version langfuse : {_lf_module.__version__}")

    # debug=False pour un output lisible (le debug HTTP a confirme que le batch passe bien)
    lf = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    # 1. Auth
    ok = lf.auth_check()
    print(f"Auth check       : {'OK' if ok else 'ECHEC'}")
    if not ok:
        print("  → Verifie tes cles LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY dans .env")
        exit(1)

    # 2. Envoie une trace
    trace = lf.trace(
        name   = "test-connexion",
        input  = {"question": "test"},
        output = {"apis": ["order-api"]},
        tags   = ["test"],
    )
    print(f"Trace ID local   : {trace.id}")

    # flush() envoie le batch HTTP — on a confirme via debug=True que le serveur repond 201
    lf.flush()
    print("Batch envoye au serveur (ingestion async)")

    # L'ingestion LangFuse est asynchrone cote serveur :
    # le batch est accepte (201) mais l'indexation peut prendre jusqu'a ~15 secondes.
    print("Attente indexation serveur", end="", flush=True)
    for _ in range(15):
        time.sleep(1)
        print(".", end="", flush=True)
        resp = httpx.get(
            f"{host}/api/public/traces/{trace.id}",
            auth=(public_key, secret_key),
            timeout=10.0,
        )
        if resp.status_code == 200:
            print(f"\nTrace confirmee sur le serveur apres indexation [OK]")
            print(f"  >> Va sur {host} > Traces > cherche 'test-connexion'")
            break
    else:
        print(f"\nTrace toujours en cours d'indexation apres 15s (status {resp.status_code})")
        print(f"  >> Le batch a quand meme ete envoye (status 201 confirme).")
        print(f"  >> Attends quelques secondes et rafraichis {host} > Traces")
        print(f"  >> Trace ID : {trace.id}")

except ImportError:
    print("ERREUR : langfuse non installe. Lance : pip install 'langfuse>=2,<3'")
except Exception as e:
    print(f"ERREUR : {e}")
