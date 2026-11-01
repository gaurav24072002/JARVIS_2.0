import truststore
truststore.inject_into_ssl()
import subprocess
import sys
import os
import ssl
from pathlib import Path
import uvicorn

def _ensure_thinking_audio():

    try:
        result = subprocess.run(
            [sys.executable, "-m", "app.generate_thinking_audio"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).parent),
        )

        if result.returncode != 0 and result.stderr:
            print(f"[startup] Thinking audio: {result.stderr.strip()}")

    except Exception as e:
        print(f"[startup] Thinking audio skipped: {e}")

def _validate_startup():

    import urllib.request
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL, CHATS_DATA_DIR, LEARNING_DATA_DIR

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/version", timeout=3) as r:
            print(f"[OK] Ollama reachable at {OLLAMA_BASE_URL} (HTTP {r.status})")
    except Exception as e:
        print(f"[WARN] Ollama is not reachable at {OLLAMA_BASE_URL}: {e}")
        print("       Start it with `ollama serve` — chat will not work until it is up.")

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            import json
            models = [m["name"] for m in json.load(r).get("models", [])]
        if OLLAMA_MODEL not in models:
            print(f"[WARN] Model '{OLLAMA_MODEL}' not found locally. Available: {models or 'none'}")
            print(f"       Pull it with: ollama pull {OLLAMA_MODEL}")
        else:
            print(f"[OK] Model '{OLLAMA_MODEL}' is available locally (no API keys needed)")
    except Exception:
        pass

    if not CHATS_DATA_DIR.exists() or not CHATS_DATA_DIR.is_dir():
        print("[WARN] CHATS_DATA_DIR does not exist or is not writable.")

    if not LEARNING_DATA_DIR.exists() or not LEARNING_DATA_DIR.is_dir():
        print("[WARN] LEARNING_DATA_DIR does not exist or is not writable.")

def _ensure_self_signed_cert():
    """Return (cert_path, key_path) for a self-signed TLS cert, generating it
    on first use. Serves the app over HTTPS so the browser treats the origin
    as a *secure context* — which is required for microphone access when the
    page is opened via a LAN IP (e.g. http://192.168.x.x:8000 is NOT secure)."""
    import shutil
    from pathlib import Path

    cert_dir = Path(__file__).parent / "certs"
    cert_dir.mkdir(exist_ok=True)
    crt = cert_dir / "jarvis.crt"
    key = cert_dir / "jarvis.key"
    if crt.exists() and key.exists():
        return str(crt), str(key)

    # 1) openssl — preferred, present on nearly every system
    if shutil.which("openssl"):
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(crt),
                "-days", "3650", "-nodes",
                "-subj", "/CN=jarvis.local",
                "-addext", "subjectAltName=DNS:localhost,DNS:jarvis.local,IP:127.0.0.1",
            ], check=True, capture_output=True, timeout=120)
            print(f"[TLS] Generated self-signed cert -> {crt}")
            return str(crt), str(key)
        except Exception as e:
            print(f"[TLS] openssl failed ({e}); trying cryptography...")

    # 2) Python cryptography fallback
    try:
        import ipaddress
        from datetime import datetime, timedelta, timezone
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key_obj = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "jarvis.local")])
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key_obj.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("jarvis.local"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]), critical=False)
            .sign(key_obj, hashes.SHA256())
        )
        crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key.write_bytes(key_obj.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        print(f"[TLS] Generated self-signed cert (cryptography) -> {crt}")
        return str(crt), str(key)
    except Exception as e:
        print(f"[TLS] Could not generate a self-signed cert: {e}")
        print("       Install openssl or `pip install cryptography`, or set JARVIS_HTTPS=0.")
        return None, None


def _tls_args():
    """uvicorn kwargs to enable HTTPS. Enabled by default; disable with JARVIS_HTTPS=0."""
    enabled = os.getenv("JARVIS_HTTPS", "1").strip().lower() not in ("0", "false", "no", "off")
    if not enabled:
        return {}
    crt, key = _ensure_self_signed_cert()
    if crt and key:
        return {"ssl_certfile": crt, "ssl_keyfile": key}
    return {}


if __name__ == "__main__":
    _validate_startup()
    _ensure_thinking_audio()

    tls_kwargs = _tls_args()
    if tls_kwargs:
        print("\n[TLS] Serving over HTTPS (https://<host>:8000).")
        print("     First visit: the browser shows a self-signed certificate warning.")
        print("     Click 'Advanced' -> 'Continue/Proceed to site' ONCE to accept it.")
        print("     (Accepting the warning is what unlocks the microphone.)")
    else:
        print("\n[HTTP] Serving over plain HTTP (http://<host>:8000).")
        print("     NOTE: browsers block the microphone over http://<ip> (not a secure context).")
        print("     For the mic to work over a LAN IP, run with HTTPS:  JARVIS_HTTPS=1 python run.py")

    try:
        uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        **tls_kwargs
        )

    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print("[ERROR] Port 8000 is already in use. Try another port or stop the other process.")
        
        else:
            print(f"[ERROR] Server failed to start: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user.")
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
