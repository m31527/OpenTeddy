"""
An https:// door to the same OpenTeddy, for what browsers allow only on a
secure page — the microphone, for the 🎙 voice mode — when OpenTeddy is
opened from another computer or a phone. (http://localhost is secure
already and needs none of this.)

    OPENTEDDY_HTTPS_PORT=8443        # off when unset

starts a second listener in this process, serving the same app, with a
certificate made here the first time (.keys/https.crt + .key next to the
database) for this machine's names and addresses — add more with
OPENTEDDY_HTTPS_NAMES="name,10.0.0.5". It is self-signed: the browser
warns once; accept it and the page counts as secure. Plain http on the
usual port keeps working for everything that uses it now (the extension,
the desktop app, the Task API).
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime
import ipaddress
import logging
import os
import socket
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_server: Any = None
_task: Optional[asyncio.Task] = None


def port() -> int:
    try:
        return int(os.environ.get("OPENTEDDY_HTTPS_PORT", "") or 0)
    except ValueError:
        return 0


def _paths() -> Tuple[str, str]:
    from config import config
    folder = os.path.join(os.path.dirname(os.path.abspath(config.db_path)) or ".", ".keys")
    os.makedirs(folder, mode=0o700, exist_ok=True)
    ignore = os.path.join(folder, ".gitignore")
    if not os.path.exists(ignore):
        with open(ignore, "w") as fh:
            fh.write("*\n")
    return os.path.join(folder, "https.crt"), os.path.join(folder, "https.key")


def _names() -> Tuple[List[str], List[str]]:
    """(DNS names, IP addresses) the certificate should cover."""
    dns = {"localhost", socket.gethostname()}
    ips = {"127.0.0.1", "::1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 80))                 # no packet sent; picks the outbound address
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    for extra in (os.environ.get("OPENTEDDY_HTTPS_NAMES") or "").split(","):
        extra = extra.strip()
        if not extra:
            continue
        try:
            ipaddress.ip_address(extra)
            ips.add(extra)
        except ValueError:
            dns.add(extra)
    return sorted(dns), sorted(i.split("%")[0] for i in ips)


def ensure_cert() -> Tuple[str, str]:
    """The certificate and key, made once (EC P-256, about two years)."""
    crt, key = _paths()
    if os.path.exists(crt) and os.path.exists(key):
        return crt, key
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    dns, ips = _names()
    private = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"OpenTeddy on {socket.gethostname()}")])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt = [x509.DNSName(d) for d in dns] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(private.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(private, hashes.SHA256()))
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    with open(crt, "wb") as fh:
        fh.write(cert.public_bytes(serialization.Encoding.PEM))
    logger.info("https: made a certificate for %s", ", ".join(dns + ips))
    return crt, key


async def start(app: Any) -> bool:
    """Serve `app` over https too, when OPENTEDDY_HTTPS_PORT is set."""
    global _server, _task
    p = port()
    if not p or _task is not None:
        return False
    import uvicorn
    crt, key = ensure_cert()
    host = os.environ.get("OPENTEDDY_HOST", "127.0.0.1")
    cfg = uvicorn.Config(app, host=host, port=p, ssl_certfile=crt, ssl_keyfile=key,
                         log_level="warning", lifespan="off", ws="auto")
    _server = uvicorn.Server(cfg)
    # The main server owns Ctrl+C / SIGTERM; this one is stopped from its shutdown.
    _server.capture_signals = contextlib.nullcontext          # type: ignore[method-assign]
    _task = asyncio.create_task(_server.serve())
    logger.info("https: also listening on https://%s:%d", host, p)
    return True


async def stop() -> None:
    global _server, _task
    if _server is not None:
        _server.should_exit = True
    if _task is not None:
        try:
            await asyncio.wait_for(_task, 5)
        except Exception:  # noqa: BLE001
            pass
    _server, _task = None, None
