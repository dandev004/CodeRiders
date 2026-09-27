"""Garda offline: la nivel de proces, blochează orice conexiune spre o adresă din afara rețelei interne.

Aceasta e o plasă de siguranță suplimentară (defense in depth) peste faptul că toate modelele
rulează local: chiar dacă o bibliotecă ar încerca să „sune acasă” (telemetrie, download de model),
conexiunea e refuzată și înregistrată în log.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket

log = logging.getLogger("secure_mom.guard")

_blocked: list[str] = []
_installed = False


def _set_offline_env() -> None:
    # Bibliotecile ML nu mai încearcă să descarce nimic la runtime.
    for k, v in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "ANONYMIZED_TELEMETRY": "False",
        "N8N_DIAGNOSTICS_ENABLED": "false",
    }.items():
        os.environ[k] = v


def install(allowed_networks: list[str], allowed_hosts: list[str]) -> None:
    global _installed
    _set_offline_env()
    if _installed:
        return
    nets = [ipaddress.ip_network(n, strict=False) for n in allowed_networks]
    hosts = {h.lower() for h in allowed_hosts}

    def is_allowed(host) -> bool:
        if host is None:
            return True
        host = str(host).lower()
        if host in hosts:
            return True
        try:
            ip = ipaddress.ip_address(host.split("%")[0])
        except ValueError:
            return False  # nume DNS extern
        return any(ip in n for n in nets)

    def record(host) -> None:
        msg = f"BLOCKED outbound connection to {host}"
        _blocked.append(str(host))
        log.warning(msg)

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not is_allowed(address[0]):
            record(address[0])
            raise ConnectionRefusedError(f"Secure MOM offline guard: {address[0]} is outside the internal network")
        return orig_connect(self, address)

    def guarded_connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not is_allowed(address[0]):
            record(address[0])
            return 111
        return orig_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host is not None and not is_allowed(host):
            record(host)
            raise socket.gaierror(f"Secure MOM offline guard: DNS lookup of {host} blocked")
        return orig_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    _installed = True
    log.info("Offline guard active: only %s allowed", ", ".join(allowed_networks))


def blocked_attempts() -> list[str]:
    return list(_blocked)


def is_installed() -> bool:
    return _installed
