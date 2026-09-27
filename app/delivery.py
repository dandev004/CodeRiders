"""Livrarea procesului-verbal: n8n self-hosted (rutare după tipul ședinței) -> SMTP local (Mailpit).

Nicio conexiune externă: webhook-ul n8n și serverul SMTP sunt pe rețeaua internă (implicit 127.0.0.1).
Dacă n8n nu răspunde, trimitem direct prin SMTP-ul local, ca procesul-verbal să ajungă oricum.
"""
from __future__ import annotations

import base64
import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path

import httpx

from .config import get_config

log = logging.getLogger("secure_mom.delivery")

SUBJECT = {"ro": "Proces-verbal", "ru": "Протокол", "en": "Minutes of Meeting"}


def recipients_for(meeting_type: str) -> list[str]:
    return list(get_config().delivery.distribution_lists.get(meeting_type, []))


def _subject(mom: dict, job: dict, lang: str) -> str:
    mt = get_config().meeting_types.get(job["meeting_type"], {}).get(lang, job["meeting_type"])
    return f"[{mt}] {SUBJECT[lang]} {job['meeting_date']} — {mom.get('title', '')}"


def send(mom: dict, job: dict, lang: str, html: str, attachments: list[Path], extra_to: list[str] | None = None,
         only_to: list[str] | None = None) -> dict:
    """Implicit: lista de distribuție a tipului de ședință (+ extra). only_to: doar adresele cerute de utilizator."""
    cfg = get_config().delivery
    to = list(only_to) if only_to else recipients_for(job["meeting_type"]) + [x for x in (extra_to or []) if x]
    subject = _subject(mom, job, lang)
    if cfg.mode == "n8n":
        try:
            return _via_n8n(cfg, job, lang, subject, html, attachments, to)
        except Exception as e:  # noqa: BLE001
            log.warning("n8n indisponibil (%s) — trimitem direct prin SMTP local", e)
            res = _via_smtp(cfg, subject, html, attachments, to)
            res["fallback_reason"] = str(e)[:200]
            return res
    return _via_smtp(cfg, subject, html, attachments, to)


def _via_n8n(cfg, job, lang, subject, html, attachments, to) -> dict:
    payload = {
        "meeting_type": job["meeting_type"],  # n8n face rutarea pe acest tag
        "job_id": job["id"], "language": lang, "subject": subject, "html": html,
        "recipients": to, "sender": cfg.sender,
        "attachments": [{"filename": p.name, "content_b64": base64.b64encode(p.read_bytes()).decode()} for p in attachments],
        "decisions": job.get("counts", {}).get("decisions", 0),
        "action_items": job.get("counts", {}).get("action_items", 0),
    }
    r = httpx.post(cfg.n8n_webhook, json=payload, timeout=60)
    r.raise_for_status()
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:200]}
    return {"via": "n8n", "recipients": to, "subject": subject, "n8n": body}


def _via_smtp(cfg, subject, html, attachments, to) -> dict:
    msg = EmailMessage()
    msg["From"] = cfg.sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    msg.set_content("Procesul-verbal este atașat / Протокол во вложении / Minutes attached.")
    msg.add_alternative(html, subtype="html")
    for p in attachments:
        sub = "vnd.openxmlformats-officedocument.wordprocessingml.document" if p.suffix == ".docx" else "octet-stream"
        msg.add_attachment(p.read_bytes(), maintype="application", subtype=sub, filename=p.name)
    with smtplib.SMTP(cfg.smtp_host, int(cfg.smtp_port), timeout=30) as s:
        s.send_message(msg)
    return {"via": "smtp", "recipients": to, "subject": subject}


def health() -> dict:
    cfg = get_config().delivery
    out = {}
    base = cfg.n8n_webhook.split("/webhook")[0]
    try:
        out["n8n"] = httpx.get(f"{base}/healthz", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        out["n8n"] = False
    try:
        with smtplib.SMTP(cfg.smtp_host, int(cfg.smtp_port), timeout=2) as s:
            s.noop()
        out["smtp"] = True
    except Exception:  # noqa: BLE001
        out["smtp"] = False
    return out
