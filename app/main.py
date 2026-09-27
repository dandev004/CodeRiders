"""Secure MOM — server intern (FastAPI). Pornire: python -m app.main  (sau scripts/start.sh)."""
from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path

from .config import ROOT, get_config
from . import offline_guard

cfg = get_config()
if cfg.security.offline_guard:
    # înainte de orice import de bibliotecă ML: nimic nu poate ieși din rețeaua internă
    offline_guard.install(cfg.security.allowed_networks, cfg.security.allowed_hosts)

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from . import delivery, pipeline, report  # noqa: E402
from .config import resolve  # noqa: E402
from .llm import Ollama  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("secure_mom")

app = FastAPI(title="Secure MOM", docs_url="/api/docs", redoc_url=None)
WEB = ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")

ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac", ".webm", ".mp4", ".mov", ".mkv",
               ".wma", ".amr", ".3gp", ".avi"}


@app.on_event("startup")
def _startup() -> None:
    pipeline.start_worker()
    log.info("Secure MOM pornit pe http://%s:%s", cfg.server.host, cfg.server.port)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def api_config() -> dict:
    return {
        "meeting_types": cfg.meeting_types,
        "languages": cfg.output.languages,
        "default_language": cfg.output.default_language,
        "distribution_lists": cfg.delivery.distribution_lists,
        "mailbox_ui": cfg.delivery.mailbox_ui,
        "auto_send": bool(cfg.delivery.get("auto_send", True)),
        "asr_model": cfg.asr.mlx_model if cfg.asr.backend in ("auto", "mlx") else cfg.asr.faster_whisper_model,
        "llm_model": cfg.llm.model,
    }


@app.get("/api/health")
def health() -> dict:
    ok, msg = Ollama().available()
    d = delivery.health()
    return {
        "offline_guard": offline_guard.is_installed(),
        "blocked_outbound_attempts": offline_guard.blocked_attempts(),
        "llm": {"ok": ok, "detail": msg, "model": cfg.llm.model},
        "asr_model_present": resolve(cfg.asr.mlx_model).exists() or resolve(cfg.asr.faster_whisper_model).exists(),
        "n8n": d["n8n"], "smtp": d["smtp"],
        "queue": pipeline._q.qsize(),
    }


_EMAIL = re.compile(r"[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+")


def _emails(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").replace(";", ",").split(",") if _EMAIL.fullmatch(x.strip())]


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    meeting_type: str = Form("auto"),  # auto = tipul e detectat din conținut de LLM (rutarea n8n îl folosește)
    output_language: str = Form(None),
    meeting_date: str = Form(None),
    send_email: bool | None = Form(None),
    extra_recipients: str = Form(""),
    num_speakers: int = Form(0),
    asr_profile: str = Form("accurate"),
) -> dict:
    ext = Path(file.filename or "audio.webm").suffix.lower() or ".webm"
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Format nesuportat: {ext}")
    if meeting_type != "auto" and meeting_type not in cfg.meeting_types:
        raise HTTPException(400, f"Tip de ședință necunoscut: {meeting_type}")
    lang = output_language or cfg.output.default_language
    if lang not in cfg.output.languages:
        raise HTTPException(400, f"Limbă necunoscută: {lang}")
    extra = _emails(extra_recipients)
    if send_email is None:  # trimitere automată la lista de distribuție a tipului de ședință (config)
        send_email = bool(cfg.delivery.get("auto_send", True))
    job = pipeline.create_job(file.filename or f"inregistrare{ext}", meeting_type, lang, meeting_date or None,
                              send_email, extra, num_speakers if num_speakers > 0 else None,
                              asr_profile if asr_profile in (cfg.asr.get("profiles") or {}) else None)
    limit = int(cfg.server.max_upload_mb) * 1024 * 1024
    size = 0
    with open(job.dir / f"input{ext}", "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close()
                shutil.rmtree(job.dir, ignore_errors=True)
                raise HTTPException(413, "Fișier prea mare")
            f.write(chunk)
    job["size_bytes"] = size
    pipeline.enqueue(job)
    return {"id": job["id"]}


@app.get("/api/jobs")
def jobs() -> list:
    return pipeline.list_jobs()


def _job(jid: str) -> pipeline.Job:
    job = pipeline.load_job(jid)
    if not job:
        raise HTTPException(404, "Job inexistent")
    return job


@app.get("/api/jobs/{jid}")
def job_status(jid: str) -> dict:
    job = _job(jid)
    job.pop("traceback", None)
    job["queue_position"] = pipeline.queue_position(jid)
    return job


@app.get("/api/jobs/{jid}/transcript")
def transcript(jid: str) -> JSONResponse:
    p = _job(jid).dir / "transcript.json"
    if not p.exists():
        raise HTTPException(404, "Transcrierea nu e gata")
    return JSONResponse(content=__import__("json").loads(p.read_text(encoding="utf-8")))


@app.get("/api/jobs/{jid}/audio")
def job_audio(jid: str) -> FileResponse:
    """Audio-ul ședinței (varianta curățată pentru ascultare), ca fragmentele incerte să fie verificate la sursă."""
    d = _job(jid).dir / "audio"
    f = next((d / n for n in ("asr.wav", "clean.wav") if (d / n).exists()), None)
    if f is None:
        raise HTTPException(404, "Audio indisponibil")
    return FileResponse(f, media_type="audio/wav")


@app.get("/api/jobs/{jid}/mom")
def mom(jid: str, lang: str | None = None) -> dict:
    job = _job(jid)
    lang = lang or job["output_language"]
    if lang not in cfg.output.languages:
        raise HTTPException(400, "Limbă necunoscută")
    if job["status"] != "done":
        raise HTTPException(409, "Procesarea nu e gata")
    m = pipeline.get_mom(job, lang)
    return {"mom": m, "html": report.render_html(m, job, lang)}


@app.get("/api/jobs/{jid}/download/{fmt}")
def download(jid: str, fmt: str, lang: str | None = None) -> FileResponse:
    job = _job(jid)
    lang = lang or job["output_language"]
    if fmt not in ("docx", "html", "md", "json"):
        raise HTTPException(400, "Format necunoscut")
    if fmt == "json":
        pipeline.get_mom(job, lang)
        return FileResponse(job.dir / f"mom.{lang}.json", filename=f"MoM_{job['meeting_date']}_{lang}.json")
    files = pipeline.build_documents(job, lang)
    f = next(p for p in files if p.suffix == f".{fmt}")
    return FileResponse(f, filename=f.name)


@app.post("/api/jobs/{jid}/send")
def send(jid: str, body: dict = Body(default={})) -> dict:
    job = _job(jid)
    if job["status"] != "done":
        raise HTTPException(409, "Procesarea nu e gata")
    lang = body.get("lang") or job["output_language"]
    if lang not in cfg.output.languages:
        raise HTTPException(400, "Limbă necunoscută")
    only = None
    if "recipients" in body:  # adresele cerute explicit de utilizator (fereastra „Trimite pe email”)
        only = _emails(body["recipients"])
        if not only:
            raise HTTPException(400, "Introduceți o adresă de email validă.")
    extra = _emails(body.get("extra_recipients", ""))
    try:
        return pipeline.deliver(job, lang, extra, only_to=only)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Trimiterea a eșuat: {e}") from e


@app.delete("/api/jobs/{jid}")
def delete(jid: str) -> dict:
    job = _job(jid)
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "Jobul e în procesare")
    shutil.rmtree(job.dir)
    return {"deleted": jid}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=cfg.server.host, port=int(cfg.server.port), log_level="info")


if __name__ == "__main__":
    sys.exit(main())
