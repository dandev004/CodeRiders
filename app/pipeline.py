"""Orchestrarea: audio -> curățare -> ASR hibrid -> diarizare -> LLM -> document -> email.

Joburile rulează secvențial într-un singur worker (un singur GPU de 16 GB: Whisper e eliberat din memorie
înainte să pornească LLM-ul). Tot ce se produce rămâne în data/jobs/<id>/ pe serverul intern.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import asr, audio, delivery, diarize, llm, report
from .config import get_config, resolve

log = logging.getLogger("secure_mom.pipeline")

STAGES = ["audio", "asr", "diarization", "llm", "report", "delivery"]


def jobs_dir() -> Path:
    p = resolve(get_config().server.data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


class Job(dict):
    @property
    def dir(self) -> Path:
        return jobs_dir() / self["id"]

    def save(self) -> None:
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(self, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.dir / "job.json")

    def stage(self, name: str, progress: float, msg: str = "") -> None:
        st = self["stages"][name]
        st["progress"] = round(progress, 3)
        if msg:
            st["message"] = msg
        if st["status"] == "pending":
            st["status"] = "running"
            st["started"] = time.time()
        self["current"] = name
        self.save()

    def done(self, name: str, **info) -> None:
        st = self["stages"][name]
        info.pop("seconds", None)
        st.update(status="done", progress=1.0, seconds=round(time.time() - st.get("started", time.time()), 1), **info)
        self.save()


def create_job(src_name: str, meeting_type: str, output_language: str, meeting_date: str | None,
               send_email: bool, extra_recipients: list[str], num_speakers: int | None = None,
               asr_profile: str | None = None) -> Job:
    jid = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = Job(
        id=jid, created=time.time(), status="queued", source_name=src_name, meeting_type=meeting_type,
        output_language=output_language, meeting_date=meeting_date or dt.date.today().isoformat(),
        send_email=send_email, extra_recipients=extra_recipients, num_speakers=num_speakers or None,
        asr_profile=asr_profile or get_config().asr.get("profile", "accurate"),
        current=None, error=None,
        stages={s: {"status": "pending", "progress": 0.0, "message": ""} for s in STAGES},
        deliveries=[],
    )
    job.dir.mkdir(parents=True)
    return job


def load_job(jid: str) -> Job | None:
    p = jobs_dir() / jid / "job.json"
    if not p.exists() or "/" in jid or ".." in jid:
        return None
    return Job(json.loads(p.read_text(encoding="utf-8")))


def list_jobs() -> list[dict]:
    out = []
    for p in sorted(jobs_dir().glob("*/job.json"), reverse=True):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append({k: j.get(k) for k in ("id", "created", "status", "source_name", "meeting_type", "meeting_date",
                                          "title", "duration_s", "total_seconds", "counts")})
    return out


def run(job: Job) -> None:
    t0 = time.time()
    job["status"] = "running"
    # la reluarea după o repornire, etapele o iau de la capăt (timpii raportați rămân corecți)
    job["stages"] = {s: {"status": "pending", "progress": 0.0, "message": ""} for s in STAGES}
    job["started"] = t0
    job.save()
    try:
        _run(job)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        log.exception("Job %s eșuat", job["id"])
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["traceback"] = traceback.format_exc()[-3000:]
        if job.get("current"):
            job["stages"][job["current"]]["status"] = "error"
    job["total_seconds"] = round(time.time() - t0, 1)
    job.save()


def _run(job: Job) -> None:
    cfg = get_config()
    src = next(job.dir.glob("input.*"))

    # 1. audio: decodare + curățare zgomot
    job.stage("audio", 0.1, "decodare + reducere zgomot")
    audio_asr, audio_clean, sr, astats = audio.prepare(src, job.dir / "audio")
    job["duration_s"] = astats["duration_s"]
    job.done("audio", **astats)

    # 2. ASR hibrid RO/RU/EN
    job.stage("asr", 0.0, "încărcare model")
    auto_type = job["meeting_type"] not in cfg.meeting_types  # „auto”: promptul ASR folosește tot glosarul
    segs, astats = asr.transcribe(audio_asr, None if auto_type else job["meeting_type"], vad_audio=audio_clean,
                                  progress=lambda p, m: job.stage("asr", p, m), profile=job.get("asr_profile"))
    if not segs:
        raise RuntimeError("Nu s-a detectat vorbire în înregistrare.")
    job.done("asr", **astats)

    # 3. diarizare
    if cfg.diarization.enabled:
        job.stage("diarization", 0.1, "amprente vocale")
        pieces, dstats = diarize.diarize(audio_clean, [(s.start, s.end) for s in segs], job.get("num_speakers"))
        for s, pc in zip(segs, pieces):
            # fără timestamp-uri pe cuvinte textul nu poate fi tăiat exact: vorbitorul majoritar al frazei
            dur: dict[str, float] = {}
            for a, b, n in pc:
                dur[n] = dur.get(n, 0.0) + b - a
            s.speaker = max(dur, key=dur.get)
            if len(pc) > 1:
                s.flags.append("multiple_speakers")
        job["diarization"] = dstats
        job.done("diarization", speakers=dstats.get("speakers"))
    else:
        for s in segs:
            s.speaker = "S1"
        job["diarization"] = {"speakers": 1}
        job.done("diarization", skipped=True)

    seg_dicts = [s.to_dict() for s in segs]
    (job.dir / "transcript.json").write_text(json.dumps(seg_dicts, ensure_ascii=False, indent=1), encoding="utf-8")
    speakers = []
    spk_langs: dict[str, list[str]] = {}
    for s in seg_dicts:
        if s["speaker"] not in speakers:
            speakers.append(s["speaker"])
        spk_langs.setdefault(s["speaker"], [])
        if s["lang"] not in spk_langs[s["speaker"]]:
            spk_langs[s["speaker"]].append(s["lang"])
    job["speaker_langs"] = spk_langs
    job["lang_seconds"] = astats.get("lang_seconds")

    # 4. LLM local
    job.stage("llm", 0.0, "analiză")
    mom, lstats = llm.generate_mom(seg_dicts, job["meeting_type"], dt.date.fromisoformat(job["meeting_date"]),
                                   speakers, job["duration_s"], progress=lambda p, m: job.stage("llm", p, m))
    if auto_type:  # tipul detectat de LLM decide rutarea n8n și lista de distribuție
        job["meeting_type"] = mom.get("meeting_type") if mom.get("meeting_type") in cfg.meeting_types \
            else next(iter(cfg.meeting_types))
        job["meeting_type_detected"] = True
    (job.dir / f"mom.{mom['language']}.json").write_text(json.dumps(mom, ensure_ascii=False, indent=1), encoding="utf-8")
    job["title"] = mom.get("title")
    job["counts"] = {"decisions": sum(len(x) for _, x in report.decision_groups(mom, mom["language"])),
                     "action_items": len(mom.get("action_items", [])),
                     "participants": len(mom.get("participants", []))}
    job.done("llm", **{k: lstats[k] for k in ("model", "chunks", "seconds")})

    # 5. document în limba aleasă
    lang = job["output_language"]
    job.stage("report", 0.2, f"document ({lang})")
    files = build_documents(job, lang)
    job.done("report", files=[p.name for p in files])

    # 6. email prin n8n / SMTP local
    if job["send_email"]:
        job.stage("delivery", 0.3, "trimitere")
        try:
            res = deliver(job, lang)
            job.done("delivery", via=res["via"], recipients=len(res["recipients"]))
        except Exception as e:  # noqa: BLE001 — documentul e gata; emailul se poate retrimite din interfață
            log.warning("Trimiterea automată a eșuat pentru %s: %s", job["id"], e)
            job["stages"]["delivery"].update(status="error", progress=1.0, message=str(e)[:200])
            job["delivery_error"] = str(e)[:200]
    else:
        job["stages"]["delivery"].update(status="skipped", progress=1.0)
    llm.Ollama().unload()


def get_mom(job: Job, lang: str) -> dict:
    p = job.dir / f"mom.{lang}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    base_lang = get_config().output.default_language
    base = json.loads((job.dir / f"mom.{base_lang}.json").read_text(encoding="utf-8"))
    mom = llm.translate_mom(base, lang)
    p.write_text(json.dumps(mom, ensure_ascii=False, indent=1), encoding="utf-8")
    return mom


def build_documents(job: Job, lang: str) -> list[Path]:
    mom = get_mom(job, lang)
    html = report.render_html(mom, job, lang)
    out = job.dir / "out"
    out.mkdir(exist_ok=True)
    stem = f"MoM_{job['meeting_type']}_{job['meeting_date']}_{lang}"
    (out / f"{stem}.html").write_text(html, encoding="utf-8")
    (out / f"{stem}.md").write_text(report.render_markdown(mom, job, lang), encoding="utf-8")
    docx = report.render_docx(mom, job, lang, out / f"{stem}.docx")
    return [out / f"{stem}.html", docx, out / f"{stem}.md"]


def deliver(job: Job, lang: str, extra: list[str] | None = None, only_to: list[str] | None = None) -> dict:
    files = build_documents(job, lang)
    mom = get_mom(job, lang)
    html = files[0].read_text(encoding="utf-8")
    res = delivery.send(mom, job, lang, html, [files[1]], (job.get("extra_recipients") or []) + (extra or []),
                        only_to=only_to)
    res.update(language=lang, at=time.time(), manual=bool(only_to))
    job.setdefault("deliveries", []).append(res)
    job.save()
    return res


# ------------------------------------------------------------------------------------------------
# Worker (un singur job pe GPU la un moment dat)
# ------------------------------------------------------------------------------------------------

_q: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()


def enqueue(job: Job) -> None:
    job.save()
    _q.put(job["id"])


def queue_position(jid: str) -> int:
    return list(_q.queue).index(jid) + 1 if jid in _q.queue else 0


def _worker() -> None:
    while True:
        jid = _q.get()
        job = load_job(jid)
        if job:
            with _lock:
                run(job)
        _q.task_done()


def start_worker() -> None:
    # joburile rămase neterminate la o repornire sunt reluate
    for p in sorted(jobs_dir().glob("*/job.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("status") in ("queued", "running"):
            _q.put(j["id"])
    threading.Thread(target=_worker, daemon=True, name="secure-mom-worker").start()


def lock() -> threading.Lock:
    return _lock
