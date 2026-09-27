"""Rulează pipeline-ul complet din linia de comandă (fără server) și afișează timpii pe etape.

    python -m scripts.run_cli data/samples/Medpark_audio.m4a --type medical --lang ro [--no-email]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import offline_guard  # noqa: E402
from app.config import get_config  # noqa: E402

cfg = get_config()
if cfg.security.offline_guard:
    offline_guard.install(cfg.security.allowed_networks, cfg.security.allowed_hosts)

from app import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--type", default="medical", choices=list(cfg.meeting_types))
    ap.add_argument("--lang", default="ro", choices=list(cfg.output.languages))
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-email", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    src = Path(a.audio)
    job = pipeline.create_job(src.name, a.type, a.lang, a.date, not a.no_email, [])
    shutil.copy(src, job.dir / f"input{src.suffix.lower()}")
    pipeline.run(job)
    print(json.dumps({k: job[k] for k in ("id", "status", "error", "total_seconds", "counts", "diarization",
                                          "lang_seconds") if k in job}, ensure_ascii=False, indent=1))
    for s, st in job["stages"].items():
        print(f"  {s:12s} {st['status']:8s} {st.get('seconds', '')}")
    print("dir:", job.dir)
    if job["status"] != "done":
        print(job.get("traceback"))
        sys.exit(1)


if __name__ == "__main__":
    main()
