"""
ingest_channel - Incremental transcript ingestion for a whole YouTube channel.

Stage 1 of the documentation pipeline. Lists a channel's uploads, downloads
captions for the videos not seen before, converts them through vtt_clean, and
records a watermark so a later run resumes exactly where this one stopped.
State is saved after every video, so an interrupted run does not re-download
what it already had.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from vtt_clean import PARAGRAPH_SECONDS, convert

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
YT_DLP = REPO_ROOT / ".venv" / "Scripts" / "yt-dlp.exe"
DOCU_ROOT = REPO_ROOT / "docu"
DEFAULT_SUB_LANGS = "fr.*,en.*"
# Channel tabs that already resolve to a video listing; anything else gets /videos.
_VIDEO_TABS = ("/videos", "/streams", "/shorts", "/playlists", "list=")
# Statuses worth attempting again on a later run when --retry-missing is passed.
_RETRYABLE = ("error", "no_captions")


def _normalize_channel_url(url: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Point a channel URL at its uploads tab, so the listing returns videos
        rather than the multi-tab channel front page.

    Inputs:
        url (str): any channel, handle, or playlist URL

    Outputs:
        url (str): a URL guaranteed to resolve to a video listing
    --------------------------------------------------------------------------
    """
    url = url.strip().rstrip("/")
    if any(tab in url for tab in _VIDEO_TABS):
        return url
    return f"{url}/videos"


def slugify(text: str, max_length: int = 60) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Build a filesystem-safe, accent-free slug from a title or channel name.

    Inputs:
        text (str): the raw title
        max_length (int): maximum slug length in characters

    Outputs:
        slug (str): lowercase hyphenated slug, never empty
    --------------------------------------------------------------------------
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug[:max_length].rstrip("-") or "sans-titre"


def _run(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """
    --------------------------------------------------------------------------
    Purpose:
        Invoke yt-dlp without raising, so the caller decides whether a non-zero
        exit ends the run or only skips the current video.

    Inputs:
        args (list[str]): arguments appended to the yt-dlp executable
        timeout (int): seconds before the call is abandoned

    Outputs:
        completed (subprocess.CompletedProcess): captured stdout and stderr
    --------------------------------------------------------------------------
    """
    return subprocess.run(
        [str(YT_DLP), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def load_state(state_path: Path) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Read the ingestion watermark, tolerating a first run with no file yet.

    Inputs:
        state_path (Path): location of index.json

    Outputs:
        state (dict): parsed state, or a fresh skeleton
    --------------------------------------------------------------------------
    """
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"channel": {}, "last_video": None, "last_run": None, "videos": {}}


def save_state(state_path: Path, state: dict) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Persist state atomically, so an interrupt cannot leave a half-written
        index that would strand the ingestion history.

    Inputs:
        state_path (Path): location of index.json
        state (dict): the state to write

    Outputs:
        None
    --------------------------------------------------------------------------
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(state_path)


def list_channel(url: str) -> tuple[dict, list[dict]]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Enumerate a channel's uploads without downloading any of them.

    Inputs:
        url (str): the channel URL, already normalized

    Outputs:
        channel (dict): channel id, display name and canonical URL
        entries (list[dict]): one {id, title} per video, newest first
    --------------------------------------------------------------------------
    """
    logger.info("[YT] listing %s", url)
    completed = _run(["--flat-playlist", "--dump-single-json", url], timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(f"listing failed: {completed.stderr.strip()[:500]}")

    payload = json.loads(completed.stdout)
    channel = {
        "id": payload.get("channel_id") or payload.get("uploader_id") or "",
        "name": payload.get("channel") or payload.get("uploader") or payload.get("title") or "",
        "url": payload.get("channel_url") or url,
    }
    entries = [
        {"id": entry["id"], "title": entry.get("title") or entry["id"]}
        for entry in payload.get("entries") or []
        if entry and entry.get("id")
    ]
    logger.info("[YT] channel '%s' -> %d videos", channel["name"], len(entries))
    return channel, entries


def fetch_metadata(video_id: str) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Read one video's metadata WITHOUT touching its captions. Keeping the
        two calls apart is what stops a rate-limited caption request from also
        costing us the title and upload date, which the chronological ordering
        of the documentation depends on.

    Inputs:
        video_id (str): the YouTube video id

    Outputs:
        info (dict): the video's metadata, including its caption inventory

    Raises:
        RuntimeError: when metadata cannot be retrieved at all
    --------------------------------------------------------------------------
    """
    completed = _run([
        "--skip-download",
        "--dump-single-json",
        "--no-warnings",
        "--extractor-retries", "3",
        "--sleep-requests", "1",
        f"https://www.youtube.com/watch?v={video_id}",
    ])
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"metadata unavailable: {completed.stderr.strip()[:200]}")
    return json.loads(completed.stdout)


def choose_language(info: dict, preferences: tuple[str, ...]) -> tuple[str | None, bool]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Pick the caption track to download from the ones the video actually
        offers, preferring a human-written track over an automatic one. Asking
        yt-dlp for a language the video does not have is what triggered the
        rate-limit failures, so the choice is made from metadata first.

    Inputs:
        info (dict): the video's metadata
        preferences (tuple[str, ...]): language prefixes, most wanted first

    Outputs:
        language (str | None): the caption language code, None when the video
            has no usable track
        manual (bool): True when the chosen track is human-written
    --------------------------------------------------------------------------
    """
    # live_chat is a replay of the chat sidebar, not speech; never a transcript.
    manual = {code: track for code, track in (info.get("subtitles") or {}).items()
              if code != "live_chat"}
    automatic = info.get("automatic_captions") or {}

    for inventory, is_manual in ((manual, True), (automatic, False)):
        for prefix in preferences:
            # sorted() puts a plain "fr" ahead of "fr-orig"/"fr-FR" variants.
            for code in sorted(inventory):
                if code.split("-")[0] == prefix:
                    return code, is_manual
    return None, False


def fetch_captions(video_id: str, language: str, manual: bool, work_dir: Path) -> Path | None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Download exactly one caption track, already known to exist.

    Inputs:
        video_id (str): the YouTube video id
        language (str): the caption language code to fetch
        manual (bool): whether that track is human-written
        work_dir (Path): scratch directory for the .vtt

    Outputs:
        path (Path | None): the caption file, None when the download failed
    --------------------------------------------------------------------------
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    completed = _run([
        "--skip-download",
        "--write-subs" if manual else "--write-auto-subs",
        "--sub-langs", language,
        "--sub-format", "vtt",
        "--no-warnings",
        "--retries", "5",
        "--extractor-retries", "3",
        "--sleep-requests", "2",
        "--paths", str(work_dir),
        "-o", "%(id)s.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ])
    matches = sorted(work_dir.glob(f"{video_id}.*.vtt"))
    if not matches:
        logger.warning("[YT] %s: caption download failed: %s",
                       video_id, completed.stderr.strip()[:160])
        return None
    return matches[0]


def write_transcript(
    target_dir: Path, info: dict, video_id: str, body: str, language: str, manual: bool
) -> Path:
    """
    --------------------------------------------------------------------------
    Purpose:
        Write one transcript with the frontmatter the knowledge notes cite and
        the updater reads back.

    Inputs:
        target_dir (Path): the transcripts/ directory
        info (dict): the video's info.json payload
        video_id (str): the YouTube video id
        body (str): cleaned prose
        language (str): caption language code
        manual (bool): whether the captions are human-written

    Outputs:
        path (Path): the file written
    --------------------------------------------------------------------------
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    upload = info.get("upload_date") or "00000000"
    dated = f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}"
    title = (info.get("title") or video_id).replace('"', "'")
    path = target_dir / f"{dated}_{slugify(title)}.md"

    lines = [
        "---",
        f'video_id: "{video_id}"',
        f'title: "{title}"',
        f'upload_date: "{dated}"',
        f"duration_min: {int(info.get('duration') or 0) // 60}",
        f'url: "https://www.youtube.com/watch?v={video_id}"',
        f'caption_language: "{language}"',
        f"caption_manual: {str(manual).lower()}",
        f'ingested_at: "{datetime.now(timezone.utc).date().isoformat()}"',
        "---",
        "",
        f"# {title}",
        "",
    ]
    path.write_text("\n".join(lines) + body + "\n", encoding="utf-8")
    return path


def _select_pending(entries: list[dict], state: dict, retry_missing: bool) -> list[dict]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Decide which listed videos still need ingestion, in chronological order
        so the documentation is built oldest-first and reads as a progression.

    Inputs:
        entries (list[dict]): the channel listing, newest first
        state (dict): the current watermark
        retry_missing (bool): also retry videos that previously failed

    Outputs:
        pending (list[dict]): videos to process, oldest first
    --------------------------------------------------------------------------
    """
    seen = state.get("videos", {})
    pending = []
    for entry in entries:
        record = seen.get(entry["id"])
        if record is None or (retry_missing and record.get("status") in _RETRYABLE):
            pending.append(entry)
    return list(reversed(pending))


def ingest(
    channel_url: str,
    limit: int | None,
    sub_langs: str,
    retry_missing: bool,
    paragraph_seconds: int,
) -> Path:
    """
    --------------------------------------------------------------------------
    Purpose:
        Run one incremental ingestion pass over a channel.

    Inputs:
        channel_url (str): channel, handle, or playlist URL
        limit (int | None): stop after this many new videos
        sub_langs (str): yt-dlp language selector
        retry_missing (bool): retry videos that previously failed
        paragraph_seconds (int): speech duration per transcript paragraph

    Outputs:
        channel_dir (Path): the documentation directory for this channel
    --------------------------------------------------------------------------
    """
    preferences = tuple(
        part.strip().split(".")[0].split("-")[0]
        for part in sub_langs.split(",") if part.strip()
    )
    channel, entries = list_channel(_normalize_channel_url(channel_url))
    channel_dir = DOCU_ROOT / slugify(channel["name"] or "chaine")
    state_path = channel_dir / ".state" / "index.json"
    work_dir = channel_dir / ".state" / "raw"

    state = load_state(state_path)
    state["channel"] = channel
    state.setdefault("videos", {})

    pending = _select_pending(entries, state, retry_missing)
    if limit is not None:
        pending = pending[:limit]
    logger.info("[YT] %d new video(s) to ingest", len(pending))

    written: list[str] = []
    for position, entry in enumerate(pending, start=1):
        video_id = entry["id"]
        logger.info("[YT] (%d/%d) %s %s", position, len(pending), video_id, entry["title"][:60])
        record = {"title": entry["title"], "ingested_at": datetime.now(timezone.utc).isoformat()}
        try:
            info = fetch_metadata(video_id)
            language, manual = choose_language(info, preferences)
            vtt_path = fetch_captions(video_id, language, manual, work_dir) if language else None
            if language is None:
                record["status"] = "no_captions"
                logger.warning("[YT] %s offers no caption track", video_id)
            elif vtt_path is None:
                record["status"] = "error"
                record["error"] = f"caption download failed for '{language}'"
            else:
                body = convert(vtt_path, paragraph_seconds)
                path = write_transcript(
                    channel_dir / "transcripts", info, video_id, body, language, manual
                )
                record.update(
                    status="ok",
                    upload_date=info.get("upload_date", ""),
                    transcript=path.relative_to(channel_dir).as_posix(),
                    caption_manual=manual,
                )
                written.append(record["transcript"])
                state["last_video"] = {
                    "id": video_id,
                    "title": entry["title"],
                    "upload_date": info.get("upload_date", ""),
                }
        except Exception as error:  # one bad video must not end the channel run
            record["status"] = "error"
            record["error"] = str(error)[:300]
            logger.error("[YT] %s failed: %s", video_id, error)

        state["videos"][video_id] = record
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, state)

        for leftover in work_dir.glob(f"{video_id}.*"):
            leftover.unlink(missing_ok=True)

    print(f"\n[YT] channel   : {channel['name']}")
    print(f"[YT] directory : {channel_dir}")
    print(f"[YT] known     : {len(state['videos'])} / {len(entries)} listed")
    print(f"[YT] new       : {len(written)} transcript(s)")
    for relative in written:
        print(f"       + {relative}")
    return channel_dir


def main() -> int:
    """
    --------------------------------------------------------------------------
    Purpose:
        Command-line entry point.

    Inputs:
        None (reads sys.argv)

    Outputs:
        code (int): process exit status
    --------------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(description="Incremental YouTube channel transcript ingestion")
    parser.add_argument("channel", nargs="?", help="channel URL; omitted, reuses the stored one")
    parser.add_argument("--docu-dir", help="existing channel directory to refresh")
    parser.add_argument("--limit", type=int, help="stop after N new videos")
    parser.add_argument("--sub-langs", default=DEFAULT_SUB_LANGS, help="yt-dlp language selector")
    parser.add_argument("--retry-missing", action="store_true", help="retry failed videos")
    parser.add_argument("--paragraph-seconds", type=int, default=PARAGRAPH_SECONDS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if not YT_DLP.exists():
        logger.error("[YT] yt-dlp not found at %s", YT_DLP)
        return 1

    channel_url = args.channel
    if not channel_url and args.docu_dir:
        stored = load_state(Path(args.docu_dir) / ".state" / "index.json")
        channel_url = (stored.get("channel") or {}).get("url")
    if not channel_url:
        logger.error("[YT] no channel URL given and none stored; pass one explicitly")
        return 2

    ingest(
        channel_url,
        args.limit,
        args.sub_langs,
        args.retry_missing,
        args.paragraph_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
