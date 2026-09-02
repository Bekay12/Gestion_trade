"""
vtt_clean - Convert a YouTube WebVTT caption file into readable prose.

Stage 2 of the channel ingestion pipeline (see ingest_channel.py). YouTube
auto-captions arrive as a rolling window: each cue repeats the tail of the
previous one and carries per-word timing tags. Stripping both is what turns
the file from a timing artifact into text a reader (or a model) can use.
"""

from __future__ import annotations

import re
from pathlib import Path

# Per-word timing spans that auto-captions interleave with the text:
#   <00:00:01.234><c>word</c>
_TAG_RE = re.compile(r"<[^>]+>")
_TIMING_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})"
)
_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&#39;": "'"}

# A new timestamped paragraph is opened every PARAGRAPH_SECONDS of speech.
# Coarse enough to read as prose, fine enough to cite a moment in the video.
PARAGRAPH_SECONDS = 90


def _to_seconds(stamp: str) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Convert a WebVTT timestamp into seconds.

    Inputs:
        stamp (str): timestamp of the form HH:MM:SS.mmm

    Outputs:
        seconds (float): the same instant expressed in seconds
    --------------------------------------------------------------------------
    """
    hours, minutes, rest = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _format_stamp(seconds: float) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Render a second count as the HH:MM:SS marker used in the output.

    Inputs:
        seconds (float): offset from the start of the video

    Outputs:
        stamp (str): zero-padded HH:MM:SS
    --------------------------------------------------------------------------
    """
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _clean_text(raw: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Strip inline timing tags and HTML entities from one cue's text.

    Inputs:
        raw (str): a single line of cue text, possibly tag-laden

    Outputs:
        text (str): the same line as plain text, whitespace-collapsed
    --------------------------------------------------------------------------
    """
    text = _TAG_RE.sub("", raw)
    for entity, replacement in _ENTITIES.items():
        text = text.replace(entity, replacement)
    return " ".join(text.split())


def parse_cues(vtt: str) -> list[tuple[float, str]]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Extract (start, text) pairs from a WebVTT document, discarding the
        header, NOTE blocks, cue settings and per-word timing spans.

    Inputs:
        vtt (str): the full contents of a .vtt file

    Outputs:
        cues (list[tuple[float, str]]): start offset in seconds and cue text,
            in document order, empty cues removed
    --------------------------------------------------------------------------
    """
    cues: list[tuple[float, str]] = []
    start: float | None = None
    buffer: list[str] = []

    for line in vtt.splitlines():
        match = _TIMING_RE.match(line.strip())
        if match:
            if start is not None and buffer:
                cues.append((start, " ".join(buffer)))
            start = _to_seconds(match.group(1))
            buffer = []
            continue
        if start is None or not line.strip() or line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
            continue
        cleaned = _clean_text(line)
        if cleaned:
            buffer.append(cleaned)

    if start is not None and buffer:
        cues.append((start, " ".join(buffer)))
    return cues


def _trim_overlap(previous: str, current: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Remove from a cue the leading words it shares with the tail of the
        previous cue. Auto-captions overlap partially as the window slides, so
        a plain prefix test leaves the shared middle duplicated in the prose.

    Inputs:
        previous (str): text already emitted
        current (str): the cue about to be emitted

    Outputs:
        text (str): current with its overlapping head removed, "" if fully
            contained in previous
    --------------------------------------------------------------------------
    """
    before, after = previous.split(), current.split()
    limit = min(len(before), len(after))
    for size in range(limit, 0, -1):
        if before[-size:] == after[:size]:
            return " ".join(after[size:])
    return current


def dedup(cues: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Collapse the rolling-window repetition of YouTube auto-captions, where
        each cue restates the tail of the one before it.

    Inputs:
        cues (list[tuple[float, str]]): parsed cues in document order

    Outputs:
        kept (list[tuple[float, str]]): cues with redundant restatements
            removed and growing cues replaced by their longest form
    --------------------------------------------------------------------------
    """
    kept: list[tuple[float, str]] = []
    for start, text in cues:
        if not kept:
            kept.append((start, text))
            continue
        previous = kept[-1][1]
        if text == previous or previous.endswith(text):
            continue
        if text.startswith(previous):
            # The cue grew in place: keep the longer form, keep the older start.
            kept[-1] = (kept[-1][0], text)
            continue
        trimmed = _trim_overlap(previous, text)
        if trimmed:
            kept.append((start, trimmed))
    return kept


def to_prose(cues: list[tuple[float, str]], paragraph_seconds: int = PARAGRAPH_SECONDS) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Join deduplicated cues into timestamped paragraphs, so a claim in the
        documentation can be traced back to a moment in the video.

    Inputs:
        cues (list[tuple[float, str]]): deduplicated cues
        paragraph_seconds (int): speech duration gathered into one paragraph

    Outputs:
        prose (str): Markdown body, each paragraph opened by an [HH:MM:SS] mark
    --------------------------------------------------------------------------
    """
    if not cues:
        return ""

    blocks: list[str] = []
    anchor = cues[0][0]
    words: list[str] = []

    for start, text in cues:
        if start - anchor >= paragraph_seconds and words:
            blocks.append(f"**[{_format_stamp(anchor)}]** " + " ".join(words))
            anchor = start
            words = []
        words.append(text)

    if words:
        blocks.append(f"**[{_format_stamp(anchor)}]** " + " ".join(words))
    return "\n\n".join(blocks)


def convert(vtt_path: Path, paragraph_seconds: int = PARAGRAPH_SECONDS) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Run the full parse -> dedup -> prose chain on a .vtt file.

    Inputs:
        vtt_path (Path): the caption file to convert
        paragraph_seconds (int): speech duration gathered into one paragraph

    Outputs:
        prose (str): the readable transcript body
    --------------------------------------------------------------------------
    """
    vtt = vtt_path.read_text(encoding="utf-8", errors="replace")
    return to_prose(dedup(parse_cues(vtt)), paragraph_seconds)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python vtt_clean.py <file.vtt>", file=sys.stderr)
        raise SystemExit(2)
    print(convert(Path(sys.argv[1])))
