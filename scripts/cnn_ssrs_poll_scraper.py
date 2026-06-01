"""
cnn_ssrs_poll_scraper.py
========================
Scrapes SSRS voter polls published by CNN.

CNN/SSRS polls are released in two forms:
  1. A topline PDF  (the full questionnaire + crosstabs)
  2. An HTML results page on CNN's polling hub

This script:
  - Fetches the CNN polling hub and finds SSRS poll links
  - Downloads the topline PDF for each poll
  - Extracts question + answer tables from the PDF
  - Saves results to a CSV and JSON

Usage
-----
    # Parse a local PDF (easiest — just print-to-PDF from your browser)
    python cnn_ssrs_poll_scraper.py --pdf /path/to/cnn_poll.pdf

    # Parse a PDF from a direct URL
    python cnn_ssrs_poll_scraper.py --pdf-url https://...ssrs_topline.pdf

    # Auto-discover from CNN polling hub (may need real browser cookies)
    python cnn_ssrs_poll_scraper.py --limit 3

Output
------
    data/cnn_ssrs_polls/
        metadata.json          # poll metadata (date, sample size, MoE)
        {poll_date}_topline.pdf
        {poll_date}_results.csv
        {poll_date}_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────────
CNN_POLL_HUB = "https://www.cnn.com/election/2024/polls"
SSRS_BASE_URL = "https://ssrs.com"
OUTPUT_DIR = Path("data/cnn_ssrs_polls")
REQUEST_DELAY = 1.5  # seconds between requests — be polite

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Helpers ────────────────────────────────────────────────────────────────────


def fetch(url: str, stream: bool = False) -> requests.Response:
    """GET with retries and a polite delay."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, stream=stream)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp
        except requests.RequestException as e:
            print(f"  [retry {attempt+1}/3] {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


def save_pdf(url: str, dest: Path) -> Path:
    """Download a PDF to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = fetch(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Saved PDF → {dest}")
    return dest


# ── Step 1: Find poll links on CNN hub ─────────────────────────────────────────


def get_poll_links(hub_url: str, limit: int = None) -> list[dict]:
    """
    Scrape the CNN polling hub for SSRS poll entries.
    Returns list of dicts: {title, date_str, poll_page_url, pdf_url}
    """
    print(f"Fetching polling hub: {hub_url}")
    resp = fetch(hub_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    polls = []

    # CNN renders polls as article cards or list items.
    # Look for links mentioning SSRS or 'poll topline'.
    # The PDF links are usually hosted at CNN's DB:
    #   https://db.polls.cnn.com/...ssrs...topline.pdf
    # or linked directly from the article page.

    # Strategy A: find direct PDF links
    pdf_pattern = re.compile(r"ssrs.*\.pdf", re.I)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pdf_pattern.search(href):
            polls.append(
                {
                    "title": a.get_text(strip=True) or href,
                    "date_str": _extract_date_from_url(href),
                    "poll_page_url": hub_url,
                    "pdf_url": href if href.startswith("http") else urljoin(hub_url, href),
                }
            )

    # Strategy B: find article links for polls, then scrape each for the PDF
    if not polls:
        article_links = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True).lower()
            href = a["href"]
            if "poll" in text and "ssrs" in text:
                article_links.append(urljoin(hub_url, href))
        for url in article_links[: limit or 10]:
            pdf_url = _find_pdf_in_article(url)
            if pdf_url:
                polls.append(
                    {
                        "title": url.split("/")[-1].replace("-", " ").title(),
                        "date_str": _extract_date_from_url(url),
                        "poll_page_url": url,
                        "pdf_url": pdf_url,
                    }
                )

    if not polls:
        print("  Could not find SSRS poll links automatically.")
        print("  Try: python cnn_ssrs_poll_scraper.py --pdf-url <direct-pdf-url>")
        return []

    polls = _deduplicate(polls)
    if limit:
        polls = polls[:limit]
    print(f"  Found {len(polls)} SSRS poll(s)")
    return polls


def _find_pdf_in_article(article_url: str) -> str | None:
    """Look for a topline PDF link inside a CNN article page."""
    try:
        resp = fetch(article_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"ssrs.*\.pdf|topline.*\.pdf|poll.*\.pdf", href, re.I):
                return href if href.startswith("http") else urljoin(article_url, href)
    except Exception as e:
        print(f"  Skipping {article_url}: {e}")
    return None


def _extract_date_from_url(url: str) -> str | None:
    """Try to pull a YYYY-MM or YYYY_MM_DD from a URL. Returns None if not found."""
    # Full date: 2024-11-05 or 2024_11_05
    m = re.search(r"(\d{4})[-_](\d{2})(?:[-_](\d{2}))?", url)
    if m:
        parts = [m.group(1), m.group(2)]
        if m.group(3):
            parts.append(m.group(3))
        return "-".join(parts)
    # Bare year: nep_2016.pdf → "2016"
    m = re.search(r"\b(20\d{2})\b", url)
    if m:
        return m.group(1)
    return None


def _deduplicate(polls: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for p in polls:
        key = p["pdf_url"]
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ── Format detection ─────────────────────────────────────────────────────────


def detect_pdf_format(pdf_path: Path) -> str:
    """Return 'cnn_exit_poll' or 'ssrs_topline'."""
    with pdfplumber.open(pdf_path) as pdf:
        text = (pdf.pages[0].extract_text() or "").lower()
    if "exit poll" in text or "total respondents" in text:
        return "cnn_exit_poll"
    return "ssrs_topline"


# ── CNN exit-poll PDF parser ──────────────────────────────────────────────────

_PAGE_CHROME = re.compile(
    r"exit poll results 2024 \| cnn politics"
    r"|https?://www\.cnn\.com"
    r"|\d+/\d+/\d+,\s+\d+:\d+\s+[ap]m"  # browser timestamp
    r"|politics subscribe"
    r"|road to 270"
    r"|filter results by"
    r"|filter by demographic"
    r"|enter a keyword"
    r"|national results\s+results by state"
    r"|^national results\b"
    r"|^ent \|"
    r"|^election \d{4}:\s+exit polls"  # page-2 nav header
    r"|^location:$"
    r"|^contest:$"
    r"|^president general$"
    r"|^no filter$"
    r"|^senate$|^house$|^governor$"
    r"|senate.{0,15}runoff|senate special"  # 2020 nav: SENATE RUNOFF / SENATE SPECIAL
    r"|^ballot measures$"
    r"|cnn en e"  # multilingual nav fragment
    r"|^(?:audio\s+)?live\s+tv\b"  # 2016/2020 nav header
    r"|^exit polls?$"  # lone "Exit Polls" nav item
    r"|election\d{4}\b"  # 2016 nav: "election2016 results..."
    r"|all states select a state"  # 2016 nav item
    r"|voters taken after they leave"  # exit poll description
    r"|pollsters use\b"  # exit poll description (various forms)
    r"|segments of voters"  # exit poll description
    r"|^ballot measure"  # nav item (broader than "ballot measures$")
    r"|^updated\s+\d{1,2}:\d{2}"  # 2016 inline timestamp (not 2024 Updated blocks)
    r"|national president search"  # 2016 nav
    r"|view as table"  # 2016 nav
    r"|^all exit polls$"  # 2016 nav item
    r"|elections senate house governor"  # 2020 nav bar containing race links
    r"|exit polls?\s+are\s+surveys"  # 2020 exit poll description
    r"|early voters are represented"  # 2020 absentee description
    r"|absentee and early voters"
    r"|pollsters use the results"
    r"|how exit polls work"
    r"|^view exit polls?\b"
    r"|^results road to 270\b",
    re.I,
)


def _strip_chrome(line: str) -> bool:
    """True if the line is page header/footer chrome that should be skipped."""
    if _PAGE_CHROME.search(line):
        return True
    # bare page-number lines like "1/27" or plain integers
    if re.match(r"^\d+/\d+$", line) or re.match(r"^\d+$", line):
        return True
    return False


# ── 2004 table parser ────────────────────────────────────────────────────────

_2004_ROW = re.compile(
    r"^(.+?)\s+\((\d+)%\)\s+"  # sub-category (sub_pct%)
    r"(\d{1,3}%|\*|n/a)\s+"  # Bush%
    r"(?:[+\-]\d+\s+|n/a\s+)?"  # optional +/- change from 2000
    r"(\d{1,3}%|\*|n/a)\s+"  # Kerry%
    r"(\d{1,3}%|\*|n/a)",  # Nader%
    re.I,
)

_2004_SECTION = re.compile(
    r"^(?:VOTE BY |ANYONE |HAVE YOU|WHEN DID|DID YOU|HOW IMPORTANT|DO YOU|"
    r"IS THIS|WHAT IS|WHICH CANDIDATE|DOES |SHOULD |HOW DO|WAS YOUR|"
    r"HOW OFTEN|ARE YOU|THINK ABOUT|WHITE EVANGELICAL|OPINION OF|"
    r"IF ECONOMY|WHICH QUALITY|MOST IMPORTANT|FIRST TIME VOTER)",
    re.I,
)


def _is_2004_format(word_rows: list[list[dict]]) -> bool:
    """True when BUSH and KERRY appear together as column headers."""
    texts = {w["text"].upper() for row in word_rows for w in row}
    return "BUSH" in texts and "KERRY" in texts


def parse_cnn_2004_exit_poll_pdf(pdf_path: Path) -> list[dict]:
    """Parse 2004 CNN.com table-format exit poll (Bush/Kerry/Nader)."""

    def _pct(s: str) -> int | None:
        if s in ("*", "n/a"):
            return None
        return int(s.rstrip("%"))

    # ── collect all non-chrome text lines ────────────────────────────────────
    lines: list[str] = []
    n_total: int | None = None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for raw in (page.extract_text() or "").split("\n"):
                line = raw.strip()
                if not line or _strip_chrome(line):
                    continue
                m = re.match(r"^([\d,]+)\s+Respondents?\b", line, re.I)
                if m:
                    n_total = int(m.group(1).replace(",", ""))
                    continue
                lines.append(line)

    # ── parse line stream ─────────────────────────────────────────────────────
    records: list[dict] = []
    category = "unknown"

    for line in lines:
        # Skip column-header lines
        if re.match(r"^(BUSH|KERRY)\b", line, re.I):
            continue
        if re.match(r"^TOTAL\s+\d{4}", line, re.I):
            continue

        # Section header
        if _2004_SECTION.match(line):
            category = line.strip().lower()
            continue

        # Data row
        m = _2004_ROW.match(line)
        if m:
            records.append(
                {
                    "category": category,
                    "n_total": n_total,
                    "sub_category": m.group(1).strip(),
                    "sub_pct": int(m.group(2)),
                    "dem_candidate": "Kerry",
                    "rep_candidate": "Bush",
                    "dem_pct": _pct(m.group(4)),  # Kerry column
                    "rep_pct": _pct(m.group(3)),  # Bush column
                }
            )

    return records


# ── 2016 card-grid parser ─────────────────────────────────────────────────────

_2016_CARD_SPLIT = 170.0  # x < 170 = left card, x >= 170 = right card
_2016_SKIP_WORDS = frozenset({"clintontrump", "other/no", "answer"})


def _is_2016_format(word_rows: list[list[dict]]) -> bool:
    """True when the merged 'clintontrump' header token is present."""
    return any(w["text"].lower() == "clintontrump" for row in word_rows for w in row)


def _parse_2016_stream(word_rows: list[list[dict]]) -> list[dict]:
    """Parse a 2016 CNN exit poll (card-grid, Clinton/Trump/Other).

    Each demographic section has pairs of side-by-side cards:
        [label]          [label]
        C%  T%  O%       C%  T%  O%
        respondent%      respondent%
    """
    S = _2016_CARD_SPLIT

    def lw(row):
        return [w for w in row if w["x0"] < S]

    def rw(row):
        return [w for w in row if w["x0"] >= S]

    def label_of(ws):
        parts = [
            w["text"]
            for w in ws
            if not re.match(r"^\d{1,3}%?$|^\d+$", w["text"])
            and w["text"].lower() not in _2016_SKIP_WORDS
        ]
        text = " ".join(parts).strip()
        return re.sub(r"\brespondents?\b", "", text, flags=re.I).strip()

    def pcts_of(ws):
        return [w for w in ws if re.match(r"^\d{1,3}%?$", w["text"])]

    def v(w):
        return int(w["text"].rstrip("%"))

    records: list[dict] = []
    category = "unknown"
    n_total: int | None = None
    lbl_left = lbl_right = ""
    pending: list[tuple[dict, str]] = []  # (record, "left"|"right")
    saw_colhdr = False  # True once we've seen clintontrump/other/no headers
    cat_buf: list[str] = []  # accumulates label rows between sections (for multi-line headers)

    for row in word_rows:
        rtext = " ".join(w["text"] for w in row)

        # Extract N total wherever it appears in the row; reset section boundary flag
        m_n = re.search(r"\b([\d,]+)\s+respondents?\b", rtext, re.I)
        if m_n:
            n_total = int(m_n.group(1).replace(",", ""))
            saw_colhdr = False  # between sections: next matching row may be a category header
            cat_buf = []

        # Skip column-header rows, finalise any buffered category, mark inside card block
        if any(w["text"].lower() in _2016_SKIP_WORDS for w in row):
            if cat_buf and not saw_colhdr:
                if pending:
                    records.extend(r for r, _ in pending)
                    pending = []
                category = " ".join(cat_buf).strip().lower()
                lbl_left = lbl_right = ""
                cat_buf = []
            saw_colhdr = True
            continue

        all_p = pcts_of(row)
        left_p = pcts_of(lw(row))
        right_p = pcts_of(rw(row))
        lbl_l = label_of(lw(row))
        lbl_r = label_of(rw(row))

        # Between sections: accumulate label rows into the category buffer.
        # Use the left-half text when different from right (candidate-name line),
        # or the shared text when both sides match (common question fragment).
        if not all_p and not saw_colhdr:
            if pending:
                records.extend(r for r, _ in pending)
                pending = []
            lbl_l_low = lbl_l.lower()
            lbl_r_low = lbl_r.lower()
            if lbl_l_low == lbl_r_low and lbl_l_low:
                # Shared text: this is the canonical fragment (e.g. "and trustworthy?")
                cat_buf.append(lbl_l)
            elif lbl_l_low:
                # Left-half only: has candidate-specific text; use left as representative
                cat_buf.append(lbl_l)
            elif lbl_r_low:
                cat_buf.append(lbl_r)
            continue

        # Label-only row inside a card block (no percentages)
        if not all_p:
            if pending:  # flush cards that never got a sub_pct
                records.extend(r for r, _ in pending)
                pending = []
            if lbl_l:
                lbl_left = lbl_l
            if lbl_r:
                lbl_right = lbl_r
            continue

        # 6-pct data row (both cards)
        if len(left_p) == 3 and len(right_p) == 3:
            if pending:
                records.extend(r for r, _ in pending)
            pending = []
            if lbl_left:
                pending.append(
                    (
                        {
                            "category": category,
                            "n_total": n_total,
                            "sub_category": lbl_left,
                            "sub_pct": None,
                            "dem_candidate": "Clinton",
                            "rep_candidate": "Trump",
                            "dem_pct": v(left_p[0]),
                            "rep_pct": v(left_p[1]),
                            "other_pct": v(left_p[2]),
                        },
                        "left",
                    )
                )
            if lbl_right:
                pending.append(
                    (
                        {
                            "category": category,
                            "n_total": n_total,
                            "sub_category": lbl_right,
                            "sub_pct": None,
                            "dem_candidate": "Clinton",
                            "rep_candidate": "Trump",
                            "dem_pct": v(right_p[0]),
                            "rep_pct": v(right_p[1]),
                            "other_pct": v(right_p[2]),
                        },
                        "right",
                    )
                )
            lbl_left = lbl_right = ""
            continue

        # 3-pct data row (left card only)
        if len(left_p) == 3 and not right_p:
            if pending:
                records.extend(r for r, _ in pending)
            pending = []
            if lbl_left:
                pending.append(
                    (
                        {
                            "category": category,
                            "n_total": n_total,
                            "sub_category": lbl_left,
                            "sub_pct": None,
                            "dem_candidate": "Clinton",
                            "rep_candidate": "Trump",
                            "dem_pct": v(left_p[0]),
                            "rep_pct": v(left_p[1]),
                            "other_pct": v(left_p[2]),
                        },
                        "left",
                    )
                )
            lbl_left = ""
            continue

        # Sub-pct row (1–2 single pcts, one per card)
        if 1 <= len(all_p) <= 2 and pending:
            for rec, side in pending:
                src = left_p if side == "left" else right_p
                if src:
                    rec["sub_pct"] = v(src[0])
            records.extend(r for r, _ in pending)
            pending = []
            continue

    # Flush anything left
    records.extend(r for r, _ in pending)
    return records


def parse_cnn_exit_poll_pdf(pdf_path: Path) -> list[dict]:
    """Dispatch to the correct exit-poll parser based on detected format.

    2016: card-grid layout, three candidates (Clinton/Trump/Other).
    2020: N-line stream, Biden/Trump two-candidate table.
    2024: Updated-boundary stream, Harris/Trump two-candidate table.
    """
    word_rows = _extract_word_rows(pdf_path)

    if _is_2004_format(word_rows):
        records = parse_cnn_2004_exit_poll_pdf(pdf_path)
        print(f"  Extracted {len(records)} sub-category rows  [2004 text format]")
        return records

    if _is_2016_format(word_rows):
        records = _parse_2016_stream(word_rows)
        print(f"  Extracted {len(records)} sub-category rows  [2016 card-grid format]")
        return records

    dem_candidate = _detect_candidate(word_rows)
    rep_candidate = _detect_rep_candidate(word_rows)

    # 2024 format uses "Updated ..." as block separators.
    # 2020 format has no "Updated" lines — detect and use N-line stream parsing.
    has_updated = any(
        re.match(r"^Updated\b", " ".join(w["text"] for w in row), re.I) for row in word_rows
    )
    if has_updated:
        blocks = _split_word_rows_into_blocks(word_rows)
        records: list[dict] = []
        for block in blocks:
            records.extend(
                _parse_block_spatial(
                    block, dem_candidate=dem_candidate, rep_candidate=rep_candidate
                )
            )
    else:
        records = _parse_stream_n_based(word_rows, dem_candidate, rep_candidate)

    print(f"  Extracted {len(records)} sub-category rows")
    return records


# ── Spatial helpers ───────────────────────────────────────────────────────────


def _extract_word_rows(pdf_path: Path) -> list[list[dict]]:
    """Return all non-chrome word rows from every page, sorted top-to-bottom."""
    all_rows: list[list[dict]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words() or []
            for row in _group_by_y(words):
                row_text = " ".join(w["text"] for w in row)
                if not _strip_chrome(row_text):
                    all_rows.append(row)
    return all_rows


def _group_by_y(words: list[dict], tol: float = 3.0) -> list[list[dict]]:
    """Group words into visual rows by y0 proximity (tol points)."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = []
    cur = [words[0]]
    for w in words[1:]:
        if abs(w["top"] - cur[0]["top"]) <= tol:
            cur.append(w)
        else:
            rows.append(sorted(cur, key=lambda w: w["x0"]))
            cur = [w]
    rows.append(sorted(cur, key=lambda w: w["x0"]))
    return rows


def _split_word_rows_into_blocks(
    word_rows: list[list[dict]],
) -> list[list[list[dict]]]:
    """Split row stream into blocks at 'Updated ...' boundaries."""
    blocks: list[list[list[dict]]] = []
    cur: list[list[dict]] = []
    for row in word_rows:
        row_text = " ".join(w["text"] for w in row)
        if re.match(r"^Updated\b", row_text, re.I):
            if cur:
                blocks.append(cur)
            cur = []
        else:
            cur.append(row)
    if cur:
        blocks.append(cur)
    return blocks


def _detect_candidate(word_rows: list[list[dict]]) -> str:
    """Return the Democratic candidate name found in the document ('Harris', 'Biden', …)."""
    known = re.compile(r"^(Harris|Biden|Clinton|Obama|Kerry|Gore)$", re.I)
    for row in word_rows:
        first = row[0]["text"] if row else ""
        if known.match(first):
            return first.capitalize()
    return "Harris"  # safe default


def _detect_rep_candidate(word_rows: list[list[dict]]) -> str:
    """Return the Republican candidate name found in the document ('Trump', 'Romney', …)."""
    known = re.compile(r"^(Trump|Romney|McCain|Bush|Dole|Reagan)$", re.I)
    for row in word_rows:
        first = row[0]["text"] if row else ""
        if known.match(first):
            return first.capitalize()
    return "Trump"  # safe default for 2016+


def _clean_category(raw: str) -> str:
    """Post-process a raw category string extracted from cat_buf."""
    # Strip trailing garbled n_total text (e.g. "Feeling if Trump elected president 22966t t l d t")
    raw = re.sub(r"\s+\d{4,}.*$", "", raw).strip()
    # Remove PDF letter-spacing artifacts: 3+ consecutive single-letter tokens each followed by a space
    # e.g. "U S f I l i U.S. support for Israel is:" → "U.S. support for Israel is:"
    raw = re.sub(r"(?:(?<=\s)|^)(?:[A-Za-z] ){3,}", "", raw).strip()
    return raw


def _parse_stream_n_based(
    word_rows: list[list[dict]], dem_candidate: str, rep_candidate: str
) -> list[dict]:
    """Parse a row stream that has no 'Updated' separators (e.g. 2020 format).

    Blocks are delimited by 'N total respondents' lines.  The category name
    accumulates in a buffer after the previous rep row (2020: category follows
    data) or before the N line (2024-style; handled by the primary parser).
    """
    records: list[dict] = []
    cat_buf: list[str] = []  # text rows between previous block end and next N
    rep_pat = re.compile(rf"^{re.escape(rep_candidate)}\b", re.I)

    i = 0
    while i < len(word_rows):
        row = word_rows[i]
        row_text = " ".join(w["text"] for w in row)

        # Match "N total respondents" — also handles garbled "total" (e.g. "t t l d t")
        m = re.match(
            r"^([\d,]+)\s+(?:total\s+respondents|t[\s.]*[o0][\s.]*t[\s.]*a[\s.]*l)", row_text, re.I
        )
        if m:
            category = _clean_category(" ".join(cat_buf))
            cat_buf = []

            # Collect rows until Republican candidate row (or end of stream)
            block: list[list[dict]] = [row]
            j = i + 1
            while j < len(word_rows):
                nrt = " ".join(w["text"] for w in word_rows[j])
                if re.match(r"^Updated\b", nrt, re.I):
                    j += 1
                    break
                if rep_pat.match(nrt):
                    block.append(word_rows[j])
                    j += 1
                    break
                block.append(word_rows[j])
                j += 1

            rows = _parse_block_spatial(
                block,
                dem_candidate=dem_candidate,
                rep_candidate=rep_candidate,
                category_override=category or None,
            )
            records.extend(rows)
            i = j
        else:
            # Text between blocks — accumulate as category label for the next block
            if not _strip_chrome(row_text) and not re.match(r"^Updated\b", row_text, re.I):
                cat_buf.append(row_text)
            i += 1

    return records


def _nearest_col(word: dict, anchors: list[float]) -> int:
    """Return the index of the anchor x-position nearest to this word's center."""
    wx = (word["x0"] + word["x1"]) / 2
    return min(range(len(anchors)), key=lambda j: abs(wx - anchors[j]))


def _is_pct_row(row: list[dict]) -> bool:
    """True when most words in the row are bare numbers or percentages."""
    pct = sum(1 for w in row if re.match(r"^\d{1,3}%?$", w["text"]))
    return pct >= max(1, len(row) * 0.6)


def _parse_block_spatial(
    rows: list[list[dict]],
    dem_candidate: str = "Harris",
    rep_candidate: str = "Trump",
    category_override: str | None = None,
) -> list[dict]:
    """Parse one spatial block into flat sub-category rows."""
    # ── Find 'N total respondents' row ────────────────────────────────────────
    n_total: int | None = None
    n_row_idx: int | None = None
    cat_lines: list[str] = []

    for i, row in enumerate(rows):
        row_text = " ".join(w["text"] for w in row)
        m = re.match(
            r"^([\d,]+)\s+(?:total\s+respondents|t[\s.]*[o0][\s.]*t[\s.]*a[\s.]*l)", row_text, re.I
        )
        if m:
            n_total = int(m.group(1).replace(",", ""))
            n_row_idx = i
            break
        cat_lines.append(row_text)

    if n_total is None:
        return []
    category = (
        category_override if category_override is not None else _clean_category(" ".join(cat_lines))
    )
    if not category:
        category = "unknown"  # first block or chrome-filtered category

    rest = rows[n_row_idx + 1 :]

    # ── Find Democratic and Republican candidate header rows ──────────────────
    dem_row: list[dict] | None = None
    rep_row: list[dict] | None = None
    dem_idx: int | None = None
    rep_idx: int | None = None
    dem_pat = re.compile(rf"^{re.escape(dem_candidate)}\b", re.I)
    rep_pat = re.compile(rf"^{re.escape(rep_candidate)}\b", re.I)

    for i, row in enumerate(rest):
        row_text = " ".join(w["text"] for w in row)
        if dem_pat.match(row_text) and dem_row is None:
            dem_row, dem_idx = row, i
        elif rep_pat.match(row_text) and rep_row is None:
            rep_row, rep_idx = row, i

    if dem_row is None:
        return []

    # ── Build column anchors from the Democratic candidate row x-positions ────
    anchors: list[float] = []
    dem_pcts: list[int] = []
    for w in dem_row:
        if dem_pat.match(w["text"]):
            continue
        m = re.match(r"^(\d{1,3})%?$", w["text"])
        if m:
            anchors.append((w["x0"] + w["x1"]) / 2)
            dem_pcts.append(int(m.group(1)))

    n_cols = len(anchors)
    if n_cols == 0:
        return []

    # ── Republican candidate percentages ──────────────────────────────────────
    rep_pcts: list[int] = []
    if rep_row:
        for w in rep_row:
            if rep_pat.match(w["text"]):
                continue
            m = re.match(r"^(\d{1,3})%?$", w["text"])
            if m:
                rep_pcts.append(int(m.group(1)))

    # ── Assign label words and sub-pcts to columns via nearest anchor ─────────
    col_labels: dict[int, list[str]] = {j: [] for j in range(n_cols)}
    col_sub_pct: dict[int, int | None] = {j: None for j in range(n_cols)}

    between = [
        row
        for i, row in enumerate(rest)
        if i != dem_idx and i != rep_idx and (dem_idx is None or i < dem_idx)
    ]

    for row in between:
        is_pct = _is_pct_row(row)
        for w in row:
            j = _nearest_col(w, anchors)
            txt = w["text"]
            if is_pct:
                m = re.match(r"^(\d{1,3})%?$", txt)
                if m and col_sub_pct[j] is None:
                    col_sub_pct[j] = int(m.group(1))
            else:
                # Skip bare numbers and bare percentages (nav-bar artefacts)
                if not re.match(r"^\d{1,3}%?$", txt):
                    col_labels[j].append(txt)

    # ── Build output rows ─────────────────────────────────────────────────────
    result: list[dict] = []
    for j in range(n_cols):
        result.append(
            {
                "category": category,
                "n_total": n_total,
                "sub_category": " ".join(col_labels[j]).strip() or f"col_{j + 1}",
                "sub_pct": col_sub_pct[j],
                "dem_candidate": dem_candidate,
                "rep_candidate": rep_candidate,
                "dem_pct": dem_pcts[j],
                "rep_pct": rep_pcts[j] if j < len(rep_pcts) else None,
            }
        )
    return result


# ── Step 2: Extract tables from PDF ────────────────────────────────────────────


def parse_topline_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract poll questions and responses from an SSRS topline PDF.

    SSRS toplines follow a consistent format:
        Q1.  [Question text]
            Response A    xx%
            Response B    xx%
            ...
        [sample size / margin note]

    Returns a list of question records:
        {question_num, question_text, responses: [{label, pct}], notes}
    """
    records = []
    current_q = None
    notes_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Detect question header: Q1. / Q1a. / Q.1 / 1.
                q_match = re.match(r"^(?:Q\.?\s*)?(\d+[a-z]?)[.\)]\s+(.*)", line, re.I)
                if q_match:
                    if current_q:
                        current_q["notes"] = " ".join(notes_lines).strip()
                        records.append(current_q)
                    notes_lines = []
                    current_q = {
                        "question_num": q_match.group(1),
                        "question_text": q_match.group(2).strip(),
                        "responses": [],
                        "notes": "",
                    }
                    continue

                # Detect response lines: "Strongly approve    42%"
                # SSRS uses varying spacing — look for a number ending in %
                resp_match = re.match(r"^(.+?)\s{2,}(\d{1,3})%\s*$", line)
                if resp_match and current_q:
                    label = resp_match.group(1).strip()
                    pct = int(resp_match.group(2))
                    # Filter out obvious non-responses (page numbers, dates)
                    if len(label) > 1 and not re.match(r"^\d+$", label):
                        current_q["responses"].append({"label": label, "pct": pct})
                    continue

                # Detect metadata / notes lines
                if re.search(r"n\s*=\s*\d+|margin|±|conducted|sample", line, re.I):
                    notes_lines.append(line)

    # Flush last question
    if current_q:
        current_q["notes"] = " ".join(notes_lines).strip()
        records.append(current_q)

    print(f"  Extracted {len(records)} question(s) from PDF")
    return records


def extract_poll_metadata(pdf_path: Path) -> dict:
    """Pull top-level metadata from the first page of the PDF."""
    meta = {
        "source": "CNN/SSRS",
        "pdf_file": pdf_path.name,
        "dates": None,
        "sample_size": None,
        "moe": None,
        "raw_header": "",
    }
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0].extract_text() or ""
        meta["raw_header"] = first_page[:600]

        # Sample size
        m = re.search(r"n\s*=\s*([\d,]+)", first_page, re.I)
        if m:
            meta["sample_size"] = int(m.group(1).replace(",", ""))

        # Margin of error
        m = re.search(r"±\s*([\d.]+)\s*(?:percentage points?)?", first_page)
        if m:
            meta["moe"] = float(m.group(1))

        # Field dates
        m = re.search(r"(?:conducted|fielded|dates?)[:\s]+(.{10,60}?\d{4})", first_page, re.I)
        if m:
            meta["dates"] = m.group(1).strip()

    return meta


# ── Step 3: Save outputs ────────────────────────────────────────────────────────


def records_to_dataframe(records: list[dict], meta: dict) -> pd.DataFrame:
    """Flatten the nested question/response structure into a tidy DataFrame."""
    rows = []
    for q in records:
        for r in q["responses"]:
            rows.append(
                {
                    "source": meta.get("source"),
                    "poll_dates": meta.get("dates"),
                    "sample_size": meta.get("sample_size"),
                    "moe": meta.get("moe"),
                    "question_num": q["question_num"],
                    "question_text": q["question_text"],
                    "response": r["label"],
                    "pct": r["pct"],
                    "notes": q.get("notes", ""),
                }
            )
    return pd.DataFrame(rows)


def save_results(
    date_slug: str,
    records: list[dict],
    meta: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {"metadata": meta, "questions": records}
    json_path = out_dir / f"{date_slug}_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON → {json_path}")

    # CSV
    df = records_to_dataframe(records, meta)
    csv_path = out_dir / f"{date_slug}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}  ({len(df)} rows)")


def _save_exit_poll_results(
    date_slug: str,
    records: list[dict],
    meta: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{date_slug}_exit_poll.json"
    with open(json_path, "w") as f:
        json.dump({"metadata": meta, "rows": records}, f, indent=2)
    print(f"  JSON → {json_path}")

    df = pd.DataFrame(records)
    csv_path = out_dir / f"{date_slug}_exit_poll.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}  ({len(df)} rows)")

    meta_path = out_dir / "metadata.json"
    existing: list[dict] = []
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, ValueError):
            existing = []
    # Replace existing entry for the same pdf_file, or append
    updated = [e for e in existing if e.get("pdf_file") != meta.get("pdf_file")]
    updated.append(meta)
    with open(meta_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"  Metadata → {meta_path}")
    print("Done.")


# ── Main ────────────────────────────────────────────────────────────────────────


def run(
    pdf_url: str = None,
    pdf_file: str = None,
    hub_url: str = CNN_POLL_HUB,
    limit: int = None,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Local file: skip all network steps ───────────────────────────────────
    if pdf_file:
        pdf_path = Path(pdf_file).expanduser().resolve()
        if not pdf_path.exists():
            print(f"ERROR: file not found: {pdf_path}")
            return
        date_slug = _extract_date_from_url(pdf_path.name) or pdf_path.stem
        fmt = detect_pdf_format(pdf_path)
        print(f"\nParsing local file: {pdf_path}  [format: {fmt}]")
        try:
            if fmt == "cnn_exit_poll":
                records = parse_cnn_exit_poll_pdf(pdf_path)
                meta = {
                    "source": "CNN/NEP Exit Poll",
                    "format": "cnn_exit_poll",
                    "pdf_file": pdf_path.name,
                    "source_file": str(pdf_path),
                }
                _save_exit_poll_results(date_slug, records, meta, OUTPUT_DIR)
                return
            else:
                meta = extract_poll_metadata(pdf_path)
                records = parse_topline_pdf(pdf_path)
        except Exception as e:
            print(f"  ERROR parsing PDF: {e}")
            return
        meta["source_file"] = str(pdf_path)
        save_results(date_slug, records, meta, OUTPUT_DIR)
        meta_path = OUTPUT_DIR / "metadata.json"
        with open(meta_path, "w") as f:
            import json as _json

            _json.dump([meta], f, indent=2)
        print(f"\nMetadata index → {meta_path}")
        print("Done.")
        return

    # ── URL or hub scrape ─────────────────────────────────────────────────────
    if pdf_url:
        polls = [
            {
                "title": "manual",
                "date_str": _extract_date_from_url(pdf_url),
                "poll_page_url": hub_url,
                "pdf_url": pdf_url,
            }
        ]
    else:
        polls = get_poll_links(hub_url, limit=limit)

    if not polls:
        return

    all_meta = []
    for poll in polls:
        print(f"\nProcessing: {poll['title']} [{poll['date_str']}]")
        date_slug = poll["date_str"]
        pdf_path = OUTPUT_DIR / f"{date_slug}_topline.pdf"

        # Download PDF
        try:
            save_pdf(poll["pdf_url"], pdf_path)
        except Exception as e:
            print(f"  ERROR downloading PDF: {e}")
            continue

        # Parse
        try:
            meta = extract_poll_metadata(pdf_path)
            records = parse_topline_pdf(pdf_path)
        except Exception as e:
            print(f"  ERROR parsing PDF: {e}")
            continue

        meta["poll_page_url"] = poll.get("poll_page_url")
        meta["pdf_url"] = poll.get("pdf_url")
        all_meta.append(meta)

        # Save
        save_results(date_slug, records, meta, OUTPUT_DIR)

    # Master metadata index
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nMetadata index → {meta_path}")
    print("Done.")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN/SSRS voter poll scraper")
    parser.add_argument(
        "--pdf",
        dest="pdf_file",
        help="Path to a local PDF file (e.g. printed from browser). Skips all network steps.",
    )
    parser.add_argument(
        "--pdf-url",
        help="Direct URL to a specific SSRS topline PDF (skips hub scrape)",
    )
    parser.add_argument(
        "--hub-url",
        default=CNN_POLL_HUB,
        help=f"CNN polling hub URL (default: {CNN_POLL_HUB})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of polls to scrape (default: all found)",
    )
    args = parser.parse_args()
    run(pdf_url=args.pdf_url, pdf_file=args.pdf_file, hub_url=args.hub_url, limit=args.limit)
