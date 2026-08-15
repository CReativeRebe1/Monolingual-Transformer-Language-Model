"""
Streaming Wikipedia dump reader: the fastest legitimate route to bulk Devanagari prose.

Why the dump beats every alternative:
  * The MediaWiki API caps `exlimit` at 1 for whole-article plain-text extracts (verified live --
    it returns "exlimit was too large for a whole article extracts request, lowered to 1"), so an
    API walk is one HTTP round-trip per article.
  * Crawling the HTML is worse still: one round-trip per article PLUS a politeness delay PLUS
    boilerplate extraction to undo the site chrome.
  * The dump is a single sequential read of one compressed file containing every article, and
    Wikimedia publishes it precisely so bulk consumers do not hammer the site. ~160k Hindi
    articles arrive in the time an API walk would take to fetch a few thousand.

Two things make this fast in practice, both implemented here:
  1. **Never decompress to disk.** `bz2.open` streams the archive, and `ElementTree.iterparse`
     yields pages as they are parsed, so peak memory stays flat regardless of dump size (the
     Hindi dump is ~1GB compressed, several GB expanded -- materializing it would be pointless
     I/O). Each parsed element is cleared immediately, which matters: without the `elem.clear()`
     below, iterparse silently retains the whole tree and memory grows to the full dump size.
  2. **Parallel markup stripping.** XML parsing is inherently sequential, but converting each
     article's wikitext to plain prose is CPU-bound and independent per article -- so pages are
     handed to a process pool via `imap_unordered` while the reader keeps streaming. On a 16-core
     machine this is the difference between one core at 100% and sixteen.

Wikitext is stripped with `mwparserfromhell` rather than regex: templates, nested links, tables,
and refs nest arbitrarily, and hand-rolled regex gets them wrong in ways that quietly poison a
training corpus with markup fragments.
"""
from __future__ import annotations

import bz2
import hashlib
import logging
import multiprocessing as mp
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("lma.wiki_dump")

DUMP_URL = "https://dumps.wikimedia.org/{wiki}/latest/{wiki}-latest-pages-articles.xml.bz2"
WIKI_CODE = {"hindi": "hiwiki", "nepali": "newiki"}

MIN_CHARS = 200          # matches lma/text/filters.py's min_chars floor
CHUNK = 200              # articles handed to a pool worker at a time

# Section headings whose bodies are reference/navigation apparatus, not prose. Dropping them
# stops "External links" URL lists and bare citation dumps from entering the corpus, where the
# heuristic filters would only catch some of them.
_DROP_SECTIONS = re.compile(
    r"^\s*(इन्हें भी देखें|सन्दर्भ|बाहरी कड़ियाँ|बाहरी कडियाँ|यो पनि हेर्नुहोस्|सन्दर्भ सामग्री|"
    r"बाह्य लिङ्कहरू|References|External links|See also|Further reading|Bibliography)\s*$",
    re.IGNORECASE,
)
_MULTI_NL = re.compile(r"\n{3,}")


def strip_wikitext(wikitext: str) -> str:
    """Converts raw wikitext to plain prose, dropping reference/navigation sections.

    Sections are filtered at the *wikitext* level, before strip_code() runs, deliberately:
    strip_code() removes the "==" heading markers, so a heading becomes an ordinary line of text
    and is no longer distinguishable from prose. Filtering afterwards therefore silently does
    nothing -- caught by test_strip_wikitext_drops_reference_sections, which found reference
    sections surviving into the output.
    """
    import mwparserfromhell  # imported inside the worker: not needed in the parent process

    try:
        code = mwparserfromhell.parse(wikitext)
        kept = []
        for section in code.get_sections(include_lead=True, flat=True):
            headings = section.filter_headings()
            if headings and _DROP_SECTIONS.match(str(headings[0].title).strip()):
                continue
            # Remove the heading line itself; keep only the section's prose.
            for h in headings:
                section.remove(h)
            kept.append(section.strip_code(normalize=True, collapse=True))
    except Exception:  # noqa: BLE001 - a single malformed page must not kill the pool
        return ""

    return _MULTI_NL.sub("\n\n", "\n".join(kept)).strip()


def _process(args: tuple[str, str, str]) -> dict | None:
    """Pool worker: (site, title, wikitext) -> shard record, or None if too short."""
    site, title, wikitext = args
    text = strip_wikitext(wikitext)
    if len(text) < MIN_CHARS:
        return None
    return {
        "url": f"https://{site}/wiki/{title.replace(' ', '_')}",
        "title": title,
        "text": text,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "host": site,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        # "manual", per course TA ruling (asked directly): a raw Wikipedia XML dump is not a
        # curated/pre-cleaned corpus file -- it is language-organized raw data that we ourselves
        # parse, strip wikitext markup from, and filter (this module + the shared cleaning
        # cascade), which the TA confirmed counts as manual collection when preprocessed by our
        # own script. This is a deliberate reversal of this module's earlier "downloaded" tag;
        # see PHASE1_LOG.md for the date/reasoning of the change.
        "source_type": "manual",
    }


def iter_pages(dump_path: Path, site: str):
    """Streams (site, title, wikitext) from a pages-articles dump, main namespace only.

    Redirects and non-article namespaces are skipped here rather than in the workers, so the
    pool is never handed work that would be discarded anyway.
    """
    with bz2.open(dump_path, "rb") as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag != "page":
                continue
            ns = elem.findtext("./{*}ns")
            redirect = elem.find("./{*}redirect")
            title = elem.findtext("./{*}title") or ""
            wikitext = elem.findtext("./{*}revision/{*}text") or ""
            # Free the element (and its now-dead siblings) before yielding: iterparse otherwise
            # keeps the entire document tree alive and memory grows to full dump size.
            elem.clear()
            if ns == "0" and redirect is None and wikitext:
                yield site, title, wikitext


def parse_dump(dump_path: Path, site: str, writer, workers: int | None = None,
               max_docs: int | None = None) -> int:
    """Parses a dump into shard records using a process pool. Returns documents written."""
    workers = workers or max(1, mp.cpu_count() - 1)
    written = 0
    logger.info("parsing %s with %d workers", dump_path.name, workers)
    with mp.Pool(workers) as pool:
        for record in pool.imap_unordered(_process, iter_pages(dump_path, site), chunksize=CHUNK):
            if record is None:
                continue
            writer.write(record)
            written += 1
            if written % 20_000 == 0:
                logger.info("%s: %d docs written", site, written)
            if max_docs and written >= max_docs:
                logger.info("%s: hit max_docs=%d", site, max_docs)
                break
    return written
