"""
All Hindi-language content the cascade needs: word lists, phrases, markers, blocklists.

Separated from the logic on purpose. Every module in hindi/clean/ imports its language content
from here, so the Nepali copy differs from the Hindi copy in *this file plus config.py* and almost
nowhere else -- which is what makes the two pipelines genuinely independent rather than one
pipeline with a language flag.
"""
from __future__ import annotations

# Closed-class function words. Used as a cheap "is this connected prose?" signal -- the rules only
# check that a few are PRESENT, so this is intentionally a short hand-list, not a lexicon.
STOPWORDS = frozenset(
    "और के की का को है हैं में से पर एक यह वह हो था थी थे कि लिए भी नहीं ने तो जो अगर लेकिन "
    "उन्होंने अपने बाद साथ किया गया गई गए कर रहा रही रहे कोई कुछ सभी जब तक हुआ हुई".split()
)

# Words that are frequent in Hindi but rare in Nepali (and vice versa in the Nepali file). Used by
# lang.check() as a script-level tiebreaker: fastText's lid.176 confuses the two, since both are
# Devanagari and share much vocabulary, so confidence alone is not enough evidence.
MARKERS = frozenset("है हैं था थे किया गया करने वाले उन्होंने रहा हुए इसके लिये कोई".split())
OTHER_LANG_MARKERS = frozenset("छ छन् गर्न भएको हुन्छ तथा नेपाल गरेको पनि हो".split())  # Nepali

# Boilerplate PHRASES: substring match, used both to reject (rules.check) and to excise
# (repair.repair). Kept short and specific to avoid false positives on prose that merely mentions
# e.g. "subscribe" as a verb.
BOILERPLATE_PHRASES = (
    "cookie policy",
    "all rights reserved",
    "subscribe to our newsletter",
    "javascript को सक्षम करें",
    "यह वेबसाइट कुकीज़ का उपयोग करती है",
    # Observed directly in this corpus: amarujala.com's cookie/data-collection notice was
    # byte-identical on 745 of ~950 heuristic-surviving documents in an early build.
    "हम डाटा संग्रह टूल्स",
    "अमर उजाला एप इंस्टॉल कर रजिस्टर करें",
    "खबरें लगातार पढ़ने के लिए अमर उजाला एप डाउनलोड करें",
)

# Boilerplate LINES: matched after stripping, so a whole line equal to one of these is deleted by
# repair() rather than condemning the document. These are nav/furniture lines that trafilatura
# leaves behind, and a document is usually fine once they are gone.
BOILERPLATE_LINES = frozenset(
    [
        "विज्ञापन",
        "विज्ञापन जारी",
        "और पढ़ें",
        "आगे पढ़ें",
        "पूरी खबर पढ़ें",
        "यह भी पढ़ें",
        "ये भी पढ़ें",
        "संबंधित खबरें",
        "ताजा खबरें",
        "शेयर करें",
        "टिप्पणियाँ",
        "एक टिप्पणी भेजें",
        "हमें फॉलो करें",
        "सदस्यता लें",
        "advertisement",
        "read more",
        "share this",
        "trending news",
    ]
)

# Advertising / commerce vocabulary. Used by safety.topics() to score a semantic cluster's
# ad-density (Yi 2.1 downsamples ad-heavy clusters rather than deleting them, since an ad-heavy
# cluster still contains fluent language).
AD_KEYWORDS = frozenset(
    "ऑफर छूट डिस्काउंट सेल खरीदें खरीदारी कीमत मूल्य रुपये मुफ्त फ्री कूपन डील बुक "
    "सब्सक्राइब डाउनलोड इंस्टॉल क्लिक लिंक व्हाट्सएप टेलीग्राम फॉलो शेयर लाइक "
    "जुआ सट्टा लॉटरी कैसीनो लोन ईएमआई बीमा".split()
)

# Safety blocklist. A course-project-appropriate keyword filter, not a content-moderation system:
# it targets explicit sexual content, slurs, and self-harm instruction, where a keyword hit is
# strong evidence. Deliberately conservative -- safety_max_hits in config means a single incidental
# occurrence in a news report does not discard an otherwise-fine document, since over-filtering a
# low-resource corpus has its own cost (Lecture 3, slide 30).
# NOTE: written as an explicit tuple, NOT `"...".split()`. Building it by splitting a string
# shredded the multi-word entries into their component words, so the blocklist ended up containing
# का ("of"), करें ("do"), कैसे ("how") and तरीका ("method") -- among the most common words in Hindi.
# With safety_max_hits = 2, every ordinary news article hit two of them: the filter silently deleted
# 157 of 2,922 perfectly good documents (5.4%) in the first real run before this was caught.
# Multi-word entries must stay intact; `safety.hits()` substring-matches those and word-matches the
# single-token ones.
SAFETY_BLOCKLIST = frozenset(
    (
        # explicit sexual content / slurs -- single tokens, word-matched
        "अश्लील", "नग्न", "सेक्सी", "वेश्या", "रंडी", "बलात्कारी",
        "चूतिया", "मादरचोद", "भोसड़ी", "हरामी",
        "xxx", "porn", "nude", "escort", "viagra",
        # self-harm instruction -- phrases, substring-matched. The individual words (आत्महत्या
        # "suicide", फांसी "hanging") appear constantly in legitimate news reporting, so only the
        # instructional phrasing is blocked.
        "आत्महत्या कैसे करें",
        "फांसी लगाने का तरीका",
        "ज़हर खाकर मरने",
        "जहर खाकर मरने",
        "आत्महत्या करने का तरीका",
    )
)

# Mojibake repair pairs: (broken, correct). These are the specific corruptions actually seen in
# this corpus, not a generic table -- each was found by grepping raw shards for U+FFFD and for
# Latin-1-misdecoded Devanagari.
MOJIBAKE = (
    ("â€™", "'"),
    ("â€œ", '"'),
    ("â€\x9d", '"'),
    ("â€“", "-"),
    ("â€”", "-"),
    ("Â ", " "),
    ("Ã—", "x"),
    ("&nbsp;", " "),
    ("&amp;", "&"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&lt;", "<"),
    ("&gt;", ">"),
)
