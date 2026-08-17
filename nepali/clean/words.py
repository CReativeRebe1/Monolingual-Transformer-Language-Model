"""
All Nepali-language content the cascade needs: word lists, phrases, markers, blocklists.

The Nepali counterpart of hindi/clean/words.py. Same structure, entirely different content -- this
file plus config.py is where the two pipelines actually diverge, which is what makes them
independent rather than one pipeline with a language flag.

Nepali is the lower-resource side of this project, and that shapes the lists: they are built from
what appears in *this* corpus (85% fast-crawl, mostly news portals) rather than transliterated from
the Hindi lists, because the boilerplate of Nepali news sites is not the boilerplate of Hindi ones.
"""
from __future__ import annotations

# Closed-class function words. Nepali marks its verbs with छ/छन्/थियो where Hindi uses है/हैं/था, so
# the two stopword lists barely overlap despite the shared script.
STOPWORDS = frozenset(
    "र को का की मा छ छन् थियो थिए भएको लागि तर यदि पनि होइन ले हो यो त्यो अनि जुन "
    "गर्न गर्ने गरेको भने तथा हुन्छ नै सँग बाट देखि सम्म धेरै केही सबै अब भन्ने "
    "गरी हुन् थिएन पर्छ भन्दा उनले उहाँले".split()
)

# Words frequent in Nepali but rare in Hindi, and vice versa. fastText's lid.176 confuses the two
# languages -- same script, shared Sanskrit vocabulary -- and this project depends on the two
# corpora being genuinely independent, so lang.check() uses this as a tiebreaker.
MARKERS = frozenset("छ छन् गर्न भएको हुन्छ गरेको नेपाल भने तथा सम्म बाट पर्छ गर्नु उहाँले".split())
OTHER_LANG_MARKERS = frozenset("है हैं था थे किया गया करने उन्होंने रहा हुए इसके लिये".split())  # Hindi

# Boilerplate PHRASES: substring-matched, used both to reject (rules.check) and to excise
# (repair.repair, which deletes the whole line containing one).
BOILERPLATE_PHRASES = (
    "cookie policy",
    "all rights reserved",
    "subscribe to our newsletter",
    "जाभास्क्रिप्ट सक्षम गर्नुहोस्",
    "जावास्क्रिप्ट सक्षम गर्नुहोस्",
    "यो वेबसाइटले कुकिज प्रयोग गर्दछ",
    "सर्वाधिकार सुरक्षित",
    "प्रतिलिपि अधिकार",
    "एप डाउनलोड गर्नुहोस्",
)

# Boilerplate LINES: a line equal to one of these is furniture, and the document is usually fine
# once it is gone. Drawn from Nepali news-portal templates (Onlinekhabar, Setopati, Ratopati and
# similar), which is where the bulk of the Nepali crawl came from.
BOILERPLATE_LINES = frozenset(
    [
        "विज्ञापन",
        "थप पढ्नुहोस्",
        "अन्य समाचार",
        "सम्बन्धित समाचार",
        "ताजा समाचार",
        "ताजा अपडेट",
        "प्रतिक्रिया दिनुहोस्",
        "प्रतिक्रिया",
        "सेयर गर्नुहोस्",
        "शेयर गर्नुहोस्",
        "हामीलाई फलो गर्नुहोस्",
        "प्रकाशित मिति",
        "एजेन्सी",
        "रासस",
        "advertisement",
        "read more",
        "share this",
    ]
)

# Advertising / commerce vocabulary, for safety.topics()'s cluster ad-density scoring.
AD_KEYWORDS = frozenset(
    "छुट अफर बिक्री किन्नुहोस् किन्न मूल्य रुपैयाँ रुपिया नि:शुल्क निःशुल्क फ्री कुपन "
    "सदस्यता डाउनलोड इन्स्टल क्लिक लिङ्क सेयर फलो लाइक भाइबर व्हाट्सएप "
    "लगानी ऋण किस्ता बीमा लटरी जुवा बाजी क्यासिनो".split()
)

# Safety blocklist. Explicit tuple, NOT `"...".split()` -- splitting a string would shred the
# multi-word entries into common words. That exact mistake in the Hindi file put का ("of") and
# करें ("do") in the blocklist and silently deleted 5.4% of good documents before it was caught;
# the Nepali equivalents (गर्ने "to do", तरिका "method") would fail the same way.
SAFETY_BLOCKLIST = frozenset(
    (
        # explicit sexual content / slurs -- single tokens, word-matched
        "अश्लील", "नग्न", "वेश्या", "बलात्कारी", "रन्डी", "मुजी", "पुक्का",
        "xxx", "porn", "nude", "escort", "viagra",
        # self-harm instruction -- phrases, substring-matched. आत्महत्या ("suicide") alone appears
        # constantly in legitimate Nepali news reporting, so only instructional phrasing is blocked.
        "आत्महत्या कसरी गर्ने",
        "झुण्डिएर मर्ने तरिका",
        "विष खाएर मर्ने",
        "आत्महत्या गर्ने तरिका",
    )
)

# Mojibake and HTML-entity repairs. Identical to the Hindi table: these are encoding failures of
# the transport layer, not of the language, so the same corruptions appear in both corpora.
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
