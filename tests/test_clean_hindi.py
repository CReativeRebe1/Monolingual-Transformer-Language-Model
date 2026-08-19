"""Tests for hindi/clean/ -- the Hindi cleaning cascade.

Deliberately a separate file from tests/test_clean_nepali.py, mirroring the pipelines themselves.
The point of splitting the pipelines is that the two languages may disagree about thresholds and
normalisation, and a shared test file would quietly pressure them back into agreement.

Several tests here are regressions for bugs that real corpus data exposed and that reading the code
did not:

* `test_ordinary_prose_is_not_flagged_as_unsafe` -- the blocklist was built with `"...".split()`,
  which shredded its multi-word entries into common words (का "of", करें "do", कैसे "how"). Every
  Hindi article hit two of them, and the filter silently deleted 5.4% of good documents.
* `test_coherence_is_nonzero_for_a_coherent_document` -- textbook IDF, log(N/(1+df)), is exactly
  zero for a term appearing in 2 of 3 paragraphs, i.e. it zeroes out the shared terms the metric
  exists to measure. Every document scored 0.0000.
* `test_repair_strips_nested_json_anywhere_in_the_document` -- the JSON stripper only handled a blob
  at the very start and could not match nested braces, so 1,285 documents were rejected for symbol
  density caused by metadata the pipeline believed it had removed.
* `test_repair_valve_is_measured_in_characters` -- a line-based safety valve read a normal article
  (one long prose line plus three furniture lines) as 60% boilerplate and refused to repair it.
"""
import pytest

from hindi.clean import dedup, lang, repair, rules, safety, score
from hindi.clean.config import DEFAULT
from hindi.clean.norm import norm

PARA_1 = (
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं। "
    "यह शहर अपनी ऐतिहासिक इमारतों के लिए जाना जाता है और हर साल लाखों पर्यटक यहाँ आते हैं।"
)
PARA_2 = (
    "दिल्ली सरकार ने प्रदूषण कम करने के लिए कई कदम उठाए हैं और मेट्रो का विस्तार किया गया है। "
    "विशेषज्ञों का मानना है कि सार्वजनिक परिवहन बढ़ाने से हवा की गुणवत्ता में सुधार होगा।"
)
PARA_3 = (
    "दिल्ली नगर निगम ने कहा कि निर्माण कार्य पर निगरानी रखी जाएगी और नियम तोड़ने वालों पर "
    "जुर्माना लगाया जाएगा। अदालत ने भी इस मामले में सख्त रुख अपनाया है।"
)
ARTICLE = "\n\n".join([PARA_1, PARA_2, PARA_3])
NEPALI_DOC = (
    "नेपालमा दशैं सबैभन्दा ठूलो चाड मानिन्छ र यस अवसरमा परिवारका सदस्यहरू भेटघाट गर्छन् भने "
    "यो चाड पन्ध्र दिनसम्म मनाइन्छ तथा विदेशमा बसोबास गर्ने नेपालीहरू पनि घर फर्कने गरेको छ।"
)


# ------------------------------------------------------------------ norm

def test_norm_is_idempotent():
    messy = "हिन्दीी  भाषाँ है ि गलत क्"
    once = norm(messy)
    assert norm(once) == once


def test_hindi_folds_chandrabindu_to_anusvara():
    """Hindi's documented divergence: modern Hindi treats the two as interchangeable, and folding
    cuts vocabulary fragmentation. Nepali must NOT do this -- see the Nepali test."""
    assert norm("भाषाँ") == "भाषां"


def test_nukta_is_preserved():
    """ज़/ज distinguish Perso-Arabic loanwords that are frequent in Hindi."""
    assert "ज़" in norm("ज़रूरी काम")


def test_repeated_matra_and_orphan_matra_are_repaired():
    assert norm("हिन्दीी") == "हिन्दी"           # doubled matra: encoding corruption
    assert "ि" not in norm("है ि गलत").split()[1]  # a matra cannot begin a word


# ------------------------------------------------------------------ lang

def test_hindi_article_passes_language_check():
    assert lang.check(ARTICLE) == (True, None)


def test_nepali_document_is_rejected_from_the_hindi_corpus():
    """The two corpora must stay independent; fastText alone confuses these languages."""
    ok, why = lang.check(NEPALI_DOC)
    assert ok is False
    assert why == "nepali_markers_dominant"


def test_latin_text_is_rejected_on_script_ratio():
    ok, why = lang.check("This is an English document about the weather in Delhi. " * 6)
    assert ok is False
    assert why == "low_devanagari_ratio"


# ------------------------------------------------------------------ rules

def test_well_formed_article_passes_the_rules():
    verdict = rules.check(ARTICLE)
    assert verdict.passed, verdict.failed


def test_keyword_spam_is_rejected():
    failed = rules.check("ऑफर ऑफर ऑफर छूट छूट छूट खरीदें खरीदें अभी क्लिक करें " * 20).failed
    assert "top_word_fraction" in failed


def test_repetitive_document_removal_catches_duplicate_paragraphs():
    doubled = "\n\n".join([PARA_1, PARA_2, PARA_1, PARA_2])
    assert "dup_paragraph_fraction" in rules.repetitive(doubled).failed


def test_all_violated_rules_are_reported_not_just_the_first():
    """The Phase-1 funnel needs per-rule counts, so the cascade must not short-circuit."""
    assert len(rules.check("{}").failed) > 1


# ------------------------------------------------------------------ repair

def test_repair_strips_nested_json_anywhere_in_the_document():
    doc = 'Live\nचमोली सुरंग हादसा\n{"_id":"6a7e","slug":"x","meta":{"a":2}}\n' + ARTICLE
    out = repair.repair(doc)
    assert "json_blob" in out.actions
    assert "_id" not in out.text
    assert PARA_2 in out.text          # the article itself survives


def test_repair_keeps_prose_that_merely_contains_braces():
    doc = ARTICLE + "\n{यह कोष्ठक में गद्य है}"
    assert "कोष्ठक" in repair.repair(doc).text


def test_repair_removes_boilerplate_lines_and_phrase_lines():
    doc = "विज्ञापन\n" + ARTICLE + "\nयह भी पढ़ें\nहम डाटा संग्रह टूल्स का उपयोग करते हैं"
    out = repair.repair(doc)
    assert "विज्ञापन" not in out.text
    assert "यह भी पढ़ें" not in out.text
    assert "हम डाटा संग्रह टूल्स" not in out.text
    assert rules.check(out.text).passed


def test_repair_valve_is_measured_in_characters():
    """One long prose line plus three furniture lines is 95% prose by volume, and must be repaired.
    A line-count valve called it 60% boilerplate and returned the document untouched."""
    doc = "विज्ञापन\n" + PARA_1 + PARA_2 + "\nयह भी पढ़ें\nadvertisement"
    out = repair.repair(doc)
    assert "skipped_mostly_boilerplate" not in out.actions
    assert out.lines_dropped == 3


def test_repair_leaves_a_mostly_boilerplate_document_alone():
    """Past the valve the document should fail the filters honestly, so the funnel blames the right
    stage instead of showing a suspiciously well-repaired document sail through."""
    out = repair.repair("विज्ञापन\nयह भी पढ़ें\nadvertisement\nशेयर करें\n")
    assert out.actions == ["skipped_mostly_boilerplate"]


def test_repair_fixes_mojibake_and_entities():
    out = repair.repair(ARTICLE + " â€™ &amp; &nbsp;")
    assert "mojibake" in out.actions
    assert "â€™" not in out.text and "&amp;" not in out.text


# ------------------------------------------------------------------ score

def test_coherence_is_nonzero_for_a_coherent_document():
    """Regression: smoothed IDF. The textbook form returned 0.0000 for every document."""
    assert score.coherence_score(ARTICLE) > 0


def test_coherent_article_scores_above_a_stitched_one():
    stitched = "\n\n".join([
        PARA_1,
        "क्रिकेट मैच में भारत ने आज शानदार प्रदर्शन करते हुए जीत दर्ज की और सभी दर्शक खुश हुए।",
        "शेयर बाजार में आज भारी गिरावट देखी गई और निवेशकों को नुकसान उठाना पड़ा।",
    ])
    assert score.coherence_score(ARTICLE) > score.coherence_score(stitched)


def test_short_blurb_bypasses_the_coherence_check_and_passes():
    """On a low-resource corpus an unmeasurable score must never become a rejection."""
    ok, value = score.coherent("दिल्ली में आज बारिश हुई। मौसम ठंडा रहा। लोग खुश हैं।")
    assert ok is True
    assert value is None           # None marks "bypassed", not "scored zero"


def test_unfitted_quality_model_is_neutral():
    assert score.QualityModel().score(ARTICLE) == 0.5


# ------------------------------------------------------------------ safety

def test_ordinary_prose_is_not_flagged_as_unsafe():
    """Regression for the blocklist that contained का, करें and कैसे and deleted 5.4% of the corpus."""
    for text in (ARTICLE, PARA_1, PARA_2, PARA_3):
        ok, _, n_hits = safety.check(text)
        assert ok is True and n_hits == 0


def test_news_report_mentioning_suicide_is_kept():
    """आत्महत्या appears constantly in legitimate reporting; only instructional phrasing is blocked."""
    report = PARA_1 + " पुलिस ने बताया कि व्यक्ति ने आत्महत्या की और जाँच जारी है।"
    assert safety.check(report)[0] is True


def test_a_single_self_harm_instruction_is_enough_to_block():
    """Instructional phrases have no innocent reading, so one hit rejects -- unlike single tokens,
    which get a tolerance. Sharing one budget let a document with TWO instructions read as safe."""
    ok, _, n_hits = safety.check(PARA_1 + " आत्महत्या कैसे करें, यह तरीका है।")
    assert n_hits >= 1 and ok is False


def test_two_instructional_phrases_are_blocked():
    ok, _, n_hits = safety.check(PARA_1 + " आत्महत्या कैसे करें और फांसी लगाने का तरीका यह है।")
    assert n_hits >= 2 and ok is False


def test_single_incidental_slur_does_not_discard_the_document():
    """A report quoting abuse is legitimate prose; the tolerance exists for exactly this."""
    assert safety.check(PARA_1 + " आरोपी ने अश्लील टिप्पणी की थी।")[0] is True


def test_pii_is_redacted():
    _, redacted, _ = safety.check(ARTICLE + " संपर्क: test@example.com 9876543210")
    assert "test@example.com" not in redacted and "[EMAIL]" in redacted
    assert "9876543210" not in redacted


def test_ad_clusters_are_downsampled_not_deleted(monkeypatch):
    """Yi 2.1 downsamples ad-heavy clusters: they still contain fluent Hindi."""
    ads = ["ऑफर छूट खरीदें कीमत रुपये मुफ्त कूपन डील " * 3] * 40
    news = [ARTICLE] * 40
    result = safety.cluster(ads + news)
    keep, result = safety.topics(ads + news, result)
    if result.ad_clusters:                       # scikit-learn present and a cluster stood out
        assert 0 < len(keep) < len(ads) + len(news)


# ------------------------------------------------------------------ dedup

def test_paragraph_shared_across_documents_is_removed():
    footer = "सर्वाधिकार सुरक्षित। इस वेबसाइट की सामग्री का उपयोग बिना अनुमति नहीं किया जा सकता है।"
    docs = ["\n\n".join([p, footer]) for p in (PARA_1, PARA_2, PARA_3)]
    out, removed, touched = dedup.dedup_paras(docs)
    assert removed == 3 and touched == 3
    assert all(footer not in d for d in out)


def test_short_repeated_paragraph_is_exempt():
    """A date stamp or two-word heading legitimately repeats; deleting those shreds real articles."""
    docs = ["\n\n".join([p, "१५ अगस्त"]) for p in (PARA_1, PARA_2, PARA_3)]
    out, removed, _ = dedup.dedup_paras(docs)
    assert removed == 0
    assert all("१५ अगस्त" in d for d in out)


def test_exact_and_near_duplicates_are_removed():
    reworded = ARTICLE.replace("बहुत बड़ा शहर", "काफी बड़ा शहर")
    texts = [ARTICLE, ARTICLE, reworded, NEPALI_DOC]
    keep, n_exact = dedup.dedup_exact([str(i) for i in range(len(texts))], texts)
    assert n_exact == 1                          # the verbatim copy
    survivors = [texts[i] for i in keep]
    keep2, n_near, _ = dedup.dedup_near([str(i) for i in range(len(survivors))], survivors)
    assert n_near == 1                           # the reworded copy
    assert len(keep2) == 2


def test_key_normalises_before_hashing():
    assert dedup.key(ARTICLE) == dedup.key("  " + ARTICLE.replace(" ", "  ") + "\n")


# ------------------------------------------------------------------ config

@pytest.mark.parametrize("field", ["min_words", "near_dup_threshold", "target_manual_fraction"])
def test_config_exposes_the_fields_the_pipeline_reads(field):
    assert hasattr(DEFAULT, field)


def test_manual_fraction_target_clears_the_assignment_floor():
    assert DEFAULT.target_manual_fraction >= 0.20


# ------------------------------------------------------------------ split writing

def test_split_writer_keeps_one_line_per_document(tmp_path):
    """Regression: `text.replace("\n", " ")` left bare carriage returns intact, and Python's
    universal-newline reading splits on those too. 21,823 real Hindi documents were written as one
    line and read back as two, desynchronising train.txt from train.manifest.jsonl and misattributing
    the source_type of every document after the first one."""
    from hindi.clean.run import SplitWriter

    writer = SplitWriter(tmp_path / "train")
    nasty = [
        {"text": "पहला\rदस्तावेज़", "content_sha256": "a", "source_type": "manual"},
        {"text": "दूसरा\nदस्तावेज़", "content_sha256": "b", "source_type": "downloaded"},
        {"text": "तीसरा\u2028दस्तावेज़\x0bचौथा", "content_sha256": "c", "source_type": "manual"},
    ]
    writer.write_all(nasty)
    writer.close()

    text_lines = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()
    manifest_lines = (tmp_path / "train.manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(text_lines) == len(manifest_lines) == 3, "one line per document, whatever it contains"
    # Reading the way every downstream consumer does must give the same alignment.
    with open(tmp_path / "train.txt", encoding="utf-8") as f:
        assert sum(1 for _ in f) == 3
