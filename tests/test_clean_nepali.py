"""Tests for nepali/clean/ -- the Nepali cleaning cascade.

Separate from tests/test_clean_hindi.py on purpose, mirroring the pipelines. The tests that matter
most here are the ones that would FAIL if Nepali were quietly run on Hindi's settings:

* `test_chandrabindu_is_preserved` -- ँ is standard Nepali spelling in गाउँ, ठाउँ, हुँदा, बाँकी. Hindi
  folds it to anusvara; doing that here would rewrite गाउँ as गाउं, which is misspelt Nepali, and
  would fragment the tokenizer's vocabulary against every correct occurrence in the corpus.
* `test_lid_confidence_bar_is_lower_than_hindis` -- lid.176 labels 96% of real Nepali correctly with
  a median confidence of only 0.69-0.76, so Hindi's 0.65 bar rejects 17.9% of correctly-identified
  Nepali Wikipedia. Nepali runs at 0.50 (3.0%).
* `test_quality_bar_is_lower_than_hindis` -- at Hindi's 0.30 the classifier keeps just 19.2% of
  Nepali crawl, and crawl is 85% of the Nepali corpus. Nepali runs at 0.10 (49.4%).

Also a regression for the copy-paste hazard that produced this package: the seven logic modules were
generated from Hindi's by text substitution, which turned the Hindi-vs-Nepali marker vote into a
comparison of Nepali against itself. `test_hindi_document_is_rejected_from_the_nepali_corpus` is what
catches that.
"""
from hindi.clean.config import DEFAULT as HINDI_DEFAULT
from nepali.clean import dedup, lang, repair, rules, safety, score
from nepali.clean.config import DEFAULT
from nepali.clean.norm import norm

PARA_1 = (
    "नेपाल दक्षिण एसियाको एक भूपरिवेष्टित देश हो र यसको उत्तरमा चीन तथा दक्षिण, पूर्व र पश्चिममा "
    "भारत पर्दछ। देशको भूगोल हिमाल, पहाड र तराई भन्ने तीन भागमा विभाजित गरिएको छ।"
)
PARA_2 = (
    "काठमाडौं उपत्यकामा बसोबास गर्ने जनसंख्या बढ्दै गएको छ र सहरी पूर्वाधारमा दबाब परेको छ। "
    "स्थानीय सरकारले सार्वजनिक यातायात सुधार गर्ने योजना अघि सारेको छ भने बजेट पनि छुट्याइएको छ।"
)
PARA_3 = (
    "कृषि क्षेत्रले अर्थतन्त्रमा महत्त्वपूर्ण योगदान दिँदै आएको छ तथा धेरै परिवार यसमै निर्भर छन्। "
    "सरकारले सिँचाइ र मल आपूर्तिमा सहयोग गर्ने प्रतिबद्धता जनाएको छ भन्ने कृषकहरूले बताएका छन्।"
)
ARTICLE = "\n\n".join([PARA_1, PARA_2, PARA_3])
HINDI_DOC = (
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं। "
    "यह शहर अपनी ऐतिहासिक इमारतों के लिए जाना जाता है और उन्होंने कहा कि यह किया गया था।"
)


# ------------------------------------------------------------------ norm: the key divergence

def test_chandrabindu_is_preserved():
    """The divergence from Hindi. Folding these would produce misspelt Nepali."""
    for word in ("गाउँ", "ठाउँ", "हुँदा", "बाँकी", "भाँडा"):
        assert norm(word) == word, f"{word} must survive normalisation unchanged"


def test_chandrabindu_is_not_turned_into_anusvara():
    assert "ं" not in norm("गाउँमा बसोबास")


def test_norm_is_idempotent():
    once = norm("नेपालीी  भाषा हो ि गलत क्")
    assert norm(once) == once


def test_repeated_chandrabindu_is_still_collapsed():
    """A doubled mark is encoding corruption in either language -- preserving ँ is not the same as
    accepting a corrupted sequence of them."""
    assert norm("गाउँँ") == "गाउँ"


# ------------------------------------------------------------------ lang

def test_nepali_article_passes_language_check():
    assert lang.check(ARTICLE) == (True, None)


def test_hindi_document_is_rejected_from_the_nepali_corpus():
    """Regression for the mechanical hindi->nepali copy, which made the marker vote compare Nepali
    against itself -- the gate looked present but tested nothing."""
    ok, why = lang.check(HINDI_DOC)
    assert ok is False
    assert why == "hindi_markers_dominant"


def test_marker_vote_returns_nepali_first():
    ne, hi = lang.marker_vote(ARTICLE)
    assert ne > hi


def test_lid_confidence_bar_is_lower_than_hindis():
    assert DEFAULT.lid_min_confidence < HINDI_DEFAULT.lid_min_confidence


# ------------------------------------------------------------------ rules

def test_well_formed_article_passes_the_rules():
    verdict = rules.check(ARTICLE)
    assert verdict.passed, verdict.failed


def test_nepali_stopwords_are_matched_not_hindis():
    """छ/छन्/भएको mark Nepali verbs where Hindi uses है/हैं/था, so the two lists barely overlap."""
    assert "insufficient_stopword_evidence" not in rules.check(ARTICLE).failed


def test_keyword_spam_is_rejected():
    failed = rules.check("छुट छुट छुट अफर अफर अफर किन्नुहोस् किन्नुहोस् अहिले क्लिक गर्नुहोस् " * 20).failed
    assert "top_word_fraction" in failed


def test_repetitive_document_removal_catches_duplicate_paragraphs():
    doubled = "\n\n".join([PARA_1, PARA_2, PARA_1, PARA_2])
    assert "dup_paragraph_fraction" in rules.repetitive(doubled).failed


# ------------------------------------------------------------------ repair

def test_repair_strips_nested_json_anywhere_in_the_document():
    doc = 'Live\nसमाचार शीर्षक\n{"_id":"6a7e","slug":"x","meta":{"a":2}}\n' + ARTICLE
    out = repair.repair(doc)
    assert "json_blob" in out.actions
    assert "_id" not in out.text
    assert PARA_2 in out.text


def test_repair_removes_nepali_boilerplate_lines():
    doc = "विज्ञापन\n" + ARTICLE + "\nथप पढ्नुहोस्\nसेयर गर्नुहोस्"
    out = repair.repair(doc)
    assert "थप पढ्नुहोस्" not in out.text
    assert "सेयर गर्नुहोस्" not in out.text
    assert rules.check(out.text).passed


def test_repair_removes_nepali_boilerplate_phrase_lines():
    doc = ARTICLE + "\nसर्वाधिकार सुरक्षित - यो वेबसाइटको सामग्री प्रयोग गर्न अनुमति आवश्यक छ।"
    assert "सर्वाधिकार सुरक्षित" not in repair.repair(doc).text


def test_repair_leaves_a_mostly_boilerplate_document_alone():
    out = repair.repair("विज्ञापन\nथप पढ्नुहोस्\nसेयर गर्नुहोस्\nप्रतिक्रिया\n")
    assert out.actions == ["skipped_mostly_boilerplate"]


# ------------------------------------------------------------------ score

def test_coherence_is_nonzero_for_a_coherent_document():
    assert score.coherence_score(ARTICLE) > 0


def test_short_blurb_bypasses_the_coherence_check_and_passes():
    ok, value = score.coherent("काठमाडौंमा आज पानी परेको छ। मौसम चिसो भयो। मानिसहरू खुसी छन्।")
    assert ok is True
    assert value is None


def test_quality_bar_is_lower_than_hindis():
    """Hindi's 0.30 keeps only 19.2% of Nepali crawl, which is 85% of the Nepali corpus."""
    assert DEFAULT.quality_min_score < HINDI_DEFAULT.quality_min_score


def test_unfitted_quality_model_is_neutral():
    assert score.QualityModel().score(ARTICLE) == 0.5


# ------------------------------------------------------------------ safety

def test_ordinary_prose_is_not_flagged_as_unsafe():
    """The Hindi blocklist bug (common words left in it by `.split()`) would fail identically here:
    गर्ने ("to do") and तरिका ("method") are as common in Nepali as करें and तरीका are in Hindi."""
    for text in (ARTICLE, PARA_1, PARA_2, PARA_3):
        ok, _, n_hits = safety.check(text)
        assert ok is True and n_hits == 0


def test_news_report_mentioning_suicide_is_kept():
    report = PARA_1 + " प्रहरीले आत्महत्या भएको जनाएको छ र अनुसन्धान जारी छ।"
    assert safety.check(report)[0] is True


def test_a_single_self_harm_instruction_is_enough_to_block():
    ok, _, n_hits = safety.check(PARA_1 + " आत्महत्या कसरी गर्ने, यो तरिका हो।")
    assert n_hits >= 1 and ok is False


def test_pii_is_redacted():
    _, redacted, _ = safety.check(ARTICLE + " सम्पर्क: test@example.com 9812345678")
    assert "test@example.com" not in redacted and "[EMAIL]" in redacted


# ------------------------------------------------------------------ dedup

def test_paragraph_shared_across_documents_is_removed():
    footer = "सर्वाधिकार सुरक्षित। यो वेबसाइटको सामग्री अनुमति बिना प्रयोग गर्न पाइने छैन।"
    docs = ["\n\n".join([p, footer]) for p in (PARA_1, PARA_2, PARA_3)]
    out, removed, touched = dedup.dedup_paras(docs)
    assert removed == 3 and touched == 3
    assert all(footer not in d for d in out)


def test_exact_and_near_duplicates_are_removed():
    reworded = ARTICLE.replace("महत्त्वपूर्ण योगदान", "उल्लेखनीय योगदान")
    texts = [ARTICLE, ARTICLE, reworded, HINDI_DOC]
    keep, n_exact = dedup.dedup_exact([str(i) for i in range(len(texts))], texts)
    assert n_exact == 1
    survivors = [texts[i] for i in keep]
    keep2, n_near, _ = dedup.dedup_near([str(i) for i in range(len(survivors))], survivors)
    assert n_near == 1
    assert len(keep2) == 2


def test_key_normalises_before_hashing():
    assert dedup.key(ARTICLE) == dedup.key("  " + ARTICLE.replace(" ", "  ") + "\n")


# ------------------------------------------------------------------ independence from Hindi's config

def test_the_two_configs_are_separate_objects():
    """Same field names, independent values -- if these were shared, neither language could be tuned
    without moving the other."""
    assert DEFAULT is not HINDI_DEFAULT
    assert (DEFAULT.lid_min_confidence, DEFAULT.quality_min_score) != (
        HINDI_DEFAULT.lid_min_confidence, HINDI_DEFAULT.quality_min_score
    )


def test_manual_fraction_target_clears_the_assignment_floor():
    assert DEFAULT.target_manual_fraction >= 0.20
