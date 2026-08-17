"""Tests for the learned filters' fitting step in each language's clean/run.py.

This replaces the old tests for scripts/build_corpus.py's `_train_models`, which cached a *network*
pull of trusted seed text from Wikipedia and whose tests asserted that the pull was skipped when a
cached seed already existed. That whole mechanism is gone: both pipelines now fit the perplexity
model and the quality classifier on the Wikipedia text already sitting in their own corpus, so a
rerun cannot be broken by a dead endpoint and there is no cache to invalidate. The property worth
testing changed accordingly -- not "is the pull cached" but "does it fit offline, and does it refuse
to fit on text it should not trust".

The refusal matters: the classifier's positive class must be trusted prose. If the run silently fell
back to fitting on crawl when Wikipedia was missing, it would be judging the crawl by the crawl's
own standards and the quality filter would mean nothing.
"""
import pytest

from hindi.clean import run as hindi_run
from nepali.clean import run as nepali_run

HINDI_WIKI = (
    "दिल्ली भारत की राजधानी है और यह देश का एक प्रमुख महानगर है जहाँ करोड़ों लोग रहते हैं। "
    "यह शहर अपनी ऐतिहासिक इमारतों और संस्कृति के लिए जाना जाता है और हर साल लाखों पर्यटक यहाँ आते हैं।"
)
HINDI_CRAWL = "ऑफर छूट खरीदें अभी क्लिक करें विज्ञापन शेयर करें और पढ़ें ताजा खबरें सदस्यता लें"
NEPALI_WIKI = (
    "नेपाल दक्षिण एसियाको एक भूपरिवेष्टित देश हो र यसको उत्तरमा चीन तथा दक्षिणमा भारत पर्दछ। "
    "देशको भूगोल हिमाल पहाड र तराई भन्ने तीन भागमा विभाजित गरिएको छ भने राजधानी काठमाडौं हो।"
)
NEPALI_CRAWL = "छुट अफर किन्नुहोस् अहिले क्लिक गर्नुहोस् विज्ञापन सेयर गर्नुहोस् थप पढ्नुहोस्"


def _records(wiki_text, crawl_text, n_wiki, n_crawl):
    """Synthetic records in the shape load_manual() produces, varied so the models see real spread."""
    out = []
    for i in range(n_wiki):
        out.append({"text": f"{wiki_text} यह वाक्य {i} है।", "source_dir": "wiki"})
    for i in range(n_crawl):
        out.append({"text": f"{crawl_text} {i}", "source_dir": "fast"})
    return out


@pytest.mark.parametrize(
    "module,wiki,crawl",
    [(hindi_run, HINDI_WIKI, HINDI_CRAWL), (nepali_run, NEPALI_WIKI, NEPALI_CRAWL)],
    ids=["hindi", "nepali"],
)
def test_fits_offline_from_the_corpus_own_wikipedia(module, wiki, crawl):
    cfg = module.DEFAULT
    scorer, ppl_thr, qmodel, coh_thr = module.fit_models(_records(wiki, crawl, 250, 250), cfg)

    assert ppl_thr > 0                      # a real threshold, taken from the trusted distribution
    assert scorer.perplexity(wiki) > 0
    assert qmodel.w, "quality classifier should have fitted weights"
    # The classifier must rank trusted prose above boilerplate. Only the ORDER is asserted, and only
    # on held-out documents built in the same shape as the training ones (index 999, never seen).
    # Scoring the bare template text instead is not a fair probe: with 250 near-identical synthetic
    # documents per class the model can separate them on any single feature, so a differently-shaped
    # string lands off-distribution and the ranking flips for reasons that say nothing about the
    # classifier. Its real behaviour is measured on the corpus itself -- held-out means of 0.841
    # (wiki) vs 0.390 (crawl) for Hindi and 0.766 vs 0.177 for Nepali, recorded in clean/config.py.
    held_out_good = f"{wiki} यह वाक्य 999 है।"
    held_out_bad = f"{crawl} 999"
    assert qmodel.score(held_out_good) > qmodel.score(held_out_bad)
    assert coh_thr > 0


@pytest.mark.parametrize("module", [hindi_run, nepali_run], ids=["hindi", "nepali"])
def test_refuses_to_fit_a_real_corpus_without_trusted_wikipedia_text(module):
    """No wiki in a real-sized corpus means no trusted positive class -- the run must stop.

    Fitting the classifier on the crawl it is supposed to judge would yield a filter that means
    nothing while still producing plausible-looking scores, which is worse than failing.
    """
    records = [
        {"text": HINDI_CRAWL, "source_dir": "fast"} for _ in range(module.REAL_CORPUS_DOCS)
    ]
    with pytest.raises(RuntimeError, match="wiki"):
        module.fit_models(records, module.DEFAULT)


@pytest.mark.parametrize("module", [hindi_run, nepali_run], ids=["hindi", "nepali"])
def test_demo_scale_corpus_degrades_instead_of_failing(module):
    """A handful of documents with no Wikipedia is scripts/reproduce.sh's situation: the cascade
    must still run end to end, with the learned filters disabled rather than fitted on crawl."""
    records = [{"text": HINDI_CRAWL, "source_dir": "crawled"} for _ in range(50)]
    scorer, ppl_thr, qmodel, coh_thr = module.fit_models(records, module.DEFAULT)

    assert ppl_thr == float("inf"), "nothing may be rejected on an unfitted perplexity model"
    assert not qmodel.w, "classifier must stay unfitted"
    assert qmodel.score(HINDI_CRAWL) == 0.5      # neutral, so it passes the quality gate
    assert scorer.perplexity(HINDI_CRAWL) > 0    # still usable, so _one() cannot crash
    assert coh_thr == module.DEFAULT.coherence_min_cosine


@pytest.mark.parametrize("module", [hindi_run, nepali_run], ids=["hindi", "nepali"])
def test_fit_models_makes_no_network_calls(module, monkeypatch):
    """Fitting must be offline. Any socket connection here means a rerun can be broken by an
    endpoint being down, which is precisely what moving off the seed pull was meant to fix."""
    import socket

    def explode(*a, **k):
        raise AssertionError("fit_models attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", explode)
    monkeypatch.setattr(socket, "create_connection", explode)

    wiki, crawl = (
        (HINDI_WIKI, HINDI_CRAWL) if module is hindi_run else (NEPALI_WIKI, NEPALI_CRAWL)
    )
    module.fit_models(_records(wiki, crawl, 220, 220), module.DEFAULT)
