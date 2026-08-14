"""
Writes small, illustrative JSONL.gz shards (in the exact format lma/crawl/async_crawler.py's
ShardWriter produces) for scripts/reproduce.sh -- a handful of real, hand-written Devanagari
paragraphs per language, NOT a claim about the real training corpus (that comes from the actual
crawler/OCR/downloaded-corpus pipeline; see PHASE1_LOG.md). This exists purely so a fresh clone
of this repository can run the full Phase-1 pipeline end to end in minutes without waiting on a
live crawl or a GPU, per the assignment's "test your code thoroughly" and reproducibility
requirements.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

# A mix of genuinely different topics/styles (news-like, encyclopedic, narrative) so the
# pipeline's filters (heuristic length/symbol checks, language ID, dedup) all have something
# realistic to do rather than trivially passing/failing everything the same way. Includes one
# near-duplicate pair per language (paraphrased, not copy-pasted) so the dedup stage has a real
# near-duplicate cluster to find, and one too-short/low-content document so the heuristic filter
# has something genuine to reject.
# FilterConfig defaults (lma/text/filters.py) require >=200 chars and >=40 words to pass
# heuristic filtering -- these paragraphs are written to comfortably clear both thresholds so
# the demo actually shows documents surviving to the dedup/split/tokenizer stages, not just
# exercising the rejection path.
HINDI_DOCS = [
    ("https://example.com/hi/1", "भारत की राजधानी दिल्ली में आज सुबह हल्की बारिश हुई, जिससे तापमान में कुछ गिरावट दर्ज की गई। मौसम विभाग ने अगले तीन दिनों तक आसमान में बादल छाए रहने का पूर्वानुमान जताया है। स्थानीय प्रशासन ने जलभराव की आशंका वाले क्षेत्रों में विशेष निगरानी दल तैनात किए हैं और नागरिकों से सतर्क रहने की अपील की है। शहर के कई हिस्सों में यातायात प्रभावित हुआ, विशेषकर पुराने शहर के संकरे इलाकों में।"),
    ("https://example.com/hi/2", "दिल्ली, भारत की राजधानी, में आज सवेरे थोड़ी बारिश हुई और तापमान में मामूली कमी आई। मौसम विभाग के अनुसार अगले तीन दिन आसमान में बादल छाए रहेंगे। नगर निगम ने जलभराव-प्रवण इलाकों में अतिरिक्त टीमें भेजी हैं और लोगों को सावधान रहने को कहा है। शहर के कई भाग में सड़कों पर वाहनों की गति धीमी रही, खासकर पुराने और संकरे मोहल्लों में।"),
    ("https://example.com/hi/3", "महात्मा गांधी का जन्म 2 अक्टूबर 1869 को पोरबंदर, गुजरात में हुआ था। उन्होंने भारत के स्वतंत्रता संग्राम में अहिंसा और सत्याग्रह के सिद्धांतों का प्रयोग किया, जो बाद में विश्व भर के आंदोलनों के लिए प्रेरणा बने। दक्षिण अफ्रीका में बिताए वर्षों के दौरान उन्होंने पहली बार सविनय अवज्ञा के तरीकों को आजमाया, और स्वदेश लौटने के बाद इन्हीं सिद्धांतों को भारतीय स्वतंत्रता आंदोलन का आधार बनाया। उनकी विरासत आज भी दुनिया भर के शांतिपूर्ण आंदोलनों को प्रभावित करती है।"),
    ("https://example.com/hi/4", "गंगा नदी हिमालय से निकलकर उत्तर भारत के मैदानी इलाकों से होते हुए बंगाल की खाड़ी में जाकर मिलती है। यह भारत की सबसे लंबी नदियों में से एक है और करोड़ों लोगों के जीवन का आधार है। इसके किनारे बसे शहरों में कृषि, उद्योग और धार्मिक पर्यटन तीनों फले-फूले हैं, लेकिन बढ़ते प्रदूषण ने नदी के पारिस्थितिकी तंत्र के लिए गंभीर चुनौतियाँ भी पैदा की हैं। सरकार द्वारा नदी की सफाई के लिए कई दीर्घकालिक योजनाएं चलाई जा रही हैं।"),
    ("https://example.com/hi/5", "क्रिकेट भारत में सबसे लोकप्रिय खेल है। भारतीय क्रिकेट टीम ने कई अंतरराष्ट्रीय टूर्नामेंट जीते हैं और देश भर में युवाओं के बीच इस खेल के प्रति गहरा उत्साह देखा जाता है। गली-मोहल्लों से लेकर बड़े स्टेडियमों तक, हर उम्र के लोग इस खेल से जुड़ाव महसूस करते हैं। घरेलू लीग टूर्नामेंटों ने भी नए खिलाड़ियों को राष्ट्रीय स्तर पर पहचान दिलाने में महत्वपूर्ण भूमिका निभाई है।"),
    ("https://example.com/hi/6", "राजस्थान अपने रेगिस्तानी परिदृश्य, ऐतिहासिक किलों और रंगीन संस्कृति के लिए प्रसिद्ध है। जयपुर, जोधपुर और उदयपुर जैसे शहर हर साल लाखों पर्यटकों को आकर्षित करते हैं। यहाँ की लोक कला, संगीत और पारंपरिक वेशभूषा राज्य की समृद्ध सांस्कृतिक विरासत को दर्शाती है। पर्यटन उद्योग राज्य की अर्थव्यवस्था में एक महत्वपूर्ण भूमिका निभाता है और हजारों परिवारों की आजीविका का साधन है।"),
    ("https://example.com/hi/7", "x x x x . . . -- -- -- 123 456 789 !!! ??? ..."),  # heuristic-filter reject: low content
]

NEPALI_DOCS = [
    ("https://example.com/ne/1", "काठमाडौं उपत्यकामा आज बिहान हल्का वर्षा भएको छ, जसका कारण तापक्रममा केही कमी आएको छ। मौसमविद्हरूका अनुसार आगामी तीन दिनसम्म आकाश बादलले ढाकिने प्रक्षेपण गरिएको छ। स्थानीय निकायले डुबान हुन सक्ने क्षेत्रहरूमा विशेष निगरानी टोली खटाएको छ र नागरिकहरूलाई सतर्क रहन आग्रह गरेको छ। सहरका धेरै भागमा यातायात प्रभावित भएको छ, विशेष गरी साँघुरा गल्लीहरूमा।"),
    ("https://example.com/ne/2", "नेपालको राजधानी काठमाडौंमा आज बिहानै अलिकति पानी पर्‍यो र तापक्रम अलि घट्यो। मौसम विभागका अनुसार आगामी तीन दिन आकाशमा बादल लागिरहने छ। नगरपालिकाले जलजमाव हुनसक्ने ठाउँहरूमा थप टोली पठाएको छ र बासिन्दालाई सावधान रहन भनेको छ। सहरका विभिन्न ठाउँमा सवारीसाधनको गति सुस्त भएको छ, खासगरी पुराना र साँघुरा टोलहरूमा।"),
    ("https://example.com/ne/3", "सगरमाथा विश्वको सबैभन्दा अग्लो हिमशिखर हो, जुन नेपाल र चीनको सीमामा अवस्थित छ। हरेक वर्ष सयौं पर्वतारोहीहरू यसलाई आरोहण गर्ने प्रयास गर्छन्। स्थानीय शेर्पा समुदायले यस अभियानमा महत्त्वपूर्ण भूमिका खेल्दै आएको छ र पर्यटन यस क्षेत्रको मुख्य आर्थिक आधार बनेको छ। जलवायु परिवर्तनका कारण हिमशिखरका हिउँ पग्लिने क्रम बढ्दै गएको वैज्ञानिकहरूले जनाएका छन्।"),
    ("https://example.com/ne/4", "नेपाली साहित्यमा भानुभक्त आचार्यलाई आदिकवि मानिन्छ। उनले रामायणलाई नेपाली भाषामा अनुवाद गरी नेपाली साहित्यको जगमा महत्त्वपूर्ण योगदान पुर्‍याएका थिए। उनको यो कृतिले पछिका पुस्ताका लेखक र कविहरूलाई ठूलो प्रेरणा दिएको छ। आजसम्म पनि उनको जन्मजयन्ती नेपालभर साहित्यिक कार्यक्रमहरूका साथ मनाइन्छ।"),
    ("https://example.com/ne/5", "नेपालमा दशैं सबैभन्दा ठूलो र महत्त्वपूर्ण चाड मानिन्छ। यस अवसरमा परिवारका सदस्यहरू एकआपसमा भेटघाट गर्छन् र ठूला-बडाबाट टीका र आशीर्वाद ग्रहण गर्छन्। यो चाड पन्ध्र दिनसम्म मनाइन्छ र यस अवधिमा घर-घरमा उत्सवको माहोल हुन्छ। विदेशमा बसोबास गर्ने नेपालीहरू पनि यस अवसरमा आफ्नो घर फर्कने प्रयास गर्छन्।"),
    ("https://example.com/ne/6", "पोखरा आफ्नो प्राकृतिक सुन्दरता, फेवा ताल र हिमाली दृश्यका लागि प्रख्यात छ। यो सहर नेपालको प्रमुख पर्यटकीय गन्तव्यहरू मध्ये एक मानिन्छ। यहाँबाट अन्नपूर्ण र माछापुच्छ्रे हिमशृंखलाको मनोरम दृश्य देख्न सकिन्छ, जसले वर्षभरि हजारौं आन्तरिक र बाह्य पर्यटकहरूलाई आकर्षित गर्छ। साहसिक पर्यटन र पदयात्राका लागि पनि पोखरा एक लोकप्रिय सुरुवात बिन्दु मानिन्छ।"),
    ("https://example.com/ne/7", "क क क । । । -- -- ९९९ ८८८ !!! ???"),  # heuristic-filter reject: low content
]


def _write_shard(docs: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for url, text in docs:
            record = {
                "url": url,
                "title": None,
                "text": text,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_sample_shards(demo_root: Path) -> None:
    """Writes sample_data-derived shards into <demo_root>/<language>/data/raw/crawled/."""
    _write_shard(HINDI_DOCS, demo_root / "hindi" / "data" / "raw" / "crawled" / "shard_00000.jsonl.gz")
    _write_shard(NEPALI_DOCS, demo_root / "nepali" / "data" / "raw" / "crawled" / "shard_00000.jsonl.gz")


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "sample_data"
    write_sample_shards(target)
    print(f"wrote sample shards under {target}")
