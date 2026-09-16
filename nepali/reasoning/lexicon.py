"""
Nepali entity pools for the synthetic reasoning dataset.

Written separately from the Hindi pools, not transliterated from them. Names are the ones common in
Nepal, in Nepali spelling -- दिपक rather than दीपक, सुनिता rather than सुनीता, नरेन्द्र with a half-n rather
than the Hindi anusvara form नरेंद्र -- and no name appears in both languages' pools, so Model H and
Model L are never tested on a shared entity (tests/test_reasoning.py checks this).

The pools are sized like Hindi's (~200 persons, ~60 objects) and for the same reason: L9 (slide 24)
shows a model trained on twenty names answering with one of those twenty names regardless of the
question, and a large pool split 70/30 into disjoint train and test names removes that shortcut.

Where the languages differ, this file follows Nepali:

* **Postpositions attach to the name** -- रामको, हरिभन्दा, सीतासँग -- so a name never changes form but
  is followed directly by a suffix. That makes the substring check stricter here than in Hindi: if
  one name were a prefix of another (मनिष / मनिषा, बिजय / बिजया), "मनिषको" could not be attributed
  to the right entity. Every such pair was removed.
* **Objects take masculine agreement.** Nepali inanimate nouns have no grammatical gender in
  everyday use (महँगो घडी, not महँगी घडी), so objects are listed without one.
* **Counted items take the classifier वटा and stay singular** (५ वटा स्याउ), unlike Hindi's plurals.
* Spellings follow the pretraining corpus, which is normalised by nepali/clean/norm.py: chandrabindu
  is kept (गह्रौँ, आँप), exactly as the model saw it.
"""
from __future__ import annotations

MALE = (
    "हरि", "श्याम", "कृष्ण", "नारायण", "बिष्णु", "सुरज", "सन्तोष", "सुदिप", "प्रदिप", "दिपक",
    "बिनोद", "राजु", "सञ्जय", "बिजय", "भीम", "भुपेन्द्र", "नबिन", "प्रबिन", "सुनिल", "बिपिन",
    "निर्मल", "उत्तम", "गोविन्द", "टंक", "खड्ग", "बहादुर", "पदम", "ईश्वर", "केशर", "तेज",
    "धर्म", "कर्ण", "मनिष", "आशिष", "रोशन", "अमृत", "प्रेम", "सुशील", "बिबेक", "अविनाश",
    "रुपेश", "दिवाकर", "चिरञ्जीवी", "युवराज", "धिरज", "नरेन्द्र", "सुरेन्द्र", "महेन्द्र", "राजेन्द्र", "जितेन्द्र",
    "उपेन्द्र", "ज्ञानेन्द्र", "बिरेन्द्र", "देवेन्द्र", "पुष्कर", "शङ्कर", "लोकराज", "हिमाल", "जीवन", "कुमार",
    "मुकुन्द", "गोकुल", "नवराज", "शिशिर", "बसन्त", "हेमन्त", "सुदर्शन", "अनुप", "टीकाराम", "डम्बर",
    "ठाकुर", "मिलन", "सुबास", "बिकाश", "प्रज्वल", "सुयोग", "प्रतिक", "रमण", "सबिन", "अजित",
    "सुजन", "निरज", "कल्याण", "भुवन", "तिलक", "पवित्र", "रुद्र", "अच्युत", "उज्ज्वल", "सौरव",
    "बिशाल", "समीर", "अभिनव", "दिलिप", "रोहन", "प्रशान्त", "शरद", "गौतम", "मनोहर", "राजन",
)

FEMALE = (
    "सरस्वती", "लक्ष्मी", "कमला", "बिमला", "सुशिला", "अनिता", "सुनिता", "सबिता", "बबिता", "रिता",
    "गिता", "पुष्पा", "मञ्जु", "शान्ति", "सावित्री", "दुर्गा", "गङ्गा", "यमुना", "तुलसी", "चमेली",
    "बिन्दु", "इन्दिरा", "सुस्मिता", "प्रतिभा", "रुपा", "दीपा", "अस्मिता", "श्रृजना", "सृष्टि", "आस्था",
    "प्रज्ञा", "सुजाता", "रेनु", "अञ्जना", "रोजा", "करुणा", "समता", "निरु", "भगवती", "जमुना",
    "हेमलता", "सिर्जना", "प्रमिला", "सम्झना", "कौशल्या", "निलम", "गोमा", "सानु", "मिना", "सरिना",
    "कुसुम", "लीला", "सुमित्रा", "तारा", "चन्दा", "अप्सरा", "मन्दिरा", "सुलोचना", "सलिना", "बर्षा",
    "दिक्षा", "प्रिती", "रश्मी", "अनुसा", "आभा", "सोनिका", "जस्मिन", "सुष्मा", "नविना", "रञ्जना",
    "उमा", "मेनुका", "शोभा", "सुवर्णा", "कञ्चन", "नन्दा", "इशा", "रीमा", "दिव्या", "स्नेहा",
    "बन्दना", "मिरा", "ज्ञानु", "सन्ध्या", "पद्मा", "लाली", "फुलमाया", "चेतना", "अर्पणा", "प्रकृति",
    "रिया", "सम्पदा", "दया", "कबिता", "अनिशा", "सुनैना", "जुना", "नर्मदा", "भुमिका", "प्रेरणा",
)

# Objects that plausibly have both a price and a weight. No gender: see the module docstring.
OBJECT_NAMES = (
    "किताब", "घडी", "कुर्सी", "टेबल", "थाल", "साइकल", "टोपी", "कमिज", "बाल्टिन", "बोतल",
    "कराई", "चङ्गा", "सिरक", "कुचो", "साँचो", "कैँची", "सियो", "लालटिन", "पेन्सिल", "डोरी",
    "केटली", "कलम", "गुन्द्री", "चकटी", "डोको", "मादल", "छाता", "कचौरा", "प्लेट", "घण्टी",
    "चप्पल", "ज्याकेट", "सारी", "माला", "झोला", "गिलास", "बत्ती", "कम्बल", "चम्चा", "रेडियो",
    "खाट", "बाकस", "मोबाइल", "बेलना", "सितार", "कुकर", "हेलमेट", "स्कुटर", "टेलिफोन", "ल्यापटप",
    "फ्रिज", "कोट", "स्वेटर", "घैँटो", "गाग्री", "थर्मस", "ऐना", "डाडु", "पन्यु", "सुराही",
)
OBJECTS = tuple((n, "m") for n in OBJECT_NAMES)

# Countable nouns for the quantity attribute. Singular after a count with the classifier वटा
# ("५ वटा स्याउ"). Shared by every split: they are wording, never an answer.
ITEMS = (
    "स्याउ", "आँप", "केरा", "अण्डा", "सुन्तला", "बेलुन", "बिरुवा", "किताब",
    "बल", "चुरा", "कापी", "चकलेट", "कलम",
)

PERSONS = tuple((n, "m") for n in MALE) + tuple((n, "f") for n in FEMALE)
GENDER = {name: g for name, g in PERSONS + OBJECTS}
