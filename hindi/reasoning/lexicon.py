"""
Hindi entity pools for the synthetic reasoning dataset.

Hand-curated rather than scraped, and large on purpose. L9 (slide 24) shows the failure a small pool
produces: a model finetuned on twenty names reached training loss 0.27 and scored exactly 0.000 on
unseen names, because it had learned to answer with one of the twenty names it had seen. With ~200
person names split 70/30 into disjoint train and test pools, memorising a name set stops being an
easier route to low loss than actually reading the premises.

Three Hindi-specific constraints shaped the lists, each checked by tests/test_reasoning.py:

* **Gender is recorded**, because Hindi adjectives agree with their subject -- "राहुल, अमित से बड़ा
  है" but "सीता, रेखा से बड़ी है" -- and a template that ignored it would be ungrammatical in half the
  examples, which "natural phrasing for that language" (assignment §3.1) rules out.
* **No masculine noun ends in -ā.** Hindi inflects those before a postposition: पंखा becomes पंखे
  in "पंखे से महंगी". The answer would then be a word that never appears in the prompt in that form,
  which breaks exact match and the attention-pointer analysis alike. Feminine nouns and masculine
  nouns ending in a consonant or -ī do not change, so the pools use only those.
* **No name is a substring of another** (राधा inside अनुराधा, नेहा inside स्नेहा, ललित inside
  ललिता were all removed for this), otherwise matching an entity in text credits the wrong one.

Spellings follow the pretraining corpus rather than a dictionary: the Hindi validation split writes
मेज and कमीज without nukta, and Phase 1's Hindi normaliser folds chandrabindu into anusvara, so
plurals are written चूड़ियां, not चूड़ियाँ -- the form the model has actually seen.
"""
from __future__ import annotations

MALE = (
    "राहुल", "अमित", "सुरेश", "रमेश", "महेश", "दिनेश", "राजेश", "मुकेश", "विकास", "आकाश",
    "प्रकाश", "अनिल", "सुनील", "कपिल", "नितिन", "सचिन", "रोहित", "मोहित", "सुमित", "अंकित",
    "विवेक", "अभिषेक", "आदित्य", "अर्जुन", "करण", "वरुण", "तरुण", "गौरव", "सौरभ", "पंकज",
    "नीरज", "धीरज", "मनोज", "सरोज", "विनोद", "प्रमोद", "अशोक", "आलोक", "दीपक", "विशाल",
    "कुणाल", "हर्ष", "शिव", "गोपाल", "विजय", "संजय", "उदय", "प्रवीण", "नवीन", "गोविंद",
    "हरीश", "जगदीश", "सतीश", "गिरीश", "भरत", "लक्ष्मण", "माधव", "केशव", "मनीष", "आशीष",
    "योगेश", "उमेश", "नरेश", "लोकेश", "शैलेश", "कमल", "अमन", "चेतन", "जतिन", "प्रदीप",
    "संदीप", "कुलदीप", "राजीव", "संजीव", "सुरेंद्र", "महेंद्र", "नरेंद्र", "जितेंद्र", "पवन", "मदन",
    "भूषण", "मोहन", "किशोर", "अनूप", "हेमंत", "बसंत", "निखिल", "सुधीर", "अजय", "अक्षय",
    "तुषार", "मयंक", "प्रतीक", "कार्तिक", "सागर", "प्रशांत", "अनुज", "दिलीप", "गणेश", "राकेश",
)

FEMALE = (
    "सीता", "गीता", "रीता", "प्रिया", "नेहा", "पूजा", "रेखा", "सुधा", "उषा", "आशा",
    "निशा", "दिशा", "श्वेता", "ममता", "कविता", "सविता", "ललिता", "सुनीता", "विनीता", "नम्रता",
    "अंजलि", "रूपाली", "दीपाली", "वैशाली", "शालिनी", "भावना", "कल्पना", "अर्चना", "वंदना", "रचना",
    "सपना", "मीना", "रीना", "हेमा", "रश्मि", "राधा", "माया", "सरिता", "सोनिया", "तान्या",
    "अदिति", "श्रुति", "स्मृति", "प्रीति", "ज्योति", "आरती", "भारती", "मालती", "शांति", "कांता",
    "पार्वती", "गौरी", "पायल", "काजल", "नंदिनी", "मोहिनी", "रागिनी", "यामिनी", "दामिनी", "शिल्पा",
    "सुरभि", "नीलम", "जया", "मधु", "इंदु", "बिंदु", "चांदनी", "रोशनी", "पल्लवी", "मोनिका",
    "प्रियंका", "दीपिका", "राधिका", "अनुष्का", "खुशबू", "तनुजा", "वर्षा", "मेघा", "मानसी", "रंजना",
    "नैना", "सलोनी", "भूमि", "पूर्णिमा", "गरिमा", "महिमा", "प्रतिमा", "सुषमा", "उर्मिला", "शर्मिला",
    "सरला", "विमला", "निर्मला", "जानकी", "यशोदा", "सुशीला", "मंजू", "रुचि", "शिखा", "कीर्ति",
)

# Objects: things that plausibly have both a price and a weight. Gender drives adjective agreement
# ("घड़ी महंगी है", "बैग महंगा है") and the genitive ("घड़ी की कीमत", "बैग का वजन").
OBJECTS = (
    ("किताब", "f"), ("घड़ी", "f"), ("कुर्सी", "f"), ("मेज", "f"), ("थाली", "f"), ("साइकिल", "f"),
    ("टोपी", "f"), ("कमीज", "f"), ("बाल्टी", "f"), ("चादर", "f"), ("बोतल", "f"), ("कड़ाही", "f"),
    ("पतंग", "f"), ("गुड़िया", "f"), ("रजाई", "f"), ("झाड़ू", "f"), ("चाबी", "f"), ("कैंची", "f"),
    ("सुई", "f"), ("लालटेन", "f"), ("पेंसिल", "f"), ("रस्सी", "f"), ("अलमारी", "f"), ("केतली", "f"),
    ("कलम", "f"), ("दरी", "f"), ("चटाई", "f"), ("टोकरी", "f"), ("ढोलक", "f"), ("छतरी", "f"),
    ("कटोरी", "f"), ("प्लेट", "f"), ("घंटी", "f"), ("कंघी", "f"), ("चप्पल", "f"), ("जैकेट", "f"),
    ("साड़ी", "f"), ("माला", "f"), ("अंगूठी", "f"), ("तिजोरी", "f"),
    ("बैग", "m"), ("गिलास", "m"), ("लैंप", "m"), ("कंबल", "m"), ("चम्मच", "m"), ("रेडियो", "m"),
    ("पलंग", "m"), ("संदूक", "m"), ("मोबाइल", "m"), ("बेलन", "m"), ("सितार", "m"), ("कुकर", "m"),
    ("हेलमेट", "m"), ("स्कूटर", "m"), ("टेलीफोन", "m"), ("पेन", "m"), ("लैपटॉप", "m"), ("फ्रिज", "m"),
    ("कोट", "m"), ("स्वेटर", "m"),
)

# Countable nouns for the quantity attribute, in the plural a count takes ("7 सेब", "5 किताबें").
# Counts start at 2 so a plural is always grammatical. Items are wording, not entities: they are
# never an answer, so they are shared by every split rather than split like the names.
ITEMS = (
    "सेब", "आम", "केले", "अंडे", "संतरे", "गुब्बारे", "पौधे", "कंचे",
    "किताबें", "गेंदें", "चूड़ियां", "कॉपियां", "टॉफियां",
)

PERSONS = tuple((n, "m") for n in MALE) + tuple((n, "f") for n in FEMALE)
GENDER = {name: g for name, g in PERSONS + OBJECTS}
