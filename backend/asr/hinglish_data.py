"""Hand-curated Hinglish spelling data.

Whisper decodes Hindi (and code-switched English) into Devanagari; the general
transliteration rules in `transliterate.py` then romanize it phonetically. Rules
alone get common words wrong in ways real Hinglish typing never does ("hogaa"
not "hoga", "naheen" not "nahi") and cannot recover an English loanword that
Whisper wrote out in Devanagari ("लाइफ" is "life", not "laaiph").

Two tables, both keyed by the *exact* Devanagari spelling (post
leading/trailing-punctuation strip) so a lookup is a dict hit with zero
ambiguity — no romanization drift to compare against.

- `COMMON_WORDS`: the highest-frequency Hindi function/content words, mapped to
  the spelling native speakers actually type. Checked first, so a real Hindi
  word (`sab`, `par`, `hai`, `main`) never falls through to loanword-skeleton
  matching and gets misread as an English word that happens to share a
  consonant skeleton ("sab" -> "sub", "par" -> "per").
- `LOANWORDS`: common English words borrowed into Hindi speech, restored to
  their English spelling. Checked second (after COMMON_WORDS, before the
  general rules + skeleton-match fallback in `transliterate.py`).
"""

from typing import Dict

# ---------------------------------------------------------------------------
# COMMON_WORDS -- the ~300 highest-frequency Hindi words.
# ---------------------------------------------------------------------------
COMMON_WORDS: Dict[str, str] = {
    # --- pronouns & determiners ---
    "मैं": "main", "मुझे": "mujhe", "मुझको": "mujhko",
    "मेरा": "mera", "मेरी": "meri", "मेरे": "mere",
    "हम": "ham", "हमें": "hamein", "हमको": "hamko",
    "हमारा": "hamara", "हमारी": "hamari", "हमारे": "hamare",
    "तू": "tu", "तुझे": "tujhe", "तेरा": "tera", "तेरी": "teri", "तेरे": "tere",
    "तुम": "tum", "तुम्हें": "tumhein", "तुम्हारा": "tumhara",
    "तुम्हारी": "tumhari", "तुम्हारे": "tumhare",
    "आप": "aap", "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke", "आपको": "aapko",
    "वह": "woh", "वो": "woh", "वे": "ve",
    "उसे": "use", "उसका": "uska", "उसकी": "uski", "उसके": "uske", "उसको": "usko",
    "उन्हें": "unhein", "उनका": "unka", "उनकी": "unki", "उनके": "unke",
    "इसे": "ise", "इसका": "iska", "इसकी": "iski", "इसके": "iske", "इसको": "isko",
    "इनका": "inka", "इनकी": "inki", "इनके": "inke",
    "यह": "yeh", "ये": "ye", "इन": "in", "उन": "un", "इस": "is", "उस": "us",
    "कोई": "koi", "कुछ": "kuch", "सब": "sab", "सभी": "sabhi", "हर": "har",
    "कौन": "kaun", "किसी": "kisi", "किसका": "kiska", "किसकी": "kiski",
    "क्या": "kya", "क्यों": "kyun", "कैसे": "kaise", "कैसा": "kaisa", "कैसी": "kaisi",
    "कहाँ": "kahan", "कहां": "kahan", "कब": "kab",
    "कितना": "kitna", "कितनी": "kitni", "कितने": "kitne",
    "जो": "jo", "जिसे": "jise", "जिसका": "jiska", "जितना": "jitna",
    "जब": "jab", "तब": "tab", "यहाँ": "yahan", "यहां": "yahan",
    "वहाँ": "wahan", "वहां": "wahan", "खुद": "khud", "अपना": "apna",
    "अपनी": "apni", "अपने": "apne",

    # --- conjunctions / discourse markers ---
    "और": "aur", "या": "ya", "लेकिन": "lekin", "मगर": "magar",
    "परंतु": "parantu", "इसलिए": "isliye", "क्योंकि": "kyunki", "कि": "ki",
    "अगर": "agar", "वरना": "warna", "फिर": "phir", "भी": "bhi", "ही": "hi",
    "तो": "to", "मतलब": "matlab", "यानी": "yani",
    "वैसे": "waise", "ऐसे": "aise", "ऐसा": "aisa", "ऐसी": "aisi",
    "वैसा": "waisa", "वैसी": "waisi", "यही": "yahi", "वही": "wahi",
    "नहीं": "nahi", "बस": "bas", "यार": "yaar",
    "वहीं": "wahin", "यहीं": "yahin", "क्यूं": "kyun", "क्यूँ": "kyun",

    # --- hona (to be) ---
    "है": "hai", "हैं": "hain", "था": "tha", "थी": "thi", "थे": "the",
    "होगा": "hoga", "होगी": "hogi", "होंगे": "honge", "हो": "ho",
    "होता": "hota", "होती": "hoti", "होते": "hote",
    "हुआ": "hua", "हुई": "hui", "हुए": "hue",
    "हूँ": "hoon", "हूं": "hoon", "रहा": "raha", "रही": "rahi", "रहे": "rahe",

    # --- karna (to do) ---
    "करता": "karta", "करती": "karti", "करते": "karte", "करना": "karna",
    "करूँगा": "karunga", "करूंगा": "karunga", "करूँगी": "karungi", "करूंगी": "karungi",
    "करेंगे": "karenge", "करेगा": "karega", "करेगी": "karegi",
    "किया": "kiya", "करी": "kari", "कर": "kar", "करके": "karke",

    # --- jana (to go) ---
    "जाना": "jaana", "जाता": "jata", "जाती": "jati", "जाते": "jate",
    "जाऊँगा": "jaunga", "जाऊंगा": "jaunga", "गया": "gaya", "गई": "gayi", "गए": "gaye",

    # --- aana (to come) ---
    "आना": "aana", "आता": "aata", "आती": "aati", "आते": "aate",
    "आया": "aaya", "आई": "aai", "आए": "aaye",

    # --- dena / lena (to give / take) ---
    "देना": "dena", "देता": "deta", "देती": "deti", "देते": "dete",
    "दिया": "diya", "दी": "di", "दिए": "diye",
    "लेना": "lena", "लेता": "leta", "लेती": "leti", "लेते": "lete",
    "लिया": "liya", "ली": "li", "लिए": "liye",

    # --- other common verbs ---
    "देखना": "dekhna", "देखता": "dekhta", "देखती": "dekhti", "देखते": "dekhte",
    "देखा": "dekha", "देखी": "dekhi", "देखे": "dekhe",
    "सोचना": "sochna", "सोचता": "sochta", "सोचती": "sochti", "सोचते": "sochte",
    "सोचा": "socha",
    "समझना": "samajhna", "समझता": "samajhta", "समझती": "samajhti",
    "समझते": "samajhte", "समझा": "samjha", "समझ": "samajh",
    "बोलना": "bolna", "बोलता": "bolta", "बोलती": "bolti", "बोलते": "bolte", "बोला": "bola",
    "चलना": "chalna", "चलता": "chalta", "चलती": "chalti", "चलते": "chalte", "चला": "chala",
    "मिलना": "milna", "मिलता": "milta", "मिलती": "milti", "मिलते": "milte", "मिला": "mila",
    "लगना": "lagna", "लगता": "lagta", "लगती": "lagti", "लगते": "lagte", "लगा": "laga",
    "रखना": "rakhna", "रखता": "rakhta", "रखती": "rakhti", "रखते": "rakhte", "रखा": "rakha",
    "बनाना": "banana", "बनाता": "banata", "बनाती": "banati", "बनाते": "banate",
    "बनाया": "banaya", "बनी": "bani", "बना": "bana",
    "निकलना": "nikalna", "निकला": "nikla", "निकली": "nikli",
    "समझाना": "samjhana", "बताना": "batana", "बताया": "bataya",
    "सीखना": "seekhna", "सीखा": "seekha", "सिखाना": "sikhana",
    "चाहिए": "chahiye", "चाहता": "chahta", "चाहती": "chahti", "चाहते": "chahte",

    # --- adverbs / adjectives ---
    "अभी": "abhi", "अब": "ab", "ज़्यादा": "zyada", "ज्यादा": "zyada",
    "कम": "kam", "थोड़ा": "thoda", "थोड़ी": "thodi", "थोड़े": "thode",
    "बहुत": "bahut", "सच": "sach", "सही": "sahi", "गलत": "galat",
    "अच्छा": "achha", "अच्छी": "achhi", "अच्छे": "achhe",
    "बुरा": "bura", "बुरी": "buri", "बुरे": "bure",
    "बड़ा": "bada", "बड़ी": "badi", "बड़े": "bade",
    "छोटा": "chota", "छोटी": "choti", "छोटे": "chote",
    "नया": "naya", "नई": "nai", "नए": "naye",
    "पुराना": "purana", "पुरानी": "purani", "पुराने": "purane",
    "जल्दी": "jaldi", "धीरे": "dheere", "आगे": "aage", "पीछे": "peeche",
    "ऊपर": "upar", "नीचे": "neeche", "अलग": "alag", "साथ": "saath",
    "पास": "paas", "दूर": "door", "पूरा": "poora", "पूरी": "poori", "पूरे": "poore",

    # --- nouns: time, place, common ---
    "समय": "samay", "वक़्त": "waqt", "वक्त": "waqt", "जगह": "jagah",
    "बात": "baat", "बातें": "baatein", "काम": "kaam", "नाम": "naam",
    "दिन": "din", "रात": "raat", "साल": "saal", "महीना": "mahina",
    "हफ़्ता": "hafta", "हफ्ता": "hafta", "घर": "ghar", "बाहर": "bahar",
    "अंदर": "andar", "सपने": "sapne", "सपना": "sapna",
    "दोस्त": "dost", "दोस्तों": "doston", "दोस्तो": "dosto",
    "लोग": "log", "आदमी": "aadmi", "औरत": "aurat", "बच्चा": "baccha", "बच्चे": "bachche",
    "कल": "kal", "आज": "aaj", "जान": "jaan", "दिल": "dil",
    "हाथ": "haath", "सिर": "sir", "मुँह": "munh", "मुंह": "munh", "चल": "chal",
    "रुख": "rukh", "हिंदी": "hindi", "हिन्दी": "hindi",
    "खोलेंगे": "kholenge", "खोलेगा": "kholega", "खोलेगी": "kholegi",

    # --- postpositions / particles ---
    "से": "se", "को": "ko", "का": "ka", "की": "ki", "के": "ke",
    "में": "mein", "पर": "par", "पे": "pe", "ने": "ne", "तक": "tak",

    # --- numbers ---
    "एक": "ek", "दो": "do", "तीन": "teen", "चार": "char",
    "पाँच": "paanch", "पांच": "paanch", "छह": "chhe", "छः": "chhe",
    "सात": "saat", "आठ": "aath", "नौ": "nau", "दस": "das",
}


# ---------------------------------------------------------------------------
# LOANWORDS -- common English words borrowed into Hindi speech, restored to
# their English spelling. Keyed by the Devanagari Whisper writes them as.
# ---------------------------------------------------------------------------
LOANWORDS: Dict[str, str] = {
    "लाइफ": "life", "स्टार्ट": "start", "ब्रांड": "brand", "प्रोसेस": "process",
    "फोकस": "focus", "चैनल": "channel", "कॉफी": "coffee", "कॉफ़ी": "coffee",
    "मास्टरपीस": "masterpiece", "बिल्ट": "built", "टाइम": "time",
    "एक्चुअली": "actually", "वीडियो": "video", "आइडिया": "idea",
    "क्रिएटिव": "creative", "कंटेंट": "content", "सब्सक्राइब": "subscribe",
    "लाइक": "like", "कमेंट": "comment", "शेयर": "share", "फॉलो": "follow",
    "फॉलोअर": "follower", "अपलोड": "upload", "डाउनलोड": "download",
    "कैमरा": "camera", "फोन": "phone", "मोबाइल": "mobile", "इंटरनेट": "internet",
    "एडिटिंग": "editing", "थंबनेल": "thumbnail", "स्क्रिप्ट": "script",
    "प्रोजेक्ट": "project", "स्टोरी": "story", "प्लान": "plan",
    "पॉजिटिव": "positive", "निगेटिव": "negative", "नेगेटिव": "negative",
    "बेसिकली": "basically", "सीरियसली": "seriously", "ऑनेस्टली": "honestly",
    "परफेक्ट": "perfect", "स्पेशल": "special", "नॉर्मल": "normal",
    "सिंपल": "simple", "प्रॉब्लम": "problem", "सॉल्यूशन": "solution",
    "रिजल्ट": "result", "सक्सेस": "success", "एक्सपीरियंस": "experience",
    "चैलेंज": "challenge", "मोटिवेशन": "motivation", "इंस्पिरेशन": "inspiration",
    "कॉन्फिडेंस": "confidence", "पर्सनालिटी": "personality", "कैरेक्टर": "character",
    "क्वालिटी": "quality", "बजट": "budget", "बिजनेस": "business",
    "बिज़नेस": "business", "कंपनी": "company", "मार्केट": "market",
    "कस्टमर": "customer", "प्रोडक्ट": "product", "सर्विस": "service",
    "टीम": "team", "मेंबर": "member", "लीडर": "leader", "मैनेजर": "manager",
    "डिसीजन": "decision", "गोल": "goal", "टारगेट": "target",
    "ग्रोथ": "growth", "डेवलपमेंट": "development", "चेंज": "change",
    "फ्यूचर": "future", "मिनट": "minute", "सेकंड": "second",
    "शेड्यूल": "schedule", "मीटिंग": "meeting", "डिस्कशन": "discussion",
    "कम्युनिकेशन": "communication", "रिलेशनशिप": "relationship",
    "पार्टनर": "partner", "सपोर्ट": "support", "गाइड": "guide",
    "एडवाइस": "advice", "सजेशन": "suggestion", "ओपिनियन": "opinion",
    "इमोशन": "emotion", "मूड": "mood", "एटीट्यूड": "attitude",
    "बिहेवियर": "behavior", "हैबिट": "habit", "रूटीन": "routine",
    "लाइफस्टाइल": "lifestyle", "हेल्थ": "health", "फिटनेस": "fitness",
    "डाइट": "diet", "एक्सरसाइज": "exercise", "स्ट्रेस": "stress",
    "टेंशन": "tension", "प्रेशर": "pressure", "बैलेंस": "balance",
    "कंट्रोल": "control", "पावर": "power", "स्ट्रेंथ": "strength",
    "स्किल": "skill", "टैलेंट": "talent", "एबिलिटी": "ability",
    "परफॉर्मेंस": "performance", "प्रैक्टिस": "practice", "ट्रेनिंग": "training",
    "एजुकेशन": "education", "नॉलेज": "knowledge", "इंफॉर्मेशन": "information",
    "टेक्नोलॉजी": "technology", "साइंस": "science", "रिसर्च": "research",
    "स्टडी": "study", "एनालिसिस": "analysis", "डेटा": "data",
    "डिटेल": "detail", "फैक्ट": "fact", "रियलिटी": "reality",
    "रीजन": "reason", "लॉजिक": "logic", "सेंस": "sense",
    "पर्पस": "purpose", "वैल्यू": "value", "बेनिफिट": "benefit",
    "एडवांटेज": "advantage", "रिस्क": "risk", "सेफ्टी": "safety",
    "सिक्योरिटी": "security", "फ्रीडम": "freedom", "सिस्टम": "system",
    "मेथड": "method", "टेक्निक": "technique", "स्टाइल": "style",
    "डिज़ाइन": "design", "डिजाइन": "design", "स्ट्रक्चर": "structure",
    "फॉर्मेट": "format", "मॉडल": "model", "वर्जन": "version",
    "अपडेट": "update", "फीचर": "feature", "फंक्शन": "function",
    "ऑप्शन": "option", "चॉइस": "choice", "एक्शन": "action",
    "एक्टिविटी": "activity", "डायरेक्शन": "direction", "पोजीशन": "position",
    "लोकेशन": "location", "एरिया": "area", "सिटी": "city",
    "वर्ल्ड": "world", "नेचर": "nature", "एनवायरनमेंट": "environment",
    "वेदर": "weather", "लाइट": "light", "कलर": "color",
    "साउंड": "sound", "वॉइस": "voice", "म्यूजिक": "music",
    "सॉन्ग": "song", "डांस": "dance", "आर्ट": "art", "मूवी": "movie",
    "फिल्म": "film", "ड्रामा": "drama", "कॉमेडी": "comedy",
    "बुक": "book", "मैगजीन": "magazine", "आर्टिकल": "article",
    "रिपोर्ट": "report", "रिव्यू": "review", "फीडबैक": "feedback",
    "लेवल": "level", "स्टैंडर्ड": "standard", "स्कोर": "score",
    "टेस्ट": "test", "आंसर": "answer", "कन्फ्यूजन": "confusion",
    "केस": "case", "सिचुएशन": "situation", "कंडीशन": "condition",
    "बैकग्राउंड": "background", "हिस्ट्री": "history", "सोर्स": "source",
    "इम्पैक्ट": "impact", "इन्फ्लुएंस": "influence",
    "टार्गेट": "target", "कैरियर": "career", "डिफरेंट": "different",
    "शॉप": "shop", "कॉम्प्लेक्स": "complex", "हॉरर": "horror", "जॉब": "job",
    "कारेक्टर": "character", "पैशन": "passion", "सीरीज़": "series", "सीरीज": "series",
}
