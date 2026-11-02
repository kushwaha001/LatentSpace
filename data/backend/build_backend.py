"""Build the deterministic mock transit backend (data/backend/*.json).

Run from the repo root:  python data/backend/build_backend.py

Station NAMES and Indian Railways station CODES are real. Everything else is SYNTHETIC:
train numbers, train names, timetables, distances, fares, seat availability, PNRs and the
metro station identifiers (``mum_andheri`` etc. are in-repo ids, not official codes).
See data/backend/DATACARD.md. The output is fully determined by this file (no randomness
beyond a fixed-seed hash), so re-running it reproduces the same JSON byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Indian Railways stations: (code, English name, city, state, Hindi, Marathi, aliases) ──
# Codes and names are real (Indian Railways). Aliases are romanised spellings people type.
RAIL = [
    ("CSMT", "Mumbai CSMT (Chhatrapati Shivaji Maharaj Terminus)", "Mumbai", "Maharashtra",
     "छत्रपति शिवाजी महाराज टर्मिनस", "छत्रपती शिवाजी महाराज टर्मिनस",
     ["csmt", "cst", "vt", "victoria terminus", "mumbai cst", "mumbai csmt", "सीएसएमटी", "सीएसटी"]),
    ("MMCT", "Mumbai Central", "Mumbai", "Maharashtra", "मुंबई सेंट्रल", "मुंबई सेंट्रल",
     ["bombay central", "mumbai central"]),
    ("LTT", "Lokmanya Tilak Terminus", "Mumbai", "Maharashtra", "लोकमान्य तिलक टर्मिनस",
     "लोकमान्य टिळक टर्मिनस", ["ltt", "kurla terminus", "lokmanya tilak", "एलटीटी"]),
    ("BDTS", "Bandra Terminus", "Mumbai", "Maharashtra", "बांद्रा टर्मिनस", "वांद्रे टर्मिनस",
     ["bandra", "bandra terminus"]),
    ("DR", "Dadar", "Mumbai", "Maharashtra", "दादर", "दादर", ["dadar"]),
    ("TNA", "Thane", "Thane", "Maharashtra", "ठाणे", "ठाणे", ["thane"]),
    ("KYN", "Kalyan Junction", "Kalyan", "Maharashtra", "कल्याण", "कल्याण", ["kalyan"]),
    ("PUNE", "Pune Junction", "Pune", "Maharashtra", "पुणे", "पुणे", ["pune", "poona"]),
    ("NDLS", "New Delhi", "Delhi", "Delhi", "नई दिल्ली", "नवी दिल्ली",
     ["new delhi", "nai dilli", "nayi dilli"]),
    ("NZM", "Hazrat Nizamuddin", "Delhi", "Delhi", "हज़रत निज़ामुद्दीन", "हजरत निजामुद्दीन",
     ["nizamuddin", "hazrat nizamuddin"]),
    ("DLI", "Delhi Junction", "Delhi", "Delhi", "दिल्ली जंक्शन", "दिल्ली जंक्शन",
     ["old delhi", "delhi junction", "purani dilli", "पुरानी दिल्ली"]),
    ("ANVT", "Anand Vihar Terminal", "Delhi", "Delhi", "आनंद विहार टर्मिनल", "आनंद विहार टर्मिनल",
     ["anand vihar"]),
    ("HWH", "Howrah Junction", "Kolkata", "West Bengal", "हावड़ा", "हावडा", ["howrah", "haora"]),
    ("SDAH", "Sealdah", "Kolkata", "West Bengal", "सियालदह", "सियालदह", ["sealdah"]),
    ("MAS", "Chennai Central (Puratchi Thalaivar Dr. M.G. Ramachandran Central)", "Chennai",
     "Tamil Nadu", "चेन्नई सेंट्रल", "चेन्नई सेंट्रल", ["chennai central", "madras central"]),
    ("MS", "Chennai Egmore", "Chennai", "Tamil Nadu", "चेन्नई एगमोर", "चेन्नई एगमोर",
     ["egmore", "chennai egmore"]),
    ("SBC", "KSR Bengaluru City", "Bengaluru", "Karnataka", "बेंगलुरु", "बंगळूरु",
     ["bengaluru", "bangalore", "bangalore city", "ksr bengaluru", "बंगलौर", "बेंगलूरु"]),
    ("YPR", "Yesvantpur Junction", "Bengaluru", "Karnataka", "यशवंतपुर", "यशवंतपूर",
     ["yesvantpur", "yeshwantpur", "yeshvantpur"]),
    ("SC", "Secunderabad Junction", "Hyderabad", "Telangana", "सिकंदराबाद", "सिकंदराबाद",
     ["secunderabad", "sikandrabad"]),
    ("HYB", "Hyderabad Deccan", "Hyderabad", "Telangana", "हैदराबाद", "हैदराबाद",
     ["hyderabad", "nampally", "hyderabad deccan"]),
    ("ADI", "Ahmedabad Junction", "Ahmedabad", "Gujarat", "अहमदाबाद", "अहमदाबाद",
     ["ahmedabad", "amdavad"]),
    ("ST", "Surat", "Surat", "Gujarat", "सूरत", "सुरत", ["surat"]),
    ("BRC", "Vadodara Junction", "Vadodara", "Gujarat", "वडोदरा", "वडोदरा", ["vadodara", "baroda"]),
    ("JP", "Jaipur Junction", "Jaipur", "Rajasthan", "जयपुर", "जयपूर", ["jaipur"]),
    ("LKO", "Lucknow Charbagh", "Lucknow", "Uttar Pradesh", "लखनऊ", "लखनौ",
     ["lucknow", "charbagh", "lakhnau"]),
    ("CNB", "Kanpur Central", "Kanpur", "Uttar Pradesh", "कानपुर सेंट्रल", "कानपूर सेंट्रल",
     ["kanpur", "kanpur central", "कानपुर"]),
    ("BSB", "Varanasi Junction", "Varanasi", "Uttar Pradesh", "वाराणसी", "वाराणसी",
     ["varanasi", "banaras", "benares", "kashi", "बनारस"]),
    ("PNBE", "Patna Junction", "Patna", "Bihar", "पटना", "पाटणा", ["patna"]),
    ("BPL", "Bhopal Junction", "Bhopal", "Madhya Pradesh", "भोपाल", "भोपाळ", ["bhopal"]),
    ("NGP", "Nagpur Junction", "Nagpur", "Maharashtra", "नागपुर", "नागपूर", ["nagpur"]),
    ("NK", "Nashik Road", "Nashik", "Maharashtra", "नासिक रोड", "नाशिक रोड",
     ["nashik", "nasik", "nashik road", "नाशिक", "नासिक"]),
    ("KOP", "Kolhapur (Chhatrapati Shahu Maharaj Terminus)", "Kolhapur", "Maharashtra",
     "कोल्हापुर", "कोल्हापूर", ["kolhapur"]),
    ("SUR", "Solapur", "Solapur", "Maharashtra", "सोलापुर", "सोलापूर", ["solapur", "sholapur"]),
    ("MAO", "Madgaon Junction", "Madgaon", "Goa", "मडगांव", "मडगाव", ["madgaon", "margao", "goa"]),
    ("ERS", "Ernakulam Junction", "Kochi", "Kerala", "एर्नाकुलम", "एर्नाकुलम",
     ["ernakulam", "kochi", "cochin"]),
    ("TVC", "Thiruvananthapuram Central", "Thiruvananthapuram", "Kerala", "तिरुवनंतपुरम",
     "तिरुवनंतपुरम", ["thiruvananthapuram", "trivandrum"]),
    ("CBE", "Coimbatore Junction", "Coimbatore", "Tamil Nadu", "कोयंबटूर", "कोईम्बतूर",
     ["coimbatore", "kovai"]),
    ("GHY", "Guwahati", "Guwahati", "Assam", "गुवाहाटी", "गुवाहाटी", ["guwahati", "gauhati"]),
    ("ASR", "Amritsar Junction", "Amritsar", "Punjab", "अमृतसर", "अमृतसर", ["amritsar"]),
    ("CDG", "Chandigarh", "Chandigarh", "Chandigarh", "चंडीगढ़", "चंदीगड", ["chandigarh"]),
    ("BBS", "Bhubaneswar", "Bhubaneswar", "Odisha", "भुवनेश्वर", "भुवनेश्वर", ["bhubaneswar"]),
    ("PRYJ", "Prayagraj Junction", "Prayagraj", "Uttar Pradesh", "प्रयागराज", "प्रयागराज",
     ["prayagraj", "allahabad", "इलाहाबाद"]),
    ("AGC", "Agra Cantt", "Agra", "Uttar Pradesh", "आगरा कैंट", "आग्रा कॅन्ट",
     ["agra", "agra cantt", "आगरा", "आग्रा"]),
    ("INDB", "Indore Junction", "Indore", "Madhya Pradesh", "इंदौर", "इंदूर", ["indore"]),
    ("GKP", "Gorakhpur Junction", "Gorakhpur", "Uttar Pradesh", "गोरखपुर", "गोरखपूर",
     ["gorakhpur"]),
]

# Ambiguous city-level names: resolve to several stations (the tool returns candidates).
AMBIGUOUS = {
    "mumbai": ["CSMT", "MMCT", "LTT", "BDTS", "DR"],
    "bombay": ["CSMT", "MMCT", "LTT", "BDTS", "DR"],
    "मुंबई": ["CSMT", "MMCT", "LTT", "BDTS", "DR"],
    "delhi": ["NDLS", "NZM", "DLI", "ANVT"],
    "dilli": ["NDLS", "NZM", "DLI", "ANVT"],
    "दिल्ली": ["NDLS", "NZM", "DLI", "ANVT"],
    "kolkata": ["HWH", "SDAH"],
    "कोलकाता": ["HWH", "SDAH"],
    "chennai": ["MAS", "MS"],
    "चेन्नई": ["MAS", "MS"],
}

# ── Metro lines. Station names are real; order follows the real line (some lines are a
# contiguous subset). Each station: (English, Hindi, Marathi or None = same as Hindi, aliases).
METRO = {
    "Mumbai": {
        "Line 1 (Blue)": [
            ("Versova", "वर्सोवा", None, []),
            ("D N Nagar", "डी एन नगर", None, ["dn nagar"]),
            ("Azad Nagar", "आज़ाद नगर", "आझाद नगर", []),
            ("Andheri", "अंधेरी", None, ["andheri metro"]),
            ("Western Express Highway", "वेस्टर्न एक्सप्रेस हाईवे", "वेस्टर्न एक्सप्रेस हायवे",
             ["weh"]),
            ("Chakala (J B Nagar)", "चकाला (जे बी नगर)", None, ["chakala", "jb nagar"]),
            ("Airport Road", "एयरपोर्ट रोड", "एअरपोर्ट रोड", []),
            ("Marol Naka", "मरोल नाका", "मरोळ नाका", ["marol"]),
            ("Saki Naka", "साकीनाका", None, ["sakinaka"]),
            ("Asalpha", "असल्फा", None, []),
            ("Jagruti Nagar", "जागृति नगर", "जागृती नगर", []),
            ("Ghatkopar", "घाटकोपर", None, ["ghatkopar metro"]),
        ],
        "Line 3 (Aqua)": [
            ("Aarey JVLR", "आरे जेवीएलआर", "आरे जेव्हीएलआर", ["aarey"]),
            ("SEEPZ", "सीप्ज़", "सीप्झ", ["seepz"]),
            ("MIDC Andheri", "एमआईडीसी अंधेरी", "एमआयडीसी अंधेरी", ["midc"]),
            ("Marol Naka", "मरोल नाका", "मरोळ नाका", ["marol"]),
            ("CSMIA T2", "एयरपोर्ट टर्मिनल 2", "विमानतळ टर्मिनल 2",
             ["airport t2", "terminal 2", "t2", "csmia terminal 2"]),
            ("Sahar Road", "सहार रोड", None, []),
            ("CSMIA T1", "एयरपोर्ट टर्मिनल 1", "विमानतळ टर्मिनल 1",
             ["airport t1", "terminal 1", "t1", "csmia terminal 1"]),
            ("Santacruz", "सांताक्रूज़", "सांताक्रूझ", ["santacruz metro"]),
            ("Bandra Colony", "बांद्रा कॉलोनी", "वांद्रे कॉलनी", []),
            ("BKC", "बीकेसी", None, ["bandra kurla complex"]),
            ("Dharavi", "धारावी", None, []),
            ("Shitaladevi Mandir", "शीतलादेवी मंदिर", None, ["shitladevi"]),
            ("Dadar", "दादर", None, ["dadar metro"]),
            ("Siddhivinayak", "सिद्धिविनायक", None, ["siddhi vinayak"]),
            ("Worli", "वर्ली", "वरळी", []),
            ("Acharya Atre Chowk", "आचार्य अत्रे चौक", None, []),
            ("Science Museum", "साइंस म्यूज़ियम", "सायन्स म्युझियम", []),
            ("Mahalaxmi", "महालक्ष्मी", None, ["mahalakshmi"]),
            ("Jagannath Shankar Seth", "जगन्नाथ शंकर सेठ", "जगन्नाथ शंकरशेट", []),
            ("Grant Road", "ग्रांट रोड", "ग्रँट रोड", []),
            ("Girgaon", "गिरगांव", "गिरगाव", []),
            ("Kalbadevi", "कालबादेवी", "काळबादेवी", []),
            ("CSMT Metro", "सीएसएमटी मेट्रो", None, ["csmt metro"]),
            ("Hutatma Chowk", "हुतात्मा चौक", None, ["fort"]),
            ("Churchgate", "चर्चगेट", None, ["churchgate metro"]),
            ("Vidhan Bhavan", "विधान भवन", None, []),
            ("Cuffe Parade", "कफ परेड", None, []),
        ],
    },
    "Pune": {
        "Purple Line": [
            ("PCMC", "पीसीएमसी", None, ["pimpri chinchwad", "pimpri"]),
            ("Sant Tukaram Nagar", "संत तुकाराम नगर", None, []),
            ("Bhosari (Nashik Phata)", "भोसरी (नासिक फाटा)", "भोसरी (नाशिक फाटा)",
             ["bhosari", "nashik phata"]),
            ("Kasarwadi", "कासारवाडी", None, []),
            ("Phugewadi", "फुगेवाडी", None, []),
            ("Dapodi", "दापोडी", None, []),
            ("Bopodi", "बोपोडी", None, []),
            ("Khadki", "खडकी", None, ["kirkee"]),
            ("Range Hill", "रेंज हिल", None, ["range hills"]),
            ("Shivaji Nagar", "शिवाजी नगर", "शिवाजीनगर", ["shivajinagar"]),
            ("Civil Court", "सिविल कोर्ट", "सिव्हिल कोर्ट", ["district court"]),
            ("Budhwar Peth", "बुधवार पेठ", None, []),
            ("Mandai", "मंडई", None, []),
            ("Swargate", "स्वारगेट", None, []),
        ],
        "Aqua Line": [
            ("Vanaz", "वनाज़", "वनाज", []),
            ("Anand Nagar", "आनंद नगर", "आनंदनगर", []),
            ("Ideal Colony", "आइडियल कॉलोनी", "आयडियल कॉलनी", []),
            ("Nal Stop", "नल स्टॉप", "नळ स्टॉप", []),
            ("Garware College", "गरवारे कॉलेज", None, []),
            ("Deccan Gymkhana", "डेक्कन जिमखाना", None, ["deccan"]),
            ("Chhatrapati Sambhaji Udyan", "छत्रपति संभाजी उद्यान", "छत्रपती संभाजी उद्यान",
             ["sambhaji udyan", "sambhaji park"]),
            ("PMC", "पीएमसी", None, ["pune municipal corporation"]),
            ("Civil Court", "सिविल कोर्ट", "सिव्हिल कोर्ट", ["district court"]),
            ("Mangalwar Peth", "मंगलवार पेठ", "मंगळवार पेठ", []),
            ("Pune Railway Station", "पुणे रेलवे स्टेशन", "पुणे रेल्वे स्टेशन",
             ["pune station", "railway station"]),
            ("Ruby Hall Clinic", "रूबी हॉल क्लिनिक", "रुबी हॉल क्लिनिक", ["ruby hall"]),
            ("Bund Garden", "बंड गार्डन", None, []),
            ("Yerawada", "यरवदा", "येरवडा", ["yerwada"]),
            ("Kalyani Nagar", "कल्याणी नगर", "कल्याणीनगर", []),
            ("Ramwadi", "रामवाडी", None, []),
        ],
    },
    "Delhi": {
        "Yellow Line": [
            ("Vishwavidyalaya", "विश्वविद्यालय", None, ["delhi university", "du"]),
            ("Vidhan Sabha", "विधान सभा", None, []),
            ("Civil Lines", "सिविल लाइन्स", "सिव्हिल लाइन्स", []),
            ("Kashmere Gate", "कश्मीरी गेट", None, ["kashmiri gate"]),
            ("Chandni Chowk", "चांदनी चौक", None, []),
            ("Chawri Bazar", "चावड़ी बाज़ार", "चावडी बाजार", ["chawdi bazar"]),
            ("New Delhi", "नई दिल्ली", "नवी दिल्ली", ["new delhi metro"]),
            ("Rajiv Chowk", "राजीव चौक", None, ["cp", "connaught place"]),
            ("Patel Chowk", "पटेल चौक", None, []),
            ("Central Secretariat", "केंद्रीय सचिवालय", None, []),
            ("Udyog Bhawan", "उद्योग भवन", None, []),
            ("Lok Kalyan Marg", "लोक कल्याण मार्ग", None, []),
            ("Jor Bagh", "जोर बाग", None, []),
            ("Dilli Haat - INA", "दिल्ली हाट - आईएनए", None, ["ina", "dilli haat"]),
            ("AIIMS", "एम्स", None, ["aiims"]),
            ("Green Park", "ग्रीन पार्क", None, []),
            ("Hauz Khas", "हौज़ खास", "हौज खास", []),
            ("Malviya Nagar", "मालवीय नगर", None, []),
            ("Saket", "साकेत", None, []),
            ("Qutab Minar", "क़ुतुब मीनार", "कुतुब मिनार", ["qutub minar"]),
            ("Chhattarpur", "छतरपुर", None, ["chhatarpur"]),
            ("Sultanpur", "सुल्तानपुर", None, []),
            ("Ghitorni", "घिटोरनी", None, []),
            ("Arjan Garh", "अर्जनगढ़", "अर्जनगड", ["arjangarh"]),
            ("Guru Dronacharya", "गुरु द्रोणाचार्य", None, []),
            ("Sikanderpur", "सिकंदरपुर", None, []),
            ("MG Road", "एमजी रोड", None, ["mg road gurugram"]),
            ("IFFCO Chowk", "इफको चौक", None, []),
            ("Millennium City Centre Gurugram", "मिलेनियम सिटी सेंटर गुरुग्राम", None,
             ["huda city centre", "millennium city centre"]),
        ],
        "Blue Line": [
            ("Dwarka Sector 21", "द्वारका सेक्टर 21", None, ["sector 21"]),
            ("Dwarka Sector 8", "द्वारका सेक्टर 8", None, []),
            ("Dwarka Sector 9", "द्वारका सेक्टर 9", None, []),
            ("Dwarka Sector 10", "द्वारका सेक्टर 10", None, []),
            ("Dwarka Sector 11", "द्वारका सेक्टर 11", None, []),
            ("Dwarka Sector 12", "द्वारका सेक्टर 12", None, []),
            ("Dwarka Sector 13", "द्वारका सेक्टर 13", None, []),
            ("Dwarka Sector 14", "द्वारका सेक्टर 14", None, []),
            ("Dwarka", "द्वारका", None, []),
            ("Dwarka Mor", "द्वारका मोड़", "द्वारका मोड", []),
            ("Nawada", "नवादा", None, []),
            ("Uttam Nagar West", "उत्तम नगर पश्चिम", None, []),
            ("Uttam Nagar East", "उत्तम नगर पूर्व", None, []),
            ("Janakpuri West", "जनकपुरी पश्चिम", None, []),
            ("Janakpuri East", "जनकपुरी पूर्व", None, []),
            ("Tilak Nagar", "तिलक नगर", "टिळक नगर", []),
            ("Subhash Nagar", "सुभाष नगर", None, []),
            ("Tagore Garden", "टैगोर गार्डन", None, []),
            ("Rajouri Garden", "राजौरी गार्डन", None, []),
            ("Ramesh Nagar", "रमेश नगर", None, []),
            ("Moti Nagar", "मोती नगर", None, []),
            ("Kirti Nagar", "कीर्ति नगर", "कीर्ती नगर", []),
            ("Shadipur", "शादीपुर", None, []),
            ("Patel Nagar", "पटेल नगर", None, []),
            ("Rajendra Place", "राजेंद्र प्लेस", None, []),
            ("Karol Bagh", "करोल बाग", None, []),
            ("Jhandewalan", "झंडेवालान", None, []),
            ("RK Ashram Marg", "आर के आश्रम मार्ग", None, ["r k ashram marg"]),
            ("Rajiv Chowk", "राजीव चौक", None, ["cp", "connaught place"]),
            ("Barakhamba Road", "बाराखंभा रोड", None, []),
            ("Mandi House", "मंडी हाउस", None, []),
            ("Supreme Court", "सुप्रीम कोर्ट", None, ["pragati maidan"]),
            ("Indraprastha", "इंद्रप्रस्थ", None, []),
            ("Yamuna Bank", "यमुना बैंक", None, []),
        ],
        "Airport Express (Orange)": [
            ("New Delhi", "नई दिल्ली", "नवी दिल्ली", ["new delhi metro"]),
            ("Shivaji Stadium", "शिवाजी स्टेडियम", None, []),
            ("Dhaula Kuan", "धौला कुआं", "धौला कुआँ", []),
            ("Delhi Aerocity", "दिल्ली एयरोसिटी", "दिल्ली एअरोसिटी", ["aerocity"]),
            ("IGI Airport", "आईजीआई एयरपोर्ट", "आयजीआय विमानतळ",
             ["airport", "igi airport", "terminal 3", "t3", "delhi airport", "एयरपोर्ट"]),
            ("Dwarka Sector 21", "द्वारका सेक्टर 21", None, ["sector 21"]),
            ("Yashobhoomi Dwarka Sector 25", "यशोभूमि द्वारका सेक्टर 25", None,
             ["yashobhoomi", "sector 25"]),
        ],
    },
}

# ── Synthetic train corridors: (stations with synthetic cumulative km, class set) ────────
SLEEPER = ["SL", "3A", "2A", "1A"]
CHAIR = ["2S", "CC", "EC"]
CORRIDORS = [
    ([("CSMT", 0), ("DR", 9), ("TNA", 34), ("KYN", 54), ("PUNE", 192)], CHAIR, "Intercity"),
    ([("CSMT", 0), ("DR", 9), ("TNA", 34), ("PUNE", 192)], SLEEPER, "Deccan Mail"),
    ([("MMCT", 0), ("ST", 263), ("BRC", 392), ("NZM", 1370), ("NDLS", 1384)], SLEEPER, "Rajdhani"),
    ([("BDTS", 0), ("ST", 255), ("BRC", 384), ("ADI", 491)], CHAIR, "Shatabdi"),
    ([("CSMT", 0), ("KYN", 54), ("NK", 187), ("NGP", 837), ("HWH", 1968)], SLEEPER, "Mail"),
    ([("CSMT", 0), ("PUNE", 192), ("SUR", 455), ("SC", 770), ("HYB", 780)], SLEEPER, "Express"),
    ([("LTT", 0), ("MAO", 765), ("ERS", 1592), ("TVC", 1813)], SLEEPER, "Konkan Express"),
    ([("PUNE", 0), ("KOP", 327)], SLEEPER, "Sahyadri Express"),
    ([("NDLS", 0), ("CNB", 440), ("PRYJ", 634), ("BSB", 756), ("PNBE", 1001), ("HWH", 1451)],
     SLEEPER, "Poorva Express"),
    ([("NDLS", 0), ("LKO", 512), ("GKP", 783)], SLEEPER, "Gorakhdham Express"),
    ([("NDLS", 0), ("JP", 308), ("ADI", 934)], SLEEPER, "Ashram Express"),
    ([("NZM", 0), ("AGC", 188), ("BPL", 701), ("NGP", 1092), ("SC", 1670)], SLEEPER,
     "Dakshin Express"),
    ([("MAS", 0), ("SBC", 362)], CHAIR, "Shatabdi"),
    ([("MAS", 0), ("CBE", 495), ("ERS", 692), ("TVC", 913)], SLEEPER, "Mail"),
    ([("NDLS", 0), ("CDG", 244), ("ASR", 449)], CHAIR, "Shatabdi"),
    ([("HWH", 0), ("BBS", 437), ("MAS", 1662)], SLEEPER, "Coromandel Express"),
    ([("NDLS", 0), ("LKO", 512), ("GHY", 1900)], SLEEPER, "North East Express"),
    ([("SDAH", 0), ("GHY", 1000)], SLEEPER, "Kanchan Express"),
    ([("INDB", 0), ("BPL", 186), ("NGP", 576)], SLEEPER, "Intercity"),
    ([("PUNE", 0), ("YPR", 1000), ("SBC", 1012)], SLEEPER, "Udyan Express"),
    ([("CSMT", 0), ("PUNE", 192), ("SUR", 455)], CHAIR, "Vande Bharat"),
    ([("DLI", 0), ("JP", 300)], CHAIR, "Double Decker"),
    ([("ANVT", 0), ("LKO", 500), ("GKP", 770)], SLEEPER, "Humsafar Express"),
    ([("MS", 0), ("CBE", 500)], SLEEPER, "Cheran Express"),
    ([("SC", 0), ("SBC", 610)], SLEEPER, "Express"),
]
# Synthetic per-km fare (INR) and flat reservation charge per class.
RATE = {"2S": 0.35, "SL": 0.55, "CC": 1.25, "3A": 1.5, "2A": 2.2, "EC": 2.6, "1A": 3.6}
RESERVATION = {"2S": 15, "SL": 20, "CC": 40, "3A": 40, "2A": 50, "EC": 60, "1A": 60}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _h(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def build_trains() -> list[dict]:
    trains = []
    n = 0
    for ci, (stops, classes, kind) in enumerate(CORRIDORS):
        for direction in (0, 1):
            seq = stops if direction == 0 else [(c, stops[-1][1] - k) for c, k in reversed(stops)]
            for k in range(2):  # two services per direction
                n += 1
                number = f"S{20000 + ci * 10 + direction * 4 + k:05d}"
                start = 5 * 60 + _h(number, "dep") % (17 * 60)  # 05:00..21:59
                start -= start % 5
                speed = 55 + _h(number, "speed") % 25  # km/h, synthetic
                daily = _h(number, "days") % 3 != 0
                days = DAYS if daily else sorted(
                    {DAYS[_h(number, d) % 7] for d in range(4)}, key=DAYS.index
                )
                origin_name = next(r[1] for r in RAIL if r[0] == seq[0][0]).split(" (")[0]
                dest_name = next(r[1] for r in RAIL if r[0] == seq[-1][0]).split(" (")[0]
                timings = []
                for j, (code, km) in enumerate(seq):
                    minutes = start + round(km / speed * 60) + j * 5
                    timings.append({"code": code, "km": km, "minute_of_journey": minutes - start,
                                    "time": f"{minutes // 60 % 24:02d}:{minutes % 60:02d}",
                                    "day_offset": minutes // (24 * 60)})
                trains.append({
                    "number": number,
                    "name": f"{origin_name}-{dest_name} {kind} (synthetic)",
                    "classes": classes,
                    "runs_on": days,
                    "stops": timings,
                })
    return trains


def build_metro() -> dict:
    out = {}
    for city, lines in METRO.items():
        stations: dict[str, dict] = {}
        line_out = {}
        for line, rows in lines.items():
            ids = []
            for en, hi, mr, aliases in rows:
                sid = f"{city[:3].lower()}_{slug(en)}"
                st = stations.setdefault(sid, {"id": sid, "name": en, "hi": hi, "mr": mr or hi,
                                               "aliases": [], "lines": []})
                st["aliases"] = sorted(set(st["aliases"]) | set(aliases))
                st["lines"].append(line)
                ids.append(sid)
            line_out[line] = ids
        out[city] = {"lines": line_out, "stations": list(stations.values())}
    return out


def build_rail() -> dict:
    return {
        "stations": [
            {"code": c, "name": n, "city": city, "state": state, "hi": hi, "mr": mr,
             "aliases": aliases}
            for c, n, city, state, hi, mr, aliases in RAIL
        ],
        "ambiguous": AMBIGUOUS,
    }


BOOKINGS = [  # synthetic PNRs, pre-existing in the mock backend
    {"pnr": "8412093355", "train": "S20000", "from": "CSMT", "to": "PUNE", "date": "2026-11-18",
     "travel_class": "CC", "passengers": 2, "status": "CNF", "coach": "C3", "seats": ["41", "42"]},
    {"pnr": "8420117746", "train": "S20020", "from": "MMCT", "to": "NDLS", "date": "2026-11-21",
     "travel_class": "3A", "passengers": 1, "status": "WL", "waitlist_position": 12},
    {"pnr": "8433508120", "train": "S20081", "from": "NDLS", "to": "BSB", "date": "2026-11-25",
     "travel_class": "SL", "passengers": 3, "status": "RAC", "rac_position": 4},
    {"pnr": "8447761092", "train": "S20050", "from": "PUNE", "to": "SC", "date": "2026-12-02",
     "travel_class": "2A", "passengers": 2, "status": "CNF", "coach": "A1", "seats": ["17", "18"]},
    {"pnr": "8451226638", "train": "S20060", "from": "LTT", "to": "MAO", "date": "2026-12-05",
     "travel_class": "SL", "passengers": 4, "status": "WL", "waitlist_position": 3},
    {"pnr": "8469904417", "train": "S20120", "from": "MAS", "to": "SBC", "date": "2026-11-29",
     "travel_class": "EC", "passengers": 1, "status": "CNF", "coach": "E1", "seats": ["9"]},
    {"pnr": "8472218803", "train": "S20140", "from": "NDLS", "to": "ASR", "date": "2026-12-10",
     "travel_class": "CC", "passengers": 2, "status": "CNF", "coach": "C5", "seats": ["60", "61"]},
    {"pnr": "8486630251", "train": "S20090", "from": "NDLS", "to": "LKO", "date": "2026-11-27",
     "travel_class": "3A", "passengers": 2, "status": "CANCELLED"},
]


def main() -> None:
    files = {
        "rail_stations.json": build_rail(),
        "metro.json": build_metro(),
        "trains.json": build_trains(),
        "bookings.json": BOOKINGS,
        "fares.json": {"per_km": RATE, "reservation": RESERVATION, "currency": "INR",
                       "synthetic": True,
                       "metro_slabs": [[2, 10], [5, 20], [12, 30], [21, 40], [32, 50],
                                       [1000, 60]],
                       "metro_airport_express_surcharge": 30},
    }
    for name, obj in files.items():
        (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
    print({k: len(v) if isinstance(v, list) else "ok" for k, v in files.items()})


if __name__ == "__main__":
    main()
