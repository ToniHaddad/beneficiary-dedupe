"""Name and place tables for the synthetic generator.

Each entry is (Arabic spelling, [Latin spellings], weight). The first Latin
spelling is the most common one; the others are transliterations that field
teams really use. Weights shape how common a name is, so that common names
("Mohammad", "Hajj") collide the way they do in real registration lists.
"""

from __future__ import annotations

MALE_GIVEN: list[tuple[str, list[str], int]] = [
    ("محمد", ["Mohammad", "Mohamed", "Muhammad", "Mohammed", "Muhammed", "Mhammad"], 14),
    ("أحمد", ["Ahmad", "Ahmed"], 7),
    ("حسين", ["Hussein", "Hussain", "Hsein", "Husayn", "Hussien"], 6),
    ("علي", ["Ali", "Aly"], 7),
    ("حسن", ["Hassan", "Hasan"], 5),
    ("خالد", ["Khaled", "Khalid"], 4),
    ("محمود", ["Mahmoud", "Mahmud", "Mahmood"], 4),
    ("يوسف", ["Youssef", "Yousef", "Yusuf", "Youssif"], 4),
    ("إبراهيم", ["Ibrahim", "Ebrahim", "Brahim"], 4),
    ("عمر", ["Omar", "Omer", "Umar"], 3),
    ("عباس", ["Abbas", "Abas"], 3),
    ("فادي", ["Fadi", "Fady"], 2),
    ("جورج", ["Georges", "George"], 2),
    ("إيلي", ["Elie", "Eli", "Ely"], 2),
    ("رامي", ["Rami", "Ramy"], 2),
    ("سمير", ["Samir", "Sameer"], 2),
    ("وليد", ["Walid", "Waleed"], 2),
    ("زياد", ["Ziad", "Ziyad"], 2),
    ("بلال", ["Bilal", "Belal"], 2),
    ("نبيل", ["Nabil", "Nabeel"], 2),
    ("جهاد", ["Jihad", "Jehad"], 2),
    ("مصطفى", ["Mustafa", "Moustafa", "Mostafa"], 3),
    ("عيسى", ["Issa", "Eissa", "Isa"], 1),
    ("عبد الله", ["Abdallah", "Abdullah", "Abd Allah", "Abdellah"], 3),
    ("إسماعيل", ["Ismail", "Ismael", "Smail"], 2),
    ("قاسم", ["Kassem", "Qasem", "Kassim", "Qassem"], 2),
    ("وائل", ["Wael", "Wa'el"], 2),
    ("أسامة", ["Oussama", "Osama", "Usama"], 2),
    ("طارق", ["Tarek", "Tariq", "Tarik"], 2),
    ("شادي", ["Chadi", "Shadi"], 1),
]

FEMALE_GIVEN: list[tuple[str, list[str], int]] = [
    ("فاطمة", ["Fatima", "Fatme", "Fatmeh", "Fatma"], 6),
    ("زينب", ["Zeinab", "Zainab", "Zaynab"], 4),
    ("مريم", ["Mariam", "Maryam", "Miriam"], 3),
    ("خديجة", ["Khadija", "Khadijeh", "Khadije"], 3),
    ("أمل", ["Amal"], 2),
    ("ليلى", ["Layla", "Leila", "Laila"], 3),
    ("حنان", ["Hanan"], 2),
    ("رانيا", ["Rania", "Ranya"], 2),
    ("نور", ["Nour", "Nur", "Noor"], 2),
    ("سميرة", ["Samira", "Sameera"], 2),
    ("هدى", ["Houda", "Huda", "Hoda"], 2),
    ("سعاد", ["Souad", "Suad", "Soaad"], 2),
]

FAMILY: list[tuple[str, list[str], int]] = [
    ("الخوري", ["Khoury", "Khouri", "El Khoury", "Al-Khouri", "El-Khoury"], 5),
    ("حداد", ["Haddad", "Hadad", "El Haddad"], 4),
    ("الحاج", ["Hajj", "El Hajj", "Al-Hajj", "Hage"], 7),
    ("صالح", ["Saleh", "Salih"], 3),
    ("موسى", ["Moussa", "Mousa", "Musa"], 3),
    ("حمدان", ["Hamdan"], 3),
    ("ناصر", ["Nasser", "Naser"], 3),
    ("شهاب", ["Chehab", "Shehab", "Shihab"], 3),
    ("شمعون", ["Chamoun", "Shamoun"], 2),
    ("الفاخوري", ["Fakhoury", "El Fakhoury", "Fakhouri"], 2),
    ("أيوب", ["Ayoub", "Ayyoub", "Ayub"], 2),
    ("درويش", ["Darwish", "Darwiche", "Darwich"], 3),
    ("سرور", ["Srour", "Sorour"], 2),
    ("الزين", ["Zein", "El Zein", "Al-Zain"], 3),
    ("المصري", ["Masri", "El Masri", "Al-Masri"], 3),
    ("الحلبي", ["Halabi", "El Halabi"], 2),
    ("جابر", ["Jaber", "Jabir"], 3),
    ("خليل", ["Khalil", "Khaleel"], 3),
    ("منصور", ["Mansour", "Mansur"], 3),
    ("سليمان", ["Sleiman", "Suleiman", "Slayman", "Soliman"], 3),
    ("حمود", ["Hammoud", "Hamoud"], 3),
    ("عواضة", ["Awada", "Awwada", "Aouada"], 2),
    ("فارس", ["Fares", "Faris"], 2),
    ("قاسم", ["Kassem", "Qassem", "Kasem"], 2),
    ("عيسى", ["Issa", "Eissa"], 2),
    ("الأسعد", ["Assaad", "El Assaad", "Al-Asaad"], 2),
    ("طالب", ["Taleb", "Talib"], 2),
    ("فقيه", ["Fakih", "Faqih", "Fakhih"], 2),
    ("شريف", ["Cherif", "Sharif", "Chérif"], 2),
    ("يونس", ["Younes", "Yunis", "Younis"], 2),
]

# (district, Arabic village name, [Latin spellings])
VILLAGES: list[tuple[str, str, list[str]]] = [
    ("Akkar", "حلبا", ["Halba", "Halbah"]),
    ("Akkar", "ببنين", ["Bebnine", "Bebnin", "Bibnin"]),
    ("Akkar", "القبيات", ["Qobayat", "Kobayat", "Al Qoubaiyat"]),
    ("Baalbek", "عرسال", ["Arsal", "Ersal", "Aarsal"]),
    ("Baalbek", "بريتال", ["Britel", "Brital"]),
    ("Baalbek", "دورس", ["Douris", "Dures"]),
    ("Zahle", "برالياس", ["Bar Elias", "Bar Elyas", "Barelias"]),
    ("Zahle", "قب الياس", ["Qabb Elias", "Kab Elias", "Qab Elias"]),
    ("Zahle", "زحلة", ["Zahle", "Zahleh", "Zahlé"]),
    ("Tyre", "قانا", ["Qana", "Kana"]),
    ("Tyre", "برج الشمالي", ["Burj el Shemali", "Borj Chemali", "Burj Shamali"]),
    ("Tyre", "معركة", ["Maarake", "Maaraka", "Marakeh"]),
    ("Nabatieh", "كفررمان", ["Kfar Roummane", "Kfarremane", "Kfar Rumman"]),
    ("Nabatieh", "حبوش", ["Habbouch", "Habboush"]),
    ("Marjayoun", "الخيام", ["Khiam", "El Khiam", "Al-Khiam"]),
    ("Bint Jbeil", "عيترون", ["Aitaroun", "Aytaroun", "Ayta Roun"]),
    ("Tripoli", "القبة", ["Qobbeh", "Kobbe", "Al Qubba"]),
    ("Tripoli", "الميناء", ["Mina", "El Mina", "Al-Mina"]),
    ("Saida", "صيدا", ["Saida", "Saïda", "Sayda"]),
    ("Jbeil", "جبيل", ["Jbeil", "Jbail"]),
    ("Aley", "عاليه", ["Aley", "Aaley", "Alay"]),
]

# Districts that empty out when fighting moves north: their households end up
# in school shelters elsewhere.
DISPLACEMENT_DISTRICTS = {"Tyre", "Nabatieh", "Marjayoun", "Bint Jbeil", "Baalbek"}
# Towns where the Organization runs school shelters.
SHELTER_TOWNS = ["Saida", "Jbeil", "Aley", "Zahle", "Qobbeh"]

# Every district has one of the Organization's centres; registrations are
# credited to the centre whose team took them.
CENTRES = {district: f"{district} centre" for district, _, _ in VILLAGES}

TEAM_SOCIAL = "Centre social department"
TEAM_SHELTER = "Mobile team, school shelter"
TEAM_DISTRIBUTION = "Distribution point"
TEAM_PAPER = "Paper list, typed up later"
TEAMS = [TEAM_SOCIAL, TEAM_SHELTER, TEAM_DISTRIBUTION, TEAM_PAPER]

# Teams that work on paper in Arabic more often than not.
ARABIC_TEAMS = {TEAM_PAPER, TEAM_SOCIAL}