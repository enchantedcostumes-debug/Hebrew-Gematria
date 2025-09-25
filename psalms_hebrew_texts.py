# Hebrew Texts of the Complete Book of Psalms
# Format: psalm_number: "hebrew_text"

psalms_hebrew_data = {
    # Start with some key psalms as examples
    1: """
    אשרי האיש אשר לא הלך בעצת רשעים 
    ובדרך חטאים לא עמד 
    ובמושב לצים לא ישב
    כי אם בתורת יהוה חפצו 
    ובתורתו יהגה יומם ולילה
    והיה כעץ שתול על פלגי מים 
    אשר פריו יתן בעתו ועלהו לא יבול 
    וכל אשר יעשה יצליח
    לא כן הרשעים כי אם כמוץ 
    אשר תדפנו רוח
    על כן לא יקמו רשעים במשפט 
    וחטאים בעדת צדיקים
    כי יודע יהוה דרך צדיקים 
    ודרך רשעים תאבד
    """,
    
    23: """
    מזמור לדוד יהוה רעי לא אחסר
    בנאות דשא ירביצני על מי מנחות ינהלני
    נפשי ישובב ינחני במעגלי צדק למען שמו
    גם כי אלך בגיא צלמות לא אירא רע 
    כי אתה עמדי שבטך ומשענתך המה ינחמני
    תערך לפני שלחן נגד צררי דשנת בשמן ראשי כוסי רויה
    אך טוב וחסד ירדפוני כל ימי חיי 
    ושבתי בבית יהוה לארך ימים
    """,
    
    # Add more psalms here as you get Hebrew texts
    # The system can analyze any number of psalms you provide
    
    # Placeholder for future psalms
    # 2: "hebrew text of psalm 2...",
    # 3: "hebrew text of psalm 3...",
    # ... continue up to psalm 150
}

# Function to get psalm text by number
def get_psalm_text(psalm_number: int) -> str:
    """Get Hebrew text for specific psalm."""
    return psalms_hebrew_data.get(psalm_number, "")

# Function to get all available psalms
def get_all_psalms() -> dict:
    """Get all available psalm texts."""
    return psalms_hebrew_data.copy()

# Statistics about available psalms
def get_psalms_stats() -> dict:
    """Get statistics about loaded psalms."""
    return {
        'total_psalms_available': len(psalms_hebrew_data),
        'psalm_numbers': sorted(psalms_hebrew_data.keys()),
        'total_characters': sum(len(text) for text in psalms_hebrew_data.values())
    }
