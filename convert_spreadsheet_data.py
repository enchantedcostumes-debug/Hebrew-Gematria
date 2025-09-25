import pandas as pd
import csv

def convert_csv_to_hebrew_data(csv_filename):
    """
    Convert your Hebrew words CSV file into Python format
    that the analysis system can use.
    """
    try:
        # Read your CSV file
        df = pd.read_csv(csv_filename, encoding='utf-8')
        
        print(f"Loaded {len(df)} Hebrew words from spreadsheet")
        print("Converting to Python format...")
        
        # Create the Python list format
        hebrew_word_list = []
        
        for index, row in df.iterrows():
            # Extract data from your spreadsheet columns
            # Adjust these column names to match your spreadsheet exactly
            hebrew_word = str(row.get('hebrew', ''))  # Your Hebrew column
            transliteration = str(row.get('transliteration', ''))  # Your transliteration column
            meaning = str(row.get('meaning', ''))  # Your meaning column
            pos = str(row.get('POS_Standardized', ''))  # Your part of speech column
            
            # Skip empty rows
            if hebrew_word and hebrew_word != 'nan':
                hebrew_word_list.append((hebrew_word, transliteration, meaning, pos))
        
        print(f"Successfully converted {len(hebrew_word_list)} words")
        
        # Generate the Python file content
        python_content = '''# Hebrew Words Database - Converted from Spreadsheet
# Generated from Tammy Casey's Hebrew Consciousness Mathematics Research

hebrew_word_list = [
'''
        
        # Add each word in proper Python format
        for hebrew, trans, meaning, pos in hebrew_word_list:
            # Escape single quotes in the data
            hebrew = hebrew.replace("'", "\\'")
            trans = trans.replace("'", "\\'")
            meaning = meaning.replace("'", "\\'")
            pos = pos.replace("'", "\\'")
            
            python_content += f"    ('{hebrew}', '{trans}', '{meaning}', '{pos}'),\n"
        
        python_content += '''
]

# Function to get all Hebrew words
def get_all_hebrew_words():
    """Return the complete list of Hebrew words for analysis."""
    return hebrew_word_list.copy()

# Statistics about the database
def get_hebrew_stats():
    """Get statistics about the Hebrew word database."""
    return {
        'total_words': len(hebrew_word_list),
        'unique_meanings': len(set(word[2] for word in hebrew_word_list)),
        'parts_of_speech': list(set(word[3] for word in hebrew_word_list if word[3]))
    }

print(f"Hebrew word database loaded: {len(hebrew_word_list)} words")
'''
        
        # Save the Python file
        with open('hebrew_words_data.py', 'w', encoding='utf-8') as f:
            f.write(python_content)
        
        print("✓ Created hebrew_words_data.py with your complete database")
        print("✓ Ready for consciousness mathematics analysis!")
        
        return len(hebrew_word_list)
        
    except Exception as e:
        print(f"Error converting data: {e}")
        print("Please check that your CSV file is in the same folder and the column names match")
        return 0

if __name__ == "__main__":
    print("HEBREW WORDS DATABASE CONVERTER")
    print("Converting spreadsheet data to Python format...")
    print("=" * 50)
    
    # Replace 'your_file.csv' with the actual name of your downloaded CSV file
    csv_filename = input("Enter your CSV filename (like 'hebrew_words_analysis.csv'): ")
    
    word_count = convert_csv_to_hebrew_data(csv_filename)
    
    if word_count > 0:
        print(f"\n🎉 SUCCESS! Converted {word_count} Hebrew words")
        print("Now you can run: python run_analysis.py")
    else:
        print("\n❌ Conversion failed. Please check your file and try again.")
