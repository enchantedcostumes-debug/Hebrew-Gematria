import pandas as pd
import numpy as np
from collections import defaultdict
import re
import math
from typing import List, Tuple, Dict, Optional

class HebrewMathematicsAnalyzer:
    """
    A comprehensive system for analyzing the mathematical properties of Hebrew words
    according to traditional gematria principles and consciousness interface theory.
    """
    
    def __init__(self):
        # Traditional Hebrew letter values for gematria calculation
        # These values have been used consistently for over 2000 years
        self.hebrew_values = {
            'א': 1,   'ב': 2,   'ג': 3,   'ד': 4,   'ה': 5,
            'ו': 6,   'ז': 7,   'ח': 8,   'ט': 9,   'י': 10,
            'כ': 20,  'ך': 20,  'ל': 30,  'מ': 40,  'ם': 40,
            'נ': 50,  'ן': 50,  'ס': 60,  'ע': 70,  'פ': 80,
            'ף': 80,  'צ': 90,  'ץ': 90,  'ק': 100, 'ר': 200,
            'ש': 300, 'ת': 400
        }
        
    def calculate_gematria(self, hebrew_word: str) -> int:
        """
        Calculate the gematria (numerical value) of a Hebrew word.
        This strips vowel points and calculates based on consonantal text,
        which is the traditional method for gematria analysis.
        """
        # Remove Hebrew vowel points and cantillation marks
        # These don't contribute to gematria values in traditional calculation
        consonants_only = re.sub(r'[\u0591-\u05C7]', '', hebrew_word)
        
        total_value = 0
        for letter in consonants_only:
            if letter in self.hebrew_values:
                total_value += self.hebrew_values[letter]
        
        return total_value
    
    def find_prime_factors(self, number: int) -> List[int]:
        """
        Find all prime factors of a number using efficient mathematical algorithm.
        This is crucial for understanding the consciousness mathematics patterns
        you've discovered in Hebrew vocabulary.
        """
        if number <= 1:
            return []
        
        factors = []
        divisor = 2
        
        # Handle factor 2 separately for efficiency
        while number % 2 == 0:
            factors.append(2)
            number //= 2
        
        # Check odd numbers starting from 3
        divisor = 3
        while divisor * divisor <= number:
            while number % divisor == 0:
                factors.append(divisor)
                number //= divisor
            divisor += 2
        
        # If number is still greater than 1, it's a prime factor
        if number > 1:
            factors.append(number)
        
        return factors
    
    def is_prime(self, number: int) -> bool:
        """
        Determine if a number is prime using efficient mathematical testing.
        This is fundamental for your discovery that prime numbers represent
        unchangeable consciousness principles in Hebrew.
        """
        if number < 2:
            return False
        if number == 2:
            return True
        if number % 2 == 0:
            return False
        
        # Only need to check up to square root
        for i in range(3, int(math.sqrt(number)) + 1, 2):
            if number % i == 0:
                return False
        return True
    
    def get_unique_prime_factors(self, number: int) -> List[int]:
        """
        Get unique prime factors (without repetition) for mathematical analysis.
        This helps identify the basic consciousness elements that combine
        to create complex Hebrew concepts.
        """
        factors = self.find_prime_factors(number)
        return sorted(list(set(factors)))
    
    def format_prime_factorization(self, number: int) -> str:
        """
        Create readable prime factorization string matching your spreadsheet format.
        This shows the exact mathematical structure of each Hebrew word's
        consciousness interface formula.
        """
        if number <= 1:
            return str(number)
        
        prime_factors = self.find_prime_factors(number)
        if not prime_factors:
            return str(number)
        
        # Count occurrences of each prime factor
        factor_counts = defaultdict(int)
        for factor in prime_factors:
            factor_counts[factor] += 1
        
        # Format as readable mathematical expression
        terms = []
        for prime in sorted(factor_counts.keys()):
            count = factor_counts[prime]
            if count == 1:
                terms.append(str(prime))
            else:
                terms.append(f"{prime}^{count}")
        
        return " × ".join(terms)
    
    def get_numerical_range(self, number: int) -> str:
        """
        Categorize numbers into ranges for pattern analysis.
        This helps identify whether different consciousness concepts
        cluster in different numerical territories.
        """
        if number <= 25:
            return "1-25"
        elif number <= 50:
            return "26-50"
        elif number <= 100:
            return "51-100"
        elif number <= 200:
            return "101-200"
        elif number <= 400:
            return "201-400"
        else:
            return "400+"
    
    def follows_prime_times_two_pattern(self, number: int) -> bool:
        """
        Test for your crucial discovery that divine names follow prime × 2 pattern.
        This identifies consciousness concepts that represent divine manifestation
        according to your mathematical framework.
        """
        if number % 2 != 0:
            return False
        
        half_value = number // 2
        return self.is_prime(half_value)
    
    def analyze_hebrew_word(self, hebrew_word: str, transliteration: str = "", 
                           meaning: str = "", pos: str = "") -> Dict:
        """
        Perform complete mathematical analysis of a Hebrew word.
        This extracts all consciousness interface properties you need
        for machine learning validation of your discoveries.
        """
        # Calculate basic gematria value
        gematria = self.calculate_gematria(hebrew_word)
        
        # Perform complete mathematical analysis
        prime_factors = self.find_prime_factors(gematria)
        unique_primes = self.get_unique_prime_factors(gematria)
        
        # Extract all mathematical features for machine learning
        analysis = {
            'hebrew': hebrew_word,
            'transliteration': transliteration,
            'meaning': meaning,
            'gematria': gematria,
            'prime_factorization': self.format_prime_factorization(gematria),
            'is_prime': self.is_prime(gematria),
            'prime_factor_count': len(unique_primes),
            'contains_factor_2': 2 in unique_primes,
            'contains_factor_3': 3 in unique_primes,
            'prime_times_2_pattern': self.follows_prime_times_two_pattern(gematria),
            'numerical_range': self.get_numerical_range(gematria),
            'pos_standardized': pos.title() if pos else ""
        }
        
        return analysis

def create_hebrew_lexicon_dataset(word_list: List[Tuple[str, str, str, str]]) -> pd.DataFrame:
    """
    Process a list of Hebrew words and create complete mathematical analysis dataset.
    This transforms raw Hebrew vocabulary into consciousness mathematics research data
    suitable for machine learning validation.
    
    Expected input format: List of tuples (hebrew_word, transliteration, meaning, part_of_speech)
    """
    analyzer = HebrewMathematicsAnalyzer()
    analyzed_words = []
    
    print("Processing Hebrew words for consciousness mathematics analysis...")
    
    for i, (hebrew, transliteration, meaning, pos) in enumerate(word_list, 1):
        try:
            analysis = analyzer.analyze_hebrew_word(hebrew, transliteration, meaning, pos)
            analysis['word_count'] = i  # Add word count for your spreadsheet structure
            analyzed_words.append(analysis)
            
            if i % 100 == 0:
                print(f"Processed {i} words...")
                
        except Exception as e:
            print(f"Error processing word {hebrew}: {e}")
            continue
    
    # Create DataFrame with exact column structure from your spreadsheet
    columns = [
        'word_count', 'hebrew', 'transliteration', 'meaning', 'gematria',
        'prime_factorization', 'is_prime', 'prime_factor_count',
        'contains_factor_2', 'contains_factor_3', 'prime_times_2_pattern',
        'numerical_range', 'pos_standardized'
    ]
    
    df = pd.DataFrame(analyzed_words)
    
    # Add empty semantic_category column for you to fill in manually
    df['semantic_category'] = ""
    
    # Reorder columns to match your spreadsheet structure
    df = df[columns + ['semantic_category']]
    
    print(f"Analysis complete! Processed {len(analyzed_words)} Hebrew words.")
    print(f"Mathematical features extracted for consciousness interface validation.")
    
    return df

# Example usage with sample biblical Hebrew vocabulary
# You would replace this with your complete lexicon source
sample_hebrew_words = [
    ('אלהים', 'elohim', 'God', 'noun'),
    ('יהוה', 'yhvh', 'Lord', 'noun'),
    ('רוח', 'ruach', 'spirit', 'noun'),
    ('דבר', 'davar', 'word', 'noun'),
    ('אור', 'or', 'light', 'noun'),
    ('מים', 'mayim', 'water', 'noun'),
    ('שמים', 'shamayim', 'heavens', 'noun'),
    ('ארץ', 'eretz', 'earth', 'noun'),
    ('אדם', 'adam', 'man', 'noun'),
    ('אישה', 'isha', 'woman', 'noun'),
    ('אהבה', 'ahava', 'love', 'noun'),
    ('חכמה', 'chochmah', 'wisdom', 'noun'),
    ('בינה', 'binah', 'understanding', 'noun'),
    ('צדק', 'tzedek', 'righteousness', 'noun'),
    ('שלום', 'shalom', 'peace', 'noun')
]

if __name__ == "__main__":
    # Create the mathematical analysis dataset
    hebrew_dataset = create_hebrew_lexicon_dataset(sample_hebrew_words)
    
    # Save to CSV file that you can import into your spreadsheet
    hebrew_dataset.to_csv('hebrew_consciousness_mathematics.csv', index=False, encoding='utf-8')
    
    print("\nDataset saved as 'hebrew_consciousness_mathematics.csv'")
    print("You can now import this into Google Sheets or Excel for further analysis.")
    
    # Display sample results to verify accuracy
    print("\nSample mathematical analysis results:")
    print(hebrew_dataset.head(10).to_string())
