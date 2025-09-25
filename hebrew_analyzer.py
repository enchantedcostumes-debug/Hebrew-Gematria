# Hebrew Consciousness Mathematics Analysis Runner
from hebrew_analyzer import create_hebrew_lexicon_dataset
from hebrew_words_data import hebrew_word_list

def main():
    print("=" * 60)
    print("HEBREW CONSCIOUSNESS MATHEMATICS ANALYSIS")
    print("Testing Tammy Casey's Consciousness Interface Theory")
    print("=" * 60)
    
    print(f"Processing {len(hebrew_word_list)} Hebrew words...")
    
    # Run complete mathematical analysis
    results = create_hebrew_lexicon_dataset(hebrew_word_list)
    
    # Save results
    filename = 'hebrew_consciousness_analysis_results.csv'
    results.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"\nResults saved as: {filename}")
    
    # Display mathematical insights
    print(f"\nMATHEMATICAL ANALYSIS SUMMARY:")
    print(f"Total words analyzed: {len(results)}")
    print(f"Prime number words: {results['is_prime'].sum()}")
    print(f"Words with factor 2: {results['contains_factor_2'].sum()}")
    print(f"Words with factor 3: {results['contains_factor_3'].sum()}")
    print(f"Prime × 2 pattern words: {results['prime_times_2_pattern'].sum()}")
    
    # Show sample results
    print(f"\nFirst 5 analysis results:")
    print(results[['hebrew', 'gematria', 'prime_factorization', 'is_prime']].head())

if __name__ == "__main__":
    main()
