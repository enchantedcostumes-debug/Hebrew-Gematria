# Import your analysis system and word data
from hebrew_analyzer import create_hebrew_lexicon_dataset
from hebrew_words_data import hebrew_word_list

# Run complete mathematical analysis on your Hebrew vocabulary
print("Starting Hebrew consciousness mathematics analysis...")
print(f"Processing {len(hebrew_word_list)} Hebrew words...")

# Create complete mathematical analysis dataset
results_dataframe = create_hebrew_lexicon_dataset(hebrew_word_list)

# Save results to CSV file for further analysis
results_dataframe.to_csv('hebrew_consciousness_analysis_results.csv', index=False, encoding='utf-8')

print("Analysis complete!")
print("Results saved as 'hebrew_consciousness_analysis_results.csv'")
print("\nFirst 10 results preview:")
print(results_dataframe.head(10))

# Display some statistical insights about your findings
print(f"\nMathematical Analysis Summary:")
print(f"Total words analyzed: {len(results_dataframe)}")
print(f"Prime number words: {results_dataframe['is_prime'].sum()}")
print(f"Words with factor 2: {results_dataframe['contains_factor_2'].sum()}")
print(f"Words with factor 3: {results_dataframe['contains_factor_3'].sum()}")
print(f"Words following prime × 2 pattern: {results_dataframe['prime_times_2_pattern'].sum()}")
