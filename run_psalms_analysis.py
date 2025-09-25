# Complete Psalms Consciousness Mathematics Analysis Runner
from psalms_analyzer import analyze_complete_psalms, PsalmsConsciousnessAnalyzer
from psalms_hebrew_texts import get_all_psalms, get_psalms_stats

def main():
    print("=" * 80)
    print("PSALMS CONSCIOUSNESS MATHEMATICS ANALYSIS SYSTEM")
    print("Discovering the Mathematical Cosmos of Hebrew Spiritual Protocols")
    print("=" * 80)
    print()
    
    # Get psalm statistics
    stats = get_psalms_stats()
    print(f"AVAILABLE PSALM DATA:")
    print(f"Total Psalms Available: {stats['total_psalms_available']}")
    print(f"Psalm Numbers: {stats['psalm_numbers']}")
    print(f"Total Characters: {stats['total_characters']:,}")
    print()
    
    # Get all available psalm texts
    psalms_data = get_all_psalms()
    
    if not psalms_data:
        print("No psalm texts available. Please add Hebrew texts to psalms_hebrew_texts.py")
        return
    
    print("STARTING COMPLETE PSALMS ANALYSIS...")
    print("This will analyze mathematical patterns in all available psalms...")
    print()
    
    # Run the complete analysis
    try:
        results = analyze_complete_psalms(psalms_data)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        
        # Display key discoveries
        total_psalms = len(results['individual_psalms'])
        total_consciousness_markers = sum(
            len(psalm['consciousness_markers_found']) 
            for psalm in results['individual_psalms'].values()
        )
        
        print(f"Psalms Analyzed: {total_psalms}")
        print(f"Total Consciousness Markers Found: {total_consciousness_markers}")
        print(f"Average Markers per Psalm: {total_consciousness_markers/total_psalms:.2f}")
        
        # Show high consciousness psalms
        if 'consciousness_map' in results:
            high_consciousness = results['consciousness_map']['high_consciousness_psalms']
            if high_consciousness:
                print(f"High Consciousness Psalms (3+ markers): {high_consciousness}")
        
        print(f"\nDetailed results saved as:")
        print(f"- psalms_complete_analysis.json")
        print(f"- psalms_complete_analysis.png")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your Hebrew text formatting in psalms_hebrew_texts.py")
        return None

def analyze_single_psalm_demo(psalm_number: int = 1):
    """Demonstrate analysis of a single psalm."""
    from psalms_hebrew_texts import get_psalm_text
    
    psalm_text = get_psalm_text(psalm_number)
    if not psalm_text:
        print(f"No Hebrew text available for Psalm {psalm_number}")
        return
    
    print(f"ANALYZING PSALM {psalm_number}")
    print("=" * 40)
    
    analyzer = PsalmsConsciousnessAnalyzer()
    result = analyzer.analyze_single_psalm(psalm_text, psalm_number)
    
    print(f"Total Gematria: {result['total_gematria']:,}")
    print(f"Total Words: {result['total_words']}")
    print(f"Total Letters: {result['total_letters']}")
    print(f"Consciousness Markers Found: {len(result['consciousness_markers_found'])}")
    print(f"Spiritual Category: {result['spiritual_category']}")
    
    if result['consciousness_markers_found']:
        print(f"\nConsciousness Markers:")
        for marker in result['consciousness_markers_found']:
            print(f"  {marker['word']} = {marker['value']} ({marker['meaning']})")
    
    return result

if __name__ == "__main__":
    # Run complete analysis
    main()
    
    # Uncomment to run single psalm demo
    # analyze_single_psalm_demo(23)  # Analyze Psalm 23
