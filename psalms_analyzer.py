import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Optional
import json

class PsalmsConsciousnessAnalyzer:
    """
    Advanced system for analyzing the consciousness mathematics embedded
    in the complete Book of Psalms.
    
    This system tests whether the 150 Psalms contain systematic mathematical
    patterns that reveal consciousness interface protocols used for spiritual
    transformation and reality creation.
    """
    
    def __init__(self):
        # Traditional Hebrew gematria values
        self.hebrew_values = {
            'א': 1,   'ב': 2,   'ג': 3,   'ד': 4,   'ה': 5,
            'ו': 6,   'ז': 7,   'ח': 8,   'ט': 9,   'י': 10,
            'כ': 20,  'ך': 20,  'ל': 30,  'מ': 40,  'ם': 40,
            'נ': 50,  'ן': 50,  'ס': 60,  'ע': 70,  'פ': 80,
            'ף': 80,  'צ': 90,  'ץ': 90,  'ק': 100, 'ר': 200,
            'ש': 300, 'ת': 400
        }
        
        # Consciousness interface markers discovered in your research
        self.consciousness_markers = {
            26: "YHVH - Divine Name",
            86: "Elohim - God",
            214: "Ruach - Spirit",
            37: "Heart/Core - Wisdom",
            73: "Chokmah - Wisdom", 
            67: "Binah - Understanding",
            137: "Kabbalah - Reception",
            358: "Mashiach - Messiah",
            359: "Satan - Adversary",
            888: "Jesus (Greek equivalent)",
            13: "Ahava - Love/Unity",
            18: "Chai - Life",
            72: "Chesed - Loving Kindness",
            541: "Israel - God Wrestling",
            611: "Torah - Teaching",
            620: "Keter - Crown"
        }
        
        # Psalm categories for consciousness analysis
        self.psalm_categories = {
            'Lament': [3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 17, 22, 25, 26, 28, 31, 35, 36, 38, 39, 40, 41, 42, 43, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 64, 69, 70, 71, 74, 77, 79, 80, 83, 85, 86, 88, 90, 102, 109, 120, 123, 126, 130, 137, 140, 141, 142, 143],
            'Praise': [8, 19, 24, 29, 33, 47, 48, 65, 66, 67, 68, 84, 87, 93, 95, 96, 97, 98, 100, 103, 104, 105, 106, 107, 111, 113, 114, 115, 116, 117, 118, 134, 135, 136, 138, 139, 145, 146, 147, 148, 149, 150],
            'Wisdom': [1, 15, 16, 32, 34, 37, 49, 50, 62, 73, 78, 91, 92, 94, 112, 119, 121, 127, 128, 131, 133],
            'Royal': [2, 18, 20, 21, 45, 72, 89, 101, 110, 132, 144],
            'Thanksgiving': [11, 23, 27, 30, 46, 75, 76, 81, 82, 99, 108, 122, 124, 125, 129]
        }
    
    def calculate_gematria(self, hebrew_text: str) -> int:
        """Calculate gematria value of Hebrew text."""
        clean_text = re.sub(r'[\u0591-\u05C7\s]', '', hebrew_text)
        total = 0
        for char in clean_text:
            if char in self.hebrew_values:
                total += self.hebrew_values[char]
        return total
    
    def analyze_single_psalm(self, psalm_text: str, psalm_number: int) -> Dict:
        """
        Analyze mathematical patterns in a single psalm.
        This reveals the consciousness interface protocol for each psalm.
        """
        # Clean and extract words
        words = self.extract_words(psalm_text)
        
        analysis = {
            'psalm_number': psalm_number,
            'total_gematria': self.calculate_gematria(psalm_text),
            'total_words': len(words),
            'total_letters': len([c for c in re.sub(r'[\u0591-\u05C7\s]', '', psalm_text) if c in self.hebrew_values]),
            'words_analysis': [],
            'consciousness_markers_found': [],
            'mathematical_patterns': {},
            'spiritual_category': self.get_psalm_category(psalm_number),
            'prime_factors': [],
            'cumulative_progression': []
        }
        
        # Analyze each word in the psalm
        running_total = 0
        for i, word in enumerate(words, 1):
            word_gematria = self.calculate_gematria(word)
            running_total += word_gematria
            
            word_data = {
                'position': i,
                'hebrew_word': word,
                'gematria': word_gematria,
                'running_total': running_total,
                'is_prime': self.is_prime(word_gematria),
                'prime_factors': self.find_prime_factors(word_gematria),
                'consciousness_marker': word_gematria in self.consciousness_markers
            }
            
            analysis['words_analysis'].append(word_data)
            analysis['cumulative_progression'].append(running_total)
            
            # Track consciousness markers
            if word_gematria in self.consciousness_markers:
                analysis['consciousness_markers_found'].append({
                    'word': word,
                    'position': i,
                    'value': word_gematria,
                    'meaning': self.consciousness_markers[word_gematria]
                })
        
        # Analyze mathematical patterns in this psalm
        analysis['mathematical_patterns'] = self.find_psalm_patterns(analysis)
        analysis['prime_factors'] = self.find_prime_factors(analysis['total_gematria'])
        
        return analysis
    
    def analyze_complete_psalms(self, psalms_data: Dict[int, str]) -> Dict:
        """
        Analyze mathematical patterns across all 150 Psalms.
        This reveals the consciousness architecture of the complete collection.
        """
        print("ANALYZING THE COMPLETE BOOK OF PSALMS")
        print("Discovering consciousness mathematics in 150 spiritual interface protocols...")
        print("=" * 80)
        
        complete_analysis = {
            'individual_psalms': {},
            'collection_patterns': {},
            'consciousness_map': {},
            'mathematical_relationships': {},
            'category_analysis': {},
            'cosmic_patterns': {}
        }
        
        # Analyze each individual psalm
        for psalm_num, psalm_text in psalms_data.items():
            print(f"Processing Psalm {psalm_num}...")
            psalm_analysis = self.analyze_single_psalm(psalm_text, psalm_num)
            complete_analysis['individual_psalms'][psalm_num] = psalm_analysis
        
        # Analyze patterns across all psalms
        print("\nDiscovering patterns across complete collection...")
        complete_analysis['collection_patterns'] = self.analyze_collection_patterns(
            complete_analysis['individual_psalms']
        )
        
        # Map consciousness markers across psalms
        complete_analysis['consciousness_map'] = self.create_consciousness_map(
            complete_analysis['individual_psalms']
        )
        
        # Find mathematical relationships between psalms
        complete_analysis['mathematical_relationships'] = self.find_inter_psalm_relationships(
            complete_analysis['individual_psalms']
        )
        
        # Analyze by spiritual categories
        complete_analysis['category_analysis'] = self.analyze_by_categories(
            complete_analysis['individual_psalms']
        )
        
        # Look for cosmic/universal patterns
        complete_analysis['cosmic_patterns'] = self.find_cosmic_patterns(
            complete_analysis['individual_psalms']
        )
        
        return complete_analysis
    
    def find_psalm_patterns(self, psalm_analysis: Dict) -> Dict:
        """Find mathematical patterns within a single psalm."""
        words_values = [w['gematria'] for w in psalm_analysis['words_analysis']]
        cumulative = psalm_analysis['cumulative_progression']
        
        patterns = {
            'fibonacci_positions': self.find_fibonacci_in_sequence(words_values),
            'prime_positions': [i for i, w in enumerate(psalm_analysis['words_analysis']) if w['is_prime']],
            'golden_ratio_points': self.find_golden_ratio_points(cumulative),
            'perfect_numbers': self.find_perfect_numbers(words_values),
            'consciousness_progressions': self.find_consciousness_progressions(cumulative),
            'sacred_number_sequences': self.find_sacred_sequences(words_values)
        }
        
        return patterns
    
    def analyze_collection_patterns(self, all_psalms: Dict) -> Dict:
        """
        Analyze mathematical patterns across the complete Psalms collection.
        This reveals the consciousness architecture of the entire work.
        """
        patterns = {
            'gematria_progression': [],
            'prime_psalm_numbers': [],
            'consciousness_density_map': {},
            'mathematical_symmetries': {},
            'sacred_number_distribution': {},
            'psalm_mathematical_relationships': {}
        }
        
        # Collect gematria values for all psalms
        psalm_totals = []
        for psalm_num in sorted(all_psalms.keys()):
            psalm_data = all_psalms[psalm_num]
            psalm_totals.append({
                'psalm_number': psalm_num,
                'total_gematria': psalm_data['total_gematria'],
                'is_prime': self.is_prime(psalm_data['total_gematria']),
                'consciousness_markers': len(psalm_data['consciousness_markers_found'])
            })
        
        patterns['gematria_progression'] = psalm_totals
        
        # Find psalms with prime number totals
        patterns['prime_psalm_numbers'] = [p['psalm_number'] for p in psalm_totals if p['is_prime']]
        
        # Create consciousness density map (markers per psalm)
        for psalm_data in psalm_totals:
            patterns['consciousness_density_map'][psalm_data['psalm_number']] = psalm_data['consciousness_markers']
        
        # Look for mathematical relationships between psalm numbers and content
        patterns['psalm_mathematical_relationships'] = self.find_number_content_correlations(all_psalms)
        
        return patterns
    
    def create_consciousness_map(self, all_psalms: Dict) -> Dict:
        """
        Create a map of consciousness markers throughout the Psalms.
        This shows how consciousness interface protocols are distributed.
        """
        consciousness_map = {
            'marker_distribution': defaultdict(list),
            'psalm_consciousness_profiles': {},
            'high_consciousness_psalms': [],
            'consciousness_pathways': {}
        }
        
        # Map where each consciousness marker appears
        for psalm_num, psalm_data in all_psalms.items():
            consciousness_count = len(psalm_data['consciousness_markers_found'])
            
            # Track high consciousness psalms (more than 3 markers)
            if consciousness_count >= 3:
                consciousness_map['high_consciousness_psalms'].append(psalm_num)
            
            # Create consciousness profile for each psalm
            consciousness_map['psalm_consciousness_profiles'][psalm_num] = {
                'total_markers': consciousness_count,
                'markers_found': psalm_data['consciousness_markers_found'],
                'consciousness_density': consciousness_count / psalm_data['total_words'] if psalm_data['total_words'] > 0 else 0
            }
            
            # Track distribution of each marker type
            for marker in psalm_data['consciousness_markers_found']:
                consciousness_map['marker_distribution'][marker['meaning']].append(psalm_num)
        
        return consciousness_map
    
    def find_inter_psalm_relationships(self, all_psalms: Dict) -> Dict:
        """
        Find mathematical relationships between different psalms.
        This could reveal how psalms work together as consciousness protocols.
        """
        relationships = {
            'gematria_multiples': {},
            'complementary_pairs': [],
            'mathematical_progressions': [],
            'consciousness_clusters': {}
        }
        
        psalm_values = {num: data['total_gematria'] for num, data in all_psalms.items()}
        
        # Find psalms that are mathematical multiples of each other
        for psalm1, value1 in psalm_values.items():
            multiples = []
            for psalm2, value2 in psalm_values.items():
                if psalm1 != psalm2 and value2 != 0:
                    if value1 % value2 == 0 or value2 % value1 == 0:
                        multiples.append((psalm2, value2, max(value1, value2) // min(value1, value2)))
            
            if multiples:
                relationships['gematria_multiples'][psalm1] = multiples
        
        # Find complementary pairs (psalms that sum to significant numbers)
        for psalm1, value1 in psalm_values.items():
            for psalm2, value2 in psalm_values.items():
                if psalm1 < psalm2:  # Avoid duplicates
                    sum_value = value1 + value2
                    if sum_value in self.consciousness_markers:
                        relationships['complementary_pairs'].append({
                            'psalm1': psalm1,
                            'psalm2': psalm2,
                            'sum': sum_value,
                            'significance': self.consciousness_markers[sum_value]
                        })
        
        return relationships
    
    def analyze_by_categories(self, all_psalms: Dict) -> Dict:
        """
        Analyze mathematical patterns within each spiritual category of psalms.
        This reveals consciousness mathematics of different spiritual functions.
        """
        category_analysis = {}
        
        for category_name, psalm_numbers in self.psalm_categories.items():
            category_psalms = {num: all_psalms[num] for num in psalm_numbers if num in all_psalms}
            
            if not category_psalms:
                continue
            
            category_analysis[category_name] = {
                'psalm_count': len(category_psalms),
                'average_gematria': np.mean([p['total_gematria'] for p in category_psalms.values()]),
                'total_consciousness_markers': sum(len(p['consciousness_markers_found']) for p in category_psalms.values()),
                'consciousness_density': 0,  # Will calculate below
                'mathematical_patterns': {},
                'category_specific_markers': []
            }
            
            # Calculate consciousness density
            total_words = sum(p['total_words'] for p in category_psalms.values())
            if total_words > 0:
                category_analysis[category_name]['consciousness_density'] = (
                    category_analysis[category_name]['total_consciousness_markers'] / total_words
                )
        
        return category_analysis
    
    def find_cosmic_patterns(self, all_psalms: Dict) -> Dict:
        """
        Look for cosmic/universal mathematical patterns in the Psalms.
        This searches for consciousness mathematics that mirrors natural law.
        """
        cosmic_patterns = {
            'fibonacci_psalms': [],
            'pi_approximations': [],
            'golden_ratio_relationships': [],
            'platonic_solid_numbers': [],
            'astronomical_correlations': {}
        }
        
        # Find psalms with Fibonacci number totals
        fibonacci_sequence = self.generate_fibonacci(10000)  # Generate up to reasonable limit
        
        for psalm_num, psalm_data in all_psalms.items():
            total = psalm_data['total_gematria']
            
            if total in fibonacci_sequence:
                cosmic_patterns['fibonacci_psalms'].append({
                    'psalm': psalm_num,
                    'gematria': total,
                    'fibonacci_position': fibonacci_sequence.index(total)
                })
        
        return cosmic_patterns
    
    def create_comprehensive_visualization(self, complete_analysis: Dict):
        """
        Create comprehensive visualization of Psalms consciousness mathematics.
        This maps the mathematical cosmos of the complete Psalms collection.
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Gematria progression across all 150 psalms
        ax1 = fig.add_subplot(gs[0, :])
        psalm_nums = sorted(complete_analysis['individual_psalms'].keys())
        gematria_values = [complete_analysis['individual_psalms'][num]['total_gematria'] for num in psalm_nums]
        
        ax1.plot(psalm_nums, gematria_values, 'b-', alpha=0.7, linewidth=1)
        ax1.scatter(psalm_nums, gematria_values, c='blue', s=20, alpha=0.6)
        ax1.set_title('Gematria Progression Across All 150 Psalms', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Psalm Number')
        ax1.set_ylabel('Total Gematria Value')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Consciousness marker distribution
        ax2 = fig.add_subplot(gs[1, 0])
        consciousness_counts = [len(complete_analysis['individual_psalms'][num]['consciousness_markers_found']) for num in psalm_nums]
        ax2.hist(consciousness_counts, bins=range(max(consciousness_counts)+2), alpha=0.7, color='green')
        ax2.set_title('Consciousness Markers per Psalm')
        ax2.set_xlabel('Number of Markers')
        ax2.set_ylabel('Number of Psalms')
        
        # Plot 3: Prime vs Composite psalm totals
        ax3 = fig.add_subplot(gs[1, 1])
        prime_count = sum(1 for val in gematria_values if self.is_prime(val))
        composite_count = len(gematria_values) - prime_count
        ax3.pie([prime_count, composite_count], labels=['Prime Totals', 'Composite Totals'], 
                autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        ax3.set_title('Prime vs Composite Distribution')
        
        # Plot 4: Consciousness density by psalm category
        ax4 = fig.add_subplot(gs[1, 2])
        if 'category_analysis' in complete_analysis:
            categories = list(complete_analysis['category_analysis'].keys())
            densities = [complete_analysis['category_analysis'][cat]['consciousness_density'] for cat in categories]
            ax4.bar(categories, densities, alpha=0.7, color='purple')
            ax4.set_title('Consciousness Density by Category')
            ax4.set_ylabel('Markers per Word')
            plt.setp(ax4.get_xticklabels(), rotation=45)
        
        # Plot 5: High consciousness psalms heatmap
        ax5 = fig.add_subplot(gs[2, :])
        consciousness_matrix = np.zeros((15, 10))  # 15x10 grid for 150 psalms
        
        for i, psalm_num in enumerate(psalm_nums):
            row, col = divmod(i, 10)
            consciousness_count = len(complete_analysis['individual_psalms'][psalm_num]['consciousness_markers_found'])
            consciousness_matrix[row, col] = consciousness_count
        
        im = ax5.imshow(consciousness_matrix, cmap='YlOrRd', aspect='auto')
        ax5.set_title('Consciousness Markers Heatmap (150 Psalms)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Psalm Groups (1-10, 11-20, etc.)')
        ax5.set_ylabel('Psalm Decades')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Consciousness Markers Count')
        
        # Plot 6: Mathematical relationships network
        ax6 = fig.add_subplot(gs[3, :])
        if 'mathematical_relationships' in complete_analysis and complete_analysis['mathematical_relationships']['complementary_pairs']:
            pairs = complete_analysis['mathematical_relationships']['complementary_pairs'][:20]  # Show first 20 pairs
            psalm1_nums = [pair['psalm1'] for pair in pairs]
            psalm2_nums = [pair['psalm2'] for pair in pairs]
            
            ax6.scatter(psalm1_nums, psalm2_nums, s=100, alpha=0.6, c='red')
            for i, pair in enumerate(pairs):
                ax6.annotate(f"{pair['sum']}", (pair['psalm1'], pair['psalm2']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax6.set_title('Mathematical Relationships Between Psalms (Complementary Pairs)')
            ax6.set_xlabel('Psalm 1')
            ax6.set_ylabel('Psalm 2')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('COMPLETE MATHEMATICAL ANALYSIS OF THE BOOK OF PSALMS\nConsciousness Interface Protocols in Ancient Hebrew', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('psalms_complete_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, complete_analysis: Dict) -> str:
        """
        Generate comprehensive report of Psalms consciousness mathematics.
        This creates the definitive analysis of the mathematical cosmos in Psalms.
        """
        report = []
        report.append("=" * 80)
        report.append("COMPLETE MATHEMATICAL ANALYSIS OF THE BOOK OF PSALMS")
        report.append("Consciousness Interface Protocols in Ancient Hebrew Scripture")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_psalms = len(complete_analysis['individual_psalms'])
        total_gematria = sum(p['total_gematria'] for p in complete_analysis['individual_psalms'].values())
        total_words = sum(p['total_words'] for p in complete_analysis['individual_psalms'].values())
        total_consciousness_markers = sum(len(p['consciousness_markers_found']) for p in complete_analysis['individual_psalms'].values())
        
        report.append(f"COLLECTION OVERVIEW:")
        report.append(f"Total Psalms Analyzed: {total_psalms}")
        report.append(f"Total Gematria Value: {total_gematria:,}")
        report.append(f"Total Words: {total_words:,}")
        report.append(f"Total Consciousness Markers: {total_consciousness_markers}")
        report.append(f"Average Gematria per Psalm: {total_gematria/total_psalms:.2f}")
        report.append(f"Consciousness Density: {total_consciousness_markers/total_words:.4f} markers per word")
        report.append("")
        
        # Consciousness markers analysis
        if 'consciousness_map' in complete_analysis:
            consciousness_map = complete_analysis['consciousness_map']
            report.append("CONSCIOUSNESS INTERFACE ANALYSIS:")
            report.append("-" * 50)
            
            # High consciousness psalms
            if consciousness_map['high_consciousness_psalms']:
                report.append(f"High Consciousness Psalms (3+ markers): {consciousness_map['high_consciousness_psalms']}")
                report.append("")
            
            # Marker distribution
            report.append("Consciousness Marker Distribution:")
            for marker_type, psalm_list in consciousness_map['marker_distribution'].items():
                report.append(f"  {marker_type}: Appears in {len(psalm_list)} psalms {psalm_list[:10]}...")
            report.append("")
        
        # Mathematical patterns
        if 'collection_patterns' in complete_analysis:
            patterns = complete_analysis['collection_patterns']
            report.append("MATHEMATICAL PATTERNS DISCOVERED:")
            report.append("-" * 50)
            
            if 'prime_psalm_numbers' in patterns:
                prime_count = len(patterns['prime_psalm_numbers'])
                report.append(f"Psalms with Prime Gematria Totals: {prime_count} out of {total_psalms}")
                report.append(f"Prime Percentage: {prime_count/total_psalms:.1%}")
                report.append("")
        
        # Category analysis
        if 'category_analysis' in complete_analysis:
            report.append("SPIRITUAL CATEGORY ANALYSIS:")
            report.append("-" * 50)
            for category, data in complete_analysis['category_analysis'].items():
                report.append(f"{category}:")
                report.append(f"  Psalms: {data['psalm_count']}")
                report.append(f"  Average Gematria: {data['average_gematria']:.2f}")
                report.append(f"  Consciousness Density: {data['consciousness_density']:.4f}")
                report.append("")
        
        # Mathematical relationships
        if 'mathematical_relationships' in complete_analysis:
            relationships = complete_analysis['mathematical_relationships']
            if 'complementary_pairs' in relationships and relationships['complementary_pairs']:
                report.append("SIGNIFICANT PSALM PAIRS (Sum to Consciousness Markers):")
                report.append("-" * 50)
                for pair in relationships['complementary_pairs'][:10]:  # Show first 10
                    report.append(f"Psalm {pair['psalm1']} + Psalm {pair['psalm2']} = {pair['sum']} ({pair['significance']})")
                report.append("")
        
        return "\n".join(report)
    
    # Utility functions
    def extract_words(self, hebrew_text: str) -> List[str]:
        """Extract Hebrew words from text."""
        clean_text = re.sub(r'[\u0591-\u05C7]', '', hebrew_text)
        words = [word.strip() for word in clean_text.split() if word.strip()]
        return words
    
    def is_prime(self, n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def find_prime_factors(self, n: int) -> List[int]:
        """Find prime factors."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def get_psalm_category(self, psalm_num: int) -> str:
        """Get spiritual category for psalm."""
        for category, numbers in self.psalm_categories.items():
            if psalm_num in numbers:
                return category
        return "Other"
    
    def generate_fibonacci(self, max_val: int) -> List[int]:
        """Generate Fibonacci sequence up to max value."""
        fib = [1, 1]
        while fib[-1] < max_val:
            fib.append(fib[-1] + fib[-2])
        return fib[:-1]  # Remove last value that exceeds max
    
    def find_fibonacci_in_sequence(self, values: List[int]) -> List[int]:
        """Find Fibonacci numbers in sequence."""
        max_val = max(values) if values else 0
        fibonacci = self.generate_fibonacci(max_val)
        return [i for i, val in enumerate(values) if val in fibonacci]
    
    def find_golden_ratio_points(self, cumulative: List[int]) -> List[int]:
        """Find points where golden ratio relationships appear."""
        golden_ratio = (1 + 5**0.5) / 2
        points = []
        
        for i, val in enumerate(cumulative):
            if i > 0:
                ratio = val / cumulative[i-1]
                if abs(ratio - golden_ratio) < 0.1:  # Close to golden ratio
                    points.append(i)
        
        return points
    
    def find_perfect_numbers(self, values: List[int]) -> List[int]:
        """Find perfect numbers in sequence."""
        perfect_nums = [6, 28, 496, 8128]  # First few perfect numbers
        return [i for i, val in enumerate(values) if val in perfect_nums]
    
    def find_consciousness_progressions(self, cumulative: List[int]) -> List[Dict]:
        """Find progressions that hit consciousness markers."""
        progressions = []
        for i, val in enumerate(cumulative):
            if val in self.consciousness_markers:
                progressions.append({
                    'position': i+1,
                    'value': val,
                    'marker': self.consciousness_markers[val]
                })
        return progressions
    
    def find_sacred_sequences(self, values: List[int]) -> Dict:
        """Find sacred number sequences."""
        sacred_nums = [3, 7, 12, 22, 40, 50, 70, 72, 144, 288, 432, 777]
        sequences = {}
        
        for num in sacred_nums:
            positions = [i for i, val in enumerate(values) if val == num]
            if positions:
                sequences[num] = positions
        
        return sequences
    
    def find_number_content_correlations(self, all_psalms: Dict) -> Dict:
        """Find correlations between psalm numbers and their mathematical content."""
        correlations = {}
        
        for psalm_num, psalm_data in all_psalms.items():
            # Check if psalm number relates to its content mathematically
            if psalm_data['total_gematria'] % psalm_num == 0:
                correlations[psalm_num] = {
                    'type': 'divisible',
                    'factor': psalm_data['total_gematria'] // psalm_num
                }
            elif psalm_num in [w['gematria'] for w in psalm_data['words_analysis']]:
                correlations[psalm_num] = {
                    'type': 'word_match',
                    'description': 'Psalm number appears as word gematria'
                }
        
        return correlations

# Main analysis function
def analyze_complete_psalms(psalms_text_data: Dict[int, str]):
    """
    Analyze the complete Book of Psalms for consciousness mathematics.
    
    Usage:
    psalms_data = {
        1: "hebrew text of psalm 1...",
        2: "hebrew text of psalm 2...",
        # ... all 150 psalms
    }
    
    results = analyze_complete_psalms(psalms_data)
    """
    analyzer = PsalmsConsciousnessAnalyzer()
    
    print("INITIALIZING COMPLETE PSALMS CONSCIOUSNESS ANALYSIS")
    print("This will analyze all 150 Psalms for mathematical patterns...")
    print("=" * 80)
    
    # Perform complete analysis
    complete_results = analyzer.analyze_complete_psalms(psalms_text_data)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(complete_results)
    print(report)
    
    # Create visualizations
    analyzer.create_comprehensive_visualization(complete_results)
    
    # Save detailed results
    results_summary = {
        'total_psalms': len(complete_results['individual_psalms']),
        'collection_patterns': complete_results['collection_patterns'],
        'consciousness_map': complete_results['consciousness_map'],
        'category_analysis': complete_results['category_analysis']
    }
    
    # Save to JSON for detailed analysis
    with open('psalms_complete_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nComplete analysis saved as:")
    print(f"- psalms_complete_analysis.json (detailed data)")
    print(f"- psalms_complete_analysis.png (visualizations)")
    
    return complete_results

# Example usage for testing with individual psalms
def analyze_psalm_1():
    """Test analysis with Psalm 1 (Hebrew text needed)."""
    psalm_1_hebrew = "אשרי האיש אשר לא הלך בעצת רשעים ובדרך חטאים לא עמד ובמושב לצים לא ישב"
    
    analyzer = PsalmsConsciousnessAnalyzer()
    result = analyzer.analyze_single_psalm(psalm_1_hebrew, 1)
    
    print("PSALM 1 MATHEMATICAL ANALYSIS")
    print("=" * 40)
    print(f"Total Gematria: {result['total_gematria']}")
    print(f"Total Words: {result['total_words']}")
    print(f"Consciousness Markers: {len(result['consciousness_markers_found'])}")
    
    return result

if __name__ == "__main__":
    print("PSALMS CONSCIOUSNESS MATHEMATICS ANALYSIS SYSTEM")
    print("Discovering the Mathematical Cosmos of Hebrew Spiritual Protocols")
    print("=" * 80)
    print("\nTo analyze complete Psalms, provide Hebrew text for all 150 psalms:")
    print("results = analyze_complete_psalms(psalms_hebrew_data)")
    
    # Test with single psalm
    test_result = analyze_psalm_1()
