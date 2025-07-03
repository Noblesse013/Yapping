import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from typing import List, Dict, Tuple, Optional
import requests
from urllib.parse import urlparse
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Required installations:
# pip install kagglehub pandas numpy requests beautifulsoup4 textstat

try:
    import kagglehub
    from bs4 import BeautifulSoup
    import textstat
except ImportError:
    print("Please install required packages:")
    print("pip install kagglehub beautifulsoup4 textstat")
    exit(1)

class SpotifyPodcastProcessor:
    def __init__(self, dataset_path: str = None):
        """Initialize the Spotify podcast dataset processor."""
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.df = None
        self.processed_data = []
        
        # Yapping detection keywords and patterns
        self.yapping_keywords = {
            'high_energy': ['react', 'rant', 'roast', 'drama', 'tea', 'gossip', 'spill', 'crazy', 'wild', 'insane'],
            'conversational': ['chat', 'talk', 'discuss', 'conversation', 'banter', 'ramble', 'stream'],
            'opinion': ['review', 'opinion', 'thoughts', 'take', 'unpopular', 'controversial', 'hot take'],
            'entertainment': ['comedy', 'funny', 'hilarious', 'jokes', 'memes', 'viral', 'trending']
        }
        
        self.normal_keywords = {
            'educational': ['learn', 'education', 'tutorial', 'guide', 'how to', 'explain', 'science'],
            'news': ['news', 'report', 'analysis', 'breaking', 'update', 'current events'],
            'professional': ['business', 'career', 'finance', 'investment', 'strategy', 'leadership'],
            'wellness': ['meditation', 'mindfulness', 'health', 'fitness', 'sleep', 'calm', 'peaceful']
        }
        
        # Common fast-talking podcast categories
        self.yapping_categories = [
            'comedy', 'entertainment', 'true crime', 'pop culture', 
            'gaming', 'sports commentary', 'reaction', 'drama'
        ]
        
        self.normal_categories = [
            'education', 'news', 'business', 'science', 'health',
            'meditation', 'history', 'documentary', 'finance'
        ]
    
    def download_dataset(self) -> str:
        """Download the Kaggle Spotify dataset."""
        print("Downloading Spotify Podcasts dataset from Kaggle...")
        try:
            path = kagglehub.dataset_download("daniilmiheev/top-spotify-podcasts-daily-updated")
            self.dataset_path = Path(path)
            print(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def load_dataset(self, csv_file: str = None) -> pd.DataFrame:
        """Load the Spotify podcasts dataset."""
        if not self.dataset_path:
            dataset_path = self.download_dataset()
            if not dataset_path:
                return None
        
        # Find CSV files in the dataset directory
        csv_files = list(self.dataset_path.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in dataset directory")
            return None
        
        # Use specified file or first CSV found
        if csv_file:
            csv_path = self.dataset_path / csv_file
        else:
            csv_path = csv_files[0]
        
        print(f"Loading dataset from: {csv_path}")
        
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Dataset loaded: {len(self.df)} podcasts")
            print(f"Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def analyze_podcast_metadata(self, row: pd.Series) -> Dict:
        """Analyze podcast metadata to determine yapping likelihood."""
        
        # Extract text fields for analysis
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))
        category = str(row.get('category', ''))
        
        # Combine text for analysis
        combined_text = f"{title} {description} {category}".lower()
        
        # Calculate yapping score
        yapping_score = 0
        normal_score = 0
        
        # Check keywords
        for category_type, keywords in self.yapping_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    yapping_score += 1
        
        for category_type, keywords in self.normal_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    normal_score += 1
        
        # Check categories
        for yap_cat in self.yapping_categories:
            if yap_cat in combined_text:
                yapping_score += 2
        
        for norm_cat in self.normal_categories:
            if norm_cat in combined_text:
                normal_score += 2
        
        # Analyze title characteristics
        title_analysis = self.analyze_title_characteristics(title)
        yapping_score += title_analysis['yapping_indicators']
        normal_score += title_analysis['normal_indicators']
        
        # Analyze description if available
        if description and len(description) > 10:
            desc_analysis = self.analyze_description(description)
            yapping_score += desc_analysis['yapping_score']
            normal_score += desc_analysis['normal_score']
        
        # Determine final classification
        if yapping_score > normal_score:
            predicted_label = 'yapping'
            confidence = yapping_score / (yapping_score + normal_score + 1)
        else:
            predicted_label = 'normal'
            confidence = normal_score / (yapping_score + normal_score + 1)
        
        return {
            'yapping_score': yapping_score,
            'normal_score': normal_score,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'title_analysis': title_analysis,
            'description_analysis': desc_analysis if 'desc_analysis' in locals() else {}
        }
    
    def analyze_title_characteristics(self, title: str) -> Dict:
        """Analyze title characteristics for yapping indicators."""
        yapping_indicators = 0
        normal_indicators = 0
        
        # Length-based indicators
        if len(title) > 50:  # Long titles often indicate rambling
            yapping_indicators += 1
        elif len(title) < 30:  # Short, concise titles
            normal_indicators += 1
        
        # Punctuation patterns
        if title.count('!') > 1:  # Multiple exclamation marks
            yapping_indicators += 1
        if title.count('?') > 1:  # Multiple questions
            yapping_indicators += 1
        if '...' in title:  # Ellipsis suggesting ongoing thought
            yapping_indicators += 1
        
        # Capitalization patterns
        if title.isupper():  # ALL CAPS suggests excitement
            yapping_indicators += 1
        
        # Word patterns
        title_lower = title.lower()
        if any(word in title_lower for word in ['vs', 'versus', 'react', 'response']):
            yapping_indicators += 1
        if any(word in title_lower for word in ['guide', 'how to', 'introduction', 'basics']):
            normal_indicators += 1
        
        return {
            'yapping_indicators': yapping_indicators,
            'normal_indicators': normal_indicators,
            'length': len(title),
            'exclamation_count': title.count('!'),
            'question_count': title.count('?')
        }
    
    def analyze_description(self, description: str) -> Dict:
        """Analyze podcast description for speech pattern indicators."""
        yapping_score = 0
        normal_score = 0
        
        # Text complexity analysis
        try:
            # Reading ease (lower = more complex)
            reading_ease = textstat.flesch_reading_ease(description)
            
            # Grade level
            grade_level = textstat.flesch_kincaid_grade(description)
            
            # Sentence characteristics
            sentences = description.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Yapping indicators
            if reading_ease > 70:  # Very easy to read (conversational)
                yapping_score += 1
            if grade_level < 8:  # Lower grade level (casual speech)
                yapping_score += 1
            if avg_sentence_length > 20:  # Long sentences (rambling)
                yapping_score += 1
            
            # Normal speech indicators
            if reading_ease < 50:  # More complex text
                normal_score += 1
            if grade_level > 12:  # Higher grade level (formal)
                normal_score += 1
            if avg_sentence_length < 15:  # Concise sentences
                normal_score += 1
            
        except:
            reading_ease = 0
            grade_level = 0
            avg_sentence_length = 0
        
        return {
            'yapping_score': yapping_score,
            'normal_score': normal_score,
            'reading_ease': reading_ease,
            'grade_level': grade_level,
            'avg_sentence_length': avg_sentence_length
        }
    
    def process_dataset(self, sample_size: int = None) -> pd.DataFrame:
        """Process the entire dataset and classify podcasts."""
        if self.df is None:
            print("Dataset not loaded. Please load dataset first.")
            return None
        
        # Sample dataset if specified
        if sample_size and sample_size < len(self.df):
            df_sample = self.df.sample(n=sample_size, random_state=42)
        else:
            df_sample = self.df.copy()
        
        print(f"Processing {len(df_sample)} podcasts...")
        
        processed_data = []
        
        for idx, row in df_sample.iterrows():
            try:
                # Analyze podcast metadata
                analysis = self.analyze_podcast_metadata(row)
                
                # Create processed row
                processed_row = {
                    'original_index': idx,
                    'title': row.get('title', ''),
                    'description': row.get('description', ''),
                    'category': row.get('category', ''),
                    'spotify_url': row.get('spotify_url', ''),
                    'predicted_label': analysis['predicted_label'],
                    'confidence': analysis['confidence'],
                    'yapping_score': analysis['yapping_score'],
                    'normal_score': analysis['normal_score'],
                    'title_length': len(str(row.get('title', ''))),
                    'description_length': len(str(row.get('description', ''))),
                    'reading_ease': analysis.get('description_analysis', {}).get('reading_ease', 0),
                    'grade_level': analysis.get('description_analysis', {}).get('grade_level', 0)
                }
                
                # Add any additional columns from original dataset
                for col in self.df.columns:
                    if col not in processed_row:
                        processed_row[col] = row.get(col, '')
                
                processed_data.append(processed_row)
                
                # Progress indicator
                if len(processed_data) % 100 == 0:
                    print(f"Processed {len(processed_data)} podcasts...")
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        print(f"\nProcessing complete!")
        print(f"Total processed: {len(processed_df)}")
        print(f"Predicted as 'yapping': {len(processed_df[processed_df['predicted_label'] == 'yapping'])}")
        print(f"Predicted as 'normal': {len(processed_df[processed_df['predicted_label'] == 'normal'])}")
        
        return processed_df
    
    def create_training_dataset(self, processed_df: pd.DataFrame, balance_dataset: bool = True) -> pd.DataFrame:
        """Create balanced training dataset from processed podcasts."""
        
        # Separate by label
        yapping_df = processed_df[processed_df['predicted_label'] == 'yapping']
        normal_df = processed_df[processed_df['predicted_label'] == 'normal']
        
        print(f"Original distribution:")
        print(f"  Yapping: {len(yapping_df)}")
        print(f"  Normal: {len(normal_df)}")
        
        if balance_dataset:
            # Balance the dataset
            min_size = min(len(yapping_df), len(normal_df))
            if min_size > 0:
                yapping_sample = yapping_df.sample(n=min_size, random_state=42)
                normal_sample = normal_df.sample(n=min_size, random_state=42)
                
                balanced_df = pd.concat([yapping_sample, normal_sample], ignore_index=True)
                balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
                
                print(f"\nBalanced dataset:")
                print(f"  Total samples: {len(balanced_df)}")
                print(f"  Yapping: {len(balanced_df[balanced_df['predicted_label'] == 'yapping'])}")
                print(f"  Normal: {len(balanced_df[balanced_df['predicted_label'] == 'normal'])}")
                
                return balanced_df
        
        # Return unbalanced dataset if balancing not requested or not possible
        return processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def save_processed_dataset(self, processed_df: pd.DataFrame, filename: str = "spotify_podcasts_processed.csv"):
        """Save processed dataset to CSV."""
        processed_df.to_csv(filename, index=False)
        print(f"Processed dataset saved to: {filename}")
    
    def generate_summary_report(self, processed_df: pd.DataFrame) -> Dict:
        """Generate a summary report of the processed dataset."""
        
        report = {
            'dataset_info': {
                'total_podcasts': len(processed_df),
                'yapping_count': len(processed_df[processed_df['predicted_label'] == 'yapping']),
                'normal_count': len(processed_df[processed_df['predicted_label'] == 'normal']),
                'yapping_percentage': len(processed_df[processed_df['predicted_label'] == 'yapping']) / len(processed_df) * 100
            },
            'confidence_stats': {
                'mean_confidence': processed_df['confidence'].mean(),
                'high_confidence_count': len(processed_df[processed_df['confidence'] > 0.7]),
                'low_confidence_count': len(processed_df[processed_df['confidence'] < 0.3])
            },
            'top_yapping_podcasts': processed_df[processed_df['predicted_label'] == 'yapping'].nlargest(10, 'confidence')[['title', 'confidence', 'yapping_score']].to_dict('records'),
            'top_normal_podcasts': processed_df[processed_df['predicted_label'] == 'normal'].nlargest(10, 'confidence')[['title', 'confidence', 'normal_score']].to_dict('records')
        }
        
        return report

def main():
    """Main function to demonstrate the Spotify podcast processor."""
    
    print("🎙️ Spotify Podcast Dataset Processor for Yapping Detection")
    print("=" * 60)
    
    # Initialize processor
    processor = SpotifyPodcastProcessor()
    
    # Download and load dataset
    df = processor.load_dataset()
    
    if df is None:
        print("Failed to load dataset. Please check your Kaggle configuration.")
        return
    
    # Process a sample of the dataset (adjust sample_size as needed)
    sample_size = 1000  # Process first 1000 podcasts for demo
    processed_df = processor.process_dataset(sample_size=sample_size)
    
    if processed_df is None:
        print("Failed to process dataset.")
        return
    
    # Create balanced training dataset
    training_df = processor.create_training_dataset(processed_df, balance_dataset=True)
    
    # Save processed dataset
    processor.save_processed_dataset(training_df, "spotify_yapping_dataset.csv")
    
    # Generate and display summary report
    report = processor.generate_summary_report(training_df)
    
    print("\n📊 DATASET SUMMARY")
    print("=" * 30)
    print(f"Total podcasts processed: {report['dataset_info']['total_podcasts']}")
    print(f"Yapping podcasts: {report['dataset_info']['yapping_count']} ({report['dataset_info']['yapping_percentage']:.1f}%)")
    print(f"Normal podcasts: {report['dataset_info']['normal_count']}")
    print(f"Average confidence: {report['confidence_stats']['mean_confidence']:.3f}")
    print(f"High confidence predictions: {report['confidence_stats']['high_confidence_count']}")
    
    print("\n🗣️ TOP YAPPING PODCASTS:")
    for i, podcast in enumerate(report['top_yapping_podcasts'][:5], 1):
        print(f"{i}. {podcast['title'][:50]}... (Score: {podcast['yapping_score']}, Confidence: {podcast['confidence']:.3f})")
    
    print("\n😌 TOP NORMAL PODCASTS:")
    for i, podcast in enumerate(report['top_normal_podcasts'][:5], 1):
        print(f"{i}. {podcast['title'][:50]}... (Score: {podcast['normal_score']}, Confidence: {podcast['confidence']:.3f})")
    
    print("\n✅ Ready for training! Use 'spotify_yapping_dataset.csv' with your YappingDetector model.")

if __name__ == "__main__":
    main()