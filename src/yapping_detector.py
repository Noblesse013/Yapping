#!/usr/bin/env python3
"""
Kaggle Spotify Podcast Dataset Processor for Yapping Detection
Processes the Spotify podcasts dataset and creates training data
"""

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

class YappingDetector:
    """Machine learning model for automated yapping detection."""
    
    def __init__(self):
        """Initialize the YappingDetector."""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Import ML libraries
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            from sklearn.preprocessing import StandardScaler
            import matplotlib.pyplot as plt
            import seaborn as sns
            import joblib
            
            self.RandomForestClassifier = RandomForestClassifier
            self.train_test_split = train_test_split
            self.cross_val_score = cross_val_score
            self.classification_report = classification_report
            self.confusion_matrix = confusion_matrix
            self.accuracy_score = accuracy_score
            self.StandardScaler = StandardScaler
            self.plt = plt
            self.sns = sns
            self.joblib = joblib
            
        except ImportError as e:
            print(f"Missing required packages: {e}")
            print("Please install: pip install scikit-learn matplotlib seaborn joblib")
            
    def prepare_dataset(self, data_dir: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare dataset for training."""
        
        # Load dataset
        dataset_path = Path(data_dir) / filename
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset: {len(df)} samples")
        
        # Prepare features
        feature_columns = [
            'yapping_score', 'normal_score', 'confidence', 
            'title_length', 'description_length'
        ]
        
        # Add reading metrics if available
        if 'reading_ease' in df.columns:
            feature_columns.append('reading_ease')
        if 'grade_level' in df.columns:
            feature_columns.append('grade_level')
            
        # Add engineered features
        df['total_score'] = df['yapping_score'] + df['normal_score']
        df['score_ratio'] = df['yapping_score'] / (df['normal_score'] + 1)
        df['title_words'] = df['title'].apply(lambda x: len(str(x).split()))
        df['description_words'] = df['description'].apply(lambda x: len(str(x).split()))
        
        # Add punctuation features
        df['exclamation_count'] = df['title'].apply(lambda x: str(x).count('!'))
        df['question_count'] = df['title'].apply(lambda x: str(x).count('?'))
        df['caps_ratio'] = df['title'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
        
        feature_columns.extend([
            'total_score', 'score_ratio', 'title_words', 'description_words',
            'exclamation_count', 'question_count', 'caps_ratio'
        ])
        
        # Select features that exist in the dataset
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        print(f"Using features: {available_features}")
        
        # Prepare feature matrix and target vector
        X = df[available_features].fillna(0)
        y = (df['predicted_label'] == 'yapping').astype(int)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X.values, y.values
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the yapping detection model."""
        
        print("🚀 Training Yapping Detection Model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = self.train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = self.StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = self.RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = self.accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"\nDetailed Classification Report:")
        print(self.classification_report(y_test, y_pred, target_names=['Normal', 'Yapping']))
        
        # Cross-validation
        cv_scores = self.cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Generate visualizations
        self._create_confusion_matrix(y_test, y_pred)
        self._create_feature_importance_plot()
        
        # Save model
        self._save_model()
        
        print("✅ Model training completed!")
        
    def _create_confusion_matrix(self, y_true, y_pred):
        """Create and save confusion matrix visualization."""
        
        plt = self.plt
        sns = self.sns
        
        # Create confusion matrix
        cm = self.confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Yapping'], 
                   yticklabels=['Normal', 'Yapping'])
        plt.title('Confusion Matrix - Yapping Detection Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save to images directory
        plt.tight_layout()
        plt.savefig('images/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Confusion matrix saved to images/confusion_matrix.png")
        
    def _create_feature_importance_plot(self):
        """Create and save feature importance visualization."""
        
        if not self.is_trained or self.feature_names is None:
            return
            
        plt = self.plt
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for plotting
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_df['feature'], feature_df['importance'])
        plt.title('Feature Importance - Yapping Detection Model')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save to images directory
        plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Feature importance plot saved to images/feature_importance.png")
        
        # Print top features
        print("\n🔍 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_df.tail(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f}")
            
    def _save_model(self):
        """Save the trained model and scaler."""
        
        if not self.is_trained:
            return
            
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Save model and scaler
        self.joblib.dump(self.model, 'models/yapping_detector_model.pkl')
        self.joblib.dump(self.scaler, 'models/feature_scaler.pkl')
        
        # Save feature names and metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_type': 'RandomForestClassifier',
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print("💾 Model saved to models/ directory")
        
    def load_model(self, model_path: str = 'models/yapping_detector_model.pkl'):
        """Load a pre-trained model."""
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.model = self.joblib.load(model_path)
        
        # Load scaler if available
        scaler_path = Path(model_path).parent / 'feature_scaler.pkl'
        if scaler_path.exists():
            self.scaler = self.joblib.load(scaler_path)
            
        # Load metadata if available
        metadata_path = Path(model_path).parent / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                
        self.is_trained = True
        print(f"✅ Model loaded from {model_path}")
        
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data."""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load_model().")
            
        # Scale features if scaler is available
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
            
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities

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