# 🎙️ Spotify Podcast Yapping Detection System

A machine learning system that classifies Spotify podcasts as "yapping" (fast, informal, conversational speech) vs "normal" (structured, educational, professional) content using natural language processing and audio content analysis.

## 📋 Overview

This project processes Spotify podcast data from Kaggle and uses various linguistic features to detect "yapping" characteristics in podcast titles and descriptions. The system can:

- Download and process Spotify podcast datasets from Kaggle
- Extract linguistic features from podcast metadata
- Classify podcasts as "yapping" vs "normal" content
- Train machine learning models for automated detection
- Generate visualizations and performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Kaggle account and API credentials configured
- Required Python packages (see Installation)

### Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```powershell
   cd d:\Yapping
   ```
3. Install the package and dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   
   Or for development installation:
   ```powershell
   pip install -e .
   ```

3. Set up Kaggle API credentials:
   - Create account at [kaggle.com](https://kaggle.com)
   - Go to Account → API → Create New API Token
   - Place `kaggle.json` in `~/.kaggle/` directory

### Basic Usage

1. **Process Spotify Dataset:**
   ```powershell
   python scripts/spotify_processor.py
   ```
   This downloads the Spotify podcast dataset and creates `data/spotify_yapping_dataset.csv`

2. **Train the Model:**
   ```powershell
   python scripts/train.py
   ```
   This trains a machine learning model using the processed dataset

## 📁 Project Structure

```
d:\Yapping\
├── src/                         # Source code
│   ├── __init__.py             
│   └── yapping_detector.py     # Main processor with feature extraction and classification
├── scripts/                    # Executable scripts
│   ├── spotify_processor.py    # Download and process Spotify dataset
│   ├── train.py               # Train the ML model
│   └── README.md              # Scripts documentation
├── data/                      # Datasets and data files
│   ├── spotify_yapping_dataset.csv  # Processed, labeled dataset (generated)
│   └── README.md              # Data documentation
├── models/                    # Trained models and artifacts
│   └── README.md              # Models documentation
├── images/                    # Generated visualizations
│   ├── confusion_matrix.png   # Model evaluation visualization
│   ├── feature_importance.png # Feature importance plot
│   └── README.md              # Images documentation
├── notebooks/                 # Jupyter notebooks for analysis
│   └── README.md              # Notebooks documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🔧 Core Components

### SpotifyPodcastProcessor (`src/yapping_detector.py`)

Main class that handles dataset processing and feature extraction:

- **Dataset Management**: Downloads Spotify podcast data from Kaggle
- **Feature Extraction**: Analyzes titles and descriptions for yapping indicators
- **Classification Logic**: Uses keyword matching and linguistic patterns
- **Data Export**: Creates balanced training datasets

#### Key Methods:
- `download_dataset()`: Downloads Kaggle dataset
- `load_dataset()`: Loads CSV data into pandas DataFrame
- `classify_podcast()`: Classifies individual podcasts
- `process_dataset()`: Processes entire dataset with features
- `create_training_dataset()`: Creates balanced training data
- `save_processed_dataset()`: Exports processed data to CSV

### YappingDetector (`scripts/train.py`)

Machine learning model for automated classification:

- Loads processed dataset from CSV
- Trains classification model
- Provides prediction capabilities
- Generates evaluation metrics and visualizations

## 🎯 Classification Logic

### Yapping Indicators
The system identifies "yapping" content using:

**Keywords:**
- High energy: react, rant, roast, drama, tea, gossip, crazy, wild
- Conversational: chat, talk, discuss, banter, ramble, stream
- Opinion-based: review, opinion, thoughts, hot take, controversial
- Entertainment: comedy, funny, hilarious, jokes, memes, viral

**Title Characteristics:**
- Length > 50 characters (suggests rambling)
- Multiple exclamation marks or question marks
- ALL CAPS text (excitement/energy)
- Ellipsis (...) suggesting ongoing thoughts
- vs/versus patterns (comparison/reaction content)

**Categories:**
- Comedy, entertainment, true crime, pop culture
- Gaming, sports commentary, reaction, drama

### Normal Content Indicators

**Keywords:**
- Educational: learn, education, tutorial, guide, how to, explain
- News: news, report, analysis, breaking, update
- Professional: business, career, finance, investment, strategy
- Wellness: meditation, mindfulness, health, fitness, calm

**Categories:**
- Education, news, business, science, health
- Meditation, history, documentary, finance

## 📊 Features Extracted

For each podcast, the system extracts:

1. **Text Features**: Keyword scores, title length, punctuation patterns
2. **Content Features**: Category classification, description analysis
3. **Linguistic Features**: Reading level, complexity metrics
4. **Metadata**: Confidence scores, prediction labels

## 🎯 Usage Examples

### Processing New Data
```python
import sys
sys.path.append('src')
from yapping_detector import SpotifyPodcastProcessor

# Initialize processor
processor = SpotifyPodcastProcessor()

# Download and process dataset
df = processor.load_dataset()
processed_df = processor.process_dataset(sample_size=1000)

# Create training dataset
training_df = processor.create_training_dataset(processed_df, balance_dataset=True)

# Save for training
processor.save_processed_dataset(training_df, "data/my_dataset.csv")
```

### Training Model
```python
import sys
sys.path.append('src')
from yapping_detector import YappingDetector

# Initialize detector
detector = YappingDetector()

# Load dataset and train
X, y = detector.prepare_dataset("data", "spotify_yapping_dataset.csv")
detector.train(X, y)
```

### Single Podcast Classification
```python
import sys
sys.path.append('src')
from yapping_detector import SpotifyPodcastProcessor

processor = SpotifyPodcastProcessor()

result = processor.analyze_podcast_metadata({
    'title': "CRAZY DRAMA ALERT! React to Celebrity Tea ☕️",
    'description': "Join us for some wild gossip and hot takes...",
    'category': 'Entertainment'
})

print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📈 Model Performance

After training, the system generates:
- **Confusion Matrix**: Visual representation of classification accuracy
- **Feature Importance**: Shows which features contribute most to predictions
- **Performance Metrics**: Accuracy, precision, recall, F1-score

### Visualizations

The trained model generates the following performance visualizations:

#### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix shows how well the model distinguishes between "yapping" and "normal" podcasts, displaying true vs predicted classifications.

#### Feature Importance
![Feature Importance](images/feature_importance.png)

This chart reveals which features (keywords, title characteristics, linguistic patterns) contribute most to the classification decisions.

## 🔄 Extending the Dataset

To add more data sources:

1. **Manual Addition**: Add new podcast data to existing CSV format
2. **Kaggle Integration**: Modify `download_dataset()` to use different datasets
3. **Custom Sources**: Implement new data collection methods in `SpotifyPodcastProcessor`

### Required CSV Format:
```csv
title,description,predicted_label,confidence,yapping_score,normal_score
"Podcast Title","Description text","yapping",0.85,12,3
```

## 🛠️ Configuration

### Adjusting Classification Sensitivity

Modify keyword lists in `SpotifyPodcastProcessor.__init__()`:

```python
# Add new yapping keywords
self.yapping_keywords['custom'] = ['new', 'keywords', 'here']

# Add new normal keywords  
self.normal_keywords['custom'] = ['serious', 'formal', 'academic']
```

### Changing Sample Size

Adjust processing in main functions:
```python
# Process more/fewer podcasts
processed_df = processor.process_dataset(sample_size=5000)
```

## 🐛 Troubleshooting

### Common Issues:

1. **Kaggle API Errors**: Ensure `kaggle.json` is properly configured
2. **Memory Issues**: Reduce `sample_size` in processing functions
3. **Missing Packages**: Install all required dependencies
4. **Empty Dataset**: Check Kaggle dataset availability and permissions

### Error Messages:

- `"No CSV files found"`: Dataset download failed or path incorrect
- `"Failed to load dataset"`: Check Kaggle configuration
- `"Please install required packages"`: Run pip install commands

## 📝 Notes

- The system uses heuristic-based classification for initial labeling
- Machine learning model improves predictions over the heuristic baseline
- Classification accuracy depends on quality and diversity of training data
- "Yapping" detection is subjective and based on observable linguistic patterns

## 🔮 Future Improvements

- Audio feature extraction for actual speech pattern analysis
- Integration with more podcast platforms (Apple Podcasts, Google Podcasts)
- Advanced NLP models (BERT, GPT) for better text understanding
- Real-time podcast classification API
- Web interface for easy interaction

## 📄 License

This project is for educational and research purposes. Ensure compliance with Kaggle and Spotify data usage policies.

---

*Happy podcast classification! 🎧*
