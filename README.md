# 🎙️ Spotify Podcast Yapping Detection

A machine learning system that classifies Spotify podcasts as "yapping" (fast, informal speech) vs "normal" (structured, professional) content.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Kaggle account with API credentials

### Installation
```bash
git clone <your-repo>
cd spotify-yapping-detector
pip install -r requirements.txt
```

### Usage
```bash
# 1. Process dataset
python scripts/spotify_processor.py

# 2. Train model
python scripts/train.py
```

## 📁 Project Structure
```
├── src/                    # Source code
├── scripts/               # Executable scripts
├── data/                  # Datasets
├── models/                # Trained models
├── images/                # Visualizations
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Dependencies
```

## 🎯 How It Works

The system identifies "yapping" podcasts using:
- **Keywords**: drama, react, gossip, tea, rant vs education, tutorial, news
- **Language patterns**: Multiple exclamations, ALL CAPS, long titles
- **Content analysis**: Reading complexity, sentence structure

## 📊 Results

![Confusion Matrix](images/confusion_matrix.png)
![Feature Importance](images/feature_importance.png)

The model achieves 99%+ accuracy with balanced datasets.

## 🔧 Customization

Add new keywords in `src/yapping_detector.py`:
```python
self.yapping_keywords['custom'] = ['new', 'keywords']
```

## 📝 License
Educational and research use only.
