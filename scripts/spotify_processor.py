import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import kagglehub
from yapping_detector import SpotifyPodcastProcessor

# Download dataset
path = kagglehub.dataset_download("daniilmiheev/top-spotify-podcasts-daily-updated")
print("Path to dataset files:", path)

# Process dataset
processor = SpotifyPodcastProcessor(path)
df = processor.load_dataset()
processed_df = processor.process_dataset(sample_size=1000)  # Adjust size as needed
training_df = processor.create_training_dataset(processed_df)

# Save to data directory
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spotify_yapping_dataset.csv')
processor.save_processed_dataset(training_df, output_path)