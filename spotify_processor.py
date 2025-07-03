import kagglehub
from spotify_processor import SpotifyPodcastProcessor


path = kagglehub.dataset_download("daniilmiheev/top-spotify-podcasts-daily-updated")
print("Path to dataset files:", path)


processor = SpotifyPodcastProcessor(path)
df = processor.load_dataset()
processed_df = processor.process_dataset(sample_size=1000)  
training_df = processor.create_training_dataset(processed_df)
processor.save_processed_dataset(training_df, "spotify_yapping_dataset.csv")