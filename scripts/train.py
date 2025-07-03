import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from yapping_detector import YappingDetector

def main():
    """Main training function."""
    detector = YappingDetector()
    
    # Load the processed Spotify dataset from data directory
    data_dir = os.path.join(current_dir, '..', 'data')
    
    try:
        X, y = detector.prepare_dataset(data_dir, "spotify_yapping_dataset.csv")
        detector.train(X, y)
        print("🎉 Training completed successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run 'python scripts/spotify_processor.py' first to generate the dataset.")
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()