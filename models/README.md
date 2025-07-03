# Models Directory

This directory stores trained machine learning models and related files.

## Generated Files:
- `yapping_detector_model.pkl`: Trained classification model
- `feature_scaler.pkl`: Feature scaling parameters
- `model_metadata.json`: Model training details and performance metrics

## Usage:
Models are automatically saved here after training. Load them for predictions:

```python
import joblib
model = joblib.load('models/yapping_detector_model.pkl')
```
