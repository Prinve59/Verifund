import os
import joblib
import sys

"""
Utility script to inspect ML model files and debug encoding issues
Run this script from command line:
python debug_util.py
"""

def inspect_pkl_files(directory):
    """Inspect the pickle files to understand their structure"""
    try:
        # Load the model files
        model_path = os.path.join(directory, 'lending_threshold_model.pkl')
        scaler_path = os.path.join(directory, 'scaler.pkl')
        encoders_path = os.path.join(directory, 'label_encoders.pkl')
        
        print(f"Looking for files in: {directory}")
        print(f"Files exist: model={os.path.exists(model_path)}, "
              f"scaler={os.path.exists(scaler_path)}, "
              f"encoders={os.path.exists(encoders_path)}")
        
        if not all([os.path.exists(p) for p in [model_path, scaler_path, encoders_path]]):
            print("Some files are missing. Please check the paths.")
            return
        
        # Load the files
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        
        # Print model information
        print("\n=== MODEL INFORMATION ===")
        print(f"Model type: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature names: {model.feature_names_in_}")
        else:
            print("Feature names not available in model")
        
        # Print scaler information
        print("\n=== SCALER INFORMATION ===")
        print(f"Scaler type: {type(scaler)}")
        print(f"Scaler mean: {scaler.mean_}")
        print(f"Number of features in scaler: {len(scaler.mean_)}")
        
        # Print label encoder information
        print("\n=== LABEL ENCODERS ===")
        print(f"Number of label encoders: {len(label_encoders)}")
        print(f"Label encoder fields: {list(label_encoders.keys())}")
        
        # Print details of each label encoder
        print("\nField classes:")
        for field, encoder in label_encoders.items():
            print(f"- {field}: {encoder.classes_[:5]}{'...' if len(encoder.classes_) > 5 else ''} "
                  f"({len(encoder.classes_)} classes)")
            
        # Cross-check if we have all the encoders we need
        if hasattr(model, 'feature_names_in_'):
            categorical_features = [f for f in model.feature_names_in_ if f in label_encoders]
            missing_encoders = [f for f in model.feature_names_in_ if f not in label_encoders and isinstance(f, str)]
            
            print("\n=== FEATURE COVERAGE ===")
            print(f"Categorical features with encoders: {categorical_features}")
            print(f"Features missing encoders: {missing_encoders}")
        
    except Exception as e:
        print(f"Error inspecting files: {e}")

if __name__ == "__main__":
    # Get the directory from command line or use default
    directory = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')
    inspect_pkl_files(directory)