import joblib

# Load the model, scaler, and label encoders
model = joblib.load('lending_threshold_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Print model info
print("Model type:", type(model))
print("Model features (if available):", getattr(model, 'feature_names_in_', 'Unknown'))
print("Model output type:", getattr(model, 'classes_', 'Regression or No Class Info'))

# Scaler info
print("\nScaler mean (if applicable):", getattr(scaler, 'mean_', 'Not a scaler?'))

# Label encoders
if isinstance(label_encoders, dict):
    print("\nLabel Encoders:")
    for col, le in label_encoders.items():
        print(f" - {col}: classes = {le.classes_}")
else:
    print("Label encoders not a dictionary.")
