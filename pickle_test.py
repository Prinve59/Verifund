import os
import joblib
import pandas as pd
import numpy as np

"""
Simple script to test loading the pickled model files and
running a test prediction to make sure everything works.

Run this script from your project root directory:
python pickle_test.py
"""

def test_model_prediction():
    """Test loading model files and making a prediction"""
    try:
        # Get directory for model files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'ml_models')
        
        # Load the model files
        print(f"Loading model files from {model_dir}...")
        model = joblib.load(os.path.join(model_dir, 'lending_threshold_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        
        print("All files loaded successfully!")
        
        # Create a test input - use actual example values based on your form
        test_input = {
            'month': 'January',  # Must be one of the values in your month encoder classes
            'age': '35',
            'occupation': 'Engineer',
            'annual_income': '50000',
            'monthly_inhand_salary': 4000.0,
            'num_bank_accounts': 2,
            'num_credit_card': 1,
            'interest_rate': 10.5,
            'num_of_loan': 1,
            'delay_from_due_date': 0,
            'num_of_delayed_payment': 0,
            'changed_credit_limit': 0.0,
            'num_credit_inquiries': 1,
            'credit_mix': 'Good',
            'outstanding_debt': 10000.0,
            'credit_utilization_ratio': 30.0,
            'credit_history_age': 60,
            'payment_of_min_amount': 'Yes',
            'total_emi_per_month': 1000.0,
            'amount_invested_monthly': 500.0,
            'payment_behaviour': 'High_spent_Medium_value_payments',
            'monthly_balance': 2500.0,
            'type_of_loan': 'Auto Loan'
        }
        
        # Process and transform the input data
        print("\nProcessing test input...")
        feature_dict = {}
        
        # Handle categorical fields with encoders
        for field, value in test_input.items():
            if field in label_encoders:
                encoder = label_encoders[field]
                if value in encoder.classes_:
                    feature_dict[field] = encoder.transform([value])[0]
                else:
                    print(f"WARNING: Value '{value}' not in encoder classes for field '{field}'")
                    print(f"Available classes: {encoder.classes_[:5]}{'...' if len(encoder.classes_) > 5 else ''}")
            else:
                # Numeric fields
                try:
                    feature_dict[field] = float(value)
                except ValueError:
                    print(f"ERROR: Could not convert {field}='{value}' to float")
        
        # Get expected features from scaler
        print("\nPreparing features array...")
        expected_features = list(label_encoders.keys())
        features_array = []
        
        # Build features array with correct ordering
        for feature in expected_features:
            if feature in feature_dict:
                features_array.append(feature_dict[feature])
            else:
                print(f"WARNING: Missing feature '{feature}' - using 0 as default")
                features_array.append(0)
        
        # Convert to numpy array
        input_array = np.array([features_array])
        print(f"Input array shape: {input_array.shape}")
        
        # Scale the data
        print("\nScaling input data...")
        print(f"Scaler expects shape: ({scaler.n_features_in_},)")
        
        if input_array.shape[1] != scaler.n_features_in_:
            print(f"ERROR: Input array has {input_array.shape[1]} features, but scaler expects {scaler.n_features_in_}")
            print("Feature dimensions don't match!")
            return
            
        try:
            input_scaled = scaler.transform(input_array)
            print("Input data scaled successfully!")
        except Exception as e:
            print(f"ERROR scaling data: {str(e)}")
            return
        
        # Make prediction
        print("\nMaking prediction...")
        try:
            prediction = model.predict(input_scaled)[0]
            print(f"Prediction successful: {prediction}")
        except Exception as e:
            print(f"ERROR making prediction: {str(e)}")
            return
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    test_model_prediction()