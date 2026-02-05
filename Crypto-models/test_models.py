import pickle
import pandas as pd
import os

print("--- ML Model Testing Script ---")

# --- 1. Configuration & Setup ---
# =================================
MODELS_DIR = 'models'

# Paths to the saved pipeline files
RF_PIPELINE_PATH = os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl')
SVM_PIPELINE_PATH = os.path.join(MODELS_DIR, 'svm_pipeline.pkl')
ISO_PIPELINE_PATH = os.path.join(MODELS_DIR, 'isolation_forest_pipeline.pkl')

# --- 2. Load the Trained Model Pipelines ---
# ===========================================
# We load the entire pipeline, which includes the preprocessor and the model.
try:
    print("\nLoading saved model pipelines...")
    with open(RF_PIPELINE_PATH, 'rb') as f:
        rf_pipeline = pickle.load(f)
    print(f"  -> ✅ Loaded Random Forest pipeline from '{RF_PIPELINE_PATH}'")
    
    with open(SVM_PIPELINE_PATH, 'rb') as f:
        svm_pipeline = pickle.load(f)
    print(f"  -> ✅ Loaded Linear SVM pipeline from '{SVM_PIPELINE_PATH}'")

    with open(ISO_PIPELINE_PATH, 'rb') as f:
        iso_pipeline = pickle.load(f)
    print(f"  -> ✅ Loaded Isolation Forest pipeline from '{ISO_PIPELINE_PATH}'")

except FileNotFoundError as e:
    print(f"\n❌ Error: Could not find a model file. {e}")
    print("Please ensure you have run the 'run_full_project.py' training script first.")
    exit()

# --- 3. Define Sample Data for Testing ---
# =========================================
# We will create two test cases: one that should be safe, and one that should be risky.

# Scenario 1: A typical, safe event
safe_event = {
    'activity': 'FileOpen',
    'role': 'ProductionLineWorker',
    'hour_of_day': 11, # 11 AM, during business hours
    'day_of_week': 2,  # Wednesday
    'is_weekend': 0,
    'avg_actions_per_day': 0.5 # A normal activity level for this user
}

# Scenario 2: A suspicious, risky event
risky_event = {
    'activity': 'FileDelete',
    'role': 'Salesperson',
    'hour_of_day': 3,  # 3 AM, outside business hours
    'day_of_week': 6,  # Sunday
    'is_weekend': 1,
    'avg_actions_per_day': 0.9 # Unusually high activity for this user
}

# Convert the dictionaries to a DataFrame for prediction
safe_df = pd.DataFrame([safe_event])
risky_df = pd.DataFrame([risky_event])


# --- 4. Test Each Model ---
# ==========================
def test_model(model_name, pipeline, data_df, expected_result):
    """A helper function to test a model and print the result."""
    print(f"\n--- Testing {model_name} ---")
    print(f"Input Data: {data_df.to_dict('records')[0]}")
    
    # The pipeline handles all preprocessing automatically
    prediction = pipeline.predict(data_df)
    
    # For Isolation Forest, convert the output (-1 for anomaly, 1 for normal)
    if model_name == "Isolation Forest":
        final_prediction = 1 if prediction[0] == -1 else 0
    else:
        final_prediction = prediction[0]
        
    verdict = "Risky" if final_prediction == 1 else "Safe"
    
    print(f"  -> Model Prediction: {verdict}")
    print(f"  -> Expected Result:  {expected_result}")
    
    test_passed = verdict == expected_result
    if test_passed:
        print("  -> ✅ Test Passed!")
    else:
        print("  -> ❌ Test Failed.")
    return test_passed

# --- Run Tests for all models and collect results ---
test_results = []
test_results.append(test_model("Random Forest", rf_pipeline, safe_df, "Safe"))
test_results.append(test_model("Random Forest", rf_pipeline, risky_df, "Risky"))

test_results.append(test_model("Linear SVM", svm_pipeline, safe_df, "Safe"))
test_results.append(test_model("Linear SVM", svm_pipeline, risky_df, "Risky"))

test_results.append(test_model("Isolation Forest", iso_pipeline, safe_df, "Safe"))
test_results.append(test_model("Isolation Forest", iso_pipeline, risky_df, "Risky"))

print("\n\n--- Model Testing Complete ---")

# --- 5. Final Verdict ---
# ========================
# Calculate and display the final satisfaction score based on test results.
total_tests = len(test_results)
passed_tests = sum(test_results) # sum() on a list of booleans counts the number of True values
satisfaction_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

print("\n==============================================")
print("--- Final Project Satisfaction Verdict ---")
print("==============================================")
print(f"Total Functional Tests Performed: {total_tests}")
print(f"Total Tests Passed: {passed_tests}")
print(f"Overall Model Satisfaction: {satisfaction_percentage:.2f}%")

if satisfaction_percentage == 100:
    print("\nConclusion: All models are fully satisfied and performing as expected. ✅")
else:
    print("\nConclusion: Some models failed functional tests. Review the logs above. ❌")

