import os
import hashlib
import random
import logging
import json
import datetime
import pickle

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_HASHES_PATH = os.path.join(_BASE_DIR, 'Crypto-models', 'model_hashes.json')

class MLRiskAnalyzer:
    """ML Risk Analyzer with file-backed integrity checking.
    
    Run ``python Crypto-models/generate_model_hashes.py`` once after training
    to produce Crypto-models/model_hashes.json and enable hash verification.
    """

    def __init__(self):
        self.models_loaded = False
        self.rf_pipeline = None
        self.svm_pipeline = None
        self.iso_pipeline = None
        self.security_alerts = []
        self._expected_hashes = self._load_expected_hashes()
        self.load_models()

    def _load_expected_hashes(self) -> dict:
        """Load expected model hashes from Crypto-models/model_hashes.json.
        Returns an empty dict when the file has not been generated yet.
        """
        if os.path.exists(MODEL_HASHES_PATH):
            try:
                with open(MODEL_HASHES_PATH, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"⚠️  Could not read model_hashes.json: {e}")
        print(
            "⚠️  MLRiskAnalyzer: model_hashes.json not found. "
            "Run 'python Crypto-models/generate_model_hashes.py' "
            "to enable model integrity verification."
        )
        return {}
        
    def _secure_load(self, filepath, model_name):
        """Load a model file, verifying its SHA-256 hash if one is registered.

        * No hash registered  → warn and load anyway (dev/setup mode).
        * Hash registered, matches → load normally.
        * Hash registered, mismatch → log critical alert and refuse to load.
        """
        with open(filepath, 'rb') as f:
            file_data = f.read()

        actual_hash = hashlib.sha256(file_data).hexdigest()
        expected_hash = self._expected_hashes.get(model_name)

        if not expected_hash:
            msg = (
                f"⚠️  No integrity hash registered for '{model_name}'. "
                "Run 'python Crypto-models/generate_model_hashes.py' to enable verification."
            )
            print(msg)
            self.security_alerts.append({
                'timestamp': str(datetime.datetime.now()),
                'level': 'warning',
                'message': msg
            })
            # Allow loading — hash file simply hasn't been generated yet.
            return pickle.loads(file_data)

        if actual_hash != expected_hash:
            msg = (
                f"🚨 SECURITY ALERT: Hash mismatch for '{model_name}'! "
                f"Expected {expected_hash[:16]}…, got {actual_hash[:16]}…"
            )
            print(msg)
            self.security_alerts.append({
                'timestamp': str(datetime.datetime.now()),
                'level': 'critical',
                'message': msg
            })
            raise ValueError(msg)

        print(f"   ✅ Integrity verified: {model_name}")
        return pickle.loads(file_data)
    
    def load_models(self):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'Crypto-models', 'models')
            
            self.rf_pipeline = self._secure_load(
                os.path.join(models_dir, 'random_forest_pipeline.pkl'), 
                'random_forest_pipeline.pkl'
            )
            self.svm_pipeline = self._secure_load(
                os.path.join(models_dir, 'svm_pipeline.pkl'), 
                'svm_pipeline.pkl'
            )
            self.iso_pipeline = self._secure_load(
                os.path.join(models_dir, 'isolation_forest_pipeline.pkl'), 
                'isolation_forest_pipeline.pkl'
            )
            self.models_loaded = True
            print("✅ All ML models loaded successfully via secure loader.")
        except Exception as e:
            self.models_loaded = False
            msg = f"❌ Failed to load ML models securely: {e}"
            print(msg)
            self.security_alerts.append({
                'timestamp': str(datetime.datetime.now()),
                'level': 'critical',
                'message': msg
            })

    def analyze_risk(self, request_features):
        """Analyzes a request and returns a risk assessment."""
        # Simulated feature vector logic if actual models aren't present
        if not self.models_loaded:
            print("⚠️ Falling back to mock ML logic (models not loaded).")
            # In mock mode, randomly return risk scores
            mock_score = round(random.uniform(0.1, 0.4), 2)
            is_risky = mock_score > 0.75
            return {
                'risk_score': mock_score,
                'is_risky': is_risky,
                'confidence': 0.85,
                'factors': ['Mock ML Engine Active'],
                'ml_status': 'MOCK MODE'
            }

        try:
            import pandas as pd
            # Create a 1-row DataFrame matching training columns
            df = pd.DataFrame([request_features])
            
            rf_pred = self.rf_pipeline.predict_proba(df)[0][1]
            svm_pred = self.svm_pipeline.predict_proba(df)[0][1]
            # IsolationForest returns 1 (inlier) or -1 (outlier)
            iso_pred = self.iso_pipeline.predict(df)[0]
            
            # Simple ensemble: average the probabilities
            ensemble_risk = (rf_pred + svm_pred) / 2.0
            
            # If Isolation Forest tags it as an anomaly (-1), push risk higher
            if iso_pred == -1:
                ensemble_risk = min(1.0, ensemble_risk + 0.3)
                
            is_risky = ensemble_risk > 0.75
            
            return {
                'risk_score': round(ensemble_risk, 2),
                'is_risky': is_risky,
                'confidence': 0.92, # Placeholder, could calculate based on model variance
                'factors': ['Ensemble Prediction', 'Isolation Forest checked'],
                'ml_status': 'Active'
            }
            
        except Exception as e:
            print(f"ML Analysis failed: {e}")
            return {
                'risk_score': 0.99,
                'is_risky': True,
                'confidence': 0.0,
                'factors': [f'Analysis Error: {str(e)}'],
                'ml_status': 'Error'
            }
