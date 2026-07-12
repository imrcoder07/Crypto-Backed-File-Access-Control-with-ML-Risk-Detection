import os
import hashlib
import random
import logging
import json
import datetime
import pickle
import time

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_HASHES_PATH = os.path.join(_BASE_DIR, 'Crypto-models', 'model_hashes.json')


class MLRiskAnalyzer:
    """ML Risk Analyzer with file-backed integrity checking.

    Run `python Crypto-models/generate_model_hashes.py` once after training
    to produce Crypto-models/model_hashes.json and enable hash verification.
    """

    HIGH_RISK_THRESHOLD = 0.75

    # Canonical feature specification: name -> expected Python type(s)
    REQUIRED_FEATURES = {
        'hour_of_day':         (int,),
        'day_of_week':         (int,),
        'is_weekend':          (int,),
        'avg_actions_per_day': (float, int),
        'activity':            (str,),
        'role':                (str,),
    }

    def __init__(self):
        self.models_loaded = False
        self.rf_pipeline = None
        self.svm_pipeline = None
        self.iso_pipeline = None
        self.security_alerts = []
        self._expected_hashes = self._load_expected_hashes()
        self.load_models()

    def _get_verdict(self, risk_score: float) -> str:
        """Centralized helper to convert a risk score to a final verdict string."""
        return "Deny" if risk_score >= self.HIGH_RISK_THRESHOLD else "Allow"

    def _load_expected_hashes(self) -> dict:
        """Load expected model hashes from Crypto-models/model_hashes.json.
        Returns an empty dict when the file has not been generated yet.
        """
        if os.path.exists(MODEL_HASHES_PATH):
            try:
                with open(MODEL_HASHES_PATH, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read model_hashes.json: {e}")
        logger.warning(
            "MLRiskAnalyzer: model_hashes.json not found. "
            "Run 'python Crypto-models/generate_model_hashes.py' "
            "to enable model integrity verification."
        )
        return {}

    def _secure_load(self, filepath, model_name):
        """Load a model file, verifying its SHA-256 hash if one is registered.

        * No hash registered  -> warn and load anyway (dev/setup mode).
        * Hash registered, matches -> load normally.
        * Hash registered, mismatch -> log critical alert and refuse to load.
        """
        with open(filepath, 'rb') as f:
            file_data = f.read()

        actual_hash = hashlib.sha256(file_data).hexdigest()
        expected_hash = self._expected_hashes.get(model_name)

        if not expected_hash:
            msg = (
                f"No integrity hash registered for '{model_name}'. "
                "Run 'python Crypto-models/generate_model_hashes.py' to enable verification."
            )
            logger.warning(msg)
            self.security_alerts.append({
                'timestamp': str(datetime.datetime.now()),
                'level': 'warning',
                'message': msg
            })
            return pickle.loads(file_data)

        if actual_hash != expected_hash:
            msg = (
                f"SECURITY ALERT: Hash mismatch for '{model_name}'! "
                f"Expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )
            logger.error(msg)
            self.security_alerts.append({
                'timestamp': str(datetime.datetime.now()),
                'level': 'critical',
                'message': msg
            })
            raise ValueError(msg)

        logger.info(f"Integrity verified: {model_name}")
        return pickle.loads(file_data)

    def _patch_njobs(self, pipeline):
        """Walk a sklearn Pipeline and set n_jobs=1 on every estimator that
        supports it.  This prevents joblib from attempting to read
        /sys/fs/cgroup/cpu.max on Windows/Docker environments where that
        virtual file does not exist, which would raise a FileNotFoundError
        and cause prediction to fall back to the error path.

        Only the estimator's runtime attribute is mutated; the serialised
        .pkl file on disk is never modified.
        """
        try:
            for _, step in pipeline.steps:
                if hasattr(step, 'n_jobs'):
                    step.n_jobs = 1
                if hasattr(step, 'transformers'):
                    for _, transformer, _ in step.transformers:
                        if hasattr(transformer, 'n_jobs'):
                            transformer.n_jobs = 1
        except Exception as e:  # pragma: no cover
            logger.debug(f"_patch_njobs encountered non-critical issue: {e}")

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

            # Environment-compatibility patch: force single-threaded prediction
            # so joblib never tries to detect CPU count via /sys/fs/cgroup/cpu.max.
            self._patch_njobs(self.rf_pipeline)
            self._patch_njobs(self.svm_pipeline)
            self._patch_njobs(self.iso_pipeline)

            self.models_loaded = True
            logger.info("All ML models loaded successfully via secure loader.")
        except Exception as e:
            self.models_loaded = False
            msg = f"Failed to load ML models securely: {e}"
            logger.error(msg)
            self.security_alerts.append({
                'timestamp': str(datetime.datetime.now()),
                'level': 'critical',
                'message': msg
            })

    def generate_features(self, filename: str, file_size: int, role: str = 'SoftwareEngineer', activity: str = 'File Copy', avg_actions_per_day: float = 15.0) -> dict:
        """Centralized helper to generate standardized ML feature vectors.
        Serves as the single source of truth for both sync and async paths.
        """
        now = datetime.datetime.now()
        day_of_week = now.weekday()
        return {
            'hour_of_day': now.hour,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'avg_actions_per_day': avg_actions_per_day,
            'activity': activity,
            'role': role
        }

    def validate_features(self, features: dict) -> None:
        """Validate the feature dictionary before prediction.

        Raises `ValueError` with a descriptive message when:
        - a required feature key is missing,
        - a value is `None`, or
        - a value is not one of the allowed types for that feature.

        This method does not modify `features`.
        """
        if not isinstance(features, dict):
            raise ValueError(
                f"features must be a dict, got {type(features).__name__}"
            )

        errors = []
        for name, allowed_types in self.REQUIRED_FEATURES.items():
            if name not in features:
                errors.append(f"missing required feature '{name}'")
                continue
            value = features[name]
            if value is None:
                errors.append(f"feature '{name}' must not be None")
                continue
            if not isinstance(value, allowed_types):
                type_names = " or ".join(t.__name__ for t in allowed_types)
                errors.append(
                    f"feature '{name}' must be {type_names}, "
                    f"got {type(value).__name__} ({value!r})"
                )

        if errors:
            raise ValueError(
                "Feature validation failed: " + "; ".join(errors)
            )

    def _validate_output(self, result: dict) -> None:
        """Assert that risk_score, final_verdict, and is_risky are internally
        consistent.  Raises ValueError if an inconsistency is detected.
        Only called when ml_status is not 'Error'.
        """
        risk_score = result.get('risk_score')
        final_verdict = result.get('final_verdict')
        is_risky = result.get('is_risky')

        if risk_score is None or final_verdict is None or is_risky is None:
            raise ValueError(
                f"Output consistency check: risk_score, final_verdict, and "
                f"is_risky must all be set; got risk_score={risk_score!r}, "
                f"final_verdict={final_verdict!r}, is_risky={is_risky!r}"
            )

        expected_verdict = self._get_verdict(risk_score)
        if final_verdict != expected_verdict:
            raise ValueError(
                f"Output consistency check: risk_score={risk_score} implies "
                f"verdict='{expected_verdict}' but got '{final_verdict}'"
            )

        expected_is_risky = risk_score >= self.HIGH_RISK_THRESHOLD
        if bool(is_risky) != expected_is_risky:
            raise ValueError(
                f"Output consistency check: risk_score={risk_score} implies "
                f"is_risky={expected_is_risky} but got {is_risky}"
            )

    def analyze_risk(self, request_features, request_id: str = None):
        """Analyzes a request and returns a risk assessment.

        Falls back to a mock low-risk score when models are not loaded,
        so the upload pipeline never hard-blocks on missing ML files.

        Args:
            request_features: dict of feature values produced by generate_features().
            request_id: optional identifier used in structured log messages.
        """
        _id = request_id or "unknown"

        if not self.models_loaded:
            logger.warning("ML models not loaded -- returning mock risk score.")
            mock_score = round(random.uniform(0.1, 0.4), 2)
            is_risky = mock_score >= self.HIGH_RISK_THRESHOLD
            return {
                'risk_score': mock_score,
                'is_risky': is_risky,
                'confidence': 0.85,
                'final_verdict': self._get_verdict(mock_score),
                'model_predictions': {},
                'factors': ['Mock ML Engine Active'],
                'ml_status': 'MOCK MODE'
            }

        t_start = time.monotonic()

        try:
            if not _PANDAS_AVAILABLE:
                raise ImportError("pandas is not installed")

            # Create a 1-row DataFrame matching training columns
            df = pd.DataFrame([request_features])

            rf_pred = self.rf_pipeline.predict_proba(df)[0][1]

            # LinearSVC has decision_function instead of predict_proba
            import math
            svm_decision = self.svm_pipeline.decision_function(df)[0]
            # Convert decision score to a probability-like value using Sigmoid
            svm_pred = 1.0 / (1.0 + math.exp(-svm_decision))

            # IsolationForest returns 1 (inlier) or -1 (outlier)
            iso_pred = self.iso_pipeline.predict(df)[0]

            # Ensemble: average RF + SVM probabilities
            ensemble_risk = (rf_pred + svm_pred) / 2.0

            # Isolation Forest anomaly flag boosts risk
            if iso_pred == -1:
                ensemble_risk = min(1.0, ensemble_risk + 0.3)

            is_risky = ensemble_risk >= self.HIGH_RISK_THRESHOLD
            rounded_risk = round(float(ensemble_risk), 2)
            final_verdict = self._get_verdict(rounded_risk)
            confidence = 0.92
            elapsed = round(time.monotonic() - t_start, 4)

            result = {
                'risk_score': rounded_risk,
                'is_risky': bool(is_risky),
                'confidence': confidence,
                'final_verdict': final_verdict,
                'model_predictions': {
                    'random_forest_score': round(float(rf_pred), 4),
                    'svm_score': round(float(svm_pred), 4),
                    'isolation_forest_anomaly': int(iso_pred == -1)
                },
                'factors': ['Ensemble Prediction', 'Isolation Forest checked'],
                'ml_status': 'Active'
            }

            # Internal consistency guard
            self._validate_output(result)

            # Structured DEBUG audit log (never exposes passwords or file content)
            logger.debug(
                "ML prediction [request_id=%s] | features=%s | "
                "rf=%.4f svm=%.4f iso=%s | ensemble=%.4f | "
                "confidence=%.2f | verdict=%s | elapsed=%.4fs",
                _id,
                {k: v for k, v in request_features.items()},
                rf_pred,
                svm_pred,
                iso_pred,
                ensemble_risk,
                confidence,
                final_verdict,
                elapsed,
            )

            return result

        except Exception as e:
            elapsed = round(time.monotonic() - t_start, 4)
            logger.error(
                "ML Analysis failed [request_id=%s, elapsed=%.4fs]: %s",
                _id,
                elapsed,
                e,
                exc_info=True,
            )
            # Return a null-safe error dict.
            # Prediction errors must NEVER be misread as High Risk.
            return {
                'risk_score': None,
                'is_risky': None,
                'confidence': None,
                'final_verdict': None,
                'model_predictions': {},
                'factors': [f'Analysis Error: {str(e)}'],
                'ml_status': 'Error',
                'error': str(e),
            }
