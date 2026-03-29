"""
Conftest patches mlflow and joblib BEFORE src.app is imported,
because the model and scaler are loaded at module level in app.py.
"""
from unittest.mock import MagicMock, patch

# --- mock model ---
_mock_model = MagicMock()
# predict_proba returns [[prob_class0, prob_class1]]
_mock_model._model_impl.sklearn_model.predict_proba.return_value = [[0.2, 0.05]]

# --- mock scaler ---
_mock_scaler = MagicMock()
_mock_scaler.transform.return_value = [[0.5]]

# Start patches (kept alive for the whole test session)
patch("mlflow.pyfunc.load_model", return_value=_mock_model).start()
patch("joblib.load", return_value=_mock_scaler).start()

# Import now so sys.modules caches the patched version for all test files
import src.app  # noqa: E402
