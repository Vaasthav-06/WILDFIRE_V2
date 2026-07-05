"""
src/api/json_utils.py

Custom JSON response utility for handling NaN, Infinity, and NumPy types
which standard FastAPI/Pydantic JSON encoders struggle with.
"""

import math
import numpy as np
from fastapi.responses import JSONResponse
import json
from typing import Any

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

def json_response(content: Any, status_code: int = 200) -> JSONResponse:
    """
    Return a FastAPI JSONResponse that properly serializes NaNs and NumPy types.
    NaNs and Infs are converted to `null` to comply with standard JSON.
    """
    return JSONResponse(
        content=sanitize_for_json(content), 
        status_code=status_code
    )
