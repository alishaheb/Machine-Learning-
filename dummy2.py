import sys
try:
    import xgboost
    print("Interpreter:", sys.executable)
    print("XGBoost version:", xgboost.__version__)
except ImportError as e:
    print("ImportError:", e)
    print("Interpreter:", sys.executable)
