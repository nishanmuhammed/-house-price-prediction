from pathlib import Path
import joblib
import pandas as pd

# Locate the project and saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# Load one real house from the dataset for a test
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
data = pd.read_csv(DATA_PATH)

# Select one house and remove its target and ID columns
house = data.drop(columns=["SalePrice", "Id"], errors="ignore").iloc[[0]]

# Predict its price
predicted_price = model.predict(house)[0]
actual_price = data["SalePrice"].iloc[0]

print("\nHOUSE PRICE PREDICTION TEST")
print("-" * 35)
print(f"Actual house price:    ${actual_price:,.2f}")
print(f"Predicted house price: ${predicted_price:,.2f}")
print(f"Absolute difference:   ${abs(actual_price - predicted_price):,.2f}")
print("\nPrediction test completed!")
