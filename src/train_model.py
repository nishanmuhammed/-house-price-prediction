from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHART_PATH = PROJECT_ROOT / "actual_vs_predicted.png"
RESULTS_PATH = PROJECT_ROOT / "model_comparison.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please place train.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

if "SalePrice" not in df.columns:
    raise ValueError("The dataset must contain a SalePrice column.")

print("=" * 60)
print("HOUSE PRICE PREDICTION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Number of houses: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0].head(15))

print("\nHouse price statistics:")
print(df["SalePrice"].describe())


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

# SalePrice is the value we want to predict.
# Id is excluded because it is an identifier, not a useful feature.
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

print(f"\nNumerical features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ============================================================
# 5. PREPROCESSING PIPELINES
# ============================================================

numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, random_state=42, n_jobs=-1
    ),
}

results = []
trained_models = {}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

for name, regressor in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append(
        {
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
        }
    )

    trained_models[name] = pipeline

    print(f"{name} evaluation:")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")


# ============================================================
# 8. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(RESULTS_PATH, index=False)

print(f"\nComparison results saved to: {RESULTS_PATH}")


# ============================================================
# 9. SELECT AND SAVE THE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

model_path = MODEL_DIR / "house_price_model.pkl"

joblib.dump(best_model, model_path)

print(f"\nBest model: {best_model_name}")
print(f"Saved model to: {model_path}")


# ============================================================
# 10. CREATE ACTUAL VS PREDICTED VISUALIZATION
# ============================================================

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(9, 6))

plt.scatter(y_test, best_predictions, alpha=0.65)

# Reference line: predictions on this line are exact.
lower = min(y_test.min(), best_predictions.min())
upper = max(y_test.max(), best_predictions.max())

plt.plot([lower, upper], [lower, upper], "r--", label="Perfect predictions")

plt.xlabel("Actual House Price ($)")
plt.ylabel("Predicted House Price ($)")

plt.title(f"Actual vs Predicted House Prices\nBest Model: {best_model_name}")

plt.legend()
plt.tight_layout()

plt.savefig(CHART_PATH, dpi=150)
plt.close()

print(f"Evaluation chart saved to: {CHART_PATH}")


# ============================================================
# 11. FINISH
# ============================================================

print("\n" + "=" * 60)
print("PROJECT TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
