# 🏠 House Price Prediction using Machine Learning

A machine learning project that predicts house sale prices
using the Ames Housing dataset.

## 🎯 Objective

Build and compare regression models to predict house prices
from property features.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Git and GitHub

## 📊 Dataset

The project uses the House Prices: Advanced Regression
Techniques dataset from Kaggle.

Dataset:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The dataset contains 1,460 labeled houses and 81 columns,
including the target column, SalePrice.

## ⚙️ Workflow

1. Load and explore the dataset.
2. Inspect missing values.
3. Separate features and target.
4. Split data into training and testing sets.
5. Impute missing numerical and categorical values.
6. Encode categorical features using OneHotEncoder.
7. Train Linear Regression and Random Forest models.
8. Compare performance using MAE, RMSE, and R².
9. Save the best-performing model.
10. Visualize actual versus predicted prices.

## 📈 Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 17360.00 | 28396.69 | 0.8949 |
| Linear Regression | 20473.63 | 31287.03 | 0.8724 |

Random Forest performed better on this test split.

## 📉 Visualization

The project generates an actual-versus-predicted price chart:

`actual_vs_predicted.png`

## 💻 How to Run

Clone the repository and enter the project directory.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate