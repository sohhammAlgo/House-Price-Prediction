# 🏠 California House Price Prediction

An end-to-end Machine Learning project that predicts California house prices using the California Housing Dataset from Scikit-Learn.

The project includes:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* XGBoost Regression
* SHAP Explainability
* Model Persistence using Joblib
* FastAPI Deployment

---

# 📌 Problem Statement

Accurately predicting house prices is an important problem in the real estate industry.

The goal of this project is to build a machine learning model that predicts median house values based on housing characteristics such as:

* Income
* House Age
* Population
* Average Rooms
* Geographic Location

---

# 📊 Dataset

Dataset Source:

```python
from sklearn.datasets import fetch_california_housing
```

Target Variable:

```text
MedHouseVal
```

Features:

| Feature    | Description       |
| ---------- | ----------------- |
| MedInc     | Median Income     |
| HouseAge   | Median House Age  |
| AveRooms   | Average Rooms     |
| AveBedrms  | Average Bedrooms  |
| Population | Block Population  |
| AveOccup   | Average Occupancy |
| Latitude   | Latitude          |
| Longitude  | Longitude         |

Dataset Size:

```text
20,640 Rows
8 Original Features
```

---

# 🔍 Exploratory Data Analysis

Performed:

* Dataset Inspection
* Missing Value Analysis
* Distribution Analysis
* Correlation Heatmap
* Outlier Detection
* Feature Relationship Analysis

Visualizations:

* Histograms
* Boxplots
* Correlation Matrix
* Feature Importance Graphs

---

# ⚙️ Feature Engineering

Created additional features to improve predictive performance.

### Rooms Per Bedroom

```python
RoomsPerBedroom = AveRooms / AveBedrms
```

### Income Per Occupant

```python
IncomePerOccupant = MedInc / AveOccup
```

### Population Density

```python
PopulationDensity = Population / AveOccup
```

### Log Transformations

```python
Population_log = log1p(Population)

AveOccup_log = log1p(AveOccup)
```

---

# 🤖 Model Training

Model Used:

```python
XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
```

Training Pipeline:

```text
Data Loading
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
XGBoost Training
      ↓
Evaluation
      ↓
Model Saving
```

---

# 📈 Evaluation Metrics

The model was evaluated using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

Example:

```text
MAE  : 0.23
RMSE : 0.39
R²   : 0.88
```

---

# 🔬 Model Explainability (SHAP)

Implemented SHAP (SHapley Additive Explanations) for model interpretability.

Generated:

* SHAP Summary Plot
* SHAP Feature Importance Plot
* SHAP Waterfall Plot

Insights:

* Median Income was the most influential feature.
* Geographic location strongly impacted house prices.
* Engineered features improved prediction quality.

---

# 💾 Model Persistence

The trained model is stored using Joblib.

```python
joblib.dump(
    model,
    "model/model.pkl"
)
```

Load model:

```python
model = joblib.load(
    "model/model.pkl"
)
```

---

# 🚀 FastAPI Deployment

The project exposes predictions through a FastAPI REST API.

### Start Server

```bash
uvicorn app:app --reload
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

# 📬 API Endpoints

## Home

```http
GET /
```

Response:

```json
{
  "message": "California House Price Prediction API Running"
}
```

---

## Health Check

```http
GET /health
```

Response:

```json
{
  "message": "Healthy"
}
```

---

## Predict House Price

```http
POST /predict
```

Request:

```json
{
  "MedInc": 8.5,
  "HouseAge": 25,
  "AveRooms": 6.5,
  "AveBedrms": 1.1,
  "Population": 1200,
  "AveOccup": 3.0,
  "Latitude": 34.05,
  "Longitude": -118.25
}
```

Response:

```json
{
  "predicted_house_price": 4.2315,
  "predicted_price_usd": 423150.00
}
```

---

# 📁 Project Structure

```text
house-price-prediction/
│
├── app.py
├── train.py
├── requirements.txt
│
├── model/
│   └── model.pkl
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
└── README.md
```

---

# 🛠️ Installation

Clone Repository:

```bash
git clone <your-repository-url>
cd house-price-prediction
```

Create Virtual Environment:

```bash
python -m venv .venv
```

Activate Environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

---

# 🧪 Run Training

```bash
python train.py
```

---

# 🌐 Run API

```bash
uvicorn app:app --reload
```

---

# 📚 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* XGBoost
* SHAP
* FastAPI
* Uvicorn
* Joblib

---

# 🚀 Future Improvements

* Hyperparameter Tuning
* Dockerization
* CI/CD Pipeline
* Cloud Deployment
* Model Monitoring
* Automated Retraining

---
