# 🎓 India Student Dropout Analysis

Analyzing and predicting student dropout rates across Indian states using UDISE+ government data.

## 📊 Dataset
- Source: [data.gov.in](https://data.gov.in) — UDISE+ State-wise Dropout Rates
- Years: 2019-20 to 2021-22
- Coverage: 37 States/UTs across Primary, Upper Primary & Secondary levels

## 🗂️ Project Structure
| File | Description |
|------|-------------|
| `01_EDA.ipynb` | Exploratory Data Analysis |
| `02_Feature_Engineering.ipynb` | Data reshaping, encoding & target variable creation |
| `03_Model_Building.ipynb` | ML model building & evaluation (all models) |
| `dropout_data.csv` | Raw dataset from UDISE+ |
| `dropout_ml_ready.csv` | ML-ready dataset (327 rows, 8 features) |

## 🔍 Key Findings So Far
- Secondary level has significantly higher dropout rates than Primary & Upper Primary
- Dataset reshaped from 37 → 327 rows using melt()
- Target: Predict high/low Secondary dropout using Primary & Upper Primary rates
- Top predictive feature: **Upper Primary 2021-22 dropout rate** — the level just before Secondary is the strongest early warning signal
- Random Forest fixes Decision Tree overfitting: CV F1 52.6% → 73.8%

## 📈 Model Results (Work in Progress)
| Model | Accuracy | F1 Score | CV F1 |
|-------|----------|----------|-------|
| Logistic Regression | 87.5% | 85.7% | 74.0% |
| Decision Tree | 87.5% | 85.7% | 52.6% |
| Random Forest | 100% | 100% | 73.8% |

## 🛠️ Tech Stack
Python, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
