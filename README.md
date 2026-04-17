# 🎓 India Student Dropout Analysis

Analyzing and predicting student dropout rates across Indian states using UDISE+ government data.

## 📊 Dataset
- Source: [data.gov.in](https://data.gov.in) — UDISE+ State-wise Dropout Rates
- Years: 2019-20 to 2021-22
- Coverage: 36 States/UTs across Primary, Upper Primary & Secondary levels

## 🗂️ Project Structure
| File | Description |
|------|-------------|
| `01_EDA.ipynb` | Exploratory Data Analysis |
| `02_Feature_Engineering.ipynb` | Data reshaping, encoding & target variable creation |
| `dropout_data.csv` | Raw dataset from UDISE+ |
| `dropout_ml_ready.csv` | ML-ready dataset (327 rows, 8 features) |

## 🔍 Key Findings So Far
- Secondary level has significantly higher dropout rates than Primary & Upper Primary
- Dataset reshaped from 37 → 327 rows using melt()
- Target variable is perfectly balanced (50.5% vs 49.5%) — no oversampling needed

## 🛠️ Tech Stack
Python, Pandas, Matplotlib, Seaborn, Scikit-learn
