

\---



\## 🔄 Data Pipeline



\*\*Source:\*\* \[US Used Cars Dataset (CarGurus, Kaggle)](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset) — 3M rows, 66 columns, 9.3 GB



\*\*Cleaning steps (`src/data/clean.py`):\*\*



| Step | Action | Rows removed |

|---|---|---|

| Fix types | Parse strings like "4 cylinders" → 4 | 0 |

| Dedup by VIN | Remove duplicate listings | 4 |

| Domain rules | Price <$500 or >$300k, year <1990, mileage >400k | 965 |

| Missingness | 9 `was\_missing` flags + group-median imputation | 0 |

| IQR outliers | Remove pricing anomalies within make-model-year groups | 10,696 |

| Log-transform | Add `log\_price` target column | 0 |

| Leakage audit | Drop `savings\_amount`, `daysonmarket`, `listed\_date` | 0 |

| VIN split | 70/15/15 train/val/test, zero VIN overlap | — |

| \*\*Final\*\* | \*\*488,335 rows, 38 columns, 0 nulls\*\* | — |



\---



\## 🤖 Modeling



The price model climbs from a dumb baseline to a neural network, so each step teaches one concept:



| Step | Model | Key concept | Val MAE |

|---|---|---|---|

| 1 | Median by make-model-year | Baseline floor | $3,518 |

| 2 | Linear regression (NumPy GD) | Gradients, learning rate | $5,206 |

| 3 | Ridge + Lasso (sklearn) | Regularization | $5,206 |

| 4 | Random Forest (200 trees) | Ensembles, bagging | $2,331 |

| 5 | LightGBM default | Boosting | $1,856 |

| 6 | LightGBM + Optuna (30 trials) | Hyperparameter search | \*\*$1,633\*\* |

| 7 | PyTorch MLP + embeddings | Backprop, embeddings | Coming |



All experiments tracked in MLflow. Target: R² ≥ 0.90, MAPE ≤ 15%. \*\*Achieved: R²=0.974, MAPE=6.25%.\*\*



\---



\## 📈 Key EDA Findings



1\. Price is right-skewed (mean $29,929 > median $26,500) — log-transform essential

2\. New cars cost 72% more than used ($33,495 vs $19,443 median)

3\. Pickup trucks are the most expensive body type ($40k median)

4\. Depreciation steepest in the first 50k miles — mileage effect is nonlinear

5\. Horsepower is the strongest numeric predictor of price (r=0.59)

6\. 238,098 rows (47.6%) have structural missingness across 5 damage columns simultaneously

7\. Ford dominates at 16% of listings — model learns Ford better than any other make



\---



\## 🔍 Model Limitations



| Segment | MAPE | Recommendation |

|---|---|---|

| $10k–$60k mainstream | 5–7% | ✅ Reliable |

| New cars (0–2 years) | 5.0% | ✅ Reliable |

| Salvage title | 11.8% | ⚠️ Use with caution |

| Cars under $10k | 15.5% | ⚠️ Use with caution |

| Cars 15+ years old | 28.9% | 🚨 Avoid — collector market |

| Luxury > $80k | 30–55% | 🚨 Avoid — insufficient training data |



> \*\*Note:\*\* Training data predates the 2021–2022 used-car price spike. A BLS CPI adjustment is applied for current listings.



\---



\## 🧰 Tech Stack



| Layer | Tools |

|---|---|

| Data | Python 3.11, Polars, pandas, Parquet |

| ML | scikit-learn, LightGBM, CatBoost, Optuna, PyTorch, SHAP |

| Tracking | MLflow, TensorBoard |

| Backend | FastAPI, Pydantic, PostgreSQL |

| Frontend | React + Tailwind |

| LLM | Claude API — profile parsing + "why this car" explanations |

| Deploy | Docker, Render (API), Vercel (frontend) |



\---



\## 🚀 Setup



```bash

git clone https://github.com/riyajclgn123/Car-Recommender.git

cd Car-Recommender

uv python install 3.11

uv venv --python 3.11 .venv

source .venv/Scripts/activate

uv pip install -r requirements.txt

python -m src.data.load

python -m src.data.clean

python -m src.data.features

python -m src.models.baseline

python -m src.models.train\_gbm

mlflow ui --host 0.0.0.0 --port 5000

```



\---



\## 📋 14-Day Build Plan



| Day | Focus | Status |

|---|---|---|

| 1 | Setup, data pipeline, 500k sample to Parquet | ✅ Done |

| 2 | EDA — 10 insights, 7 plots | ✅ Done |

| 3 | `clean.py` — 8-step pipeline, VIN split | ✅ Done |

| 4 | Feature engineering + baseline | ✅ Done |

| 5 | Linear regression from scratch + Ridge/Lasso | ✅ Done |

| 6 | Random Forest + LightGBM + Optuna | ✅ Done |

| 7 | Error analysis + SHAP + model card | ✅ Done |

| 8 | PyTorch MLP + TensorBoard | 🔄 In progress |

| 9 | Cost model (5-year TCO) | ⬜ Upcoming |

| 10 | NHTS life-stage classifier | ⬜ Upcoming |

| 11 | Recommender engine + persona tests | ⬜ Upcoming |

| 12 | FastAPI — `/predict-price`, `/tco`, `/recommend` | ⬜ Upcoming |

| 13 | Frontend + LLM intake + "why this car" | ⬜ Upcoming |

| 14 | Docker, deploy, demo video | ⬜ Upcoming |



\---



\## 👤 About



Built by \*\*Riyaj Chaulagain\*\* — B.S. Computer Science (Mathematics minor) at Southeastern Louisiana University, graduating December 2027.



This project is how I learned the full ML lifecycle by doing it: EDA, data cleaning, feature engineering, gradient descent from scratch, tree ensembles, neural networks, explainability (SHAP), concept drift, and deployment.



\- 🌐 \[riyajchaulagain.com.np](https://riyajchaulagain.com.np)

\- 💼 \[LinkedIn](https://linkedin.com/in/riyajchaulagain)

\- 🐙 \[GitHub](https://github.com/riyajclgn123)

\- 📧 criyaj2017@gmail.com



\---



\## 📄 License



MIT License — see \[LICENSE](LICENSE) for details.

