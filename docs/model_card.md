\# Model Card — Car Price Prediction Model v1



\## Model details

\- \*\*Type:\*\* LightGBM gradient boosting regressor

\- \*\*Target:\*\* log(price), reported in USD

\- \*\*Trained:\*\* November 2026

\- \*\*Author:\*\* Riyaj Chaulagain



\## Training data

\- Source: CarGurus US listings (Kaggle), model years 2006–2020

\- Sample: 488,335 listings after cleaning (from 3M raw)

\- Train/val/test: 70/15/15 split by VIN

\- Price range in training data: \~$500–$300,000



\## Performance (test set, 73,251 listings)

| Metric | Value |

|--------|-------|

| MAE    | $1,635 |

| MAPE   | 6.25% |

| RMSE   | $2,745 |

| R²     | 0.974 |



\## Performance by segment

| Segment        | MAPE   | Note |

|----------------|--------|------|

| $10k–$60k      | \~5–7%  | Reliable range |

| New cars (0-2 yr) | 5.0% | Best performance |

| < $10k         | 15.5%  | Use with caution |

| Salvage title  | 11.8%  | Damage severity unknown |

| 15+ years old  | 28.9%  | Collector market, avoid |

| Luxury > $80k  | 30-55% | Insufficient training data |



\## Top features (SHAP)

1\. year — most important; newer = more expensive

2\. horsepower — proxy for luxury tier

3\. mileage — higher mileage = lower price (nonlinear)

4\. highway\_fuel\_economy, wheel\_system, make\_name



\## Known limitations

1\. \*\*Training data predates 2021–2022 price spike.\*\* Predictions 

&#x20;  may be 20–40% below current market. Apply BLS CPI adjustment 

&#x20;  for current listings.

2\. \*\*Luxury vehicles (>$80k):\*\* MAPE 30–55%. Too few training 

&#x20;  examples. Do not use for Porsche, Ferrari, McLaren, Rolls-Royce.

3\. \*\*Classic/vintage cars (15+ years):\*\* MAPE 28.9%. Collector 

&#x20;  market pricing is not captured.

4\. \*\*Salvage titles:\*\* MAPE 11.8%. Damage severity unknown to model.

5\. \*\*Asking prices, not sale prices.\*\* Trained on listing prices 

&#x20;  which run higher than actual transaction prices.

6\. \*\*Geographic bias:\*\* Dealer locations skew toward certain states. 

&#x20;  Remote/rural area pricing may be less accurate.



\## Ethical considerations

\- This model provides guidance only, not financial advice.

\- Age, income, and location are used for filtering only — 

&#x20; never to limit which cars a user is shown.

\- All recommendations show their underlying numbers and reasons.



\## Leakage audit

\- Passed. Columns excluded: savings\_amount, daysonmarket, 

&#x20; listed\_date, mmr.

\- VIN-grouped split: same car never in train and test.

