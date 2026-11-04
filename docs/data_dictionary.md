\# Data Dictionary — CarGurus Sample (500k rows)



Generated: 2026-11-04

Source: us-used-cars-dataset (Kaggle / CarGurus)

Sample: 500,000 rows from 3,000,040, seed=42

Parquet: data/processed/sample\_500k.parquet (15.1 MB)



\## ⚠️ Columns missing from CSV (need real names)

\- make → actual name: TBD

\- model → actual name: TBD

\- trim → actual name: TBD

\- salvage\_title → actual name: TBD

\- fleet\_used → actual name: TBD



\## Target

| Column | Type | Nulls | Description |

|--------|------|-------|-------------|

| price | Float64 | 0 | Asking price USD — NOT sale price. Will be log-transformed. |



\## Features kept

| Column | Type | Nulls | Notes |

|--------|------|-------|-------|

| vin | String | 0 | Dedup key + VIN-grouped split key. Drop before training. |

| year | Int64 | 0 | Model year |

| body\_type | String | 2,246 | Sedan, SUV/Crossover, etc. |

| mileage | Float64 | ? | |

| fuel\_type | String | ? | |

| engine\_type | String | ? | |

| transmission | String | ? | |

| wheel\_system | String | ? | FWD / AWD / 4WD |

| horsepower | Float64 | ? | |

| city\_fuel\_economy | Float64 | ? | |

| highway\_fuel\_economy | Float64 | ? | |

| has\_accidents | Boolean | ? | |

| frame\_damaged | Boolean | ? | |

| theft\_title | Boolean | ? | |

| owner\_count | Float64 | ? | |

| dealer\_zip | Int64 | ? | |

| is\_new | Boolean | 0 | |

| listed\_date | String | 0 | Drop before training — unknown at prediction time |

| latitude | Float64 | 0 | Dealer location |

| longitude | Float64 | 0 | Dealer location |



\## Columns dropped (reason)

| Column | Reason |

|--------|--------|

| mmr | LEAKAGE — computed from price |

| days\_on\_market | Unknown at prediction time |

| make, model, trim | Not found under these names — finding real names |

## Key observations from load.py (Day 1)



1\. 3,000,040 total rows; sampled 500,000 (seed=42)

2\. Price range $299–$2,749,950 — extremes are bad data; domain rules

&#x20;  will drop <$500 and IQR will catch upper outliers per make-model-year group

3\. Year range 1921–2021 — will drop years before 1990 (too old to be

&#x20;  useful; too few listings to generalize)

4\. is\_new is \~50/50 (245k new, 254k used) — good balance

5\. has\_accidents, frame\_damaged, salvage, fleet all 47.6% missing —

&#x20;  KEEP (title/damage flags strongly affect price); add was\_missing flags

6\. owner\_count 50.6% missing — decision needed (keep or drop)

7\. make\_name, model\_name, vin, price, year, is\_new, listed\_date,

&#x20;  lat/lon have 0 nulls — solid foundation columns

8\. Columns excluded for leakage: savings\_amount, daysonmarket

9\. Parquet (18 MB) vs raw CSV (9.3 GB) — compression + columnar format +

&#x20;  selecting 29 of 66 columns = \~500x smaller on disk, loads in <1s

## 10 EDA Insights — Day 2



1\. \*\*Price is right-skewed.\*\* Mean ($29,929) > median ($26,500) by $3,400. 

&#x20;  A handful of luxury/exotic listings pull the mean up. Model log(price) 

&#x20;  to make errors proportional rather than absolute.



2\. \*\*New cars cost 72% more than used\*\* ($33,495 vs $19,442 median). 

&#x20;  The model must treat is\_new as a strong feature — it's essentially 

&#x20;  a different market segment.



3\. \*\*Pickup trucks are the most expensive body type\*\* ($40k median), 

&#x20;  nearly 2.4x a hatchback ($17k). Body type must be encoded — 

&#x20;  it carries strong price signal independent of make.



4\. \*\*Depreciation is steepest in the first 50k miles.\*\* All 8 makes 

&#x20;  drop \~40–60% from 0→100k miles. The first 50k miles loses more 

&#x20;  value than the next 150k combined. Mileage is nonlinear — 

&#x20;  consider log(mileage) or mileage² as a feature.



5\. \*\*Horsepower is the strongest numeric predictor of price (r=0.59).\*\* 

&#x20;  It's a proxy for luxury — expensive cars have big engines. 

&#x20;  Mileage is next at -0.45. Year is weaker than expected at 0.36.



6\. \*\*Fuel economy is negatively correlated with price (r≈-0.26).\*\* 

&#x20;  Efficient cars tend to be cheaper economy models; gas-guzzlers 

&#x20;  tend to be expensive trucks and performance cars.



7\. \*\*238,098 rows (47.6%) have structural missingness\*\* across 5 columns 

&#x20;  simultaneously (has\_accidents, frame\_damaged, salvage, fleet, theft\_title). 

&#x20;  Same count = same source, not random dropout. 

&#x20;  was\_missing flag is essential for all five.



8\. \*\*owner\_count is 50.6% missing\*\* — decision: keep with was\_missing flag. 

&#x20;  Higher owner count correlates with lower price; missing likely means 

&#x20;  dealers hide unflattering history.



9\. \*\*Mileage has 24,023 nulls (4.8%)\*\* — discovered during depreciation 

&#x20;  curve plotting. Impute with median within make-model-year group. 

&#x20;  Add was\_missing\_mileage flag.



10\. \*\*Ford dominates at 16% of listings (79k rows).\*\* Model will learn 

&#x20;   Ford better than any other make. Watch for make-level bias in 

&#x20;   error analysis on Day 7.




