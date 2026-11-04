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




