"""
src/recommender/rank.py

Rank scored cars and apply diversity pass.
Returns top 5 with at most 2 of the same model.
"""

import polars as pl
import logging

log = logging.getLogger(__name__)


def diversity_rank(df: pl.DataFrame,
                   top_n: int = 5,
                   max_per_model: int = 2) -> pl.DataFrame:
    """
    Return top_n cars by score with diversity constraint:
    at most max_per_model cars of the same make+model.
    """
    df = df.sort("total_score", descending=True)

    selected = []
    model_counts: dict[str, int] = {}

    for row in df.iter_rows(named=True):
        key = f"{row['make_name']}_{row['model_name']}"
        count = model_counts.get(key, 0)

        if count < max_per_model:
            selected.append(row)
            model_counts[key] = count + 1

        if len(selected) == top_n:
            break

    return pl.DataFrame(selected)


def format_results(top5: pl.DataFrame) -> None:
    """Print the top 5 recommendations in a readable format."""
    log.info("\n" + "="*65)
    log.info("TOP 5 RECOMMENDATIONS")
    log.info("="*65)

    display_cols = [
        "make_name", "model_name", "year", "price",
        "predicted_price", "tco_5yr", "total_score",
        "s_tco_fit", "s_deal", "s_reliability",
        "s_lifestage", "s_climate"
    ]

    for i, row in enumerate(top5.iter_rows(named=True), 1):
        log.info(f"\n  #{i} — {row.get('make_name','')} "
                 f"{row.get('model_name','')} {row.get('year','')}")
        log.info(f"       Listing price:    ${row.get('price',0):>10,.0f}")
        log.info(f"       Predicted fair:   ${row.get('predicted_price',0):>10,.0f}")
        log.info(f"       5-yr TCO:         ${row.get('tco_5yr',0):>10,.0f}")
        log.info(f"       Total score:      {row.get('total_score',0):>10.3f}")
        log.info(f"       Scores: TCO={row.get('s_tco_fit',0):.2f} "
                 f"Deal={row.get('s_deal',0):.2f} "
                 f"Rel={row.get('s_reliability',0):.2f} "
                 f"Life={row.get('s_lifestage',0):.2f} "
                 f"Clim={row.get('s_climate',0):.2f}")