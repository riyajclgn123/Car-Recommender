"""
tests/test_api.py
pytest tests for all three endpoints.

Run:
    pytest tests/ -v
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Predict price ─────────────────────────────────────────────────────────────
def test_predict_price_valid():
    r = client.post("/predict-price", json={
        "make_name":   "Toyota",
        "model_name":  "Camry",
        "year":        2019,
        "mileage":     45000,
        "body_type":   "Sedan",
        "horsepower":  203,
        "is_new":      False,
        "has_accidents": False,
        "owner_count": 1,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_price"] > 0
    assert data["make_name"] == "Toyota"
    print(f"  Toyota Camry 2019: ${data['predicted_price']:,.0f}")


def test_predict_price_invalid_year():
    r = client.post("/predict-price", json={
        "make_name":  "Toyota",
        "model_name": "Camry",
        "year":       2050,       # future year — should fail
        "mileage":    45000,
        "body_type":  "Sedan",
        "horsepower": 203,
    })
    assert r.status_code == 422   # validation error


def test_predict_price_negative_mileage():
    r = client.post("/predict-price", json={
        "make_name":  "Honda",
        "model_name": "Civic",
        "year":       2018,
        "mileage":    -100,       # invalid
        "body_type":  "Sedan",
        "horsepower": 158,
    })
    assert r.status_code == 422


# ── TCO ───────────────────────────────────────────────────────────────────────
def test_tco_valid():
    r = client.post("/tco", json={
        "predicted_price": 24000,
        "make_name":       "Toyota",
        "year":            2020,
        "city_mpg":        29,
        "highway_mpg":     38,
        "state":           "TX",
        "annual_miles":    12000,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["total_tco"] > data["predicted_price"]
    assert data["monthly_cost"] > 0
    print(f"  Toyota Camry TCO: ${data['total_tco']:,.0f} "
          f"(${data['monthly_cost']:,.0f}/mo)")


def test_tco_state_lowercase():
    """State should be auto-uppercased."""
    r = client.post("/tco", json={
        "predicted_price": 20000,
        "make_name":       "Honda",
        "year":            2019,
        "city_mpg":        30,
        "highway_mpg":     38,
        "state":           "ca",   # lowercase — validator uppercases
    })
    assert r.status_code == 200


# ── Recommend ─────────────────────────────────────────────────────────────────
def test_recommend_family():
    r = client.post("/recommend", json={
        "monthly_income":          95000 / 12,
        "max_maintenance_monthly": 300,
        "household_size":          4,
        "age_group":               "35-54",
        "income_band":             "high",
        "area_type":               "suburban",
        "state":                   "TX",
        "prefer_new":              None,
        "n_candidates":            1000,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["recommendations"]) == 5

    # Hard check: no result over budget
    for rec in data["recommendations"]:
        assert rec["listing_price"] <= data["budget_max_price"], \
            f"Budget violation: {rec['make_name']} {rec['model_name']} " \
            f"${rec['listing_price']:,.0f} > ${data['budget_max_price']:,.0f}"
        assert rec["why"] != ""
    print(f"\n  Family top pick: "
          f"{data['recommendations'][0]['make_name']} "
          f"{data['recommendations'][0]['model_name']} "
          f"{data['recommendations'][0]['year']}")


def test_recommend_budget_never_violated():
    """Critical test — budget must never be exceeded."""
    profiles = [
        {"monthly_income": 25000/12, "max_maintenance_monthly": 150,
         "household_size": 1, "age_group": "18-34",
         "income_band": "low", "area_type": "urban",
         "state": "LA", "n_candidates": 500},
        {"monthly_income": 60000/12, "max_maintenance_monthly": 200,
         "household_size": 2, "age_group": "65+",
         "income_band": "middle", "area_type": "suburban",
         "state": "FL", "n_candidates": 500},
    ]
    for profile in profiles:
        r = client.post("/recommend", json=profile)
        if r.status_code == 404:
            continue  # no results — that's ok
        assert r.status_code == 200
        data = r.json()
        for rec in data["recommendations"]:
            assert rec["listing_price"] <= data["budget_max_price"]


def test_recommend_invalid_age_group():
    r = client.post("/recommend", json={
        "monthly_income": 4000,
        "household_size": 1,
        "age_group":      "15-17",   # invalid literal
        "income_band":    "low",
        "area_type":      "urban",
        "state":          "CA",
    })
    assert r.status_code == 422

# ── Parse profile (rule-based, optional Groq fallback) ───────────────────────
def test_parse_profile_basic():
    r = client.post("/parse-profile", json={
        "text": "I am 32, make 70k a year in Dallas TX, family of 3, used is fine"})
    assert r.status_code == 200
    d = r.json()
    assert round(d["monthly_income"]) == 5833
    assert d["age_group"] == "18-34"
    assert d["household_size"] == 3
    assert d["state"] == "TX"
    assert d["area_type"] == "urban"
    assert d["income_band"] == "middle"
    assert d["prefer_new"] is None


def test_parse_profile_needs_income(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)   # test the rules alone
    r = client.post("/parse-profile", json={"text": "I need a cheap car"})
    assert r.status_code == 422


def test_parse_profile_pension():
    r = client.post("/parse-profile", json={
        "text": "retired with a pension of 4500, live on a farm in Iowa, want a new car"})
    assert r.status_code == 200
    d = r.json()
    assert d["monthly_income"] == 4500
    assert d["age_group"] == "65+"
    assert d["area_type"] == "rural"
    assert d["state"] == "IA"
    assert d["prefer_new"] is True