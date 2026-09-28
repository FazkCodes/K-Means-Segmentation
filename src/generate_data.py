"""Generate a synthetic customer dataset with hidden segments."""
import numpy as np
import pandas as pd

SEGMENTS = {
    # name: (n, age, income_k, spending_score, visits_per_month, avg_basket)
    "Young Spenders":   (250, 24, 45, 80, 9, 60),
    "Affluent Loyal":   (200, 45, 110, 75, 6, 180),
    "Budget Families":  (300, 38, 40, 35, 4, 70),
    "Savers":           (200, 52, 85, 20, 2, 90),
    "Occasional Shoppers": (150, 30, 55, 50, 1, 45),
}

def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, (n, age, inc, spend, visits, basket) in SEGMENTS.items():
        rows.append(pd.DataFrame({
            "age": rng.normal(age, 5, n).clip(18, 80),
            "annual_income_k": rng.normal(inc, 10, n).clip(10, 200),
            "spending_score": rng.normal(spend, 8, n).clip(1, 100),
            "visits_per_month": rng.normal(visits, 1.5, n).clip(0, 30),
            "avg_basket_value": rng.normal(basket, 15, n).clip(5, 500),
        }))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "customer_id", range(1, len(df) + 1))
    return df.round(2)

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/customers.csv", index=False)
    print(f"Saved {len(df)} customers to data/customers.csv")
