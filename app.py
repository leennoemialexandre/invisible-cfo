import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta

def simulate_cash_forecast(
    starting_cash: float,
    daily_revenue_avg: float,
    daily_revenue_std: float,
    monthly_rent: float,
    biweekly_payroll: float,
    inventory_purchase_amount: float,
    inventory_purchase_day: int,  # 1–90
    days: int = 90,
    revenue_multiplier: float = 1.0,
    payroll_multiplier: float = 1.0,
    inventory_day_shift: int = 0,
    seed: int = 42,
):
    if not (1 <= inventory_purchase_day <= days):
        raise ValueError("inventory_purchase_day must be between 1 and 90.")

    rng = np.random.default_rng(seed)
    day = np.arange(1, days + 1)

    revenue = rng.normal(daily_revenue_avg * revenue_multiplier, daily_revenue_std, size=days)
    revenue = np.maximum(revenue, 0)

    rent_days = {1, 31, 61}
    rent = np.array([monthly_rent if d in rent_days else 0 for d in day], dtype=float)

    payroll_days = set(range(14, days + 1, 14))
    payroll = np.array([biweekly_payroll * payroll_multiplier if d in payroll_days else 0 for d in day], dtype=float)

    inv_day = inventory_purchase_day + inventory_day_shift
    inventory = np.zeros(days, dtype=float)
    if 1 <= inv_day <= days:
        inventory[inv_day - 1] = inventory_purchase_amount

    expenses = rent + payroll + inventory
    net = revenue - expenses

    cash = np.zeros(days, dtype=float)
    cash[0] = starting_cash + net[0]
    for i in range(1, days):
        cash[i] = cash[i - 1] + net[i]

    start_dt = date.today()
    dates = [start_dt + timedelta(days=int(d - 1)) for d in day]

    return pd.DataFrame({
        "day": day,
        "date": dates,
        "revenue": revenue,
        "rent": rent,
        "payroll": payroll,
        "inventory": inventory,
        "expenses_total": expenses,
        "net_cashflow": net,
        "cash_balance": cash
    })

def compute_metrics(df: pd.DataFrame):
    cash = df["cash_balance"].values
    min_cash = float(np.min(cash))
    below_zero_idx = np.where(cash < 0)[0]
    if len(below_zero_idx) == 0:
        runway_days = int(df["day"].max())
        shortfall_day = None
        shortfall_date = None
    else:
        shortfall_day = int(df.iloc[below_zero_idx[0]]["day"])
        shortfall_date = df.iloc[below_zero_idx[0]]["date"]
        runway_days = shortfall_day - 1
    return {
        "runway_days": runway_days,
        "min_projected_cash": round(min_cash, 2),
        "shortfall_day": shortfall_day,
        "shortfall_date": shortfall_date
    }

def make_decision_cards(baseline, scenarios):
    base_min = baseline["min_projected_cash"]
    cards = []
    for key, m in scenarios.items():
        delta_min = round(m["min_projected_cash"] - base_min, 2)

        if key == "delay_inventory_7d":
            action = "Delay inventory purchase by 7 days"
            tradeoffs = "Possible stockouts or supplier timing issues."
        elif key == "increase_revenue_5pct":
            action = "Increase revenue by ~5% (pricing/upsells/promos)"
            tradeoffs = "Price sensitivity; promos may reduce margin; requires execution."
        elif key == "reduce_payroll_10pct":
            action = "Reduce payroll by ~10% (shift scheduling/part-time)"
            tradeoffs = "Service quality risk; staff morale; peak coverage constraints."
        else:
            action = key
            tradeoffs = "Depends on implementation."

        confidence = "High" if delta_min >= 5000 else ("Medium" if delta_min >= 1500 else "Low")
        rationale = (
            f"Improves minimum cash buffer by ${delta_min:,.2f} "
            f"(from ${base_min:,.2f} to ${m['min_projected_cash']:,.2f})."
        )

        cards.append({
            "action": action,
            "rationale": rationale,
            "impact_min_cash_delta": delta_min,
            "confidence": confidence,
            "tradeoffs": tradeoffs
        })

    cards.sort(key=lambda x: x["impact_min_cash_delta"], reverse=True)
    return cards

st.set_page_config(page_title="Invisible CFO", layout="wide")
st.title("Invisible CFO — 90-Day Cash Decision Engine")

with st.sidebar:
    st.header("Inputs")
    starting_cash = st.number_input("Starting cash ($)", min_value=0.0, value=18000.0, step=500.0)
    daily_revenue_avg = st.number_input("Avg daily revenue ($)", min_value=0.0, value=900.0, step=50.0)
    daily_revenue_std = st.number_input("Daily revenue volatility (std)", min_value=0.0, value=250.0, step=25.0)
    monthly_rent = st.number_input("Monthly rent ($)", min_value=0.0, value=4000.0, step=100.0)
    biweekly_payroll = st.number_input("Biweekly payroll ($)", min_value=0.0, value=12000.0, step=250.0)
    inventory_purchase_amount = st.number_input("Inventory purchase amount ($)", min_value=0.0, value=6000.0, step=250.0)
    inventory_purchase_day = st.slider("Inventory purchase day (1–90)", min_value=1, max_value=90, value=20)
    seed = st.number_input("Simulation seed (keeps results stable)", min_value=0, value=42, step=1)

baseline_df = simulate_cash_forecast(
    starting_cash=starting_cash,
    daily_revenue_avg=daily_revenue_avg,
    daily_revenue_std=daily_revenue_std,
    monthly_rent=monthly_rent,
    biweekly_payroll=biweekly_payroll,
    inventory_purchase_amount=inventory_purchase_amount,
    inventory_purchase_day=int(inventory_purchase_day),
    seed=int(seed),
)
baseline_metrics = compute_metrics(baseline_df)

scenario_dfs = {
    "delay_inventory_7d": simulate_cash_forecast(
        starting_cash, daily_revenue_avg, daily_revenue_std, monthly_rent, biweekly_payroll,
        inventory_purchase_amount, int(inventory_purchase_day),
        inventory_day_shift=7, seed=int(seed)
    ),
    "increase_revenue_5pct": simulate_cash_forecast(
        starting_cash, daily_revenue_avg, daily_revenue_std, monthly_rent, biweekly_payroll,
        inventory_purchase_amount, int(inventory_purchase_day),
        revenue_multiplier=1.05, seed=int(seed)
    ),
    "reduce_payroll_10pct": simulate_cash_forecast(
        starting_cash, daily_revenue_avg, daily_revenue_std, monthly_rent, biweekly_payroll,
        inventory_purchase_amount, int(inventory_purchase_day),
        payroll_multiplier=0.90, seed=int(seed)
    ),
}
scenario_metrics = {k: compute_metrics(v) for k, v in scenario_dfs.items()}
cards = make_decision_cards(baseline_metrics, scenario_metrics)

col1, col2, col3 = st.columns(3)
col1.metric("Runway (days)", baseline_metrics["runway_days"])
col2.metric("Minimum projected cash", f"${baseline_metrics['min_projected_cash']:,.2f}")
risk = "High" if baseline_metrics["min_projected_cash"] < 1000 else ("Medium" if baseline_metrics["min_projected_cash"] < 5000 else "Low")
col3.metric("Liquidity Risk", risk)

st.subheader("90-Day Cash Forecast (Baseline)")
fig = plt.figure()
plt.plot(baseline_df["day"], baseline_df["cash_balance"])
plt.axhline(0)
plt.xlabel("Day")
plt.ylabel("Cash Balance ($)")
st.pyplot(fig)

st.subheader("Decision Cards (Ranked by Liquidity Buffer Improvement)")
for c in cards:
    with st.expander(
        f"{c['action']}  •  Confidence: {c['confidence']}  •  Δ Min Cash: ${c['impact_min_cash_delta']:,.2f}",
        expanded=False
    ):
        st.write("**Why:**", c["rationale"])
        st.write("**Tradeoffs:**", c["tradeoffs"])

st.subheader("Scenario Metrics")
st.dataframe(pd.DataFrame(scenario_metrics).T)
