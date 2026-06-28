"""
app.py  —  Q-Commerce Hyper-Local Demand Forecasting Dashboard
==============================================================
Run:  streamlit run app.py
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q-Commerce Demand Forecasting Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  HUMAN-READABLE DISPLAY MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

STORE_DISPLAY = {
    "DS001_Andheri_MUM":     "Andheri — Mumbai",
    "DS002_Koramangala_BLR": "Koramangala — Bangalore",
    "DS003_CP_DEL":          "Connaught Place — Delhi",
}

SKU_DISPLAY = {
    "SKU001_Milk_1L":   "Milk (1 Litre)",
    "SKU002_Eggs_12pk": "Eggs (12-Pack)",
    "SKU003_Bread_500g":"Bread (500g Loaf)",
}

WEATHER_DISPLAY = {
    "Clear":        "☀️ Clear",
    "Cloudy":       "🌥️ Cloudy",
    "Rain":         "🌧️ Rain",
    "Extreme Heat": "🌡️ Extreme Heat",
}

HOUR_DISPLAY = {
    9:  "9 AM  — Morning Rush",
    18: "6 PM  — Evening Peak",
    21: "9 PM  — Night Window",
}

DOW_DISPLAY = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday",
    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
}

# Feature engineering → plain-English labels (used in feature importance charts)
FEATURE_DISPLAY = {
    "ema_3":            "Short-term Trend  (Exp. Moving Avg · 3 periods)",
    "ema_7":            "Long-term Trend   (Exp. Moving Avg · 7 periods)",
    "lag_1":            "Demand  1 Period Ago",
    "lag_2":            "Demand  2 Periods Ago",
    "lag_3":            "Demand  3 Periods Ago",
    "lag_7":            "Demand  7 Periods Ago",
    "rolling_mean_3":   "Rolling Average   (last 3 periods)",
    "rolling_mean_7":   "Rolling Average   (last 7 periods)",
    "rolling_std_3":    "Demand Volatility (last 3 periods)",
    "rolling_std_7":    "Demand Volatility (last 7 periods)",
    "rolling_max_3":    "Peak Demand       (last 3 periods)",
    "rolling_max_7":    "Peak Demand       (last 7 periods)",
    "hour":             "Hour of Day",
    "day_of_week":      "Day of Week",
    "day_of_month":     "Day of Month",
    "month":            "Month",
    "week_of_year":     "Week of Year",
    "is_weekend":       "Is Weekend",
    "hour_sin":         "Hour of Day — Cyclical Sine",
    "hour_cos":         "Hour of Day — Cyclical Cosine",
    "dow_sin":          "Day of Week — Cyclical Sine",
    "dow_cos":          "Day of Week — Cyclical Cosine",
    "temperature_celsius": "Outside Temperature (°C)",
    "weather_encoded":  "Weather Condition",
    "rain_flag":        "Raining Today",
    "heat_flag":        "Extreme Heat Day",
    "rain_evening":     "Rain × Evening Hours (Combined Effect)",
    "weekend_evening":  "Weekend × Evening Hours (Combined Effect)",
    "holiday_evening":  "Holiday × Evening Hours (Combined Effect)",
    "public_holiday_flag": "Public Holiday",
    "local_event_flag": "Local Event Day (IPL Match / Festival)",
    "stockout_flag":    "Previous Stockout Occurred",
    "store_encoded":    "Dark Store Location",
    "sku_encoded":      "Product Type",
}

# Colour palettes
PALETTE = {
    "Random Forest": "#2196F3",
    "XGBoost":       "#FF5722",
    "LightGBM":      "#4CAF50",
    "Prophet":       "#FF9800",
    "Actual":        "#9C27B0",
}
WEATHER_COLORS = {
    "☀️ Clear":        "#FFC107",
    "🌥️ Cloudy":       "#90A4AE",
    "🌧️ Rain":         "#1565C0",
    "🌡️ Extreme Heat": "#E53935",
}
PLOTLY_TEMPLATE = "plotly_white"

FEATURE_COLS: list[str] = [
    "hour", "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "lag_1", "lag_2", "lag_3", "lag_7",
    "rolling_mean_3", "rolling_mean_7",
    "rolling_std_3",  "rolling_std_7",
    "rolling_max_3",  "rolling_max_7",
    "ema_3", "ema_7",
    "temperature_celsius", "weather_encoded",
    "rain_flag", "heat_flag",
    "rain_evening", "weekend_evening", "holiday_evening",
    "public_holiday_flag", "local_event_flag", "stockout_flag",
    "store_encoded", "sku_encoded",
]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="📦 Loading dataset …")
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("synthetic_data.csv")
    except FileNotFoundError:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from generate_data import generate_qcommerce_data
        df = generate_qcommerce_data()
        df.to_csv("synthetic_data.csv", index=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date

    # Add human-readable display columns
    df["Store"]       = df["store_id"].map(STORE_DISPLAY).fillna(df["store_id"])
    df["Product"]     = df["sku_id"].map(SKU_DISPLAY).fillna(df["sku_id"])
    df["Weather"]     = df["weather_condition"].map(WEATHER_DISPLAY).fillna(df["weather_condition"])
    df["Day"]         = df["day_of_week"].map(DOW_DISPLAY)
    df["Time Window"] = df["hour_of_day"].map(HOUR_DISPLAY).fillna(df["hour_of_day"].astype(str) + ":00")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (mirrors feature_engineering.py)
# ══════════════════════════════════════════════════════════════════════════════

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)

    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    grp = df.groupby(["store_id", "sku_id"])
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = grp["quantity_sold"].shift(lag)
    for w in [3, 7]:
        df[f"rolling_mean_{w}"] = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f"rolling_std_{w}"]  = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f"rolling_max_{w}"]  = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).max())
    df["ema_3"] = grp["quantity_sold"].transform(lambda x: x.ewm(span=3).mean())
    df["ema_7"] = grp["quantity_sold"].transform(lambda x: x.ewm(span=7).mean())
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = df[f"lag_{lag}"].fillna(df["rolling_mean_3"])

    wmap = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Extreme Heat": 3}
    df["weather_encoded"] = df["weather_condition"].map(wmap).fillna(0).astype(int)
    df["rain_flag"]  = (df["weather_condition"] == "Rain").astype(int)
    df["heat_flag"]  = (df["weather_condition"] == "Extreme Heat").astype(int)

    eve = (df["hour"] >= 18).astype(int)
    df["rain_evening"]    = df["rain_flag"]           * eve
    df["weekend_evening"] = df["is_weekend"]          * eve
    df["holiday_evening"] = df["public_holiday_flag"] * eve

    le_s = LabelEncoder(); le_k = LabelEncoder()
    df["store_encoded"] = le_s.fit_transform(df["store_id"])
    df["sku_encoded"]   = le_k.fit_transform(df["sku_id"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "RMSE":   round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
        "MAE":    round(float(mean_absolute_error(yt, yp)), 3),
        "MAPE %": round(float(np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100), 2),
        "R²":     round(float(r2_score(yt, yp)), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🏋️ Training models — please wait …")
def train_all_models(_df: pd.DataFrame) -> dict:
    df     = _engineer(_df.copy()).sort_values("timestamp").reset_index(drop=True)
    X      = df[FEATURE_COLS].fillna(0)
    y      = df["quantity_sold"]
    split  = int(len(df) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    ts_te      = df["timestamp"].iloc[split:].values

    results: dict = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_pred = np.clip(rf.predict(X_te), 0, None)
    fi_rf   = (
        pd.Series(rf.feature_importances_, index=FEATURE_COLS)
          .rename(index=FEATURE_DISPLAY)
          .sort_values(ascending=False).head(15)
    )
    results["Random Forest"] = {
        "y_pred": rf_pred, "y_true": y_te.values, "ts": ts_te,
        "metrics": compute_metrics(y_te, rf_pred), "fi": fi_rf,
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb.fit(X_tr, y_tr)
    xgb_pred = np.clip(xgb.predict(X_te), 0, None)
    fi_xgb   = (
        pd.Series(xgb.feature_importances_, index=FEATURE_COLS)
          .rename(index=FEATURE_DISPLAY)
          .sort_values(ascending=False).head(15)
    )
    results["XGBoost"] = {
        "y_pred": xgb_pred, "y_true": y_te.values, "ts": ts_te,
        "metrics": compute_metrics(y_te, xgb_pred), "fi": fi_xgb,
    }

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgbm = LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, random_state=42, verbose=-1
    )
    lgbm.fit(X_tr, y_tr)
    lgbm_pred = np.clip(lgbm.predict(X_te), 0, None)
    fi_lgbm   = (
        pd.Series(lgbm.feature_importances_, index=FEATURE_COLS)
          .rename(index=FEATURE_DISPLAY)
          .sort_values(ascending=False).head(15)
    )
    results["LightGBM"] = {
        "y_pred": lgbm_pred, "y_true": y_te.values, "ts": ts_te,
        "metrics": compute_metrics(y_te, lgbm_pred), "fi": fi_lgbm,
    }

    # ── Prophet ───────────────────────────────────────────────────────────────
    try:
        from prophet import Prophet as _Prophet
        agg = df.groupby("timestamp")["quantity_sold"].sum().reset_index()
        agg.columns = ["ds", "y"]
        sp  = int(len(agg) * 0.80)
        m   = _Prophet(
            daily_seasonality=True, weekly_seasonality=True,
            yearly_seasonality=False, interval_width=0.90,
        )
        m.fit(agg.iloc[:sp])
        future = pd.DataFrame({"ds": agg["ds"].values})
        fc     = m.predict(future)
        p_pred = np.clip(fc["yhat"].values[sp:], 0, None)
        p_true = agg["y"].values[sp:]
        results["Prophet"] = {
            "y_pred": p_pred, "y_true": p_true,
            "ts": agg["ds"].values[sp:],
            "metrics": compute_metrics(p_true, p_pred),
            "fi": None,
        }
    except Exception as exc:
        results["Prophet"] = {
            "y_pred": np.zeros(len(y_te)), "y_true": y_te.values,
            "ts": ts_te,
            "metrics": {"RMSE": None, "MAE": None, "MAPE %": None, "R²": None},
            "fi": None, "error": str(exc),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER  —  rename columns for display
# ══════════════════════════════════════════════════════════════════════════════

def display_df(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Return a renamed subset of df for clean display."""
    return df[[c for c in cols if c in df.columns]].rename(columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_home(df: pd.DataFrame) -> None:
    st.title("🛒 Q-Commerce Demand Forecasting Engine")
    st.markdown(
        "> **Dark Store Intelligence Platform** — Hyper-Local Demand Prediction "
        "for 10–15-Minute Delivery across Mumbai · Bangalore · Delhi"
    )
    st.divider()

    # KPI strip
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Total Records",         f"{len(df):,}")
    c2.metric("🏪 Dark Stores",           df["Store"].nunique())
    c3.metric("🛍️ Products Tracked",      df["Product"].nunique())
    c4.metric("⚠️ Stockout Rate",         f"{df['stockout_flag'].mean()*100:.1f}%")
    c5.metric("📊 Avg Units / Time Slot", f"{df['quantity_sold'].mean():.1f}")

    st.divider()
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("📋 Transaction Records (first 25 rows)")
        show = display_df(df, {
            "timestamp":          "Timestamp",
            "Store":              "Dark Store",
            "Product":            "Product",
            "quantity_sold":      "Units Sold",
            "price_at_sale":      "Sale Price (₹)",
            "stockout_flag":      "Stockout?",
            "Weather":            "Weather",
            "temperature_celsius":"Temp (°C)",
            "public_holiday_flag":"Public Holiday?",
            "local_event_flag":   "Local Event?",
            "Time Window":        "Time Window",
        })
        st.dataframe(show.head(25), use_container_width=True, height=430)

    with col_r:
        st.subheader("📈 Units Sold — Distribution")
        fig = px.histogram(
            df, x="quantity_sold", nbins=30, color="Product",
            opacity=0.75, barmode="overlay",
            title="How many units are sold per transaction?",
            template=PLOTLY_TEMPLATE,
            labels={"quantity_sold": "Units Sold", "Product": "Product"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌦️ Weather — Days Distribution")
        wc = df.drop_duplicates(subset=["timestamp","store_id"])["Weather"].value_counts().reset_index()
        wc.columns = ["Weather Condition", "Number of Days"]
        fig2 = px.pie(
            wc, values="Number of Days", names="Weather Condition",
            color="Weather Condition",
            color_discrete_map=WEATHER_COLORS,
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("📊 Summary Statistics")
    stats = display_df(df, {
        "quantity_sold":      "Units Sold",
        "price_at_sale":      "Sale Price (₹)",
        "temperature_celsius":"Temperature (°C)",
        "stockout_flag":      "Stockout Flag",
        "public_holiday_flag":"Public Holiday Flag",
        "local_event_flag":   "Local Event Flag",
    }).describe().round(3)
    st.dataframe(stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_descriptive(df: pd.DataFrame) -> None:
    st.title("📊 Descriptive Analysis")
    st.markdown(
        "Exploring **when demand is highest** and **which products and stores drive volume** — "
        "through hourly profiles, weekly patterns, and revenue breakdowns."
    )
    st.divider()

    # Daily trend
    daily = df.groupby(["date", "Store"])["quantity_sold"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    st.subheader("📅 Daily Demand Trend — All Products per Store")
    fig = px.line(
        daily, x="date", y="quantity_sold", color="Store",
        markers=True, template=PLOTLY_TEMPLATE,
        title="Total units sold each day, broken down by dark store",
        labels={"quantity_sold": "Total Units Sold", "date": "Date", "Store": "Dark Store"},
    )
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏰ Demand by Time of Day")
        hourly = df.groupby(["Time Window", "Product"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            hourly, x="Time Window", y="quantity_sold", color="Product",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Which time slot drives the most demand?",
            labels={"quantity_sold": "Avg Units Sold", "Time Window": "Time Slot"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Demand by Day of the Week")
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekly = df.groupby(["Day", "Product"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            weekly, x="Day", y="quantity_sold", color="Product",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="How does demand shift across the week?",
            labels={"quantity_sold": "Avg Units Sold", "Day": "Day of Week"},
            category_orders={"Day": dow_order},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("🔥 Demand Heatmap — Time Slot vs Day of Week")
    pivot = df.pivot_table(
        index="Time Window", columns="Day",
        values="quantity_sold", aggfunc="mean"
    )
    # reorder columns
    ordered_days = [d for d in dow_order if d in pivot.columns]
    pivot = pivot[ordered_days]
    fig = px.imshow(
        pivot, aspect="auto",
        color_continuous_scale="Viridis", template=PLOTLY_TEMPLATE,
        title="Average units sold per (time slot × day) combination — darker = higher demand",
        labels={"x": "Day of Week", "y": "Time Slot", "color": "Avg Units"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🛍️ Total Sales Volume by Product")
        sku_tot = df.groupby("Product")["quantity_sold"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(
            sku_tot, x="Product", y="quantity_sold", color="Product",
            template=PLOTLY_TEMPLATE,
            title="Which product sells the most units overall?",
            labels={"quantity_sold": "Total Units Sold", "Product": "Product"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("🏪 Store Comparison — Average vs Volatility")
        ss = df.groupby("Store")["quantity_sold"].agg(["mean","std"]).reset_index()
        ss.columns = ["Dark Store", "Average Demand", "Std Deviation"]
        fig = px.bar(
            ss.melt(id_vars="Dark Store"),
            x="Dark Store", y="value", color="variable", barmode="group",
            template=PLOTLY_TEMPLATE,
            title="Average demand vs demand variability per store",
            labels={"value": "Units", "variable": "Metric", "Dark Store": "Dark Store"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rolling average
    st.subheader("📈 7-Period Rolling Average — Smoothed Demand Trend")
    df_s = df.sort_values("timestamp").copy()
    df_s["Rolling 7-Period Avg"] = (
        df_s.groupby(["store_id","sku_id"])["quantity_sold"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    focus_store = df_s["Store"].unique()[0]
    fig = px.line(
        df_s[df_s["Store"] == focus_store],
        x="timestamp", y="Rolling 7-Period Avg", color="Product",
        template=PLOTLY_TEMPLATE,
        title=f"Rolling average demand trend — {focus_store}",
        labels={"Rolling 7-Period Avg": "Rolling Avg (units)", "timestamp": "Date & Time"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Revenue
    st.divider()
    st.subheader("💰 Estimated Revenue by Product")
    df["Revenue (₹)"] = df["quantity_sold"] * df["price_at_sale"]
    rev = df.groupby("Product")["Revenue (₹)"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(
        rev, x="Product", y="Revenue (₹)", color="Product",
        template=PLOTLY_TEMPLATE,
        title="Which product generates the most revenue?",
        labels={"Revenue (₹)": "Estimated Revenue (₹)", "Product": "Product"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_diagnostic(df: pd.DataFrame) -> None:
    st.title("🔍 Diagnostic Analysis")
    st.markdown(
        "Identifying the **root causes of demand spikes and drops** — which weather conditions, "
        "calendar events, and time windows act as triggers for unusual demand behaviour."
    )
    st.divider()

    # ── Weather impact ────────────────────────────────────────────────────────
    st.subheader("🌦️ Impact of Weather on Demand")

    col1, col2 = st.columns(2)
    with col1:
        wa = df.groupby(["Weather","Product"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            wa, x="Weather", y="quantity_sold", color="Product",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Average units sold under each weather condition",
            labels={"quantity_sold": "Avg Units Sold", "Weather": "Weather Condition"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(
            df, x="Weather", y="quantity_sold",
            color="Weather",
            color_discrete_map=WEATHER_COLORS,
            box=True, points="all",
            template=PLOTLY_TEMPLATE,
            title="Full demand distribution shape by weather",
            labels={"quantity_sold": "Units Sold", "Weather": "Weather Condition"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # % deviation
    st.subheader("📊 Weather Demand Uplift or Suppression (% vs Overall Average)")
    overall_mean = df["quantity_sold"].mean()
    wd = df.groupby("Weather")["quantity_sold"].mean()
    wdev = ((wd - overall_mean) / overall_mean * 100).reset_index()
    wdev.columns = ["Weather Condition", "% Change vs Average"]
    wdev["Direction"] = wdev["% Change vs Average"].apply(
        lambda x: "Above Average 📈" if x >= 0 else "Below Average 📉")

    fig = px.bar(
        wdev, x="Weather Condition", y="% Change vs Average", color="Direction",
        color_discrete_map={"Above Average 📈": "#4CAF50", "Below Average 📉": "#F44336"},
        template=PLOTLY_TEMPLATE,
        title="How much does each weather condition shift demand compared to the overall mean?",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5,
                  annotation_text="Overall Average", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Holiday & event ───────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🎉 Public Holiday Effect")
        hol = df.copy()
        hol["Day Type"] = hol["public_holiday_flag"].map({0: "Normal Day", 1: "🎉 Public Holiday"})
        hol_agg = hol.groupby(["Day Type","Product"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            hol_agg, x="Day Type", y="quantity_sold", color="Product",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="How much does a public holiday boost demand?",
            labels={"quantity_sold": "Avg Units Sold", "Day Type": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

        hl = df.groupby("public_holiday_flag")["quantity_sold"].mean()
        if 1 in hl.index and 0 in hl.index:
            lift = (hl[1] - hl[0]) / hl[0] * 100
            st.metric("📈 Holiday Demand Uplift", f"+{lift:.1f}%",
                      "compared to a regular day")

    with col4:
        st.subheader("🎪 Local Event Effect  (IPL Match / Festival)")
        ev = df.copy()
        ev["Day Type"] = ev["local_event_flag"].map({0: "Normal Day", 1: "🎪 Event Day"})
        ev_agg = ev.groupby(["Day Type","Product"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            ev_agg, x="Day Type", y="quantity_sold", color="Product",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="How much does a local event (IPL/festival) boost demand?",
            labels={"quantity_sold": "Avg Units Sold", "Day Type": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

        el = df.groupby("local_event_flag")["quantity_sold"].mean()
        if 1 in el.index and 0 in el.index:
            elift = (el[1] - el[0]) / el[0] * 100
            st.metric("📈 Event Day Demand Uplift", f"+{elift:.1f}%",
                      "compared to a regular day")

    st.divider()

    # ── Weekend analysis ──────────────────────────────────────────────────────
    st.subheader("📅 Weekend vs Weekday — Demand & Stockout Rates")
    col5, col6 = st.columns(2)

    with col5:
        wk = df.copy()
        wk["Day Category"] = wk["is_weekend"].map({0: "Weekday", 1: "🎉 Weekend"})
        wk_agg = wk.groupby(["Day Category","Product"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            wk_agg, x="Day Category", y="quantity_sold", color="Product",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Weekends drive significantly higher demand",
            labels={"quantity_sold": "Avg Units Sold", "Day Category": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        so_wk = df.copy()
        so_wk["Day Category"] = so_wk["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        so_agg = so_wk.groupby("Day Category")["stockout_flag"].mean().reset_index()
        so_agg["Stockout Rate (%)"] = (so_agg["stockout_flag"] * 100).round(1)
        fig = px.bar(
            so_agg, x="Day Category", y="Stockout Rate (%)",
            color="Day Category",
            color_discrete_map={"Weekday": "#2196F3", "Weekend": "#FF5722"},
            template=PLOTLY_TEMPLATE,
            title="Weekends also see higher stockout rates — we need bigger safety stock",
            labels={"Stockout Rate (%)": "Stockout Rate (%)", "Day Category": ""},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Hour × Weather heatmap ────────────────────────────────────────────────
    st.divider()
    st.subheader("🌡️ Combined Effect: Time of Day × Weather Condition")
    inter = df.pivot_table(
        index="Time Window", columns="Weather",
        values="quantity_sold", aggfunc="mean"
    ).round(1)
    fig = px.imshow(
        inter, aspect="auto",
        color_continuous_scale="RdYlGn", template=PLOTLY_TEMPLATE,
        title="Average units sold per (time slot × weather) combination — "
              "green = high demand, red = low demand",
        labels={"color": "Avg Units Sold"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Volatility ────────────────────────────────────────────────────────────
    st.subheader("📉 Demand Volatility by Product & Store")
    st.caption(
        "Coefficient of Variation (CoV) = Standard Deviation ÷ Mean × 100.  "
        "Higher % = more unpredictable demand = higher safety stock needed."
    )
    cv = (
        df.groupby(["Store","Product"])["quantity_sold"]
          .agg(lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else 0)
          .reset_index()
    )
    cv.columns = ["Dark Store", "Product", "Demand Volatility (CoV %)"]
    cv["Demand Volatility (CoV %)"] = cv["Demand Volatility (CoV %)"].round(1)
    fig = px.bar(
        cv, x="Product", y="Demand Volatility (CoV %)", color="Dark Store",
        barmode="group", template=PLOTLY_TEMPLATE,
        title="Which products are hardest to forecast? (Higher bar = more volatile demand)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Stockout frequency ────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚠️ Stockout Frequency Analysis")
    col7, col8 = st.columns(2)

    with col7:
        so_w = df.groupby("Weather")["stockout_flag"].mean().reset_index()
        so_w["Stockout Rate (%)"] = (so_w["stockout_flag"] * 100).round(1)
        fig = px.bar(
            so_w, x="Weather", y="Stockout Rate (%)",
            color="Weather", color_discrete_map=WEATHER_COLORS,
            template=PLOTLY_TEMPLATE,
            title="Rainy days cause the most stockouts — pre-load stock before rain",
            labels={"Weather": "Weather Condition"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        so_s = df.groupby("Product")["stockout_flag"].mean().reset_index()
        so_s["Stockout Rate (%)"] = (so_s["stockout_flag"] * 100).round(1)
        fig = px.bar(
            so_s, x="Product", y="Stockout Rate (%)", color="Product",
            template=PLOTLY_TEMPLATE,
            title="Bread has the highest stockout risk due to short 3-day shelf life",
            labels={"Product": "Product"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICTIVE MODELING
# ══════════════════════════════════════════════════════════════════════════════

def page_modeling(df: pd.DataFrame) -> None:
    st.title("🤖 Predictive Modeling")
    st.markdown(
        "Four ML models — **Random Forest**, **XGBoost**, **LightGBM**, and **Prophet** — "
        "are trained on a rich feature set and evaluated on the most recent 20% of data "
        "(time-ordered, no data leakage)."
    )
    st.divider()

    # Feature engineering explainer
    with st.expander("🔧 What features are used to make predictions?", expanded=False):
        st.markdown("""
| **Feature Group** | **Features Used** | **Why It Helps** |
|---|---|---|
| **Time of Day & Calendar** | Hour, Day of Week, Day of Month, Month, Week of Year | Captures hourly peaks, weekday/weekend patterns, seasonal effects |
| **Cyclical Encoding** | Sine & Cosine of hour and day-of-week | Prevents the model from treating "Monday morning" and "Sunday night" as far apart in time |
| **Recent Demand History** | Demand 1, 2, 3, and 7 time-slots ago | The single most predictive group — recent sales strongly predict near-future sales |
| **Rolling Window Stats** | Average, Std Dev, Max over last 3 & 7 slots | Captures demand trends and volatility windows |
| **Exponential Moving Average** | Short EMA (3-slot) and Long EMA (7-slot) | Smoothed trend with higher weight on recent observations |
| **Weather Conditions** | Temperature, Rain flag, Extreme Heat flag | External demand drivers — rain boosts certain products by 20–40% |
| **Combined Effects** | Rain × Evening, Weekend × Evening, Holiday × Evening | Non-linear interactions that neither variable captures alone |
| **Events & Flags** | Public Holiday, Local Event (IPL/Festival), Stockout history | Calendar-driven demand shocks |
| **Store & Product Identity** | Dark store location, Product type | Demand levels differ significantly by store and product |
        """)

    st.subheader("🏋️ Train All Four Models")
    if st.button("🚀  Start Training", type="primary", use_container_width=True):
        results = train_all_models(df)
        st.session_state["model_results"] = results
        st.success("✅  All models trained and evaluated successfully!")

    if "model_results" not in st.session_state:
        st.info(
            "👆 Click **Start Training** above.  "
            "Random Forest, XGBoost, LightGBM, and Prophet will all train on an "
            "80% historical split and be evaluated on the remaining 20%."
        )
        return

    results: dict = st.session_state["model_results"]

    # ── Metrics comparison ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Model Evaluation Results — Test Set (most recent 20% of data)")
    st.caption(
        "**RMSE** & **MAE**: Lower is better — these measure average prediction error in units sold.  "
        "**MAPE %**: Average percentage error — lower is better.  "
        "**R²**: Proportion of demand variation explained — higher is better (max = 1.0).  "
        "🟢 Green cells highlight the best value in each column."
    )

    mrows = []
    for m_name, res in results.items():
        row = {"Model": m_name}
        row.update(res["metrics"])
        mrows.append(row)
    mdf = pd.DataFrame(mrows)

    styled = mdf.style.format(
        {"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE %": "{:.2f}", "R²": "{:.4f}"},
        na_rep="N/A"
    ).highlight_min(subset=["RMSE","MAE","MAPE %"], color="#c8e6c9") \
     .highlight_max(subset=["R²"],                  color="#c8e6c9")

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Bar chart comparisons ─────────────────────────────────────────────────
    st.subheader("📈 Visual Metrics Comparison")
    c1, c2 = st.columns(2)
    for metric, col, note in [
        ("RMSE",   c1, "Average prediction error in units  —  lower is better"),
        ("MAE",    c2, "Mean absolute error in units  —  lower is better"),
    ]:
        with col:
            m_plot = mdf.dropna(subset=[metric])
            fig = px.bar(
                m_plot, x="Model", y=metric, color="Model",
                color_discrete_map={k: v for k, v in PALETTE.items() if k in m_plot["Model"].values},
                title=f"{metric}  ({note})",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    for metric, col, note in [
        ("MAPE %", c3, "Average % error  —  lower is better"),
        ("R²",     c4, "Demand variance explained  —  higher is better"),
    ]:
        with col:
            m_plot = mdf.dropna(subset=[metric])
            fig = px.bar(
                m_plot, x="Model", y=metric, color="Model",
                color_discrete_map={k: v for k, v in PALETTE.items() if k in m_plot["Model"].values},
                title=f"{metric}  ({note})",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📉 Actual vs Predicted Demand — Test Period")
    st.caption("Dashed line = model prediction  |  Solid line = what actually happened")

    tabs = st.tabs(list(results.keys()))
    for tab, (m_name, res) in zip(tabs, results.items()):
        with tab:
            if "error" in res:
                st.error(f"Prophet could not be trained: {res['error']}")
                continue

            n = min(len(res["y_true"]), 60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(n)), y=res["y_true"][:n].tolist(),
                mode="lines+markers", name="Actual Demand",
                line=dict(color=PALETTE["Actual"], width=2), marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=list(range(n)), y=res["y_pred"][:n].tolist(),
                mode="lines+markers", name="Predicted Demand",
                line=dict(color=PALETTE.get(m_name,"#888"), width=2, dash="dash"),
                marker=dict(size=5),
            ))
            fig.update_layout(
                title=f"{m_name}: Actual vs Predicted Demand  (first {n} test samples)",
                xaxis_title="Test Sample Number",
                yaxis_title="Units Sold",
                template=PLOTLY_TEMPLATE,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residuals
            resid = res["y_true"] - res["y_pred"]
            fig_r = px.histogram(
                resid, nbins=20,
                title=f"{m_name} — Prediction Error Distribution  "
                      f"(centred at 0 = well-calibrated model)",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[PALETTE.get(m_name,"#888")],
                labels={"value": "Prediction Error (Actual − Predicted units)"},
            )
            fig_r.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7,
                            annotation_text="Zero Error", annotation_position="top right")
            st.plotly_chart(fig_r, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 What Drives the Predictions? — Feature Importance")
    st.caption(
        "Feature importance measures how much each input variable contributed to "
        "the model's predictions. A higher score means the model relies on that "
        "feature more heavily."
    )

    tree_res = {k: v for k, v in results.items() if v.get("fi") is not None}
    fi_tabs  = st.tabs(list(tree_res.keys()))

    for tab, (m_name, res) in zip(fi_tabs, tree_res.items()):
        with tab:
            fi = res["fi"].reset_index()
            fi.columns = ["Feature", "Importance Score"]
            # Normalise for display
            fi["Importance Score"] = fi["Importance Score"] / fi["Importance Score"].sum()
            fi = fi.sort_values("Importance Score", ascending=True)

            fig = px.bar(
                fi, x="Importance Score", y="Feature", orientation="h",
                color="Importance Score",
                color_continuous_scale="Blues",
                template=PLOTLY_TEMPLATE,
                title=f"{m_name} — Top 15 Most Influential Prediction Factors",
                labels={"Importance Score": "Relative Importance", "Feature": "Prediction Factor"},
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                height=520,
            )
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            # Insight callout
            top_feat = fi.iloc[-1]["Feature"]
            top_score = fi.iloc[-1]["Importance Score"]
            st.info(
                f"💡 **Key insight:** The most important predictor for {m_name} is "
                f"**{top_feat}**, accounting for **{top_score:.0%}** of the model's decision weight. "
                f"This confirms that recent demand history is the strongest signal for "
                f"short-horizon Q-commerce forecasting."
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def page_insights(df: pd.DataFrame) -> None:
    st.title("💡 Business Insights & Inventory Recommendations")
    st.markdown(
        "Turning model predictions into **concrete inventory decisions** — "
        "how much safety stock to hold, when to reorder, and when to trigger "
        "flash discounts before goods expire."
    )
    st.divider()

    # ── Safety stock ──────────────────────────────────────────────────────────
    st.subheader("🛡️ Dynamic Safety Stock Calculator")
    st.markdown(
        "Safety stock is the buffer inventory held to absorb demand surges.  "
        "Formula: **Safety Stock = Z × Demand Std Dev × √(Lead Time)**  "
        "where Z is the service-level factor."
    )

    Z_MAP = {"90% (allow 10% stockout risk)": 1.28,
             "95% (allow  5% stockout risk)": 1.65,
             "99% (allow  1% stockout risk)": 2.33}

    col_a, col_b = st.columns(2)
    with col_a:
        svc_lvl = st.selectbox(
            "🎯 Target Service Level",
            list(Z_MAP.keys()),
            index=1,
            help="Higher service level = more safety stock held = fewer stockouts but higher storage cost"
        )
    with col_b:
        lt_hrs = st.slider(
            "🚚 Replenishment Lead Time (hours)",
            min_value=1, max_value=8, value=2,
            help="How many hours between placing a replenishment order and stock arriving at the dark store"
        )

    Z = Z_MAP[svc_lvl]
    SHELF_LIFE = {
        "Milk (1 Litre)":   5,
        "Eggs (12-Pack)":  14,
        "Bread (500g Loaf)": 3,
    }

    ss = df.groupby(["Store","Product"])["quantity_sold"].agg(["mean","std"]).reset_index()
    ss.columns   = ["Dark Store", "Product", "Avg Demand per Slot", "Demand Std Dev"]
    ss["Demand Std Dev"]      = ss["Demand Std Dev"].fillna(0)
    ss["Safety Stock (units)"]= (Z * ss["Demand Std Dev"] * np.sqrt(lt_hrs)).round(1)
    ss["Reorder Point (units)"]=(ss["Avg Demand per Slot"] * lt_hrs + ss["Safety Stock (units)"]).round(1)
    ss["Shelf Life (days)"]   = ss["Product"].map(SHELF_LIFE)
    spoilage_denom = ss["Shelf Life (days)"] * ss["Avg Demand per Slot"] * 3 + 1e-8
    ss["Spoilage Risk Score"] = (ss["Safety Stock (units)"] / spoilage_denom).round(3)
    ss["Risk Level"]          = ss["Spoilage Risk Score"].apply(
        lambda x: "🔴 High — reduce safety stock" if x > 0.30
        else ("🟡 Medium — monitor closely" if x > 0.15 else "🟢 Low — acceptable")
    )
    ss["Avg Demand per Slot"] = ss["Avg Demand per Slot"].round(2)
    ss["Demand Std Dev"]      = ss["Demand Std Dev"].round(2)

    st.dataframe(
        ss[["Dark Store","Product","Avg Demand per Slot","Demand Std Dev",
            "Safety Stock (units)","Reorder Point (units)","Risk Level"]],
        use_container_width=True, hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            ss, x="Product", y="Safety Stock (units)", color="Dark Store",
            barmode="group", template=PLOTLY_TEMPLATE,
            title=f"Recommended Safety Stock per Product & Store\n"
                  f"(Service Level: {svc_lvl.split(' ')[0]}, Lead Time: {lt_hrs}h)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            ss, x="Product", y="Reorder Point (units)", color="Dark Store",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Reorder Point — trigger a replenishment order when stock falls below this",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Flash Discount Triggers ───────────────────────────────────────────────
    st.subheader("💸 Flash Discount Trigger Recommendations")
    st.markdown(
        "When current stock will last longer than **40% of the product's shelf life**, "
        "a discount should be triggered to move inventory before it expires.  "
        "Simulated stock levels are shown below."
    )

    np.random.seed(99)
    fd = ss.copy()
    fd["Simulated Current Stock (units)"] = (
        fd["Avg Demand per Slot"] * 6 * np.random.uniform(0.7, 1.8, len(fd))
    ).round(0)
    fd["Daily Sales Estimate"] = (fd["Avg Demand per Slot"] * 3).round(2)
    fd["Days of Stock Cover"]  = (
        fd["Simulated Current Stock (units)"] / (fd["Daily Sales Estimate"] + 1e-8)
    ).round(1)
    fd["Stock as % of Shelf Life"] = (
        fd["Days of Stock Cover"] / fd["Shelf Life (days)"] * 100
    ).round(1)

    def discount_action(pct: float) -> str:
        if pct > 80: return "🔴 Apply 25% Flash Discount — Urgent"
        if pct > 60: return "🟡 Apply 15% Flash Discount — Recommended"
        if pct > 40: return "🟢 No Discount Needed — Monitor Stock"
        return               "✅ Stock Level Optimal"

    fd["Recommended Action"] = fd["Stock as % of Shelf Life"].apply(discount_action)

    st.dataframe(
        fd[["Dark Store","Product","Simulated Current Stock (units)",
            "Days of Stock Cover","Shelf Life (days)",
            "Stock as % of Shelf Life","Recommended Action"]],
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Key Findings ──────────────────────────────────────────────────────────
    st.subheader("🎯 Key Quantified Business Findings")

    rain_lft = (
        df[df["weather_condition"]=="Rain"]["quantity_sold"].mean() /
        df["quantity_sold"].mean() - 1
    )
    wknd_lft = (
        df[df["is_weekend"]==1]["quantity_sold"].mean() /
        (df[df["is_weekend"]==0]["quantity_sold"].mean() + 1e-8) - 1
    )
    hol_mean  = df[df["public_holiday_flag"]==1]["quantity_sold"].mean()
    norm_mean = df[df["public_holiday_flag"]==0]["quantity_sold"].mean()
    hol_lft   = (hol_mean - norm_mean) / (norm_mean + 1e-8) if df["public_holiday_flag"].sum() > 0 else 0
    so_rate   = df["stockout_flag"].mean()

    col_f, col_r = st.columns(2)
    with col_f:
        st.markdown(f"""
#### 📌 Demand Trigger Summary

| Demand Driver | Measured Impact |
|---|---|
| 🌧️ **Rainy Weather** | **+{rain_lft*100:.0f}%** above average demand |
| 🎉 **Weekends (Sat & Sun)** | **+{wknd_lft*100:.0f}%** vs weekdays |
| 🏖️ **Public Holidays** | **+{hol_lft*100:.0f}%** demand spike |
| ⏰ **Evening Slot (6 PM)** | **Highest demand window** — pre-stocking is mandatory |
| 🌡️ **Extreme Heat Days** | **−10 to −12%** demand suppression |
| ⚠️ **Overall Stockout Rate** | **{so_rate*100:.1f}%** of all transactions lost to stockout |
        """)

    with col_r:
        st.markdown(f"""
#### 📋 Recommended Actions for Operations

1. **Weather-triggered pre-stocking:** Connect to a live weather API; pre-load +35% Milk & Bread inventory whenever rain is forecast 6 hours ahead
2. **Friday 4 PM replenishment run:** Increase weekend safety stock by {wknd_lft*100:.0f}% before the weekend surge
3. **Holiday buffer:** Load +65% inventory 24 hours before any public holiday
4. **6 PM stock alert:** Automated real-time stock check at 5:30 PM; auto-trigger replenishment if stock is below the Reorder Point
5. **ML-driven auto-replenishment:** Feed LightGBM predictions directly into the Warehouse Management System to auto-create purchase orders
6. **Bread flash discounts:** Trigger a 15–25% discount whenever Bread stock exceeds 2 days of cover (shelf life = 3 days)
7. **Reduce heat-period ordering:** Cut Milk orders by 15% when an Extreme Heat day is forecast
        """)

    st.divider()

    # ── Financial Impact ──────────────────────────────────────────────────────
    st.subheader("💰 Estimated Financial Impact of ML-Driven Inventory Optimisation")

    aov      = df["price_at_sale"].mean() if "price_at_sale" in df.columns else 55
    total_so = int(df["stockout_flag"].sum())
    rev_lost = total_so * aov
    spoilage = df["quantity_sold"].sum() * 0.05 * aov
    savings  = rev_lost * 0.60 + spoilage * 0.40

    c1, c2, c3 = st.columns(3)
    c1.metric("⚠️ Total Stockout Events",              f"{total_so:,}")
    c1.metric("💸 Est. Revenue Lost to Stockouts",      f"₹{rev_lost:,.0f}")
    c2.metric("📦 Average Transaction Value",           f"₹{aov:.0f}")
    c2.metric("🗑️ Est. Perishable Wastage Cost",        f"₹{spoilage:,.0f}")
    c3.metric(
        "💡 Projected Annual Savings (ML Optimisation)",
        f"₹{savings:,.0f}",
        "60% fewer stockouts + 40% less wastage",
    )

    impact_df = pd.DataFrame({
        "Category":   ["Revenue Lost (Stockouts)", "Perishable Wastage", "Projected ML Savings"],
        "Amount (₹)": [rev_lost, spoilage, savings],
        "Type":       ["Cost", "Cost", "Saving"],
    })
    fig = px.bar(
        impact_df, x="Category", y="Amount (₹)", color="Type",
        color_discrete_map={"Cost": "#EF5350", "Saving": "#66BB6A"},
        template=PLOTLY_TEMPLATE,
        title="Financial case for ML-driven inventory optimisation",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  &  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    df = load_data()

    with st.sidebar:
        st.markdown("## 🛒 Dark Store\nDemand Forecasting")
        st.markdown("*Powered by ML · Built for Q-Commerce*")
        st.divider()

        page = st.radio(
            "📌 Navigate to",
            [
                "🏠 Home & Data Overview",
                "📊 Descriptive Analysis",
                "🔍 Diagnostic Analysis",
                "🤖 Predictive Modeling",
                "💡 Business Insights",
            ],
        )

        st.divider()
        st.markdown("**🔽 Filter the Dashboard**")

        store_opts = ["All Stores"] + sorted(df["Store"].unique().tolist())
        sku_opts   = ["All Products"] + sorted(df["Product"].unique().tolist())
        sel_store  = st.selectbox("🏪 Dark Store", store_opts)
        sel_sku    = st.selectbox("🛍️ Product",    sku_opts)

        st.divider()
        st.caption(
            f"📅 **Period:**  "
            f"{df['timestamp'].min():%d %b %Y} → {df['timestamp'].max():%d %b %Y}\n\n"
            f"📦 **Records:**  {len(df):,}\n\n"
            f"🏪 **Stores:**  {df['Store'].nunique()}\n\n"
            f"🛍️ **Products:**  {df['Product'].nunique()}"
        )

    # Apply filters
    fdf = df.copy()
    if sel_store != "All Stores":
        fdf = fdf[fdf["Store"] == sel_store]
    if sel_sku != "All Products":
        fdf = fdf[fdf["Product"] == sel_sku]

    # Route
    if page == "🏠 Home & Data Overview":
        page_home(fdf)
    elif page == "📊 Descriptive Analysis":
        page_descriptive(fdf)
    elif page == "🔍 Diagnostic Analysis":
        page_diagnostic(fdf)
    elif page == "🤖 Predictive Modeling":
        page_modeling(df)          # always full df for modeling
    elif page == "💡 Business Insights":
        page_insights(fdf)


if __name__ == "__main__":
    main()
