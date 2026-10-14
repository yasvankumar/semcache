"""Streamlit dashboard.  Run: streamlit run dashboard/app.py"""
import sqlite3, sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import load_config

st.set_page_config(page_title="Semantic cache + router", layout="wide")
st.title("Semantic caching & cost-aware routing")
cfg = load_config()
con = sqlite3.connect(cfg.db_path)
df = pd.read_sql("SELECT * FROM requests ORDER BY id", con)
if df.empty:
    st.info("No requests logged yet. Run `python -m scripts.benchmark` or call POST /ask."); st.stop()

modes = st.multiselect("Modes", sorted(df["mode"].unique()), default=sorted(df["mode"].unique()))
df = df[df["mode"].isin(modes)].copy()
df["n"] = df.groupby("mode").cumcount()

st.subheader("Headline numbers")
agg = df.groupby("mode").agg(requests=("id", "count"), cost_per_request=("cost", "mean"), total_cost=("cost", "sum"),
                             hit_rate=("cache_hit", "mean"), escalation_rate=("escalated", "mean"),
                             p50_ms=("latency_ms", lambda s: s.quantile(.5)), p95_ms=("latency_ms", lambda s: s.quantile(.95)),
                             quality=("quality", "mean"))
st.dataframe(agg.style.format({"cost_per_request": "${:.5f}", "total_cost": "${:.4f}", "hit_rate": "{:.1%}",
                               "escalation_rate": "{:.1%}", "p50_ms": "{:.1f}", "p95_ms": "{:.1f}", "quality": "{:.3f}"}))
if {"baseline", "optimized"} <= set(agg.index):
    st.metric("Cost reduction vs baseline", f"{1 - agg.loc['optimized','total_cost'] / agg.loc['baseline','total_cost']:.1%}")

w = st.slider("Rolling window (requests)", 5, 100, 20)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Cost per request (rolling)")
    p = df.pivot_table(index="n", columns="mode", values="cost").rolling(w, min_periods=1).mean()
    st.line_chart(p)
    st.subheader("Cache hit rate (rolling)")
    o = df[df["mode"] == "optimized"].set_index("n")["cache_hit"].rolling(w, min_periods=1).mean()
    st.line_chart(o)
with c2:
    st.subheader("Route split (optimized)")
    st.bar_chart(df[df["mode"] == "optimized"]["route"].value_counts())
    st.subheader("Latency p50 / p95 (rolling, ms)")
    for m in modes:
        s = df[df["mode"] == m].set_index("n")["latency_ms"]
        st.line_chart(pd.DataFrame({f"{m} p50": s.rolling(w, min_periods=1).quantile(.5),
                                    f"{m} p95": s.rolling(w, min_periods=1).quantile(.95)}))
st.subheader("Quality score per route over time")
q = df.dropna(subset=["quality"]).pivot_table(index=df["n"] // w, columns="route", values="quality")
st.line_chart(q)
st.subheader("Recent requests")
st.dataframe(df.tail(50)[["mode", "query", "route", "model", "cache_hit", "escalated", "cost", "latency_ms", "quality", "similarity"]])
