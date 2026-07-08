"""
Sales Forecasting & Demand Intelligence Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="ISO-8859-1")
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')
    if df['Order Date'].isna().mean() > 0.3:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    return df


@st.cache_data
def monthly_series(df, category=None, region=None):
    sub = df.copy()
    if category and category != "All":
        sub = sub[sub['Category'] == category]
    if region and region != "All":
        sub = sub[sub['Region'] == region]
    ts = sub.set_index('Order Date').resample('MS')['Sales'].sum()
    return ts


@st.cache_data
def run_sarima_forecast(_ts, steps=3):
    ts = _ts
    train = ts.iloc[:-steps] if len(ts) > steps + 12 else ts
    seasonal_order = (1, 1, 1, 12) if len(train) >= 24 else (0, 0, 0, 0)
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc_obj = model.get_forecast(steps=steps)
    forecast = fc_obj.predicted_mean
    ci = fc_obj.conf_int()

    mae = rmse = None
    if len(ts) > steps + 12:
        test = ts.iloc[-steps:]
        mae = mean_absolute_error(test, forecast)
        rmse = mean_squared_error(test, forecast) ** 0.5
    return forecast, ci, mae, rmse


df = load_data()

st.title("📊 Sales Forecasting & Demand Intelligence")

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ---------------------------------------------------------------------------
# PAGE 1 — Sales Overview
# ---------------------------------------------------------------------------
if page == "Sales Overview":
    st.header("Sales Overview")

    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby('Year')['Sales'].sum().reset_index()
        fig = px.bar(yearly, x='Year', y='Sales', title="Total Sales by Year")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly = df.set_index('Order Date').resample('MS')['Sales'].sum().reset_index()
        fig = px.line(monthly, x='Order Date', y='Sales', title="Monthly Sales Trend")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filter by Region & Category")
    c1, c2 = st.columns(2)
    with c1:
        region_filter = st.multiselect("Region", options=sorted(df['Region'].unique()),
                                        default=sorted(df['Region'].unique()))
    with c2:
        category_filter = st.multiselect("Category", options=sorted(df['Category'].unique()),
                                          default=sorted(df['Category'].unique()))

    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(category_filter)]
    grouped = filtered.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    fig = px.bar(grouped, x='Region', y='Sales', color='Category', barmode='group',
                 title="Sales by Region & Category")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.header("Forecast Explorer")

    dim_type = st.selectbox("Forecast by", ["Category", "Region"])
    if dim_type == "Category":
        options = ["All"] + sorted(df['Category'].unique().tolist())
        selected = st.selectbox("Select Category", options)
        ts = monthly_series(df, category=selected)
    else:
        options = ["All"] + sorted(df['Region'].unique().tolist())
        selected = st.selectbox("Select Region", options)
        ts = monthly_series(df, region=selected)

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    if len(ts) < 6:
        st.warning("Not enough history for this selection to forecast reliably.")
    else:
        forecast, ci, mae, rmse = run_sarima_forecast(ts, steps=horizon)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Historical", mode="lines"))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                                  mode="lines+markers", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
            fill='toself', fillcolor='rgba(99,110,250,0.2)', line=dict(color='rgba(255,255,255,0)'),
            name="Confidence Interval"
        ))
        fig.update_layout(title=f"{horizon}-Month Forecast: {dim_type} = {selected}")
        st.plotly_chart(fig, use_container_width=True)

        if mae is not None:
            c1, c2 = st.columns(2)
            c1.metric("MAE", f"{mae:,.2f}")
            c2.metric("RMSE", f"{rmse:,.2f}")
        else:
            st.info("Not enough held-out history to compute MAE/RMSE for this segment; forecast uses the full series.")

# ---------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.header("Anomaly Report")

    weekly = df.set_index('Order Date').resample('W-MON')['Sales'].sum().reset_index()
    weekly.columns = ['Week', 'Sales']

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly['iso_anomaly'] = iso.fit_predict(weekly[['Sales']])

    weekly['rolling_mean'] = weekly['Sales'].rolling(8, min_periods=4).mean()
    weekly['rolling_std'] = weekly['Sales'].rolling(8, min_periods=4).std()
    weekly['zscore'] = (weekly['Sales'] - weekly['rolling_mean']) / weekly['rolling_std']
    weekly['z_anomaly'] = weekly['zscore'].abs() > 2

    anomalies = weekly[(weekly['iso_anomaly'] == -1) | (weekly['z_anomaly'])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly['Week'], y=weekly['Sales'], name="Weekly Sales", mode="lines"))
    fig.add_trace(go.Scatter(x=anomalies['Week'], y=anomalies['Sales'], name="Anomaly",
                              mode="markers", marker=dict(color="red", size=10)))
    fig.update_layout(title="Weekly Sales with Detected Anomalies")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomaly Dates")
    st.dataframe(
        anomalies[['Week', 'Sales', 'iso_anomaly', 'z_anomaly']]
        .rename(columns={'iso_anomaly': 'Isolation Forest Flag', 'z_anomaly': 'Z-Score Flag'})
        .assign(**{'Isolation Forest Flag': lambda d: d['Isolation Forest Flag'] == -1})
    )

# ---------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.header("Product Demand Segments")

    sub_df = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().reset_index()

    def cagr(series):
        series = series.sort_index()
        if len(series) < 2 or series.iloc[0] == 0:
            return 0
        years = series.index.max() - series.index.min()
        if years == 0:
            return 0
        return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100

    rows = []
    for name, g in sub_df.groupby('Sub-Category'):
        yearly = g.set_index('Year')['Sales']
        total_volume = yearly.sum()
        growth_rate = cagr(yearly)
        volatility = df[df['Sub-Category'] == name].set_index('Order Date').resample('MS')['Sales'].sum().std()
        avg_order_value = df[df['Sub-Category'] == name]['Sales'].mean()
        rows.append([name, total_volume, growth_rate, volatility, avg_order_value])

    feat_df = pd.DataFrame(
        rows, columns=['Sub-Category', 'TotalVolume', 'GrowthRate', 'Volatility', 'AvgOrderValue']
    ).fillna(0)

    X_scaled = StandardScaler().fit_transform(
        feat_df[['TotalVolume', 'GrowthRate', 'Volatility', 'AvgOrderValue']]
    )

    k = st.slider("Number of clusters (k)", 2, 6, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    feat_df['Cluster'] = kmeans.fit_predict(X_scaled)

    profile = feat_df.groupby('Cluster')[['TotalVolume', 'GrowthRate', 'Volatility', 'AvgOrderValue']].mean()

    def label_cluster(row):
        vol_high = row['TotalVolume'] > profile['TotalVolume'].median()
        growth_high = row['GrowthRate'] > profile['GrowthRate'].median()
        volatile = row['Volatility'] > profile['Volatility'].median()
        if vol_high and not volatile:
            return 'High Volume, Stable Demand'
        if not vol_high and volatile:
            return 'Low Volume, High Volatility'
        if growth_high:
            return 'Growing Demand'
        return 'Declining Demand'

    cluster_labels = {c: label_cluster(profile.loc[c]) for c in profile.index}
    feat_df['ClusterLabel'] = feat_df['Cluster'].map(cluster_labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    feat_df['PC1'], feat_df['PC2'] = coords[:, 0], coords[:, 1]

    fig = px.scatter(feat_df, x='PC1', y='PC2', color='ClusterLabel', text='Sub-Category',
                      title="Product Sub-Category Clusters (PCA-reduced)")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category → Cluster Assignment")
    st.dataframe(feat_df[['Sub-Category', 'ClusterLabel', 'TotalVolume', 'GrowthRate', 'Volatility']])
