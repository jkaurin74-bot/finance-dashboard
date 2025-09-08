import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Financijski Dashboard", layout="wide")

st.sidebar.header("Postavke podataka")
excel_file = st.sidebar.file_uploader("Učitaj Excel datoteku", type=["xlsx", "xls"])

def pick_col(cols, *candidates):
    for cand in candidates:
        for c in cols:
            if cand in c.lower():
                return c
    return None

def coerce_amount(df: pd.DataFrame) -> pd.Series:
    cols = df.columns
    amount_col = pick_col(cols, "amount", "iznos")
    prihod_col = pick_col(cols, "prihod", "income")
    rashod_col = pick_col(cols, "rashod", "trošak", "trosak", "expense")
    def to_num(s):
        return pd.to_numeric(
            s.astype(str)
             .str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False)
             .str.replace("€", "", regex=False)
             .str.replace("eur", "", case=False, regex=True),
            errors="coerce"
        )
    if amount_col:
        return to_num(df[amount_col])
    if prihod_col or rashod_col:
        prihod = to_num(df[prihod_col]) if prihod_col else 0
        rashod = to_num(df[rashod_col]) if rashod_col else 0
        return prihod.fillna(0) - rashod.fillna(0)
    return pd.Series([np.nan]*len(df))

def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)

def load_workbook(uploaded_file) -> pd.DataFrame:
    xl = pd.ExcelFile(uploaded_file)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df["__sheet__"] = sheet
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    cols = df.columns
    date_col = pick_col(cols, "datum", "date")
    cat_col = pick_col(cols, "kategorija", "category") or "__sheet__"
    desc_col = pick_col(cols, "opis", "description")
    df["Date"] = parse_dates(df[date_col]) if date_col else pd.NaT
    df["Category"] = df[cat_col].astype(str) if cat_col in cols else ""
    df["Description"] = df[desc_col].astype(str) if desc_col in cols else ""
    df["Amount"] = coerce_amount(df)
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["Type"] = np.where(df["Amount"] >= 0, "Prihod", "Rashod")
    return df[["Date", "Month", "Category", "Description", "Amount", "Type", "__sheet__"]]

if excel_file is None:
    st.info("📭 Učitaj Excel datoteku u sidebaru.")
    st.stop()

df = load_workbook(excel_file)
if df.empty:
    st.info("📭 Nema podataka za prikaz.")
    st.stop()

total_income = df.loc[df["Amount"] > 0, "Amount"].sum()
total_expense = -df.loc[df["Amount"] < 0, "Amount"].sum()
net = df["Amount"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Ukupni prihodi", f"{total_income:,.2f} €")
col2.metric("Ukupni rashodi", f"{total_expense:,.2f} €")
col3.metric("Neto", f"{net:,.2f} €")

st.sidebar.header("Filtri")
min_date = pd.to_datetime(df["Date"].min())
max_date = pd.to_datetime(df["Date"].max())
if pd.isna(min_date):
    min_date = pd.Timestamp.today().normalize()
if pd.isna(max_date) or max_date < min_date:
    max_date = min_date

date_range = st.sidebar.date_input("Raspon datuma", (min_date.date(), max_date.date()))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

type_filter = st.sidebar.multiselect("Tip transakcije", ["Prihod", "Rashod"], default=["Prihod", "Rashod"])
categories = sorted(df["Category"].dropna().astype(str).unique().tolist())
cat_filter = st.sidebar.multiselect("Kategorije", categories, default=categories)

mask = (
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)) &
    (df["Type"].isin(type_filter)) &
    (df["Category"].astype(str).isin(cat_filter))
)
fdf = df.loc[mask]

st.subheader("Trend neto iznosa po mjesecima")
monthly = fdf.groupby("Month", as_index=False)["Amount"].sum()
if not monthly.empty:
    fig_line = px.line(monthly, x="Month", y="Amount", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

colA, colB = st.columns(2)

with colA:
    st.subheader("Top kategorije")
    cat_sum = fdf.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False).head(10)
    if not cat_sum.empty:
        fig_bar = px.bar(cat_sum, x="Category", y="Amount")
        st.plotly_chart(fig_bar, use_container_width=True)

with colB:
    st.subheader("Udio rashoda po kategorijama")
    cat_share = fdf[fdf["Amount"] < 0]
    if not cat_share.empty:
        pie = cat_share.groupby("Category", as_index=False)["Amount"].sum()
        pie["Abs"] = pie["Amount"].abs()
        fig_pie = px.pie(pie, names="Category", values="Abs")
        st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Transakcije")
st.dataframe(
    fdf.sort_values("Date", ascending=False)[["Date", "Category", "Description", "Amount", "__sheet__"]]
       .rename(columns={"__sheet__": "Sheet"}),
    use_container_width=True,
    hide_index=True
)
