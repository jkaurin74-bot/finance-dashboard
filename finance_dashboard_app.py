
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Financijski Dashboard", layout="wide")

# -------------------------------
# Sidebar - Učitavanje podataka
# -------------------------------
st.sidebar.header("Postavke podataka")
excel_file = st.sidebar.file_uploader("Učitaj Excel datoteku", type=["xlsx", "xls"])

# -------------------------------
# Helper funkcije
# -------------------------------
def pick_col(cols, *candidates):
    low = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(low):
            if cand in c:
                return cols[i]
    return None

def to_num(s):
    if s is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str)
         .str.replace("\u00a0", "", regex=False)  # non-breaking space
         .str.replace(".", "", regex=False)       # tisućice
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False)      # decimalni zarez
         .str.replace("€", "", regex=False)
         .str.replace("eur", "", case=False, regex=True),
        errors="coerce"
    )

def coerce_amount(df: pd.DataFrame, amount_col=None, prihod_col=None, rashod_col=None) -> pd.Series:
    if amount_col and amount_col in df.columns:
        return to_num(df[amount_col])
    if (prihod_col and prihod_col in df.columns) or (rashod_col and rashod_col in df.columns):
        prihod = to_num(df[prihod_col]) if (prihod_col and prihod_col in df.columns) else 0.0
        rashod = to_num(df[rashod_col]) if (rashod_col and rashod_col in df.columns) else 0.0
        return (pd.Series(prihod).fillna(0) - pd.Series(rashod).fillna(0))
    # fallback: last numeric-like column
    numeric_like = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_like:
        return to_num(df[numeric_like[-1]])
    return pd.Series([np.nan]*len(df))

def parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    s = pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if s.isna().mean() > 0.6:
        s = pd.to_datetime(series, errors="coerce", dayfirst=False, infer_datetime_format=True)
    return s

# -------------------------------
# Učitavanje workbooka
# -------------------------------
if excel_file is None:
    st.info("📭 Učitaj Excel datoteku u sidebaru.")
    st.stop()

try:
    xl = pd.ExcelFile(excel_file)
except Exception as e:
    st.error(f"Greška pri otvaranju Excela: {e}")
    st.stop()

sheet = st.sidebar.selectbox("Odaberi list (sheet)", xl.sheet_names, index=0)
raw = xl.parse(sheet)
if raw.empty:
    st.warning("Odabrani list je prazan.")
    st.stop()

st.expander("🔎 Pregled sirovih podataka", expanded=False).dataframe(raw.head(50), use_container_width=True)

# -------------------------------
# Mapiranje stupaca (ručno + auto prijedlozi)
# -------------------------------
cols = list(raw.columns.astype(str))
# Auto-prijedlozi
guess_date = pick_col(cols, "datum", "date", "value date", "posting date", "book date")
guess_amount = pick_col(cols, "amount", "iznos", "iznos (eur)", "iznos (kn)", "value", "total", "sum")
guess_prihod = pick_col(cols, "prihod", "income", "credit")
guess_rashod = pick_col(cols, "rashod", "trošak", "trosak", "expense", "debit")
guess_cat = pick_col(cols, "kategorija", "category", "vrsta", "tag")
guess_desc = pick_col(cols, "opis", "description", "naziv", "merchant", "detalj", "detalji", "counterparty", "partner")

st.sidebar.markdown("### Mapiranje stupaca")
date_col = st.sidebar.selectbox("Datum (obavezno)", ["— odaberi —"] + cols, index=(cols.index(guess_date)+1) if guess_date in cols else 0)
amount_col = st.sidebar.selectbox("Iznos (ako nema Prihod/Rashod)", ["— odaberi —"] + cols, index=(cols.index(guess_amount)+1) if guess_amount in cols else 0)
prihod_col = st.sidebar.selectbox("Prihod (opcionalno)", ["— odaberi —"] + cols, index=(cols.index(guess_prihod)+1) if guess_prihod in cols else 0)
rashod_col = st.sidebar.selectbox("Rashod (opcionalno)", ["— odaberi —"] + cols, index=(cols.index(guess_rashod)+1) if guess_rashod in cols else 0)
cat_col = st.sidebar.selectbox("Kategorija (opcionalno)", ["— odaberi —"] + cols, index=(cols.index(guess_cat)+1) if guess_cat in cols else 0)
desc_col = st.sidebar.selectbox("Opis (opcionalno)", ["— odaberi —"] + cols, index=(cols.index(guess_desc)+1) if guess_desc in cols else 0)
use_sheet_as_category = st.sidebar.checkbox("Ako nema kategorije, koristi ime sheeta kao kategoriju", value=True)

# Normalize selections
def norm_choice(x):
    return None if (x is None or x == "— odaberi —") else x

date_col = norm_choice(date_col)
amount_col = norm_choice(amount_col)
prihod_col = norm_choice(prihod_col)
rashod_col = norm_choice(rashod_col)
cat_col = norm_choice(cat_col)
desc_col = norm_choice(desc_col)

if not date_col or date_col not in raw.columns:
    st.error("❗ Molim odaberi stupac s datumom u sidebaru (polje 'Datum').")
    st.stop()

# -------------------------------
# Čišćenje i standardizacija
# -------------------------------
df = raw.copy()
df["Date"] = parse_dates(df[date_col])
df["Amount"] = coerce_amount(df, amount_col=amount_col, prihod_col=prihod_col, rashod_col=rashod_col)

if cat_col and cat_col in df.columns:
    df["Category"] = df[cat_col].astype(str)
else:
    df["Category"] = sheet if use_sheet_as_category else ""

if desc_col and desc_col in df.columns:
    df["Description"] = df[desc_col].astype(str)
else:
    df["Description"] = ""

before = len(df)
df = df.dropna(subset=["Date"])
dropped = before - len(df)

df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
df["Type"] = np.where(df["Amount"] >= 0, "Prihod", "Rashod")

parsed_preview = df.head(50)[["Date", "Category", "Description", "Amount", "Type"]]
info = st.container()
with info:
    st.success(f"✅ Učitano {len(df):,} redaka (odbačeno bez datuma: {dropped:,}).")
    st.dataframe(parsed_preview, use_container_width=True, hide_index=True)

# Ako nakon mapiranja i dalje prazno:
if df.empty:
    st.warning("Nakon mapiranja stupaca nema valjanih redaka (provjeri 'Datum' i 'Iznos' / 'Prihod'/'Rashod').")
    st.stop()

# -------------------------------
# KPI pokazatelji
# -------------------------------
total_income = df.loc[df["Amount"] > 0, "Amount"].sum()
total_expense = -df.loc[df["Amount"] < 0, "Amount"].sum()
net = df["Amount"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Ukupni prihodi", f"{total_income:,.2f} €")
col2.metric("Ukupni rashodi", f"{total_expense:,.2f} €")
col3.metric("Neto", f"{net:,.2f} €")

# -------------------------------
# Filtri
# -------------------------------
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
    (df["Date"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)) &
    (df["Type"].isin(type_filter)) &
    (df["Category"].astype(str).isin(cat_filter))
)
fdf = df.loc[mask]

# -------------------------------
# Grafovi
# -------------------------------
st.subheader("Trend neto iznosa po mjesecima")
monthly = fdf.groupby("Month", as_index=False)["Amount"].sum()
if not monthly.empty:
    fig_line = px.line(monthly, x="Month", y="Amount", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("Nema podataka za odabrane filtre.")

colA, colB = st.columns(2)

with colA:
    st.subheader("Top kategorije")
    cat_sum = fdf.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False).head(15)
    if not cat_sum.empty:
        fig_bar = px.bar(cat_sum, x="Category", y="Amount")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Nema podataka.")

with colB:
    st.subheader("Udio rashoda po kategorijama")
    cat_share = fdf[fdf["Amount"] < 0]
    if not cat_share.empty:
        pie = cat_share.groupby("Category", as_index=False)["Amount"].sum()
        pie["Abs"] = pie["Amount"].abs()
        fig_pie = px.pie(pie, names="Category", values="Abs")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Nema rashoda za odabrane filtre.")

# -------------------------------
# Tablica
# -------------------------------
st.subheader("Transakcije")
st.dataframe(
    fdf.sort_values("Date", ascending=False)[["Date", "Category", "Description", "Amount", "__sheet__"]]
       .rename(columns={"__sheet__": "Sheet"}),
    use_container_width=True,
    hide_index=True
)

# Download
st.download_button("⬇️ Preuzmi filtrirane podatke (CSV)", fdf.to_csv(index=False).encode("utf-8"),
                   file_name="transakcije_filtrirano.csv", mime="text/csv")
