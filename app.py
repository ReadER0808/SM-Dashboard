# Sales & Marketing Dashboard
# By: Aniket Varbude

import re
from typing import Optional, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ================================
# ---------- Utilities -----------
# ================================

def coalesce_cols(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column name from candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None

def _to_int_safe(x: object) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"(\d{3,4})", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def map_postcode_to_state(pc: object) -> Optional[str]:
    """AU postcode → state (approx ranges)."""
    n = _to_int_safe(pc)
    if n is None:
        return None
    if 200 <= n <= 299 or 2600 <= n <= 2618 or 2900 <= n <= 2920: return "ACT"
    if (1000 <= n <= 1999) or (2000 <= n <= 2599) or (2619 <= n <= 2899): return "NSW"
    if (3000 <= n <= 3999) or (8000 <= n <= 8999): return "VIC"
    if 4000 <= n <= 4999: return "QLD"
    if 5000 <= n <= 5799: return "SA"
    if 6000 <= n <= 6999: return "WA"
    if 7000 <= n <= 7999: return "TAS"
    if 800 <= n <= 999: return "NT"
    return None

def normalize_postcode_state(df: pd.DataFrame,
                             postcode_candidates: List[str],
                             state_candidates: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Ensure a clean uppercase 'State' where possible; infer from postcode if needed."""
    pc_col = coalesce_cols(df, postcode_candidates)
    st_col = coalesce_cols(df, state_candidates)
    if pc_col and (not st_col or df[st_col].isna().all()):
        df["__State"] = df[pc_col].apply(map_postcode_to_state)
        st_col = "__State"
    elif st_col:
        df[st_col] = df[st_col].astype(str).str.upper().str.strip()
        df[st_col] = df[st_col].replace({"N/A": None, "": None, "NONE": None})
    return pc_col, st_col

def coerce_spv_column(df: pd.DataFrame, candidates=None, out_col="SPV") -> pd.DataFrame:
    """Find & parse SPV-like column into numeric df['SPV'] (prefers excl. GST)."""
    if candidates is None:
        candidates = [
            # prefer excl. GST first
            "SPV (excl. GST)", "SPV excl. GST", "SPV excl GST", "SPV ex GST", "SPV (ex GST)",
            # fallbacks
            "SPV (inc. GST)", "SPV inc GST", "SPV (Inc GST)",
            "SPV", "Sales Purchase Value", "Sale Value", "Contract Value",
            "Deal Value", "Total Value", "Total Amount", "Amount", "Value"
        ]
    src = coalesce_cols(df, candidates)
    if not src:
        df[out_col] = 0.0
        return df

    raw = df[src].astype(str).str.strip()
    # negatives like "(1,234.56)"
    neg_mask = raw.str.match(r"^\(.*\)$", na=False)
    raw = raw.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    # keep digits . , - ; drop everything else (AUD, $, spaces, letters)
    cleaned = raw.str.replace(r"[^0-9\.,\-]", "", regex=True)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.replace({"": np.nan, "-": np.nan, ".": np.nan})
    num = pd.to_numeric(cleaned, errors="coerce")
    num = pd.Series(np.where(neg_mask, -num, num), index=df.index, dtype="float64")
    df[out_col] = num.fillna(0.0)
    return df

# --- name unification helpers (first-name → full name when unambiguous) ---
def _normalize_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower().replace(".", " ")
    s = re.sub(r"[^a-z\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _titlecase_name(s: str) -> str:
    return " ".join(p.capitalize() for p in s.split())

def build_fullname_index(all_names: pd.Series) -> dict:
    idx = defaultdict(set)
    for raw in all_names.dropna().astype(str):
        norm = _normalize_name(raw)
        parts = norm.split()
        if len(parts) >= 2:
            idx[parts[0]].add(_titlecase_name(norm))
    return idx

def canonicalize_series(series: pd.Series, fullname_index: dict, manual_aliases: dict | None = None) -> pd.Series:
    aliases = { _normalize_name(k): v for k, v in (manual_aliases or {}).items() }
    out = []
    for raw in series.fillna("").astype(str):
        norm = _normalize_name(raw)
        if not norm:
            out.append(np.nan); continue
        if norm in aliases:
            out.append(aliases[norm]); continue
        parts = norm.split()
        if len(parts) >= 2:
            out.append(_titlecase_name(norm)); continue
        first = parts[0]
        cands = sorted(fullname_index.get(first, []))
        out.append(cands[0] if len(cands) == 1 else first.capitalize())
    return pd.Series(out, index=series.index)

# ================================
# ---------- Cleaning ------------
# ================================

@st.cache_data(show_spinner=False)
def clean_leads(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    person = coalesce_cols(df, ["assigned_to","Assigned To","Owner","Consultant","Sales Agent"])
    _, state_col = normalize_postcode_state(df, ["postal_code","Postcode","postcode","Zip","Mailing Zip"],
                                               ["state","State","Mailing State"])
    if person and person != "Assigned To":
        df = df.rename(columns={person: "Assigned To"})
    if "Assigned To" not in df.columns:
        df["Assigned To"] = pd.Series(pd.NA, dtype="string")
    df["Assigned To"] = df["Assigned To"].astype("string").str.strip()
    if state_col and state_col != "State":
        df = df.rename(columns={state_col: "State"})
    return df

@st.cache_data(show_spinner=False)
def clean_contacts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    person = coalesce_cols(df, ["Assigned To","assigned_to","Owner","Consultant","Sales Agent"])
    _, state_col = normalize_postcode_state(df, ["Mailing Zip","Postal Code","Postcode","Zip"],
                                               ["Mailing State","State"])
    if person and person != "Assigned To":
        df = df.rename(columns={person: "Assigned To"})
    if "Assigned To" not in df.columns:
        df["Assigned To"] = pd.Series(pd.NA, dtype="string")
    df["Assigned To"] = df["Assigned To"].astype("string").str.strip()
    if state_col and state_col != "State":
        df = df.rename(columns={state_col: "State"})
    return df

@st.cache_data(show_spinner=False)
def clean_pa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    person = coalesce_cols(df, ["Sales Agent","Assigned To","Consultant","Owner","Sales TEAM"])
    _, state_col = normalize_postcode_state(df, ["Postcode","Zip","Mailing Zip"],
                                               ["State"])
    df = coerce_spv_column(df)  # ensures numeric df["SPV"]
    df["SPV"] = pd.to_numeric(df["SPV"], errors="coerce").fillna(0.0).astype(float)
    if person and person != "Sales Agent":
        df = df.rename(columns={person: "Sales Agent"})
    if "Sales Agent" not in df.columns:
        df["Sales Agent"] = pd.Series(pd.NA, dtype="string")
    df["Sales Agent"] = df["Sales Agent"].astype("string").str.strip()
    if state_col and state_col != "State":
        df = df.rename(columns={state_col: "State"})
    return df

# ================================
# ---------- Validations ---------
# ================================

def file_checks(df: pd.DataFrame, kind: str) -> tuple[bool, list[str]]:
    """Return (ok, messages) per file type."""
    msgs = []

    def have(label: str, candidates: list[str], required: bool = True) -> bool:
        col = coalesce_cols(df, candidates)
        if col:
            msgs.append(f"✅ **{label}**: found `{col}`")
            return True
        if required:
            msgs.append(f"❌ **{label}**: missing. Expected one of: `{', '.join(candidates)}`")
            return False
        msgs.append(f"⚠️ **{label}**: not found (optional)")
        return True

    if kind == "leads":
        ok = True
        ok &= have("Assigned To / Owner", ["assigned_to", "Assigned To", "Owner", "Consultant", "Sales Agent"], True)
        ok &= have("State or Postcode", ["State","Mailing State","postal_code","Postcode","postcode","Zip","Mailing Zip"], False)
        return ok, msgs
    if kind == "contacts":
        ok = True
        ok &= have("Assigned To / Owner", ["Assigned To","assigned_to","Owner","Consultant","Sales Agent"], True)
        ok &= have("State or Postcode", ["Mailing State","State","Mailing Zip","Postal Code","Postcode","Zip"], False)
        return ok, msgs
    if kind == "pa":
        ok = True
        ok &= have("Sales Agent", ["Sales Agent","Assigned To","Consultant","Owner","Sales TEAM"], True)
        ok &= have("SPV / Value", [
            "SPV (excl. GST)", "SPV excl. GST", "SPV excl GST", "SPV ex GST", "SPV (ex GST)",
            "SPV (inc. GST)", "SPV inc GST", "SPV (Inc GST)",
            "SPV", "Sales Purchase Value", "Sale Value", "Contract Value",
            "Deal Value", "Total Value", "Total Amount", "Amount", "Value"
        ], True)
        ok &= have("State or Postcode", ["State","Postcode","Zip","Mailing Zip"], False)
        return ok, msgs
    return True, ["(unknown file type)"]

# ================================
# ---------- Metrics -------------
# ================================

def stage_counts(df: pd.DataFrame, person_col: str) -> pd.DataFrame:
    """Group counts by person; returns empty-typed frame if missing."""
    if df is None or df.empty or person_col not in df.columns:
        return pd.DataFrame({person_col: pd.Series(dtype="string"), "Count": pd.Series(dtype=int)})
    out = (
        df[df[person_col].notna()]
        .groupby(person_col, dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    out[person_col] = out[person_col].astype("string")
    return out

from typing import Tuple

def kpis(leads: pd.DataFrame, contacts: pd.DataFrame, pa: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # --- Normalize join keys to StringDtype (prevents object/float64 merge issues) ---
    for df, col in ((leads, "Assigned To"), (contacts, "Assigned To"), (pa, "Sales Agent")):
        if col in df.columns:
            df[col] = df[col].astype("string")
        else:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="string")

    # --- Per-consultant counts ---
    leads_ct    = stage_counts(leads,   "Assigned To").rename(columns={"Count": "Leads Allocated"})
    contacts_ct = stage_counts(contacts, "Assigned To").rename(columns={"Count": "Contact Conversions"})
    pa_ct       = stage_counts(pa,      "Sales Agent").rename(columns={"Count": "PA Conversions"})

    # --- SPV totals per salesperson (robust, never a scalar) ---
    if "SPV" in pa.columns:
        pa["SPV"] = pd.to_numeric(pa["SPV"], errors="coerce").fillna(0.0).astype(float)
    else:
        pa["SPV"] = pd.Series(0.0, index=pa.index, dtype=float)

    # Ensure join key exists and is string-typed
    if "Sales Agent" not in pa.columns:
        pa["Sales Agent"] = pd.Series(pd.NA, index=pa.index, dtype="string")
    else:
        pa["Sales Agent"] = pa["Sales Agent"].astype("string")

    spv_totals = (
        pa.loc[pa["Sales Agent"].notna() & (pa["Sales Agent"] != "")]
          .groupby("Sales Agent", dropna=False)["SPV"]
          .sum()
          .reset_index()
          .rename(columns={"SPV": "Total SPV"})
    )
    spv_totals["Sales Agent"] = spv_totals["Sales Agent"].astype("string")
    spv_totals["Total SPV"]   = pd.to_numeric(spv_totals["Total SPV"], errors="coerce").fillna(0.0)

    # --- Merge counts into summary ---
    summary = (
        leads_ct
        .merge(contacts_ct, how="outer", on="Assigned To")
        .merge(pa_ct,      how="outer", left_on="Assigned To", right_on="Sales Agent")
    )

    # Canonical salesperson name
    summary["Sales Agent"] = summary["Assigned To"].combine_first(summary.get("Sales Agent"))
    if "Assigned To" in summary.columns:
        summary = summary.drop(columns=["Assigned To"])
    summary["Sales Agent"] = summary["Sales Agent"].astype("string")

    # Fill numeric holes
    for c in ["Leads Allocated", "Contact Conversions", "PA Conversions"]:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce").fillna(0).astype(int)

    # Merge SPV totals (single time)
    summary = summary.merge(spv_totals, how="left", on="Sales Agent")
    if "Total SPV" in summary.columns:
        summary["Total SPV"] = pd.to_numeric(summary["Total SPV"], errors="coerce").fillna(0.0)

    # --- Highlights ---
    highlights = {}
    if not leads_ct.empty:
        highlights["Top by Leads"] = tuple(leads_ct.iloc[0])
    if not contacts_ct.empty:
        highlights["Top by Contacts"] = tuple(contacts_ct.iloc[0])
    if not pa_ct.empty:
        highlights["Top by PAs"] = tuple(pa_ct.iloc[0])
    if not spv_totals.empty and spv_totals["Total SPV"].sum() > 0:
        top_row = spv_totals.sort_values("Total SPV", ascending=False).iloc[0]
        highlights["Top by SPV"] = (top_row["Sales Agent"], float(top_row["Total SPV"]))

    # --- States (optional) ---
    top_state_by_leads = None
    if "State" in leads.columns and not leads["State"].dropna().empty:
        ts = leads["State"].value_counts().reset_index()
        ts.columns = ["State", "Lead Count"]
        top_state_by_leads = tuple(ts.iloc[0]) if not ts.empty else None

    top_state_by_sales = None
    if "State" in pa.columns and not pa["State"].dropna().empty:
        ps = pa["State"].value_counts().reset_index()
        ps.columns = ["State", "PA Count"]
        top_state_by_sales = tuple(ps.iloc[0]) if not ps.empty else None

    highlights["Top State by Leads"] = top_state_by_leads
    highlights["Top State by Sales"] = top_state_by_sales

    # --- Sort summary ---
    sort_cols = [c for c in ["PA Conversions", "Contact Conversions", "Leads Allocated", "Total SPV"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=[False]*len(sort_cols), kind="mergesort").reset_index(drop=True)

    return summary, highlights


# ================================
# -------------- UI --------------
# ================================

st.set_page_config(page_title="Sales & Marketing Dashboard", layout="wide")
st.title("📊 Sales & Marketing Dashboard — Upload & Analyze")
st.caption("Lead allocation, conversions, SPV, and state performance (no date filter).")

with st.expander("ℹ️ Instructions", expanded=False):
    st.markdown("""
    - Upload **Leads.csv**, **Contacts.csv**, and **PA.csv** (any time period).
    - The app auto-detects people/state and parses **SPV** (prefers *SPV (excl. GST)*).
    - If a file lacks expected columns, you’ll get guidance below.
    """)

col1, col2, col3 = st.columns(3)
with col1: leads_file    = st.file_uploader("Leads CSV",    type=["csv"], key="leads")
with col2: contacts_file = st.file_uploader("Contacts CSV", type=["csv"], key="contacts")
with col3: pa_file       = st.file_uploader("PA Conversions CSV", type=["csv"], key="pa")

# Read raw for checks
leads_raw = pd.read_csv(leads_file) if leads_file else pd.DataFrame()
contacts_raw = pd.read_csv(contacts_file) if contacts_file else pd.DataFrame()
pa_raw = pd.read_csv(pa_file) if pa_file else pd.DataFrame()

with st.expander("✅ File checks & guidance", expanded=False):
    if not leads_raw.empty:
        ok, msgs = file_checks(leads_raw, "leads")
        st.markdown("**Leads.csv**")
        for m in msgs:
            st.markdown(m)
        if not ok:
            st.warning("Leads file is missing a required person column. KPIs may be incomplete.")

    if not contacts_raw.empty:
        ok, msgs = file_checks(contacts_raw, "contacts")
        st.markdown("**Contacts.csv**")
        for m in msgs:
            st.markdown(m)

    if not pa_raw.empty:
        ok_pa, msgs = file_checks(pa_raw, "pa")
        st.markdown("**PA.csv**")
        for m in msgs:
            st.markdown(m)
        if not ok_pa:
            st.error("PA file is missing `Sales Agent` and/or an SPV column. SPV totals will be zero.")

# Clean
leads    = clean_leads(leads_raw)     if not leads_raw.empty    else pd.DataFrame()
contacts = clean_contacts(contacts_raw) if not contacts_raw.empty else pd.DataFrame()
pa       = clean_pa(pa_raw)           if not pa_raw.empty       else pd.DataFrame()

# Guard
if all(df.empty for df in [leads, contacts, pa]):
    st.info("Upload at least one CSV to begin.")
    st.stop()

# -------- Name unification (after cleaning) --------
name_series = []
if not leads.empty and "Assigned To" in leads.columns:
    name_series.append(leads["Assigned To"].astype(str))
if not contacts.empty and "Assigned To" in contacts.columns:
    name_series.append(contacts["Assigned To"].astype(str))
if not pa.empty and "Sales Agent" in pa.columns:
    name_series.append(pa["Sales Agent"].astype(str))

all_names = pd.concat(name_series, ignore_index=True) if name_series else pd.Series([], dtype=str)
fullname_index = build_fullname_index(all_names)

# optional manual overrides
MANUAL_ALIASES = {
     "mark": "Mark Hayward",
     "charles": "Charles Christodoulou",
     "doug": "Doug Edwards",
     "zach": "Zach Lagettie",
     "zachary" : "Zach Lagettie",
}

if not leads.empty and "Assigned To" in leads.columns:
    leads["Assigned To"] = canonicalize_series(leads["Assigned To"], fullname_index, MANUAL_ALIASES).astype("string")
if not contacts.empty and "Assigned To" in contacts.columns:
    contacts["Assigned To"] = canonicalize_series(contacts["Assigned To"], fullname_index, MANUAL_ALIASES).astype("string")
if not pa.empty and "Sales Agent" in pa.columns:
    pa["Sales Agent"] = canonicalize_series(pa["Sales Agent"], fullname_index, MANUAL_ALIASES).astype("string")

ambiguous = {k: sorted(v) for k, v in fullname_index.items() if len(v) > 1}
if ambiguous:
    with st.expander("🔧 Name merge suggestions / conflicts"):
        st.write("These first names map to multiple full names. Add MANUAL_ALIASES entries to disambiguate:")
        st.json(ambiguous)

# Preview
with st.expander("Preview (first 6 rows per file)"):
    if not leads.empty:    st.subheader("Leads");     st.dataframe(leads.head(6), use_container_width=True)
    if not contacts.empty: st.subheader("Contacts");  st.dataframe(contacts.head(6), use_container_width=True)
    if not pa.empty:       st.subheader("PA");        st.dataframe(pa.head(6), use_container_width=True)

# KPIs & summary
summary, highlights = kpis(leads, contacts, pa)

st.markdown("---")
st.subheader("Consultant Summary")
st.dataframe(summary, use_container_width=True)

# KPI tiles
kpi_cols = st.columns(4)
with kpi_cols[0]:
    if highlights.get("Top by Leads"):
        n, c = highlights["Top by Leads"]; st.metric("Top by Leads", f"{n}", delta=f"{c} leads")
with kpi_cols[1]:
    if highlights.get("Top by Contacts"):
        n, c = highlights["Top by Contacts"]; st.metric("Top by Contacts", f"{n}", delta=f"{c}")
with kpi_cols[2]:
    if highlights.get("Top by PAs"):
        n, c = highlights["Top by PAs"]; st.metric("Top by PAs", f"{n}", delta=f"{c}")
with kpi_cols[3]:
    if highlights.get("Top by SPV"):
        n, v = highlights["Top by SPV"]; st.metric("Top by SPV", f"{n}", delta=f"${v:,.0f}")
    else:
        st.metric("Top by SPV", "No SPV data", delta="$0")

# State performance (optional)
st.markdown("---")
st.subheader("State Performance")
state_cols = st.columns(2)
with state_cols[0]:
    if "State" in leads.columns and not leads.empty:
        s_leads = leads["State"].value_counts().reset_index()
        s_leads.columns = ["State", "Lead Count"]
        st.markdown("**Leads by State**"); st.bar_chart(s_leads.set_index("State"))
with state_cols[1]:
    if "State" in pa.columns and not pa.empty:
        s_sales = pa["State"].value_counts().reset_index()
        s_sales.columns = ["State", "PA Count"]
        st.markdown("**PA Conversions by State**"); st.bar_chart(s_sales.set_index("State"))

# Downloads
st.markdown("---")
st.subheader("Downloads")
outputs = {"consultant_summary.csv": summary}
if not leads.empty and "State" in leads.columns:
    outputs["leads_by_state.csv"] = leads["State"].value_counts().rename_axis("State").reset_index(name="Lead Count")
if not pa.empty and "State" in pa.columns:
    outputs["pa_by_state.csv"]  = pa["State"].value_counts().rename_axis("State").reset_index(name="PA Count")
for fname, df_out in outputs.items():
    st.download_button(label=f"⬇️ Download {fname}",
                       data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv")

# ================================
# Conversion rates & SPV visuals
# ================================
st.markdown("### Conversion rates & funnels")

conv = summary.copy()
for c in ["Sales Agent", "Leads Allocated", "Contact Conversions", "PA Conversions", "Total SPV"]:
    if c not in conv.columns: conv[c] = 0
conv["Total SPV"] = pd.to_numeric(conv["Total SPV"], errors="coerce").fillna(0.0)

# rates
conv["CR_LC"] = np.where(conv["Leads Allocated"] > 0, conv["Contact Conversions"] / conv["Leads Allocated"], 0.0)
conv["CR_CP"] = np.where(conv["Contact Conversions"] > 0, conv["PA Conversions"] / conv["Contact Conversions"], 0.0)
conv["CR_LP"] = np.where(conv["Leads Allocated"] > 0, conv["PA Conversions"] / conv["Leads Allocated"], 0.0)

c1, c2 = st.columns([3, 2])
with c1:
    n_max = max(1, int(len(conv)))
    top_n = st.slider("Show top N consultants (by PA conversions)", 1, n_max, min(10, n_max))
with c2:
    min_leads = st.number_input("Min leads to include", min_value=0, value=5, step=1)

conv_f = conv[conv["Leads Allocated"] >= min_leads].sort_values("PA Conversions", ascending=False).head(top_n).copy()

# 1) Horizontal bar — choose what the bar shows
if not conv_f.empty:
    bar_metric = st.radio(
        "Bar shows",
        ["Conversion rate (PA/Leads)", "Total SPV"],
        index=0,
        horizontal=True,
        key="bar_metric_mode",
    )

    if bar_metric == "Conversion rate (PA/Leads)":
        conv_bar = conv_f.sort_values("CR_LP", ascending=False).copy()
        conv_bar["RateLabel"] = (conv_bar["CR_LP"] * 100).round(1).astype(str) + "%"
        fig_bar = px.bar(
            conv_bar,
            y="Sales Agent",
            x="CR_LP",
            orientation="h",
            hover_data={
                "Leads Allocated": True,
                "Contact Conversions": True,
                "PA Conversions": True,
                "Total SPV": ":,.0f",
            },
            labels={"CR_LP": "PA from Leads %"},
        )
        # preserve our sort order top→bottom and clean formatting
        fig_bar.update_layout(
            xaxis_tickformat=".0%",
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis={"categoryorder": "array", "categoryarray": conv_bar["Sales Agent"].tolist()},
        )
        # show the same metric as the bar (rate) to avoid confusion
        fig_bar.update_traces(
            text=conv_bar["RateLabel"],
            textposition="outside",
            cliponaxis=False,
        )

    else:  # "Total SPV"
        conv_bar = conv_f.sort_values("Total SPV", ascending=False).copy()
        conv_bar["CRLabel"] = (conv_bar["CR_LP"] * 100).round(1).astype(str) + "%"
        fig_bar = px.bar(
            conv_bar,
            y="Sales Agent",
            x="Total SPV",
            orientation="h",
            color="CR_LP",  # color by conversion rate for extra context
            hover_data={
                "Leads Allocated": True,
                "Contact Conversions": True,
                "PA Conversions": True,
                "CR_LP": ":.1%",
            },
            labels={"Total SPV": "Total SPV ($)", "CR_LP": "PA/Leads"},
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(
            xaxis_tickformat="$,.0f",
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis={"categoryorder": "array", "categoryarray": conv_bar["Sales Agent"].tolist()},
            coloraxis_colorbar=dict(title="PA/Leads"),
        )
        # label with dollars
        fig_bar.update_traces(
            text=[f"${v:,.0f}" for v in conv_bar["Total SPV"]],
            textposition="outside",
            cliponaxis=False,
        )

    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No consultants match the current filters to draw conversion bars.")

# 2) Bubble — volume vs conversion (size = SPV, color = CR_LP)
if not conv_f.empty:
    fig_bbl = px.scatter(
        conv_f, x="Leads Allocated", y="PA Conversions",
        size="Total SPV", color="CR_LP", hover_name="Sales Agent",
        size_max=48, labels={"CR_LP": "PA/Leads"},
    )
    fig_bbl.update_layout(coloraxis_colorbar=dict(title="PA/Leads"), height=440, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bbl, use_container_width=True)

# 3) Per-consultant funnel with SPV
if not conv_f.empty:
    agent = st.selectbox("Funnel by consultant", options=conv_f["Sales Agent"].dropna().tolist())
    row = conv_f.loc[conv_f["Sales Agent"] == agent].iloc[0]
    fig_fun = go.Figure(go.Funnel(
        y=["Leads", "Contacts", "PAs"],
        x=[int(row["Leads Allocated"]), int(row["Contact Conversions"]), int(row["PA Conversions"])],
        textinfo="value+percent initial",
    ))
    st.plotly_chart(fig_fun, use_container_width=True)
    st.caption(f"Total SPV for {agent}: ${row['Total SPV']:,.0f}")

# 4) Team funnel
if not conv_f.empty:
    agg = conv_f[["Leads Allocated", "Contact Conversions", "PA Conversions", "Total SPV"]].sum()
    fig_fun_all = go.Figure(go.Funnel(
        y=["Leads", "Contacts", "PAs"],
        x=[int(agg["Leads Allocated"]), int(agg["Contact Conversions"]), int(agg["PA Conversions"])],
        textinfo="value+percent initial",
    ))
    st.plotly_chart(fig_fun_all, use_container_width=True)
    st.caption(f"Aggregate SPV (shown consultants): ${agg['Total SPV']:,.0f}")

# 5) Grouped bars — conversion rate by agent across all stages
st.markdown("### Conversion rate by sales agent (Leads → Contacts → PAs)")
cr = summary.copy()
for c in ["Sales Agent", "Leads Allocated", "Contact Conversions", "PA Conversions", "Total SPV"]:
    if c not in cr.columns: cr[c] = 0
cr["Leads Allocated"] = pd.to_numeric(cr["Leads Allocated"], errors="coerce").fillna(0).astype(int)
cr["Contact Conversions"] = pd.to_numeric(cr["Contact Conversions"], errors="coerce").fillna(0).astype(int)
cr["PA Conversions"] = pd.to_numeric(cr["PA Conversions"], errors="coerce").fillna(0).astype(int)
cr["Total SPV"] = pd.to_numeric(cr.get("Total SPV", 0), errors="coerce").fillna(0.0)

cr = cr[cr["Leads Allocated"] > 0].copy()
if cr.empty:
    st.info("No consultants with leads to compute conversion rates.")
else:
    cr["CR_L→C"] = np.where(cr["Leads Allocated"] > 0, cr["Contact Conversions"] / cr["Leads Allocated"], 0.0)
    cr["CR_C→PA"] = np.where(cr["Contact Conversions"] > 0, cr["PA Conversions"] / cr["Contact Conversions"], 0.0)
    cr["CR_L→PA"] = np.where(cr["Leads Allocated"] > 0, cr["PA Conversions"] / cr["Leads Allocated"], 0.0)
    order = cr.sort_values("CR_L→PA", ascending=False)["Sales Agent"].tolist()
    long = cr.melt(
        id_vars=["Sales Agent", "Leads Allocated", "Contact Conversions", "PA Conversions", "Total SPV"],
        value_vars=["CR_L→C", "CR_C→PA", "CR_L→PA"],
        var_name="Stage", value_name="Conversion Rate",
    )
    long["Sales Agent"] = pd.Categorical(long["Sales Agent"], categories=order, ordered=True)
    stage_order = ["CR_L→C", "CR_C→PA", "CR_L→PA"]
    stage_labels = {"CR_L→C": "Leads → Contacts", "CR_C→PA": "Contacts → PAs", "CR_L→PA": "Leads → PAs"}

    fig = px.bar(
        long, x="Sales Agent", y="Conversion Rate", color="Stage",
        category_orders={"Sales Agent": order, "Stage": stage_order},
        labels={"Conversion Rate": "Rate", "Sales Agent": "", "Stage": "Stage"},
        barmode="group",
        hover_data={"Leads Allocated": True, "Contact Conversions": True, "PA Conversions": True, "Total SPV": ":,.0f", "Stage": False},
    )
    fig.for_each_trace(lambda t: t.update(name=stage_labels.get(t.name, t.name)))
    fig.update_layout(yaxis_tickformat=".0%", height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
