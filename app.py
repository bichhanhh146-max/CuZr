# app.py — Alloy Tg/Tm Predictor (ExtraTrees, 42 elements, English UI)
import re
from typing import Dict
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ================== CONFIG ================== #
FEATURES = [
    'Fe','Ni','Co','Cr','Hf','Sn','Zr','Ce','In','Ga','Gd','Si','Zn','Bi','Ti',
    'Mo','W','P','Pd','Ta','V','Nb','Sc','Al','Cu','La','Mg','Tb','Y','Be','Pr',
    'Er','Tm','Nd','B','Au','Ho','Ca','Dy','Mn','Ag','C'
]

TG_MODEL_PATH = "model_Tg_extratrees.pkl"
TM_MODEL_PATH = "model_Tm_extratrees.pkl"

# ================== UTILS ================== #
EL_PATTERN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")

def parse_composition(text: str) -> Dict[str, float]:
    """Parse 'Cu50Zr40Al10' or 'Cu:50, Zr:40, Al:10' → {'Cu':50,'Zr':40,'Al':10} (not normalized)."""
    if not text:
        return {}
    cleaned = re.sub(r"[:=,;\s-]+", "", text.strip())
    parts = EL_PATTERN.findall(cleaned)
    if not parts:
        alt = re.findall(r"([A-Z][a-z]?)\s*[:=]\s*([0-9]+\.?[0-9]*)", text)
        if not alt:
            raise ValueError("Failed to parse composition string. Examples: 'Cu50Zr40Al10' or 'Cu:50, Zr:40, Al:10'.")
        parts = alt
    comp: Dict[str, float] = {}
    for sym, num in parts:
        val = float(num) if num else 1.0
        comp[sym] = comp.get(sym, 0.0) + val
    return comp

def normalize_to_percent(comp: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, v) for v in comp.values()))
    if s <= 0:
        raise ValueError("Sum of ratios must be > 0.")
    return {k: (max(0.0, v) / s) * 100.0 for k, v in comp.items()}

def to_vector(comp_pct: Dict[str, float]) -> np.ndarray:
    """Return (1, 42) vector ordered by FEATURES."""
    vec = np.zeros(len(FEATURES), dtype=float)
    for i, el in enumerate(FEATURES):
        vec[i] = comp_pct.get(el, 0.0)
    return vec.reshape(1, -1)

# ================== MODELS ================== #
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None

model_Tg = load_model(TG_MODEL_PATH)
model_Tm = load_model(TM_MODEL_PATH)

def predict_Tg(comp_pct: dict):
    if model_Tg is None:
        return None
    return float(model_Tg.predict(to_vector(comp_pct))[0])

def predict_Tm(comp_pct: dict):
    if model_Tm is None:
        return None
    return float(model_Tm.predict(to_vector(comp_pct))[0])

# ================== APP ================== #
st.set_page_config(page_title="Alloy Tg/Tm Predictor", page_icon="🧪", layout="wide")
st.title("🧪 Alloy Tg/Tm Predictor — ExtraTrees (42 elements)")
st.caption("Enter alloy composition (atomic %) to predict Tg and Tm.")

mode = st.sidebar.radio("Input mode", ["Formula string", "Builder", "Batch CSV"])
auto_norm = st.sidebar.checkbox("Normalize to 100 at%", value=True)

# ---------- Mode 1: Formula string ----------
if mode == "Formula string":
    s = st.text_input("Composition (e.g., Cu50Zr40Al10 or 'Cu:50, Zr:40, Al:10')", value="Cu50Zr50")
    if st.button("🔮 Predict (string)"):
        try:
            comp_raw = parse_composition(s)
            unknown = [e for e in comp_raw if e not in FEATURES]
            if unknown:
                st.error(f"Unsupported elements: {', '.join(unknown)}")
            else:
                comp_pct = normalize_to_percent(comp_raw) if auto_norm else comp_raw
                tg = predict_Tg(comp_pct)
                tm = predict_Tm(comp_pct)
                c1, c2 = st.columns(2)
                with c1: st.metric("Tg (K)", f"{tg:.1f}" if tg is not None else "N/A")
                with c2: st.metric("Tm (K)", f"{tm:.1f}" if tm is not None else "N/A")
                st.markdown("**Normalized composition (at%)**")
                st.dataframe(pd.DataFrame({"Element": list(comp_pct.keys()), "at%": list(comp_pct.values())})
                             .sort_values("Element").reset_index(drop=True),
                             use_container_width=True)
        except Exception as e:
            st.error(str(e))

# ---------- Mode 2: Builder ----------
if mode == "Builder":
    picked = st.multiselect("Pick elements", FEATURES, default=["Cu", "Zr"], max_selections=12)
    comp_inputs = {}
    for el in picked:
        comp_inputs[el] = st.number_input(el, min_value=0.0, value=(100.0/len(picked) if picked else 0.0),
                                          step=0.5, key=f"n_{el}")
    if st.button("🔮 Predict (builder)"):
        if not picked:
            st.error("Please choose at least two elements.")
        else:
            comp_pct = normalize_to_percent(comp_inputs) if auto_norm else comp_inputs
            tg = predict_Tg(comp_pct)
            tm = predict_Tm(comp_pct)
            c1, c2 = st.columns(2)
            with c1: st.metric("Tg (K)", f"{tg:.1f}" if tg is not None else "N/A")
            with c2: st.metric("Tm (K)", f"{tm:.1f}" if tm is not None else "N/A")
            st.dataframe(pd.DataFrame({"Element": list(comp_pct.keys()), "at%": list(comp_pct.values())})
                         .sort_values("Element").reset_index(drop=True),
                         use_container_width=True)

# ---------- Mode 3: Batch CSV ----------
if mode == "Batch CSV":
    up = st.file_uploader("Upload CSV", type="csv")
    st.caption("CSV with a `composition` column (string) **or** 42 columns named exactly as features.")
    if up:
        df = pd.read_csv(up)
        if "composition" in df.columns:
            rows = []
            for s in df["composition"].astype(str):
                try:
                    comp = normalize_to_percent(parse_composition(s))
                    rows.append(to_vector(comp).reshape(-1))
                except Exception:
                    rows.append(np.zeros(len(FEATURES)))
            X = np.vstack(rows)
            out = pd.DataFrame({"composition": df["composition"]})
        else:
            for el in FEATURES:
                if el not in df.columns:
                    df[el] = 0.0
            X = df[FEATURES].fillna(0.0).to_numpy(float)
            out = pd.DataFrame()
        if model_Tg is not None:
            out["Tg_pred"] = model_Tg.predict(X)
        if model_Tm is not None:
            out["Tm_pred"] = model_Tm.predict(X)
        st.dataframe(out.head(), use_container_width=True)
        st.download_button("Download predictions (CSV)", out.to_csv(index=False), "predictions.csv")

# ---------- Binary phase (scatter) ----------
st.divider()
st.subheader("Binary phase sweep (scatter)")
colA, colB, colStep = st.columns([1, 1, 1])
elA = colA.selectbox("Element A", FEATURES, index=FEATURES.index("Cu"))
elB = colB.selectbox("Element B", FEATURES, index=FEATURES.index("Zr"))
step_bin = colStep.number_input("Step (%)", 1, 20, 5)
if st.button("Generate binary phase"):
    xs, tg_list, tm_list = [], [], []
    for a in range(0, 101, int(step_bin)):
        comp = {elA: float(a), elB: float(100 - a)}
        xs.append(a)
        tg_list.append(predict_Tg(comp))
        tm_list.append(predict_Tm(comp))
    df_bin = pd.DataFrame({f"{elA}_at%": xs, "Tg_pred": tg_list, "Tm_pred": tm_list})
    st.dataframe(df_bin.head(), use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(df_bin[f"{elA}_at%"], df_bin["Tg_pred"], s=20, label="Tg (K)", alpha=0.85)
    ax.scatter(df_bin[f"{elA}_at%"], df_bin["Tm_pred"], s=20, label="Tm (K)", alpha=0.85)
    ax.set_xlabel(f"{elA} at%  ( {elB} = 100 − {elA} )")
    ax.set_ylabel("Temperature (K)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ---------- Universal Phase Explorer (PCA scatter for any N) ----------
st.divider()
st.header("Universal Phase Explorer (PCA scatter)")

top = st.columns([1, 2, 1, 1.2])
with top[0]:
    max_n = 12
    n_elems = st.number_input("Number of elements", min_value=2, max_value=max_n, value=3, step=1)
with top[1]:
    picked_u = st.multiselect("Pick elements", FEATURES,
                              default=(["Cu", "Zr", "Al"] if n_elems >= 3 else ["Cu", "Zr"]),
                              max_selections=n_elems)
with top[2]:
    step_u = st.number_input("Step (%)", min_value=1, max_value=20, value=5, step=1,
                             help="Grid resolution. Smaller step → more points.")
with top[3]:
    color_target = st.radio("Color by", ["Tg", "Tm"], horizontal=True)

def integer_compositions(K: int, parts: int):
    """All tuples (len=parts) of non-negative integers summing to K."""
    if parts == 1:
        yield (K,)
    else:
        for i in range(K + 1):
            for rest in integer_compositions(K - i, parts - 1):
                yield (i,) + rest

if len(picked_u) != n_elems:
    st.info(f"Select exactly {n_elems} elements.")
else:
    units = int(100 // step_u)
    try:
        est_points = math.comb(units + n_elems - 1, n_elems - 1)
    except ValueError:
        est_points = 0
    if est_points > 30000:
        st.warning(f"Estimated grid size ≈ {est_points:,}. Consider increasing the step to reduce points.")

    if st.button("Generate grid, predict & plot (PCA)"):
        rows = []
        names = picked_u[:]
        for tup in integer_compositions(units, n_elems):
            comp = {names[i]: float(tup[i] * step_u) for i in range(n_elems)}
            if abs(sum(comp.values()) - 100.0) > 1e-9:
                continue
            tg = predict_Tg(comp)
            tm = predict_Tm(comp)
            rows.append({**comp,
                         "Tg_pred": (tg if tg is not None else np.nan),
                         "Tm_pred": (tm if tm is not None else np.nan)})
        if not rows:
            st.error("No points generated. Try a larger step or fewer elements.")
        else:
            df_phase = pd.DataFrame(rows)
            st.write(f"Generated points: **{len(df_phase):,}**")
            st.dataframe(df_phase.head(), use_container_width=True)

            # PCA 2D via SVD (no extra deps)
            X = df_phase[names].to_numpy(float)
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            Z = U[:, :2] * S[:2]

            # scatter colored by Tg or Tm
            color_col = "Tg_pred" if color_target == "Tg" else "Tm_pred"
            label_col = "Tg (K)" if color_target == "Tg" else "Tm (K)"

            fig, ax = plt.subplots()
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=df_phase[color_col].to_numpy(),
                            s=16, alpha=0.9, edgecolors='none')
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label(label_col)
            ax.set_xlabel("PCA-1")
            ax.set_ylabel("PCA-2")
            ax.set_title(f"PCA scatter — elements: {', '.join(names)} | step={step_u}% | color={label_col}")
            ax.grid(True, alpha=0.25)
            st.pyplot(fig)

            # export
            out = df_phase.copy()
            out["PCA1"] = Z[:, 0]
            out["PCA2"] = Z[:, 1]
            st.download_button("Download grid (CSV)",
                               data=out.to_csv(index=False).encode("utf-8"),
                               file_name=f"phase_grid_{'_'.join(names)}_step{step_u}.csv")
