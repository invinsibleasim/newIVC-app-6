# damp_t_app.py
# Streamlit app for DAmP‑T + CUSP‑T (Morphology‑preserving) translation
# License: For research & evaluation only. Validate before production use.

import io
import json
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# APP CONFIG / TITLE
# =========================
st.set_page_config(page_title="DAmP‑T + CUSP‑T: Degradation‑Aware I‑V Translation", layout="wide")
st.title("DAmP‑T + CUSP‑T — Degradation‑Aware multi‑modal I‑V Translation")
st.caption(
    "Upload Light I‑V, Dark I‑V, and optional Suns‑Voc CSVs. "
    "Choose **Fast initializer**, **Precision Fit (MO)**, or **CUSP‑T (Morphology)** to get translated 'as‑is' and 'neutralized' curves at target conditions."
)

# =========================
# OPTIONAL: MO IMPORT
# =========================
HAS_MO = False
try:
    # If present, enables Precision Fit (MO)
    from damp_t.optim_mo import damp_t_pipeline_mo, MOWeights  # noqa: F401
    HAS_MO = True
except Exception:
    HAS_MO = False  # App still runs with Fast and CUSP‑T

# =========================
# CSV HELPERS (common)
# =========================
REQUIRED_LIGHT = {"V", "I", "G", "T"}
REQUIRED_DARK  = {"V", "I"}
REQUIRED_SV    = {"G", "T", "Voc"}

def load_csv(upload, required_cols):
    df = pd.read_csv(upload)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.dropna().copy()

def split_light_curves(df_light):
    """Split concatenated light I‑V into individual sweeps by detecting V wrap."""
    V = df_light["V"].values
    idx = [0]
    for i in range(1, len(V)):
        if V[i] < V[i-1] - 1e-9:
            idx.append(i)
    idx.append(len(V))
    curves = []
    for s, e in zip(idx[:-1], idx[1:]):
        seg = df_light.iloc[s:e].copy()
        # Force scalar G, T per sweep (median)
        Gm = float(seg["G"].median())
        Tm = float(seg["T"].median())
        seg["G"], seg["T"] = Gm, Tm
        curves.append(seg.reset_index(drop=True))
    return curves

# =========================
# DIAGNOSTICS (for Fast path)
# =========================
def detect_kinks(V, I, window=9, thresh=5.0):
    """Simple 2nd-derivative peak finder; returns 0/1/2 approximate kink count."""
    if len(V) < window + 2:
        return 0
    # Smooth (moving average with odd window)
    k = max(3, window | 1)
    pad = k // 2
    def smooth(x):
        xx = np.pad(x, (pad, pad), mode="edge")
        ker = np.ones(k) / k
        return np.convolve(xx, ker, mode="valid")
    Vs = smooth(V)
    Is = smooth(I)
    d1 = np.gradient(Is, Vs)
    d2 = np.gradient(d1, Vs)
    ref = np.median(np.abs(d2) + 1e-9)
    pk = int(np.sum(np.abs(d2) > (thresh * ref)))
    if pk > 2:
        return 2
    if pk > 0:
        return 1
    return 0

# =========================
# DARK‑IV FITS (Fast path)
# =========================
def fit_shunt_powerlaw(V, I, v_max=2.0):
    """Fit I ~ a*|V|^m in low‑voltage region (dark IV); return dict(a, m)."""
    V = np.asarray(V); I = np.asarray(I)
    Vabs = np.abs(V)
    mask = (Vabs > 1e-4) & (Vabs < v_max)
    if mask.sum() < 10:
        return {"a": 1e-5, "m": 1.5}
    x = np.log(Vabs[mask])
    y = np.log(np.abs(I[mask]) + 1e-15)
    A = np.vstack([np.ones_like(x), x]).T
    c, m = np.linalg.lstsq(A, y, rcond=None)[0]
    a = float(np.exp(c))
    return {"a": a, "m": float(m)}

def estimate_series_resistance(V, I, top_frac=0.1):
    """Estimate Rs from dark I‑V at high current (linearized slope dV/dI)."""
    V = np.asarray(V); I = np.asarray(I)
    n = len(V)
    if n < 10:
        return 0.2
    idx = np.argsort(I)
    sel = idx[int((1 - top_frac) * n):]
    dI = I[sel] - I[sel].mean()
    dV = V[sel] - V[sel].mean()
    denom = (dI**2).sum()
    if denom < 1e-12:
        return 0.2
    slope = (dI * dV).sum() / denom
    return float(max(0.0, slope))

# =========================
# SUNS‑VOC (Fast path coarse fit)
# =========================
kB = 1.380649e-23
q = 1.602176634e-19

def fit_suns_voc(G, T, Voc, Ns=60):
    """Coarse ln(G) Voc fit (used only as soft check in Fast path)."""
    G = np.asarray(G); T = np.asarray(T); Voc = np.asarray(Voc)
    if len(G) < 2:
        return {"n_eff": 1.5, "J0_ref": 1e-10, "Tref": 25.0, "c_isc": 8.0}
    Tref = float(np.median(T))
    Vt = Ns * kB * (Tref + 273.15) / q
    x = np.log(np.maximum(G, 1e-3))
    y = Voc
    A = np.vstack([np.ones_like(x), x]).T
    c0, c1 = np.linalg.lstsq(A, y, rcond=None)[0]
    n_eff = float(np.clip(c1 / max(1e-9, Vt), 1.0, 2.2))
    c_isc = 8.0
    Gm = float(np.median(G))
    Vocm = float(np.median(Voc))
    Isc = c_isc * (Gm / 1000.0)
    J0 = max(1e-15, Isc / (np.exp(Vocm / (n_eff * Vt)) - 1.0))
    return {"n_eff": n_eff, "J0_ref": float(J0), "Tref": Tref, "c_isc": c_isc}

# =========================
# SIMPLE DEVICE MODEL (Fast)
# =========================
def Vth(TC, Ns=60):
    return Ns * kB * (TC + 273.15) / q

class SubmoduleParams:
    def __init__(self):
        self.Rs = 0.3
        self.I01 = 1e-9
        self.n1 = 1.4
        self.I02 = 1e-7
        self.n2 = 2.0
        self.a_sh = 1e-5
        self.m_sh = 1.5
        self.phi = 1.0
        self.eta_inactive = 0.0
        self.b_bypass = 0.0

class EnvCoeffs:
    def __init__(self):
        self.alpha_I = 0.0005
        self.beta_Voc = -0.002

class DegradationState:
    def __init__(self, theta=1.0):
        self.theta = theta

def photocurrent(G, TC, sp: SubmoduleParams, env: EnvCoeffs, degr: DegradationState):
    return sp.phi * (G / 1000.0) * (1.0 + env.alpha_I * (TC - 25.0)) * (1.0 - sp.eta_inactive) * degr.theta

def shunt_current(Vk, sp: SubmoduleParams):
    if Vk <= 0:
        return 0.0
    return sp.a_sh * (Vk ** sp.m_sh)

def diode_currents(Vk, I, TC, sp: SubmoduleParams, Ns=60):
    Vt = Vth(TC, Ns)
    arg1 = (Vk + I * sp.Rs) / (sp.n1 * Vt)
    arg2 = (Vk + I * sp.Rs) / (sp.n2 * Vt)
    arg1 = np.clip(arg1, -50, 50); arg2 = np.clip(arg2, -50, 50)
    return sp.I01 * (np.exp(arg1) - 1.0) + sp.I02 * (np.exp(arg2) - 1.0)

def submodule_residual(I, Vk, G, TC, sp: SubmoduleParams, env: EnvCoeffs, degr: DegradationState, Ns=60):
    Vk_eff = (1.0 - sp.b_bypass) * Vk
    Il = photocurrent(G, TC, sp, env, degr)
    Id = diode_currents(Vk_eff, I, TC, sp, Ns)
    Ish = shunt_current(Vk_eff, sp)
    return I - (Il - Id - Ish)

def module_current(V, G, TC, subs, env: EnvCoeffs, degr: DegradationState, Ns=60):
    I = 0.0
    for _ in range(60):
        Vsum = 0.0
        for sp in subs:
            lo, hi = -5.0, max(5.0, V + 5.0)
            for __ in range(30):
                mid = 0.5 * (lo + hi)
                f = submodule_residual(I, mid, G, TC, sp, env, degr, Ns)
                if f > 0:
                    hi = mid
                else:
                    lo = mid
            Vk = 0.5 * (lo + hi)
            Vsum += Vk + I * sp.Rs
        errV = Vsum - V
        I -= 0.5 * errV / max(1e-6, sum(sp.Rs for sp in subs))
        if abs(errV) < 1e-4:
            break
    return max(0.0, I)

def translate_curve(V_grid, G2, T2, subs, env, degr):
    return np.array([module_current(v, G2, T2, subs, env, degr) for v in V_grid])

def init_submodules(kinks=0, M=3):
    subs = []
    for i in range(M):
        sp = SubmoduleParams()
        if kinks >= 1 and i == (M - 1):
            sp.b_bypass = 0.5  # relaxed prior for kinked case
        subs.append(sp)
    return subs

def fit_composite(anchor_df, Rs_est, shunt_dict, subs):
    a_sh, m_sh = shunt_dict["a"], shunt_dict["m"]
    for sp in subs:
        sp.Rs = max(0.0, Rs_est / len(subs))
        sp.a_sh = max(1e-8, a_sh / len(subs))
        sp.m_sh = float(m_sh)
    env = EnvCoeffs()
    degr = DegradationState(theta=1.0)
    V = anchor_df["V"].values; I = anchor_df["I"].values
    top = max(1, int(0.02 * len(I)))
    Isc_proxy = float(np.sort(I)[-top:].mean())
    Gm = float(anchor_df["G"].median())
    phi = (Isc_proxy / (Gm / 1000.0)) / len(subs)
    for sp in subs:
        sp.phi = float(np.clip(phi, 0.1, 20.0))
    return subs, env, degr

def damp_t_pipeline(df_light, df_dark, df_sv=None, target_G=1000.0, target_T=25.0):
    sweeps = split_light_curves(df_light)
    anchor = sweeps[0]
    V_anchor = anchor["V"].values; I_anchor = anchor["I"].values
    kx = detect_kinks(V_anchor, I_anchor)
    subs = init_submodules(kx, M=3)
    sh = fit_shunt_powerlaw(df_dark["V"].values, df_dark["I"].values)
    Rs = estimate_series_resistance(df_dark["V"].values, df_dark["I"].values)
    if df_sv is not None:
        sv_fit = fit_suns_voc(df_sv["G"].values, df_sv["T"].values, df_sv["Voc"].values)
    else:
        sv_fit = {"n_eff": 1.5, "J0_ref": 1e-10, "Tref": 25.0, "c_isc": 8.0}
    subs, env, degr = fit_composite(anchor, Rs, sh, subs)
    V_grid = np.linspace(float(anchor["V"].min()), float(anchor["V"].max()), 200)
    I_as_is = translate_curve(V_grid, target_G, target_T, subs, env, degr)
    subsN = []
    for sp in subs:
        s2 = SubmoduleParams()
        s2.__dict__.update(sp.__dict__)
        s2.b_bypass = 0.0
        s2.eta_inactive = 0.0
        s2.a_sh = min(s2.a_sh, 1e-8)
        subsN.append(s2)
    I_neutral = translate_curve(V_grid, target_G, target_T, subsN, env, degr)
    out = {
        "anchor": anchor,
        "translated": pd.DataFrame({"V": V_grid, "I": I_as_is}),
        "neutralized": pd.DataFrame({"V": V_grid, "I": I_neutral}),
        "kinks_detected": int(kx),
        "Rs_estimate": float(Rs),
        "shunt_powerlaw": sh,
        "suns_voc_fit": sv_fit,
        "subs": [sp.__dict__ for sp in subs],
    }
    return out

# =========================
# INLINE CUSP‑T LIBRARY
# =========================
def _cuspt_split_by_wrap(df: pd.DataFrame):
    V = df['V'].values
    cuts = [0]
    for i in range(1, len(V)):
        if V[i] < V[i-1] - 1e-9:
            cuts.append(i)
    cuts.append(len(V))
    sweeps=[]
    for s,e in zip(cuts[:-1], cuts[1:]):
        seg = df.iloc[s:e].copy().reset_index(drop=True)
        if 'G' in seg.columns: seg['G'] = float(seg['G'].median())
        if 'T' in seg.columns: seg['T'] = float(seg['T'].median())
        sweeps.append(seg)
    return sweeps

def _cuspt_detect_kink_knots(V: np.ndarray, I: np.ndarray, window: int=9, thresh: float=5.0):
    n = len(V)
    if n < window + 2: return []
    k = max(3, window | 1)
    pad = k//2
    def smooth(x):
        xx = np.pad(x, (pad,pad), mode='edge')
        ker = np.ones(k)/k
        return np.convolve(xx, ker, mode='valid')
    Vs = smooth(V); Is = smooth(I)
    d1 = np.gradient(Is, Vs)
    d2 = np.gradient(d1, Vs)
    ref = np.median(np.abs(d2) + 1e-12)
    peaks = np.where(np.abs(d2) > thresh*ref)[0]
    if peaks.size == 0: return []
    picks = [int(peaks[0])]
    for i in range(1, len(peaks)):
        if peaks[i] != peaks[i-1] + 1:
            picks.append(int(peaks[i]))
        if len(picks) == 2: break
    return [float(Vs[j]) for j in picks]

def _cuspt_fit_monotone_spline_fc(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2: raise ValueError("need >=2 points")
    h = np.diff(x)
    delta = np.diff(y)/np.where(h==0, 1e-12, h)
    m = np.zeros_like(x)
    m[0]=delta[0]; m[-1]=delta[-1]
    m[1:-1] = (delta[:-1]+delta[1:])/2.0
    for i in range(x.size-1):
        if delta[i] == 0.0:
            m[i]=0.0; m[i+1]=0.0
        else:
            a = m[i]/delta[i]; b = m[i+1]/delta[i]; s = a*a + b*b
            if s > 9.0:
                t = 3.0/np.sqrt(s)
                m[i]   = t*a*delta[i]
                m[i+1] = t*b*delta[i]
    def eval(u):
        u = np.asarray(u, float)
        u_cl = np.clip(u, x[0], x[-1])
        idx = np.searchsorted(x, u_cl) - 1
        idx = np.clip(idx, 0, x.size-2)
        x0=x[idx]; x1=x[idx+1]; y0=y[idx]; y1=y[idx+1]
        m0=m[idx]; m1=m[idx+1]
        h=(x1-x0); t=np.where(h==0, 0.0, (u_cl-x0)/h)
        t2=t*t; t3=t2*t
        h00=2*t3-3*t2+1; h10=t3-2*t2+t
        h01=-2*t3+3*t2; h11=t3-t2
        return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1
    return eval

def _cuspt_fit_powerlaw_shunt(V: np.ndarray, I: np.ndarray, v_max=2.0):
    V=np.asarray(V); I=np.asarray(I)
    Vabs=np.abs(V)
    m=(Vabs>1e-4)&(Vabs<v_max)
    if m.sum()<10: return 1e-5, 1.5
    x=np.log(Vabs[m]); y=np.log(np.abs(I[m])+1e-15)
    A=np.vstack([np.ones_like(x), x]).T
    c, mexp = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.exp(c)), float(mexp)

def _cuspt_estimate_rs(V: np.ndarray, I: np.ndarray, top_frac=0.1):
    n=len(V); 
    if n<10: return 0.2
    idx=np.argsort(I); sel=idx[int((1-top_frac)*n):]
    dI=I[sel]-I[sel].mean(); dV=V[sel]-V[sel].mean()
    denom=(dI**2).sum()
    if denom<1e-12: return 0.2
    slope=(dI*dV).sum()/denom
    return float(max(0.0, slope))

def _cuspt_fit_sunsvoc(G: np.ndarray, T: np.ndarray, Voc: np.ndarray, beta_V=-0.08):
    G=np.asarray(G); T=np.asarray(T); Voc=np.asarray(Voc)
    if G.size<2:
        return {'A': float(Voc.mean() if Voc.size else 40.0), 'B': 0.0, 'Tref': 25.0, 'beta_V': float(beta_V)}
    Tref=float(np.median(T))
    x=np.log(np.maximum(G,1e-3)); y=Voc
    A=np.vstack([np.ones_like(x), x]).T
    a,b=np.linalg.lstsq(A,y,rcond=None)[0]
    return {'A': float(a), 'B': float(b), 'Tref': Tref, 'beta_V': float(beta_V)}

def _cuspt_predict_voc(model: dict, G: float, T: float):
    a=model.get('A',40.0); b=model.get('B',0.0); beta_V=model.get('beta_V',-0.08)
    return float(a + b*np.log(max(1e-3,G)) + beta_V*(T-25.0))

def cuspt_translate_inline(light_df: pd.DataFrame,
                           dark_df: pd.DataFrame,
                           sv_df: pd.DataFrame | None,
                           G2: float = 1000.0,
                           T2: float = 25.0,
                           alpha_I: float = 0.0005,
                           beta_V: float = -0.08,
                           rs_cap: float = 0.35,
                           a_bounds=(1e-6,5e-3),
                           m_bounds=(1.1,2.0)):
    """
    CUSP‑T morphology‑preserving translation with non‑ohmic shunt + Rs‑free (Suns‑Voc) anchoring.
    Returns: {"translated": df(V,I), "neutralized": df(V,I), "diagnostics": {...}}
    """
    # 1) Anchor sweep
    sweeps=_cuspt_split_by_wrap(light_df)
    if not sweeps: raise ValueError("No Light I‑V sweep found.")
    anchor=sweeps[0]
    V1=anchor['V'].values; I1=anchor['I'].values
    G1=float(anchor['G'].median()); T1=float(anchor['T'].median())

    # 2) Isc, Voc
    top=max(1,int(0.02*len(I1)))
    Isc1=float(np.sort(I1)[-top:].mean())
    idx0=int(np.argmin(np.abs(I1))); Voc1=float(V1[idx0])

    # 3) Morphology M(V)=I/Isc
    M=I1/max(1e-9, Isc1)
    order=np.argsort(V1); V1s=V1[order]; Ms=M[order]
    shape=_cuspt_fit_monotone_spline_fc(V1s, Ms)

    # 4) Kink detection
    knots=_cuspt_detect_kink_knots(V1, I1)
    Vk1=float(knots[0]) if knots else 0.6*Voc1
    Vk1=float(np.clip(Vk1, 0.2*Voc1, 0.9*Voc1)) if Voc1>0 else Vk1

    # 5) Dark fits
    a_sh, m_sh=_cuspt_fit_powerlaw_shunt(dark_df['V'].values, dark_df['I'].values)
    a_sh=float(np.clip(a_sh, a_bounds[0], a_bounds[1]))
    m_sh=float(np.clip(m_sh, m_bounds[0], m_bounds[1]))
    Rs_est=min(rs_cap, _cuspt_estimate_rs(dark_df['V'].values, dark_df['I'].values))

    # 6) Suns‑Voc model + target anchors
    if sv_df is not None and len(sv_df)>=2:
        sv_model=_cuspt_fit_sunsvoc(sv_df['G'].values, sv_df['T'].values, sv_df['Voc'].values, beta_V=beta_V)
        Voc2=_cuspt_predict_voc(sv_model, G2, T2)
    else:
        Voc2=Voc1 + beta_V*((T2-25.0)-(T1-25.0)) + 0.65*np.log((G2/max(1e-3,G1))+1.0)
        sv_model={'A':Voc1,'B':0.65,'Tref':T1,'beta_V':beta_V}

    Isc2=Isc1*(G2/G1)*(1.0+alpha_I*(T2-25.0))/(1.0+alpha_I*(T1-25.0))

    # 7) Local‑stiffness warp (piecewise)
    r = Vk1/max(1e-9,Voc1)
    Vk2 = r*Voc2
    V2_grid = np.linspace(0.0, max(V1.max(), Voc2), len(V1))
    def W_inv(v2):
        if v2<=Vk2:
            t=v2/max(1e-9,Vk2); return t*Vk1
        t=(v2-Vk2)/max(1e-9,(Voc2-Vk2)); return Vk1 + t*max(1e-9,(Voc1-Vk1))
    V1_back = np.array([W_inv(v) for v in V2_grid])

    # 8) As‑is + neutralized
    M_interp = np.clip(shape(V1_back), 0.0, 1.05)
    I2_shape = Isc2*M_interp
    Ish2 = a_sh*np.maximum(0.0, V2_grid)**m_sh
    I2_as_is = np.maximum(0.0, I2_shape - Ish2 - 0.02*Rs_est*V2_grid)
    I2_neu   = np.maximum(0.0, Isc2*M_interp - 0.02*min(1e-3, Rs_est)*V2_grid)

    # 9) KPIs (basic)
    def kpi(V,I):
        P=V*I; j=int(np.argmax(P))
        return {"Pmax": float(P[j]), "Vmp": float(V[j]), "Imp": float(I[j]),
                "Voc": float(V[np.argmin(np.abs(I))]), "Isc": float(I[0])}

    return {
        "translated":  pd.DataFrame({"V": V2_grid, "I": I2_as_is}),
        "neutralized": pd.DataFrame({"V": V2_grid, "I": I2_neu}),
        "diagnostics": {
            "Isc1": Isc1, "Voc1": Voc1, "Isc2": Isc2, "Voc2": Voc2,
            "Vk1": Vk1, "Vk2": Vk2, "a_sh": a_sh, "m_sh": m_sh, "Rs_est": Rs_est,
            "sunsvoc_model": sv_model,
            "kpi_translated":  kpi(V2_grid, I2_as_is),
            "kpi_neutralized": kpi(V2_grid, I2_neu)
        }
    }

# =========================
# PLOTTING & KPIs (common)
# =========================
def plot_results(anchor_df, tr_df, neu_df, title="Results"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # I‑V
    ax[0].plot(anchor_df["V"], anchor_df["I"], "k.", label="Measured (anchor)")
    ax[0].plot(tr_df["V"], tr_df["I"], "C0-", label="Translated (as‑is)")
    ax[0].plot(neu_df["V"], neu_df["I"], "C1--", label="Neutralized")
    ax[0].set_xlabel("Voltage [V]"); ax[0].set_ylabel("Current [A]"); ax[0].grid(True)
    ax[0].legend(); ax[0].set_title("I‑V")
    # P‑V
    ax[1].plot(anchor_df["V"], anchor_df["V"] * anchor_df["I"], "k.", label="Measured")
    ax[1].plot(tr_df["V"], tr_df["V"] * tr_df["I"], "C0-", label="Translated")
    ax[1].plot(neu_df["V"], neu_df["V"] * neu_df["I"], "C1--", label="Neutralized")
    ax[1].set_xlabel("Voltage [V]"); ax[1].set_ylabel("Power [W]"); ax[1].grid(True)
    ax[1].legend(); ax[1].set_title("P‑V")
    fig.suptitle(title); fig.tight_layout()
    return fig

def kpi(curve_df):
    P = curve_df["V"] * curve_df["I"]
    idx = int(np.argmax(P.values))
    return {"Pmax": float(P.iloc[idx]), "Vmp": float(curve_df["V"].iloc[idx]), "Imp": float(curve_df["I"].iloc[idx])}

# =========================
# SIDEBAR UI
# =========================
st.sidebar.header("Data upload")
f_light = st.sidebar.file_uploader("Light I‑V CSV (V,I,G,T)", type=["csv"])
f_dark  = st.sidebar.file_uploader("Dark I‑V CSV (V,I)", type=["csv"])
f_suns  = st.sidebar.file_uploader("Suns‑Voc CSV (optional: G,T,Voc)", type=["csv"])
target_G = st.sidebar.number_input("Target irradiance G₂ [W/m²]", 200.0, 1400.0, 1000.0, 10.0)
target_T = st.sidebar.number_input("Target temperature T₂ [°C]", -20.0, 100.0, 25.0, 0.5)

fit_mode = st.sidebar.radio(
    "Fit mode",
    ["Fast initializer", "Precision Fit (MO)", "CUSP‑T (Morphology)"],
    index=0,
    help="Fast: quick single‑pass; MO: robust Light+Dark+Suns‑Voc fit; CUSP‑T: morphology‑preserving + non‑ohmic shunt."
)
if fit_mode == "Precision Fit (MO)" and not HAS_MO:
    st.sidebar.warning(
        "Multi‑objective add‑on not found. Falling back to Fast initializer. "
        "Drop 'damp_t/optim_mo.py' into your project or use damp_t_mo_addon.zip."
    )
    if fit_mode == "Precision Fit (MO)":
        fit_mode = "Fast initializer"

# MO weights (if active)
mo_kwargs = {}
if fit_mode == "Precision Fit (MO)" and HAS_MO:
    st.sidebar.subheader("MO weights & robust deltas")
    w_light = st.sidebar.slider("w_light (Light I‑V)", 0.1, 3.0, 1.0, 0.1)
    w_dark  = st.sidebar.slider("w_dark  (Dark I‑V)", 0.0, 3.0, 0.6, 0.1)
    w_suns  = st.sidebar.slider("w_suns  (Suns‑Voc)", 0.0, 3.0, 0.5, 0.1)
    dL = st.sidebar.slider("delta_light (A)", 0.05, 2.0, 0.5, 0.05)
    dD = st.sidebar.slider("delta_dark (A)", 0.02, 1.0, 0.2, 0.02)
    dS = st.sidebar.slider("delta_suns (V)", 0.02, 0.2, 0.08, 0.01)
    lam = st.sidebar.slider("lambda_prior", 0.0, 0.5, 0.05, 0.01)
    mo_kwargs["weights"] = MOWeights(
        w_light=w_light, w_dark=w_dark, w_suns=w_suns,
        delta_light=dL, delta_dark=dD, delta_suns=dS, lambda_prior=lam
    )

# CUSP‑T parameters (always visible)
with st.sidebar.expander("CUSP‑T (Morphology) parameters"):
    cuspt_alpha_I = st.number_input("α_I (Isc temp coeff) [1/°C]", value=0.0005, format="%.6f", key="cuspt_alpha")
    cuspt_beta_V  = st.number_input("β_V for Voc temperature shift [V/°C]", value=-0.08, format="%.3f", key="cuspt_beta")
    cuspt_rs_cap  = st.number_input("Rs cap (proxy upper bound) [Ω‑module‑proxy]", value=0.35, format="%.3f", key="cuspt_rs")
    cuspt_a_min   = st.number_input("a_sh min", value=1e-6, format="%.6f", key="cuspt_amin")
    cuspt_a_max   = st.number_input("a_sh max", value=5e-3, format="%.6f", key="cuspt_amax")
    cuspt_m_min   = st.number_input("m_sh min", value=1.1, format="%.2f", key="cuspt_mmin")
    cuspt_m_max   = st.number_input("m_sh max", value=2.0, format="%.2f", key="cuspt_mmax")

run = st.sidebar.button("Run Translation", type="primary")

# =========================
# MAIN RUN
# =========================
if run:
    if not (f_light and f_dark):
        st.error("Please upload both Light I‑V and Dark I‑V CSV files. Suns‑Voc is optional.")
        st.stop()

    try:
        df_light = load_csv(f_light, REQUIRED_LIGHT)
        df_dark  = load_csv(f_dark, REQUIRED_DARK)
        df_suns  = load_csv(f_suns, REQUIRED_SV) if f_suns else None
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    # 1) Fast initializer (always run as baseline)
    try:
        base = damp_t_pipeline(df_light, df_dark, df_suns, target_G, target_T)
    except Exception as e:
        st.exception(e)
        st.stop()

    # 2) Optional MO refinement
    using_mo = (fit_mode == "Precision Fit (MO)") and HAS_MO
    objective_info = None
    subs_opt = None
    if using_mo:
        try:
            sweeps = split_light_curves(df_light)
            anchor = sweeps[0]
            light_arr = {"V": anchor["V"].values, "I": anchor["I"].values, "G": anchor["G"].values, "T": anchor["T"].values}
            dark_arr  = {"V": df_dark["V"].values, "I": df_dark["I"].values}
            if df_suns is not None:
                suns_arr = {"G": df_suns["G"].values, "T": df_suns["T"].values, "Voc": df_suns["Voc"].values}
            else:
                idx0 = int(np.argmin(np.abs(anchor["I"].values)))
                Voc_est = float(anchor["V"].values[idx0])
                Gm, Tm = float(anchor["G"].median()), float(anchor["T"].median())
                Gs = np.array([Gm, 0.9*Gm, 0.8*Gm], dtype=float)
                Ts = np.full_like(Gs, Tm, dtype=float)
                Voc = np.full_like(Gs, Voc_est, dtype=float)
                suns_arr = {"G": Gs, "T": Ts, "Voc": Voc}

            res_mo = damp_t_pipeline_mo(light=light_arr, dark=dark_arr, suns=suns_arr,
                                        target_G=target_G, target_T=target_T, **mo_kwargs)
            translated = pd.DataFrame(res_mo["translated"])
            neutralized = pd.DataFrame(res_mo["neutralized"])
            objective_info = res_mo["objective_info"]
            subs_opt = [sp.__dict__ for sp in res_mo["subs_opt"]]
        except Exception as e:
            st.warning("MO refinement failed; showing Fast initializer results instead.")
            st.exception(e)
            using_mo = False

    # 3) CUSP‑T (Morphology)
    using_cuspt = (fit_mode == "CUSP‑T (Morphology)")
    cuspt_diag = None
    if using_cuspt:
        try:
            res_cuspt = cuspt_translate_inline(
                df_light, df_dark, df_suns,
                G2=target_G, T2=target_T,
                alpha_I=cuspt_alpha_I, beta_V=cuspt_beta_V,
                rs_cap=cuspt_rs_cap, a_bounds=(cuspt_a_min, cuspt_a_max),
                m_bounds=(cuspt_m_min, cuspt_m_max)
            )
            translated = res_cuspt["translated"]
            neutralized = res_cuspt["neutralized"]
            cuspt_diag = res_cuspt.get("diagnostics", None)
            objective_info = None  # not applicable for this prototype
        except Exception as e:
            st.warning("CUSP‑T failed; showing Fast initializer results instead.")
            st.exception(e)
            using_cuspt = False

    # 4) Choose what to display
    anchor_df = base["anchor"]
    if using_mo:
        tr_df = translated
        neu_df = neutralized
    elif using_cuspt:
        tr_df = translated
        neu_df = neutralized
    else:
        tr_df = base["translated"]
        neu_df = base["neutralized"]

    st.success("Translation complete." + (" (MO refined)" if using_mo else (" (CUSP‑T)" if using_cuspt else "")))
    fig = plot_results(anchor_df, tr_df, neu_df, title=f"Translation to (G={target_G:.0f} W/m², T={target_T:.1f} °C)")
    st.pyplot(fig)

    # KPIs (cards)
    c1, c2, c3 = st.columns(3)
    k_meas = kpi(anchor_df)
    k_tr   = kpi(tr_df)
    k_neu  = kpi(neu_df)

    with c1:
        st.subheader("Measured (anchor)")
        st.metric("Pmax [W]", f"{k_meas['Pmax']:.2f}")
        st.caption(f"Vmp: {k_meas['Vmp']:.2f} V  |  Imp: {k_meas['Imp']:.2f} A")

    with c2:
        tag = " · MO" if using_mo else (" · CUSP‑T" if using_cuspt else "")
        st.subheader(f"Translated (as‑is{tag})")
        st.metric("Pmax [W]", f"{k_tr['Pmax']:.2f}")
        st.caption(f"Vmp: {k_tr['Vmp']:.2f} V  |  Imp: {k_tr['Imp']:.2f} A")

    with c3:
        tag = " · MO" if using_mo else (" · CUSP‑T" if using_cuspt else "")
        st.subheader(f"Neutralized{tag}")
        st.metric("Pmax [W]", f"{k_neu['Pmax']:.2f}")
        st.caption(f"Vmp: {k_neu['Vmp']:.2f} V  |  Imp: {k_neu['Imp']:.2f} A")

    # Diagnostics JSON
    st.divider()
    st.subheader("Diagnostics & Parameters")
    diag = {
        "kinks_detected": base["kinks_detected"],
        "Rs_estimate": base["Rs_estimate"],
        "shunt_powerlaw": base["shunt_powerlaw"],
        "suns_voc_fit": base["suns_voc_fit"]
    }
    if objective_info is not None:
        diag["objective_info"] = objective_info
    if using_cuspt and cuspt_diag is not None:
        diag["cuspt_diagnostics"] = cuspt_diag
    st.json(diag)

    # Downloads
    def csv_button(label, df, fname):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=fname, mime="text/csv")

    st.subheader("Download corrected data")
    colA, colB = st.columns(2)
    with colA: csv_button("Translated (as‑is) CSV", tr_df, "translated_as_is.csv")
    with colB: csv_button("Neutralized CSV", neu_df, "translated_neutralized.csv")

    # Bundle everything
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("anchor_measured.csv", anchor_df.to_csv(index=False))
        z.writestr("translated_as_is.csv", tr_df.to_csv(index=False))
        z.writestr("translated_neutralized.csv", neu_df.to_csv(index=False))
        params = {
            "kinks_detected": base["kinks_detected"],
            "Rs_estimate": base["Rs_estimate"],
            "shunt_powerlaw": base["shunt_powerlaw"],
            "suns_voc_fit": base["suns_voc_fit"],
            "subs": (subs_opt if (using_mo and objective_info is not None) else base["subs"])
        }
        if objective_info is not None:
            params["objective_info"] = objective_info
        if using_cuspt and cuspt_diag is not None:
            params["cuspt_diagnostics"] = cuspt_diag
        z.writestr("damp_t_params.json", json.dumps(params, indent=2))
    st.download_button("Download all (ZIP)", data=buf.getvalue(), file_name="damp_t_results_bundle.zip", mime="application/zip")

# =========================
# HELP / NOTES
# =========================
with st.expander("CSV schemas and tips"):
    st.markdown("""
**Light I‑V CSV:** `V, I, G, T`  
- Voltage [V], Current [A], Irradiance G [W/m²], Module temperature T [°C].  
- You can concatenate multiple sweeps; the app splits them by voltage wrap (first sweep is the anchor).

**Dark I‑V CSV:** `V, I`  
- Include ±(1–2) V around 0 V for non‑ohmic shunt fit; include a forward high‑current region for \(R_s\).

**Suns‑Voc CSV (optional):** `G, T, Voc`  
- If missing, MO synthesizes a small grid; CUSP‑T uses a mild ln(G) fallback but real Suns‑Voc improves \(V_{oc}\) accuracy.
  
**Outputs:**  
- **Translated (as‑is):** keeps bypass, non‑ohmic shunt, and degradation state.  
- **Neutralized:** bypass=off, inactive area=0, and shunt suppressed to a high‑Rsh limit (for “what‑if”).
""")
