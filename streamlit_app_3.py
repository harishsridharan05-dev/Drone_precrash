"""
DroneGuard — Ag Drone Pre-Crash Detection
==========================================
Run with:  streamlit run streamlit_app.py
"""

import os, io, gc, json, zipfile, subprocess, tempfile, warnings, time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DroneGuard — Pre-Crash Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0b0e14; }
  [data-testid="stHeader"]           { background: transparent; }
  section[data-testid="stSidebar"]   { display: none; }
  h1,h2,h3,h4,p,label,div           { color: #e8edf5 !important; }
  .block-container                   { padding-top: 2rem; }
  .stFileUploader > div              { background: #131820 !important; border: 2px dashed #1e2735 !important; border-radius: 12px; }
  .stSlider > div > div > div       { background: #00d4a0 !important; }
  .metric-card {
    background: #131820;
    border: 1px solid #1e2735;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-label { font-size: 11px; color: #6b7a96 !important; text-transform: uppercase; letter-spacing: 1px; font-family: monospace; }
  .metric-value { font-size: 36px; font-weight: 800; font-family: monospace; line-height: 1.1; margin-top: 4px; }
  .metric-sub   { font-size: 11px; color: #6b7a96 !important; margin-top: 4px; }
  .verdict-danger { background: rgba(255,71,87,0.12); border: 1px solid rgba(255,71,87,0.4); border-radius: 30px; padding: 10px 28px; color: #ff4757 !important; font-size: 16px; font-weight: 700; display: inline-block; }
  .verdict-safe   { background: rgba(46,213,115,0.1); border: 1px solid rgba(46,213,115,0.4); border-radius: 30px; padding: 10px 28px; color: #2ed573 !important; font-size: 16px; font-weight: 700; display: inline-block; }
  .warn-card { background: #131820; border: 1px solid #1e2735; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
  .warn-critical { border-left: 3px solid #ff4757 !important; }
  .warn-warning  { border-left: 3px solid #ffa502 !important; }
  .badge-critical { background: rgba(255,71,87,0.15); color: #ff4757; font-size: 10px; font-family: monospace; font-weight: 700; padding: 2px 8px; border-radius: 4px; letter-spacing: 1px; }
  .badge-warning  { background: rgba(255,165,2,0.15);  color: #ffa502; font-size: 10px; font-family: monospace; font-weight: 700; padding: 2px 8px; border-radius: 4px; letter-spacing: 1px; }
  .stButton > button { background: #00d4a0 !important; color: #000 !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important; padding: 12px 32px !important; font-size: 15px !important; width: 100%; }
  .stButton > button:hover { background: #00a87e !important; }
  div[data-testid="stFileUploadDropzone"] p { color: #6b7a96 !important; }
  .stAlert { background: #131820 !important; border-radius: 10px !important; }
  hr { border-color: #1e2735 !important; }

  /* ── RTL Warning Overlay ─────────────────────────────────── */
  #rtl-overlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 999999;
    background: rgba(0, 0, 0, 0.82);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    animation: fadeIn 0.3s ease;
  }
  #rtl-overlay.visible { display: flex; align-items: center; justify-content: center; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  #rtl-modal {
    background: #0f1318;
    border: 2px solid #ff4757;
    border-radius: 20px;
    padding: 48px 52px;
    max-width: 540px;
    width: 90vw;
    text-align: center;
    box-shadow: 0 0 80px rgba(255, 71, 87, 0.35), 0 24px 64px rgba(0,0,0,0.7);
    animation: popIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
  }
  @keyframes popIn {
    from { transform: scale(0.82); opacity: 0; }
    to   { transform: scale(1);    opacity: 1; }
  }

  #rtl-pulse-ring {
    width: 88px; height: 88px;
    border-radius: 50%;
    background: rgba(255,71,87,0.12);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 24px;
    position: relative;
  }
  #rtl-pulse-ring::before {
    content: '';
    position: absolute; inset: -8px;
    border-radius: 50%;
    border: 2px solid rgba(255,71,87,0.4);
    animation: pulseRing 1.6s ease-out infinite;
  }
  #rtl-pulse-ring::after {
    content: '';
    position: absolute; inset: -18px;
    border-radius: 50%;
    border: 2px solid rgba(255,71,87,0.2);
    animation: pulseRing 1.6s ease-out 0.4s infinite;
  }
  @keyframes pulseRing {
    0%   { transform: scale(0.9); opacity: 1; }
    100% { transform: scale(1.4); opacity: 0; }
  }

  #rtl-icon { font-size: 40px; line-height: 1; }

  #rtl-title {
    font-size: 28px;
    font-weight: 800;
    color: #ff4757 !important;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
    font-family: monospace;
    text-transform: uppercase;
  }
  #rtl-subtitle {
    font-size: 13px;
    color: #6b7a96 !important;
    font-family: monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 28px;
  }
  #rtl-command {
    background: rgba(255,71,87,0.08);
    border: 1px solid rgba(255,71,87,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 28px;
  }
  #rtl-command-label {
    font-size: 10px;
    color: #6b7a96 !important;
    font-family: monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  #rtl-command-text {
    font-size: 22px;
    font-weight: 800;
    color: #ff4757 !important;
    font-family: monospace;
    letter-spacing: 2px;
  }
  #rtl-crash-type {
    font-size: 12px;
    color: #ffa502 !important;
    font-family: monospace;
    background: rgba(255,165,2,0.08);
    border: 1px solid rgba(255,165,2,0.25);
    border-radius: 8px;
    padding: 8px 16px;
    display: inline-block;
    margin-bottom: 28px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  #rtl-dismiss {
    background: #ff4757;
    color: #fff !important;
    border: none;
    border-radius: 10px;
    padding: 13px 36px;
    font-size: 14px;
    font-weight: 700;
    font-family: monospace;
    letter-spacing: 1px;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    text-transform: uppercase;
    width: 100%;
  }
  #rtl-dismiss:hover { background: #e03545; transform: translateY(-1px); }
  #rtl-dismiss:active { transform: translateY(0); }
  #rtl-timer {
    font-size: 11px;
    color: #6b7a96 !important;
    font-family: monospace;
    margin-top: 14px;
  }
  #rtl-countdown {
    color: #ffa502 !important;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)

# ── RTL Warning Popup ─────────────────────────────────────────────────────────
def show_rtl_popup(crash_type: str, secs_before: float):
    """Full-screen RTL warning popup — self-contained in one iframe."""
    components.html(f"""<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{
  width:100%; height:100%;
  background: rgba(0,0,0,0.88);
  display:flex; align-items:center; justify-content:center;
  font-family: monospace;
}}
#modal {{
  background: #0f1318;
  border: 2px solid #ff4757;
  border-radius: 20px;
  padding: 44px 48px;
  max-width: 520px;
  width: 88vw;
  text-align: center;
  box-shadow: 0 0 80px rgba(255,71,87,0.45), 0 24px 64px rgba(0,0,0,0.9);
  animation: popIn 0.35s cubic-bezier(0.34,1.56,0.64,1);
}}
@keyframes popIn {{
  from {{ transform:scale(0.8); opacity:0; }}
  to   {{ transform:scale(1);   opacity:1; }}
}}
.ring {{
  width:88px; height:88px; border-radius:50%;
  background:rgba(255,71,87,0.12);
  display:flex; align-items:center; justify-content:center;
  margin:0 auto 22px; position:relative;
}}
.ring::before {{
  content:''; position:absolute; inset:-8px; border-radius:50%;
  border:2px solid rgba(255,71,87,0.45);
  animation:pulse 1.6s ease-out infinite;
}}
.ring::after {{
  content:''; position:absolute; inset:-20px; border-radius:50%;
  border:2px solid rgba(255,71,87,0.2);
  animation:pulse 1.6s ease-out 0.5s infinite;
}}
@keyframes pulse {{
  0%   {{ transform:scale(0.9); opacity:1; }}
  100% {{ transform:scale(1.5); opacity:0; }}
}}
.icon  {{ font-size:38px; line-height:1; }}
.title {{ font-size:26px; font-weight:800; color:#ff4757;
          letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; }}
.sub   {{ font-size:11px; color:#6b7a96;
          letter-spacing:2px; text-transform:uppercase; margin-bottom:24px; }}
.cmd   {{ background:rgba(255,71,87,0.08); border:1px solid rgba(255,71,87,0.35);
          border-radius:12px; padding:18px 22px; margin-bottom:20px; }}
.cmd-lbl {{ font-size:10px; color:#6b7a96;
            letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }}
.cmd-txt {{ font-size:22px; font-weight:800; color:#ff4757; letter-spacing:3px; }}
.badge {{ font-size:12px; color:#ffa502;
          background:rgba(255,165,2,0.08); border:1px solid rgba(255,165,2,0.3);
          border-radius:8px; padding:7px 16px; display:inline-block;
          margin-bottom:26px; text-transform:uppercase; letter-spacing:1px; }}
.btn   {{ background:#ff4757; color:#fff; border:none;
          border-radius:10px; padding:14px 36px; font-size:14px;
          font-weight:700; font-family:monospace; letter-spacing:1px;
          cursor:pointer; width:100%; text-transform:uppercase; }}
.btn:hover {{ background:#e03545; }}
.timer {{ font-size:11px; color:#6b7a96; margin-top:14px; }}
.cd    {{ color:#ffa502; font-weight:700; }}
</style>
</head>
<body>
<div id="modal">
  <div class="ring"><div class="icon">&#128680;</div></div>
  <div class="title">PRE-CRASH DETECTED</div>
  <div class="sub">Immediate action required</div>
  <div class="cmd">
    <div class="cmd-lbl">Trigger Command</div>
    <div class="cmd-txt">&#9664; RETURN TO LAUNCH</div>
  </div>
  <div class="badge">&#9888; Predicted: {crash_type}</div>
  <br><br>
  <button class="btn" onclick="dismiss()">&#10003; &nbsp; Acknowledged &mdash; Initiating RTL</button>
  <div class="timer">Auto-dismiss in <span class="cd" id="cd">15</span>s</div>
</div>
<script>
  var secs = 15;
  var iv = setInterval(function() {{
    secs--;
    var el = document.getElementById('cd');
    if (el) el.textContent = secs;
    if (secs <= 0) dismiss();
  }}, 1000);
  function dismiss() {{
    clearInterval(iv);
    document.body.style.transition = 'opacity 0.3s';
    document.body.style.opacity = '0';
    setTimeout(function() {{
      document.body.innerHTML = '';
      document.body.style.background = 'transparent';
      // Hide the iframe in the parent page
      try {{
        var iframes = window.parent.document.querySelectorAll('iframe');
        for (var i = 0; i < iframes.length; i++) {{
          try {{
            if (iframes[i].contentWindow === window) {{
              iframes[i].style.display = 'none';
              break;
            }}
          }} catch(e) {{}}
        }}
      }} catch(e) {{}}
    }}, 350);
  }}
</script>
</body>
</html>""", height=800, scrolling=False)
    # Push this iframe to cover the full screen using CSS on the parent
    st.markdown("""
<style>
section[data-testid="stMain"] > div > div > div > div:last-child iframe {
  position: fixed !important;
  top: 0 !important; left: 0 !important;
  width: 100vw !important; height: 100vh !important;
  z-index: 2147483647 !important;
  border: none !important;
}
</style>""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    def _load(name):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
            return None

    hgb_binary    = _load("hgb_binary.pkl")
    scaler_bin    = _load("scaler_bin.pkl")
    hgb_mc        = _load("hgb_mc.pkl")
    scaler_mc     = _load("scaler_mc.pkl")
    label_encoder = _load("label_encoder.pkl")

    feat_path = os.path.join(MODELS_DIR, "feature_cols_v3.json")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [
            "roll_abs","pitch_abs","errRP","climb","hdop","volt","vibe_total",
            "roll_rate","pitch_rate","yaw_rate","errRP_rate","volt_rate","hdop_rate",
            "roll_std_5s","pitch_std_5s","errRP_std_5s","vibe_std_5s",
            "roll_mean_5s","pitch_mean_5s","climb_std_5s",
            "roll_std_10s","pitch_std_10s",
            "rcou_mean","rcou_asymmetry","rcou_max_dev",
            "roll_bias_persistence","errRP_cumulative",
        ]

    loaded = all(x is not None for x in [hgb_binary, scaler_bin, hgb_mc, scaler_mc, label_encoder])
    return hgb_binary, scaler_bin, hgb_mc, scaler_mc, label_encoder, feature_cols, loaded

hgb_binary, scaler_bin, hgb_mc, scaler_mc, label_encoder, FEATURE_COLS, MODELS_LOADED = load_models()

# ── Parameter warnings ────────────────────────────────────────────────────────
PARAM_WARNINGS = {
    "roll_abs":              {"threshold": 8.0,   "direction": "above",     "message": "Roll angle {val:.1f}° (normal <5°) — asymmetric lift or wind"},
    "pitch_abs":             {"threshold": 8.0,   "direction": "above",     "message": "Pitch angle {val:.1f}° (normal <5°) — longitudinal instability"},
    "errRP":                 {"threshold": 0.15,  "direction": "above",     "message": "Attitude control error ErrRP={val:.3f} (normal <0.05) — FC struggling"},
    "vibe_total":            {"threshold": 15.0,  "direction": "above",     "message": "Vibration {val:.1f} m/s² (normal <10) — motor/prop damage or loose mount"},
    "roll_std_5s":           {"threshold": 4.0,   "direction": "above",     "message": "Roll oscillation std {val:.2f}° /5s (normal <2°) — turbulence building"},
    "pitch_std_5s":          {"threshold": 4.0,   "direction": "above",     "message": "Pitch oscillation std {val:.2f}° /5s — longitudinal instability"},
    "rcou_asymmetry":        {"threshold": 40.0,  "direction": "above",     "message": "Motor asymmetry {val:.0f}µs std (normal <20µs) — ESC/motor failure"},
    "rcou_max_dev":          {"threshold": 120.0, "direction": "above",     "message": "Motor spread {val:.0f}µs max-min (normal <60µs) — one motor overloaded"},
    "hdop":                  {"threshold": 3.0,   "direction": "above",     "message": "GPS HDOP {val:.2f} (good <1.5) — position estimate unreliable"},
    "hdop_rate":             {"threshold": 0.5,   "direction": "above",     "message": "GPS HDOP changing {val:.2f}/s — multipath or interference"},
    "volt":                  {"threshold": 10.5,  "direction": "below",     "message": "Battery {val:.2f}V (critical <10.5V) — imminent power loss"},
    "volt_rate":             {"threshold": -0.1,  "direction": "below",     "message": "Voltage dropping {val:.3f}V/step — high draw or battery failure"},
    "roll_bias_persistence": {"threshold": 5.0,   "direction": "abs_above", "message": "Persistent roll bias {val:.1f}° — payload shift or motor imbalance"},
    "errRP_cumulative":      {"threshold": 0.5,   "direction": "above",     "message": "Cumulative attitude error {val:.2f} — sustained control struggle"},
}

def sev(param, val):
    crit = {"roll_abs":20,"pitch_abs":20,"errRP":0.4,"vibe_total":40,
            "roll_std_5s":10,"pitch_std_5s":10,"rcou_asymmetry":100,"rcou_max_dev":250,
            "hdop":6,"roll_bias_persistence":12}
    if param in crit:
        t = crit[param]
        return "CRITICAL" if (abs(val)>t if "bias" in param else val>t) else "WARNING"
    return "CRITICAL" if param=="volt" else "WARNING"

def get_param_warnings(feat_series):
    out = []
    for param, cfg in PARAM_WARNINGS.items():
        if param not in feat_series.index: continue
        val = float(feat_series[param])
        triggered = (
            (cfg["direction"]=="above"     and val > cfg["threshold"]) or
            (cfg["direction"]=="below"     and val < cfg["threshold"]) or
            (cfg["direction"]=="abs_above" and abs(val) > cfg["threshold"])
        )
        if triggered:
            out.append({"param":param,"value":round(val,4),
                        "severity":sev(param,val),
                        "message":cfg["message"].format(val=val)})
    return sorted(out, key=lambda x: 0 if x["severity"]=="CRITICAL" else 1)

# ── Feature engineering ───────────────────────────────────────────────────────
def compute_features(df):
    f = pd.DataFrame(index=df.index)
    roll  = df.get("roll",  pd.Series(0.0, index=df.index))
    pitch = df.get("pitch", pd.Series(0.0, index=df.index))
    yaw   = df.get("yaw",   pd.Series(0.0, index=df.index))
    errRP = df.get("errRP", pd.Series(0.0, index=df.index)).fillna(0)
    climb = df.get("climb", pd.Series(0.0, index=df.index)).fillna(0)
    hdop  = df.get("hdop",  pd.Series(1.2, index=df.index)).fillna(1.2).clip(0,20)
    volt  = df.get("volt",  pd.Series(12.0,index=df.index)).fillna(12.0).clip(5,63)
    vx    = df.get("vx",    pd.Series(0.0, index=df.index)).fillna(0)
    vy    = df.get("vy",    pd.Series(0.0, index=df.index)).fillna(0)
    vz    = df.get("vz",    pd.Series(0.0, index=df.index)).fillna(0)
    clip_ = df.get("clip",  pd.Series(0.0, index=df.index)).fillna(0)

    rcou_cols   = [df.get(f"c{i}", pd.Series(1000.0,index=df.index)).fillna(1000) for i in range(1,7)]
    rcou_stack  = pd.concat(rcou_cols,axis=1); rcou_stack.columns=[f"c{i}" for i in range(1,7)]
    # Treat disarmed idle (all motors = same low value, typically 1050) as NaN
    # so post-crash rows don't collapse rcou_asymmetry to 0
    all_same = rcou_stack.std(axis=1) < 1
    rcou_stack_clean = rcou_stack.copy()
    rcou_stack_clean[all_same & (rcou_stack.max(axis=1) < 1100)] = np.nan
    rcou_active = rcou_stack_clean.where(rcou_stack_clean>1100)

    f["roll_abs"]  = roll.abs(); f["pitch_abs"] = pitch.abs()
    f["errRP"]     = errRP;      f["climb"]     = climb
    f["hdop"]      = hdop;       f["volt"]      = volt
    f["vibe_total"]= np.sqrt(vx**2+vy**2+vz**2)+clip_*10

    f["roll_rate"] = roll.diff().abs().fillna(0);  f["pitch_rate"]= pitch.diff().abs().fillna(0)
    f["yaw_rate"]  = yaw.diff().abs().fillna(0);   f["errRP_rate"]= errRP.diff().abs().fillna(0)
    f["volt_rate"] = volt.diff().fillna(0);         f["hdop_rate"] = hdop.diff().abs().fillna(0)

    t_max = max(df.get("t",pd.Series([len(df)])).max(),1)
    W5  = max(3, min(50,  int(len(df)*5 /t_max)))
    W10 = max(5, min(100, int(len(df)*10/t_max)))

    f["roll_std_5s"]  = roll.rolling(W5, min_periods=3).std().fillna(0)
    f["pitch_std_5s"] = pitch.rolling(W5,min_periods=3).std().fillna(0)
    f["errRP_std_5s"] = errRP.rolling(W5,min_periods=3).std().fillna(0)
    f["vibe_std_5s"]  = f["vibe_total"].rolling(W5,min_periods=3).std().fillna(0)
    f["roll_mean_5s"] = roll.abs().rolling(W5,min_periods=3).mean().fillna(0)
    f["pitch_mean_5s"]= pitch.abs().rolling(W5,min_periods=3).mean().fillna(0)
    f["climb_std_5s"] = climb.rolling(W5,min_periods=3).std().fillna(0)
    f["roll_std_10s"] = roll.rolling(W10,min_periods=5).std().fillna(0)
    f["pitch_std_10s"]= pitch.rolling(W10,min_periods=5).std().fillna(0)

    f["rcou_mean"]      = rcou_active.mean(axis=1).fillna(1100)
    f["rcou_asymmetry"] = rcou_active.std(axis=1).fillna(0)
    f["rcou_max_dev"]   = (rcou_active.max(axis=1)-rcou_active.min(axis=1)).fillna(0)
    f["roll_bias_persistence"] = roll.rolling(W10,min_periods=5).mean().fillna(0)
    f["errRP_cumulative"]      = errRP.rolling(W5, min_periods=3).sum().fillna(0)
    return f

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_bin(fpath):
    from pymavlink import DFReader
    log  = DFReader.DFReader_binary(fpath, zero_time_base=True)
    rows = {k:[] for k in ["ATT","BAT","GPS","VIBE","BARO","RCOU"]}
    WANTED = set(rows.keys())
    try:
        while True:
            msg = log.recv_msg()
            if msg is None: break
            mt = msg.get_type()
            if mt not in WANTED: continue
            t = msg.TimeUS/1e6
            if mt=="ATT":
                rows["ATT"].append({"t":t,"roll":msg.Roll,"pitch":msg.Pitch,"yaw":msg.Yaw,
                                     "errRP":msg.ErrRP,"errYaw":msg.ErrYaw})
            elif mt=="BAT":
                rows["BAT"].append({"t":t,"volt":msg.Volt,"curr":msg.Curr})
            elif mt=="GPS":
                rows["GPS"].append({"t":t,"hdop":msg.HDop,"nsats":msg.NSats})
            elif mt=="VIBE":
                rows["VIBE"].append({"t":t,"vx":msg.VibeX,"vy":msg.VibeY,"vz":msg.VibeZ,"clip":msg.Clip})
            elif mt=="BARO":
                rows["BARO"].append({"t":t,"alt":msg.Alt,"climb":msg.CRt})
            elif mt=="RCOU":
                r={"t":t}
                for i in range(1,7): r[f"c{i}"]=getattr(msg,f"C{i}",1000)
                rows["RCOU"].append(r)
    finally:
        try:
            if hasattr(log,'filehandle') and log.filehandle: log.filehandle.close()
            elif hasattr(log,'fp') and log.fp: log.fp.close()
        except: pass
        del log; gc.collect()

    dfs = {k:pd.DataFrame(v) for k,v in rows.items() if v}
    if "ATT" not in dfs or len(dfs["ATT"])==0:
        raise ValueError("No ATT messages found in log")

    t0 = dfs["ATT"]["t"].min()
    for df in dfs.values(): df["t"]=df["t"]-t0

    att = dfs["ATT"].sort_values("t").set_index("t")
    att = att[~att.index.duplicated(keep="last")]

    def resample(df, cols):
        if df is None or len(df)==0: return None
        df=df.sort_values("t").set_index("t"); df=df[~df.index.duplicated(keep="last")]
        cols=[c for c in cols if c in df.columns]
        if not cols: return None
        return df[cols].reindex(att.index,method="nearest",tolerance=0.5)

    base = att.copy()
    for key,cols in [("BAT",["volt","curr"]),("GPS",["hdop","nsats"]),
                     ("VIBE",["vx","vy","vz","clip"]),("BARO",["alt","climb"]),
                     ("RCOU",["c1","c2","c3","c4","c5","c6"])]:
        if key in dfs:
            r=resample(dfs[key],cols)
            if r is not None: base=base.join(r,how="left")

    base=base.ffill().bfill()
    if "volt" in base.columns:
        # Support S3 (11V) through S12 (50V) packs. Values of 1.0 are "no data" placeholders.
        base.loc[base["volt"]<3.0,"volt"]=np.nan; base.loc[base["volt"]>63,"volt"]=np.nan
        base["volt"]=base["volt"].ffill().bfill().fillna(12.0)
    base.reset_index(inplace=True)
    if "index" in base.columns: base.rename(columns={"index":"t"},inplace=True)
    return base

def parse_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        bins=[n for n in z.namelist() if n.lower().endswith(".bin")]
        if bins:
            tmp=tempfile.mktemp(suffix=".bin")
            with z.open(bins[0]) as src, open(tmp,"wb") as dst: dst.write(src.read())
            try: return parse_bin(tmp)
            finally:
                try: os.remove(tmp)
                except: pass
    raise ValueError("No .bin file found inside ZIP")

# ── Inference ─────────────────────────────────────────────────────────────────
def _multi_sensor_confirmed(feat_df):
    """
    Per-row boolean: True only when 3+ distinct sensor categories are
    simultaneously in violation. Kills noise from a single noisy sensor
    (rcou_asymmetry firing alone T=0-5s) and only passes rows where
    multiple independent signals agree.

    Categories (each counts as 1):
      A - motor outputs  : rcou_asymmetry > 40  OR  rcou_max_dev > 120
      B - attitude error : errRP > 0.15          OR  errRP_cumulative > 8
      C - vibration      : vibe_total > 15        OR  vibe_std_5s > 5
      D - roll/pitch     : roll_abs > 8           OR  pitch_abs > 8
      E - roll instab.   : roll_std_5s > 4        OR  roll_std_10s > 4
    """
    A = (feat_df["rcou_asymmetry"] > 40) | (feat_df["rcou_max_dev"] > 120)
    B = (feat_df["errRP"] > 0.15)        | (feat_df["errRP_cumulative"] > 8)
    C = (feat_df["vibe_total"] > 15)     | (feat_df.get("vibe_std_5s", pd.Series(0, index=feat_df.index)) > 5)
    D = (feat_df["roll_abs"] > 8)        | (feat_df["pitch_abs"] > 8)
    E = (feat_df["roll_std_5s"] > 4)     | (feat_df["roll_std_10s"] > 4)
    n_categories = A.astype(int) + B.astype(int) + C.astype(int) + D.astype(int) + E.astype(int)
    return n_categories >= 3


def _late_flight_gate(risk_s, raw_df, early_fraction=0.70):
    """
    Suppress alerts that fire in the first `early_fraction` of the flight
    UNLESS the risk score is extremely high (>0.85) — which would indicate
    a genuine very-early catastrophic failure.

    For a 43s flight:
      - Rows in the first 70% (≈ first 30 rows) are suppressed unless risk > 0.85
      - Only the last 30% (≈ rows 30–43) can trigger a normal alert

    This is the primary fix for the "alert fires at second 5 of a 43s flight"
    problem: even if the model assigns high risk to early rows (due to training
    artefacts), those rows are gated out.
    """
    n = len(risk_s)
    gate = np.ones(n, dtype=bool)  # True = alert allowed

    early_cutoff = int(n * early_fraction)
    for i in range(early_cutoff):
        if risk_s[i] <= 0.85:   # suppress unless catastrophically high
            gate[i] = False

    return gate


def _rule_based_crash_risk(feat_df):
    """
    Hard-rule fallback risk signal for unambiguous crash signatures
    that the model may miss (e.g. distribution mismatch, high-voltage packs).

    Returns a per-row risk array in [0, 1]. Values > 0.60 are treated as
    high-confidence crash indicators and will override the model gate logic.

    Rules (each contributes to a severity score, capped at 1.0):
      +0.50  roll_abs > 45° for >3 consecutive rows  (definite flip/tumble)
      +0.30  vibe_total > 30 m/s²                    (severe vibration)
      +0.25  rcou_asymmetry > 200µs                  (massive motor imbalance)
      +0.25  roll_std_5s > 30°                       (violent angular instability)
      +0.20  errRP > 0.20                            (EKF attitude error high)
    """
    roll = feat_df["roll_abs"]
    vibe = feat_df["vibe_total"]
    rcou = feat_df["rcou_asymmetry"]
    rstd = feat_df["roll_std_5s"]
    errp = feat_df["errRP"]

    flip_mask = (roll > 45).astype(int).rolling(3, min_periods=1).sum() >= 3

    score = (
        flip_mask.astype(float) * 0.50 +
        (vibe > 30).astype(float) * 0.30 +
        (rcou > 200).astype(float) * 0.25 +
        (rstd > 30).astype(float) * 0.25 +
        (errp > 0.20).astype(float) * 0.20
    ).clip(0, 1.0)

    return score.values


def predict_flight(raw_df, threshold=0.60):
    feat_df = compute_features(raw_df)
    X = np.zeros((len(feat_df), len(FEATURE_COLS)))
    for i,col in enumerate(FEATURE_COLS):
        if col in feat_df.columns: X[:,i]=feat_df[col].fillna(0).values

    X_s    = scaler_bin.transform(X)
    risk   = hgb_binary.predict_proba(X_s)[:,1]

    # Blend model risk with hard-rule risk (take the max of both).
    # This ensures definite crashes (roll>45°, severe vibe) are caught even
    # when the model's training distribution doesn't match this log's scale.
    rule_risk = _rule_based_crash_risk(feat_df)
    risk      = np.maximum(risk, rule_risk)

    risk_s = pd.Series(risk).rolling(5,min_periods=1).mean().values

    # Gate 1: multi-sensor confirmation (3+ independent sensor categories)
    confirmed = _multi_sensor_confirmed(feat_df).values

    # Gate 2: late-flight gate — only allow alerts in last 30% of flight
    # (or if risk is extremely high >0.85, allow earlier)
    late_gate = _late_flight_gate(risk_s, raw_df, early_fraction=0.70)

    # Alert fires only when: model score > threshold AND multi-sensor confirmed
    # AND we are in the late portion of the flight
    alert = (risk_s > threshold) & confirmed & late_gate

    X_mc_s   = scaler_mc.transform(X)
    mc_probs = hgb_mc.predict_proba(X_mc_s)
    mc_idx   = mc_probs.argmax(axis=1)

    first_alert = int(np.where(alert)[0][0]) if alert.any() else -1

    # Peak risk: look only in the late-gate region for the display peak
    risk_s_gated = risk_s.copy()
    risk_s_gated[~late_gate] *= 0.3   # dampen early-flight peaks for display
    peak_idx    = int(np.argmax(risk_s_gated))

    crash_type  = label_encoder.inverse_transform([mc_idx[peak_idx]])[0]
    confidence  = float(mc_probs[peak_idx].max())
    top3_idx    = np.argsort(mc_probs[peak_idx])[-3:][::-1]
    top3        = [(label_encoder.classes_[i],round(float(mc_probs[peak_idx,i]),3)) for i in top3_idx]
    param_warnings = get_param_warnings(feat_df.iloc[peak_idx])

    t_col    = raw_df.get("t", pd.Series(range(len(raw_df))))
    duration = float(t_col.max()) if len(t_col) > 0 else 0

    # alert_seconds: seconds from first alert to end of flight (in time, not rows)
    if first_alert >= 0 and "t" in raw_df.columns:
        t_first_alert = float(raw_df["t"].iloc[first_alert])
        alert_secs    = round(duration - t_first_alert, 1)
    elif first_alert >= 0:
        alert_secs = len(risk) - first_alert
    else:
        alert_secs = 0

    rs = risk_s.tolist()
    if len(rs) > 500:
        step = max(1, len(rs) // 500)
        rs   = rs[::step]

    return {"risk_series":    rs,
            "peak_risk":      float(risk_s.max()),
            "alert":          bool(alert.any()),
            "first_alert":    first_alert,
            "alert_seconds":  alert_secs,
            "crash_type":     crash_type,
            "confidence":     confidence,
            "top3":           top3,
            "warnings":       param_warnings,
            "total_rows":     len(raw_df),
            "duration":       round(duration, 1),
            "threshold":      threshold,
            "peak_idx":       peak_idx}

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;padding-bottom:24px;border-bottom:1px solid #1e2735;margin-bottom:32px;">
  <div style="width:44px;height:44px;background:#00d4a0;border-radius:10px;display:grid;place-items:center;flex-shrink:0;">
    <span style="font-size:22px;">🛸</span>
  </div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#e8edf5;">DroneGuard</div>
    <div style="font-size:13px;color:#6b7a96;">Ag Drone Pre-Crash Detection — ArduPilot Log Analysis</div>
  </div>
  <div style="margin-left:auto;font-family:monospace;font-size:11px;color:#00d4a0;border:1px solid #00d4a0;padding:3px 10px;border-radius:20px;">v3.0</div>
</div>
""", unsafe_allow_html=True)

# Model status
if not MODELS_LOADED:
    st.error(f"⚠️ Models not found in `{MODELS_DIR}`. Copy your 5 `.pkl` files and `feature_cols_v3.json` there and restart.")
    st.stop()

# Upload + threshold
col_upload, col_thresh = st.columns([3,1])
with col_upload:
    uploaded = st.file_uploader(
        "Drop your flight log here",
        type=["bin","log","zip"],
        help="ArduPilot .bin, text .log, or .zip containing a .bin"
    )
with col_thresh:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    threshold = st.slider("Alert Threshold", 0.30, 0.90, 0.60, 0.05,
                          help="Risk score above this triggers a crash alert")

analyse = st.button("🔍 Analyse Flight", disabled=(uploaded is None))

if analyse and uploaded:
    ext = os.path.splitext(uploaded.name)[1].lower()

    with st.spinner(f"Parsing `{uploaded.name}` — large files may take 20–30s…"):
        try:
            # Save upload to temp file
            tmp = tempfile.mktemp(suffix=ext)
            with open(tmp,"wb") as f: f.write(uploaded.read())

            try:
                if   ext==".bin": raw_df = parse_bin(tmp)
                elif ext==".log":
                    try:    raw_df = parse_bin(tmp)
                    except: raise ValueError("Could not parse .log")
                elif ext==".zip": raw_df = parse_zip(tmp)
                else: raise ValueError(f"Unsupported: {ext}")
            finally:
                gc.collect()
                try: os.remove(tmp)
                except: pass

            if len(raw_df)<10:
                st.error("Log too short — fewer than 10 telemetry rows parsed.")
                st.stop()

            result = predict_flight(raw_df, threshold)
            st.session_state["raw_df"]  = raw_df
            st.session_state["feat_df"] = compute_features(raw_df)
            st.session_state["result"]  = result

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Verdict ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if result["alert"]:
        st.markdown(f'<div class="verdict-danger">⚠ CRASH RISK DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-safe">✓ FLIGHT NORMAL</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='color:#6b7a96;font-size:12px;margin-top:8px;'>{uploaded.name}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Stat cards ────────────────────────────────────────────────────────────
    risk_pct = result["peak_risk"]*100
    risk_color = "#ff4757" if risk_pct>70 else "#ffa502" if risk_pct>45 else "#2ed573"
    warn_color = "#ffa502" if result["alert"] else "#2ed573"
    warn_val   = f"{result['alert_seconds']}s" if result["alert"] else "none"

    c1,c2,c3,c4 = st.columns(4)
    for col, label, value, color, sub in [
        (c1,"Peak Risk",        f"{risk_pct:.0f}%",                     risk_color, "probability of pre-crash state"),
        (c2,"Flight Duration",  f"{result['duration']}s",               "#00d4a0",  "seconds in log"),
        (c3,"Alert Warning",    warn_val,                                warn_color, "seconds before predicted crash"),
        (c4,"Log Rows",         f"{result['total_rows']:,}",            "#00d4a0",  "telemetry samples parsed"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{color}">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Risk chart ────────────────────────────────────────────────────────────
    rs = result["risk_series"]
    xs = list(range(len(rs)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=rs, mode="lines", name="Risk Score",
        line=dict(color="#00d4a0", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,160,0.12)"
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ff4757",
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_position="top right",
                  annotation_font_color="#ff4757")
    if result["first_alert"] >= 0:
        alert_x = result["first_alert"] * len(rs) // result["total_rows"]
        fig.add_vline(x=alert_x, line_dash="dot", line_color="#ffa502",
                      annotation_text="first alert", annotation_font_color="#ffa502")
    fig.update_layout(
        paper_bgcolor="#131820", plot_bgcolor="#131820",
        margin=dict(l=48,r=16,t=16,b=36),
        height=200,
        showlegend=False,
        xaxis=dict(showgrid=False, color="#6b7a96", title="samples"),
        yaxis=dict(showgrid=True,  gridcolor="#1e2735", color="#6b7a96",
                   range=[0,1], tickformat=".0%"),
        font=dict(family="monospace", color="#6b7a96"),
    )
    st.markdown('<div style="background:#131820;border:1px solid #1e2735;border-radius:12px;padding:20px;">'
                '<div style="font-size:11px;font-family:monospace;color:#6b7a96;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Risk Score Timeline</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Crash type + top 3 ────────────────────────────────────────────────────
    col_type, col_top3 = st.columns(2)
    with col_type:
        crash_name = result["crash_type"].replace("_"," ").title()
        conf_pct   = result["confidence"]*100
        st.markdown(f"""
        <div style="background:#131820;border:1px solid #1e2735;border-radius:12px;padding:24px;height:100%">
          <div style="font-size:11px;font-family:monospace;color:#6b7a96;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;">Predicted Crash Type</div>
          <div style="font-size:26px;font-weight:700;color:#00d4a0;margin-bottom:8px;">{crash_name}</div>
          <div style="font-size:13px;color:#6b7a96;">{"Detected with " + f"{conf_pct:.0f}% model confidence" if result["alert"] else "No crash signature detected"}</div>
        </div>""", unsafe_allow_html=True)

    with col_top3:
        bars_html = '<div style="background:#131820;border:1px solid #1e2735;border-radius:12px;padding:24px;">'
        bars_html += '<div style="font-size:11px;font-family:monospace;color:#6b7a96;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;">Top 3 Candidates</div>'
        for label,prob in result["top3"]:
            pct = prob*100
            bars_html += f"""
            <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;font-family:monospace;margin-bottom:5px;">
                <span style="color:#e8edf5;">{label.replace("_"," ").title()}</span>
                <span style="color:#00d4a0;">{pct:.1f}%</span>
              </div>
              <div style="height:6px;background:#1e2735;border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:#00d4a0;border-radius:3px;"></div>
              </div>
            </div>"""
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Parameter warnings ────────────────────────────────────────────────────
    st.markdown('<div style="background:#131820;border:1px solid #1e2735;border-radius:12px;padding:24px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;font-family:monospace;color:#6b7a96;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;">Parameter Warnings</div>', unsafe_allow_html=True)

    if not result["warnings"]:
        st.markdown('<div style="text-align:center;color:#2ed573;padding:16px;">✓ All monitored parameters within normal range</div>', unsafe_allow_html=True)
    else:
        for w in result["warnings"]:
            badge_cls   = "badge-critical" if w["severity"]=="CRITICAL" else "badge-warning"
            card_border = "warn-critical"  if w["severity"]=="CRITICAL" else "warn-warning"
            st.markdown(f"""
            <div class="warn-card {card_border}">
              <span class="{badge_cls}">{w["severity"]}</span>
              <span style="font-size:13px;color:#e8edf5;margin-left:10px;">{w["message"]}</span>
              <div style="font-family:monospace;font-size:11px;color:#6b7a96;margin-top:4px;margin-left:2px;">{w["param"]} = {w["value"]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Flight Replay Simulation ──────────────────────────────────────────────────
if "raw_df" in st.session_state and "result" in st.session_state:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="background:#131820;border:1px solid #1e2735;border-radius:12px;padding:24px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <div style="width:32px;height:32px;background:rgba(0,212,160,0.15);border-radius:8px;display:grid;place-items:center;">
        <span style="font-size:16px;">▶</span>
      </div>
      <div>
        <div style="font-size:14px;font-weight:700;color:#e8edf5;">Flight Replay Simulation</div>
        <div style="font-size:11px;color:#6b7a96;font-family:monospace;">Step through the flight and watch the model's crash prediction unfold in real time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    r_df   = st.session_state["raw_df"]
    f_df   = st.session_state["feat_df"]
    res    = st.session_state["result"]

    # Downsample so replay runs at a reasonable speed
    # Target ~200 steps max for the UI
    n_rows    = len(r_df)
    step_size = max(1, n_rows // 200)
    indices   = list(range(0, n_rows, step_size))
    n_steps   = len(indices)

    total_dur = float(r_df["t"].max()) if "t" in r_df.columns else n_rows
    first_alert_row = res["first_alert"]  # row index in original data

    # Speed selector
    speed_col, _, btn_col = st.columns([2,3,2])
    with speed_col:
        speed = st.select_slider(
            "Playback speed",
            options=["0.5×", "1×", "2×", "5×", "10×"],
            value="5×",
            help="Controls how fast the replay runs. Higher = faster."
        )
    speed_map = {"0.5×": 0.10, "1×": 0.05, "2×": 0.025, "5×": 0.01, "10×": 0.004}
    delay = speed_map[speed]

    with btn_col:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        play_btn = st.button("▶  Play Replay", use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Static placeholder widgets — all updated in place during replay
    ph_status   = st.empty()
    ph_timer    = st.empty()
    ph_bars     = st.empty()
    ph_chart    = st.empty()
    ph_warnings = st.empty()

    def render_frame(step_i, idx, alert_fired, crashed):
        """Render one frame of the replay into the placeholders."""
        t_now    = float(r_df["t"].iloc[idx]) if "t" in r_df.columns else idx
        pct_done = t_now / max(total_dur, 1)
        risk_now = res["risk_series"][min(step_i, len(res["risk_series"])-1)]

        # Risk colour
        if risk_now > 0.75:   rc = "#ff4757"
        elif risk_now > 0.50: rc = "#ffa502"
        else:                  rc = "#00d4a0"

        # ── Status banner ──
        if crashed:
            status_html = '<div style="background:rgba(255,71,87,0.18);border:1px solid rgba(255,71,87,0.5);border-radius:10px;padding:14px 20px;text-align:center;"><span style="color:#ff4757;font-size:18px;font-weight:800;">💥 CRASH — END OF LOG</span></div>'
        elif alert_fired:
            rows_left = n_rows - idx
            secs_left = round(float(r_df["t"].max()) - t_now, 1) if "t" in r_df.columns else "?"
            status_html = f'<div style="background:rgba(255,165,2,0.12);border:1px solid rgba(255,165,2,0.4);border-radius:10px;padding:14px 20px;text-align:center;"><span style="color:#ffa502;font-size:16px;font-weight:700;">⚠ CRASH PREDICTED — {secs_left}s until impact</span></div>'
        else:
            status_html = '<div style="background:rgba(0,212,160,0.08);border:1px solid rgba(0,212,160,0.2);border-radius:10px;padding:14px 20px;text-align:center;"><span style="color:#00d4a0;font-size:15px;font-weight:600;">✈ FLIGHT NOMINAL</span></div>'
        ph_status.markdown(status_html, unsafe_allow_html=True)

        # ── Timer bar ──
        bar_pct = int(pct_done * 100)
        bar_color = "#ff4757" if crashed else "#ffa502" if alert_fired else "#00d4a0"
        ph_timer.markdown(f"""
        <div style="margin:10px 0 6px;">
          <div style="display:flex;justify-content:space-between;font-size:11px;font-family:monospace;color:#6b7a96;margin-bottom:5px;">
            <span>T + {t_now:.1f}s</span>
            <span>{bar_pct}%</span>
            <span>{total_dur:.1f}s total</span>
          </div>
          <div style="height:8px;background:#1e2735;border-radius:4px;overflow:hidden;">
            <div style="height:100%;width:{bar_pct}%;background:{bar_color};border-radius:4px;transition:width 0.1s;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Live metric bars ──
        row_r  = r_df.iloc[idx]
        row_f  = f_df.iloc[idx]
        roll   = abs(float(row_r.get("roll",  0)))
        pitch  = abs(float(row_r.get("pitch", 0)))
        vibe   = float(row_f.get("vibe_total", 0)) if "vibe_total" in row_f.index else 0.0
        volt   = float(row_r.get("volt", 12.0))
        errRP  = float(row_r.get("errRP", 0.0))

        def mini_bar(label, val, max_val, unit, warn_thresh, crit_thresh, invert=False):
            pct = min(100, val / max_val * 100)
            if invert:
                color = "#ff4757" if val < crit_thresh else "#ffa502" if val < warn_thresh else "#00d4a0"
            else:
                color = "#ff4757" if val > crit_thresh else "#ffa502" if val > warn_thresh else "#00d4a0"
            return f"""
            <div style="flex:1;background:#0d1119;border:1px solid #1e2735;border-radius:8px;padding:10px 14px;">
              <div style="font-size:10px;font-family:monospace;color:#6b7a96;letter-spacing:1px;text-transform:uppercase;">{label}</div>
              <div style="font-size:20px;font-weight:700;color:{color};font-family:monospace;">{val:.1f}<span style="font-size:11px;color:#6b7a96;"> {unit}</span></div>
              <div style="height:4px;background:#1e2735;border-radius:2px;margin-top:6px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{color};border-radius:2px;"></div>
              </div>
            </div>"""

        bars_html = '<div style="display:flex;gap:10px;margin:10px 0;">'
        bars_html += mini_bar("Roll",  roll,  45,  "°",   8,  20)
        bars_html += mini_bar("Pitch", pitch, 45,  "°",   8,  20)
        bars_html += mini_bar("Vibe",  vibe,  80,  "m/s²",15, 40)
        # Max voltage is 63V (S12 LiPo fully charged). Warn <3.5V/cell (auto-detect cell count)
        _volt_max = max(25.0, volt * 1.15)   # 15% headroom above seen max
        _cell_count = max(1, round(volt / 4.2)) if volt > 5 else 3
        _warn_v = _cell_count * 3.5; _crit_v = _cell_count * 3.3
        bars_html += mini_bar("Volt",  volt,  _volt_max,  "V",   _warn_v, _crit_v, invert=True)
        bars_html += mini_bar("ErrRP", errRP, 0.6, "",    0.15, 0.40)
        bars_html += f"""
        <div style="flex:1.2;background:#0d1119;border:1px solid #1e2735;border-radius:8px;padding:10px 14px;">
          <div style="font-size:10px;font-family:monospace;color:#6b7a96;letter-spacing:1px;text-transform:uppercase;">Risk Score</div>
          <div style="font-size:20px;font-weight:700;color:{rc};font-family:monospace;">{risk_now*100:.0f}<span style="font-size:11px;color:#6b7a96;"> %</span></div>
          <div style="height:4px;background:#1e2735;border-radius:2px;margin-top:6px;overflow:hidden;">
            <div style="height:100%;width:{risk_now*100:.0f}%;background:{rc};border-radius:2px;"></div>
          </div>
        </div>"""
        bars_html += "</div>"
        ph_bars.markdown(bars_html, unsafe_allow_html=True)

        # ── Mini risk chart (up to current step) ──
        rs_so_far = res["risk_series"][:step_i+1]
        xs        = list(range(len(rs_so_far)))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=xs, y=rs_so_far, mode="lines", name="Risk",
            line=dict(color=rc, width=2),
            fill="tozeroy", fillcolor=f"rgba(0,212,160,0.10)"
        ))
        fig2.add_hline(y=res["threshold"], line_dash="dash", line_color="#ff4757", line_width=1)
        fig2.update_layout(
            paper_bgcolor="#131820", plot_bgcolor="#131820",
            margin=dict(l=40,r=10,t=10,b=30), height=130,
            showlegend=False,
            xaxis=dict(showgrid=False, color="#6b7a96"),
            yaxis=dict(showgrid=True, gridcolor="#1e2735", color="#6b7a96", range=[0,1], tickformat=".0%"),
            font=dict(family="monospace", color="#6b7a96"),
        )
        ph_chart.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # ── Live warnings (only once alert fires) ──
        if alert_fired:
            active_warns = get_param_warnings(row_f)
            if active_warns:
                warn_html = '<div style="margin-top:8px;">'
                for w in active_warns[:4]:  # show top 4
                    bc  = "warn-critical" if w["severity"]=="CRITICAL" else "warn-warning"
                    bdg = "badge-critical" if w["severity"]=="CRITICAL" else "badge-warning"
                    warn_html += f'<div class="warn-card {bc}"><span class="{bdg}">{w["severity"]}</span><span style="font-size:12px;color:#e8edf5;margin-left:10px;">{w["message"]}</span></div>'
                warn_html += "</div>"
                ph_warnings.markdown(warn_html, unsafe_allow_html=True)

    if play_btn:
        alert_fired = False
        rtl_popup_shown = False
        for step_i, idx in enumerate(indices):
            crashed = (step_i == n_steps - 1) and res["alert"]
            # Check if alert fires at this step
            if first_alert_row >= 0 and idx >= first_alert_row:
                if not alert_fired:
                    alert_fired = True
                    # Fire RTL popup the first time alert triggers during replay
                    if not rtl_popup_shown:
                        rtl_popup_shown = True
                        crash_label = res["crash_type"].replace("_", " ").title()
                        secs_left = round(float(r_df["t"].max()) - float(r_df["t"].iloc[idx]), 1) if "t" in r_df.columns else 0
                        show_rtl_popup(crash_label, secs_left)
            render_frame(step_i, idx, alert_fired, crashed)
            time.sleep(delay)
    else:
        # Show static first frame before play
        render_frame(0, 0, False, False)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
st.markdown('<hr><div style="text-align:center;font-family:monospace;font-size:11px;color:#6b7a96;padding:16px 0;">DroneGuard · HistGradientBoosting · 27 features · ATT / BAT / GPS / VIBE / BARO / RCOU</div>', unsafe_allow_html=True)
