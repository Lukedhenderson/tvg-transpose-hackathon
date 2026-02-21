"""
Homeostatic Grid Edge — Virtual Power Plant Dashboard
=====================================================
Run:  streamlit run app.py
"""

from __future__ import annotations

import time
import random
from collections import deque

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Design tokens (light / sleek, high contrast) ──────────────────────────

BG = "#f8f9fa"
CARD = "#ffffff"
BORDER = "#e2e8f0"
BORDER_SUB = "#f1f5f9"
TEXT = "#0f172a"
TEXT_MUTED = "#475569"
TEXT_SUB = "#64748b"

INDIGO = "#6366f1"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"

STATUS_COLOR = {
    "Online": GREEN, "Shedding": AMBER, "Offline": RED, "Ramping": INDIGO,
}
FLEET_ACCENT = {
    "NOMINAL": GREEN, "RESPONDING": AMBER, "EMERGENCY": RED, "RECOVERING": INDIGO,
}

# ── Simulation config ────────────────────────────────────────────────────

NUM_AGENTS = 50
HISTORY_LEN = 200
FREQ_NOMINAL = 60.0
GRID_COLS = 10
GRID_ROWS = 5

# ── EdgeAgent ────────────────────────────────────────────────────────────

class EdgeAgent:
    """Autonomous HVAC edge node with individualised droop thresholds."""

    __slots__ = (
        "agent_id", "base_load", "current_load", "status",
        "freq_shed", "freq_offline", "ramp_rate",
    )

    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self.base_load = random.uniform(50, 100)
        self.current_load = self.base_load
        self.status = "Online"
        self.freq_shed = random.uniform(59.92, 59.98)
        self.freq_offline = self.freq_shed - random.uniform(0.03, 0.07)
        self.ramp_rate = random.uniform(0.12, 0.30)

    def step(self, freq: float) -> None:
        if freq < self.freq_offline:
            self.current_load = 0.0
            self.status = "Offline"
        elif freq < self.freq_shed:
            self.current_load = min(self.current_load, self.base_load * 0.5)
            self.status = "Shedding"
        else:
            self.current_load = min(
                self.base_load,
                self.current_load + self.base_load * self.ramp_rate,
            )
            if self.current_load >= self.base_load * 0.995:
                self.current_load = self.base_load
                self.status = "Online"
            else:
                self.status = "Ramping"

# ── Frequency simulation ─────────────────────────────────────────────────

def next_frequency(prev: float, target: float | None = None) -> float:
    if target is not None:
        drift = 0.35 * (target - prev)
        noise = random.gauss(0, 0.004)
    else:
        drift = 0.15 * (FREQ_NOMINAL - prev)
        noise = random.gauss(0, 0.004)
        if random.random() < 0.03:
            noise -= random.uniform(0.06, 0.14)
    return round(max(59.80, min(60.06, prev + drift + noise)), 4)


# ── UI component helpers ─────────────────────────────────────────────────

_FONT = "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif"

def _kpi(label: str, value: str, delta: str = "",
         delta_color: str = TEXT_MUTED, accent: str | None = None) -> str:
    bdr = f"border-left:3px solid {accent};" if accent else ""
    return (
        f'<div style="background:{CARD};border:1px solid {BORDER};{bdr}'
        f'border-radius:8px;padding:20px 24px;height:100%;box-shadow:0 1px 2px rgba(0,0,0,.04)">'
        f'<div style="font:500 11px/1 {_FONT};text-transform:uppercase;'
        f'letter-spacing:.06em;color:{TEXT_MUTED};margin-bottom:10px">{label}</div>'
        f'<div style="font:600 28px/1.1 {_FONT};color:{TEXT};'
        f'font-variant-numeric:tabular-nums;letter-spacing:-.02em">{value}</div>'
        f'<div style="font:400 12px/1.4 {_FONT};color:{delta_color};'
        f'margin-top:8px">{delta}</div></div>'
    )

def _section(text: str) -> None:
    st.markdown(
        f'<div style="font:600 11px/1 {_FONT};text-transform:uppercase;'
        f'letter-spacing:.08em;color:{TEXT_MUTED};margin:0 0 8px">{text}</div>',
        unsafe_allow_html=True,
    )

def _gap(px: int = 16) -> None:
    st.markdown(f'<div style="height:{px}px"></div>', unsafe_allow_html=True)


# ── Chart builders ────────────────────────────────────────────────────────

_AXIS = dict(
    gridcolor=BORDER, zerolinecolor=BORDER,
    tickfont=dict(family=_FONT, size=10, color=TEXT_MUTED),
    title_font=dict(family=_FONT, size=11, color=TEXT_SUB),
)
_PLOTLY_CFG = {"displayModeBar": False}


def build_freq_chart(
    ticks: list, freqs: list,
    shed_lo: float, shed_hi: float,
    off_lo: float, off_hi: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=shed_lo, y1=shed_hi,
                  fillcolor="rgba(251,191,36,.06)", line_width=0)
    fig.add_hrect(y0=off_lo, y1=off_hi,
                  fillcolor="rgba(248,113,113,.06)", line_width=0)
    fig.add_trace(go.Scatter(
        x=ticks, y=freqs, mode="lines",
        line=dict(color=INDIGO, width=2, shape="spline"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=TEXT_MUTED, size=11),
        height=195, margin=dict(l=48, r=8, t=6, b=24),
        xaxis={**_AXIS, "title": None},
        yaxis={**_AXIS, "title": None, "range": [59.78, 60.08]},
        showlegend=False,
    )
    return fig


def build_load_chart(ticks: list, loads: list, max_load: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ticks, y=loads, mode="lines",
        line=dict(color=GREEN, width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(74,222,128,.04)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=TEXT_MUTED, size=11),
        height=195, margin=dict(l=48, r=8, t=6, b=24),
        xaxis={**_AXIS, "title": None},
        yaxis={**_AXIS, "title": None, "range": [0, max_load * 1.05]},
        showlegend=False,
    )
    return fig


def build_agent_grid(agents: list[EdgeAgent]) -> go.Figure:
    xs = [a.agent_id % GRID_COLS for a in agents]
    ys = [GRID_ROWS - 1 - a.agent_id // GRID_COLS for a in agents]
    colors = [STATUS_COLOR[a.status] for a in agents]
    util = [a.current_load / a.base_load for a in agents]
    sizes = [max(16, 38 * u) for u in util]
    border_c = [
        STATUS_COLOR[a.status] if a.status != "Online" else BORDER
        for a in agents
    ]
    border_w = [2.0 if a.status != "Online" else 0.5 for a in agents]
    hovers = [
        f"<b>HVAC-{a.agent_id:03d}</b><br>"
        f"Status: {a.status}<br>"
        f"Load: {a.current_load:.1f}/{a.base_load:.1f} kW<br>"
        f"Shed @ {a.freq_shed:.3f} Hz<br>"
        f"Off @ {a.freq_offline:.3f} Hz"
        for a in agents
    ]

    ex: list[float | None] = []
    ey: list[float | None] = []
    for a in agents:
        ax, ay = a.agent_id % GRID_COLS, GRID_ROWS - 1 - a.agent_id // GRID_COLS
        if ax < GRID_COLS - 1:
            ex += [ax, ax + 1, None]; ey += [ay, ay, None]
        if a.agent_id // GRID_COLS < GRID_ROWS - 1:
            ex += [ax, ax, None]; ey += [ay, ay - 1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="rgba(0,0,0,.12)", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=sizes, color=colors, opacity=0.85,
            line=dict(color=border_c, width=border_w),
        ),
        text=[str(a.agent_id) for a in agents],
        textfont=dict(family=_FONT, size=7, color="rgba(0,0,0,.5)"),
        textposition="middle center",
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=6, r=6, t=6, b=6),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-.8, GRID_COLS - .2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-.8, GRID_ROWS - .2], scaleanchor="x"),
    )
    return fig


# ── Page config & CSS ─────────────────────────────────────────────────────

st.set_page_config(page_title="VPP Control Room", page_icon="⚡", layout="wide")

st.markdown(f"""
<style>
:root {{color-scheme:light}}
[data-testid="stAppViewContainer"] {{background:{BG}}}
.block-container {{padding:0.5rem 2rem 1.5rem;max-width:1440px;font-family:{_FONT};background:{BG}}}
#MainMenu,footer {{visibility:hidden}}

/* remove the default header entirely — it creates a white block over the title */
header[data-testid="stHeader"] {{display:none !important}}

/* sidebar */
section[data-testid="stSidebar"] {{background:{CARD};border-right:1px solid {BORDER}}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    font-size:11px !important;font-weight:600 !important;
    letter-spacing:.06em !important;text-transform:uppercase;color:{TEXT} !important;
}}

/* title */
h1 {{font-size:22px !important;font-weight:600 !important;letter-spacing:-.02em !important;color:{TEXT} !important}}

/* force all text/controls dark so nothing is invisible on white */
p, span, label, .stMarkdown, .stRadio label,
div[data-baseweb="select"] *,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {{color:{TEXT} !important}}
.stSlider label, .stSlider span {{color:{TEXT} !important}}
div[data-baseweb="radio"] label span {{color:{TEXT} !important}}

/* hide st.metric (we render our own) */
[data-testid="stMetric"] {{display:none}}

/* buttons — all variants including sidebar event buttons */
button, .stButton button, .stButton>button,
div[data-testid="stButton"] button,
section[data-testid="stSidebar"] button {{
    background:{CARD} !important;border:1px solid {BORDER} !important;color:{TEXT} !important;
    border-radius:8px !important;font:500 13px {_FONT} !important;
    transition:background .15s,border-color .15s;
}}
button span, .stButton button span,
section[data-testid="stSidebar"] button span,
button p, .stButton button p {{color:{TEXT} !important}}
button:hover, .stButton button:hover,
section[data-testid="stSidebar"] button:hover {{
    background:{BORDER_SUB} !important;border-color:{TEXT_SUB} !important;
}}

/* expander */
.streamlit-expanderHeader {{
    font:500 13px {_FONT} !important;color:{TEXT} !important;
    background:{CARD} !important;border:1px solid {BORDER} !important;
    border-radius:8px !important;
}}

/* radio pills */
div[data-baseweb="radio"] label span {{font-size:13px !important;color:{TEXT}}}

/* dataframe */
[data-testid="stDataFrame"] {{border:1px solid {BORDER};border-radius:8px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
hr {{border-color:{BORDER} !important}}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────

st.markdown(
    f'<h1 style="margin-bottom:4px">VPP Control Room</h1>'
    f'<p style="font:400 13px {_FONT};color:{TEXT_MUTED};margin:0 0 20px">'
    f'50 autonomous edge agents &nbsp;·&nbsp; decentralized frequency response &nbsp;·&nbsp; zero-latency load shed</p>',
    unsafe_allow_html=True,
)


# ── Session-state initialisation ──────────────────────────────────────────

if "agents" not in st.session_state:
    _rng = random.getstate()
    random.seed(42)
    st.session_state.agents = [EdgeAgent(i) for i in range(NUM_AGENTS)]
    random.setstate(_rng)

    st.session_state.freq = FREQ_NOMINAL
    st.session_state.tick = 0
    st.session_state.freq_hist = deque(maxlen=HISTORY_LEN)
    st.session_state.load_hist = deque(maxlen=HISTORY_LEN)
    st.session_state.tick_hist = deque(maxlen=HISTORY_LEN)
    st.session_state.mode = "auto"
    st.session_state.paused = False
    st.session_state.scenario_target = None
    st.session_state.scenario_ticks = 0
    st.session_state.scenario_shock = False


# ── Sidebar ───────────────────────────────────────────────────────────────

step_btn = False
interval = 1.0

with st.sidebar:
    st.subheader("Mode")
    mode_label = st.radio(
        "mode_", ["Auto", "Manual", "Step-by-Step"],
        horizontal=True, label_visibility="collapsed",
        index=["auto", "manual", "step"].index(st.session_state.mode),
    )
    st.session_state.mode = {
        "Auto": "auto", "Manual": "manual", "Step-by-Step": "step",
    }[mode_label]

    if st.session_state.mode == "auto":
        interval = st.slider("Refresh (s)", 0.3, 3.0, 1.0, 0.1)
        paused = st.toggle("Pause", value=st.session_state.paused)
        st.session_state.paused = paused
    elif st.session_state.mode == "manual":
        st.markdown(
            f'<p style="font:400 12px {_FONT};color:{TEXT_MUTED};margin:0 0 8px">'
            f"Drag to set frequency — agents with a threshold above "
            f"the value will respond.</p>", unsafe_allow_html=True,
        )
        manual_freq = st.slider(
            "Frequency (Hz)", 59.80, 60.06,
            float(st.session_state.freq), 0.01, format="%.2f",
        )
        st.session_state.freq = manual_freq
    else:
        step_btn = st.button("Advance one tick",
                             use_container_width=True, type="primary")

    st.divider()
    st.subheader("Events")

    if st.button("Data Center Surge", use_container_width=True,
                 help="DC powers up cooling — dip to ~59.94 Hz"):
        st.session_state.scenario_target = 59.94
        st.session_state.scenario_ticks = 25
        st.session_state.scenario_shock = True
    if st.button("Generator Trip", use_container_width=True,
                 help="Gas turbine offline — dip to ~59.88 Hz"):
        st.session_state.scenario_target = 59.88
        st.session_state.scenario_ticks = 30
        st.session_state.scenario_shock = True
    if st.button("Cascading Failure", use_container_width=True,
                 help="Multi-unit trip — severe dip to ~59.83 Hz"):
        st.session_state.scenario_target = 59.83
        st.session_state.scenario_ticks = 45
        st.session_state.scenario_shock = True
    if st.button("Restore Normal", use_container_width=True,
                 help="Recover to 60.00 Hz"):
        st.session_state.scenario_target = 60.0
        st.session_state.scenario_ticks = 30
        st.session_state.scenario_shock = False

    st.divider()
    st.subheader("Legend")

    _legend = " &nbsp;&nbsp; ".join(
        f'<span style="color:{c}">●</span>&thinsp;{s}'
        for s, c in STATUS_COLOR.items()
    )
    st.markdown(
        f'<div style="font:400 12px {_FONT};color:{TEXT_MUTED};line-height:1.8">'
        f'{_legend}</div>',
        unsafe_allow_html=True,
    )

    agents_ref = st.session_state.agents
    shed_lo = min(a.freq_shed for a in agents_ref)
    shed_hi = max(a.freq_shed for a in agents_ref)
    off_lo = min(a.freq_offline for a in agents_ref)
    off_hi = max(a.freq_offline for a in agents_ref)
    st.markdown(
        f'<div style="font:400 12px {_FONT};color:{TEXT_MUTED};margin-top:12px;line-height:1.7">'
        f'Shed &nbsp;{shed_lo:.3f} – {shed_hi:.3f} Hz<br>'
        f'Offline &nbsp;{off_lo:.3f} – {off_hi:.3f} Hz<br>'
        f'<span style="color:{TEXT_SUB}">Hover any node to inspect thresholds.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Simulation tick ───────────────────────────────────────────────────────

should_tick = (
    (st.session_state.mode == "auto" and not st.session_state.paused)
    or st.session_state.mode == "manual"
    or (st.session_state.mode == "step" and step_btn)
)

if should_tick:
    if st.session_state.mode == "manual":
        freq = st.session_state.freq
    elif st.session_state.scenario_ticks > 0:
        if st.session_state.scenario_shock:
            gap = st.session_state.scenario_target - st.session_state.freq
            freq = st.session_state.freq + 0.7 * gap + random.gauss(0, 0.003)
            freq = round(max(59.80, min(60.06, freq)), 4)
            st.session_state.scenario_shock = False
        else:
            freq = next_frequency(st.session_state.freq,
                                  target=st.session_state.scenario_target)
        st.session_state.scenario_ticks -= 1
    else:
        st.session_state.scenario_target = None
        freq = next_frequency(st.session_state.freq)

    st.session_state.freq = freq
    st.session_state.tick += 1

    for agent in st.session_state.agents:
        agent.step(freq)

    st.session_state.freq_hist.append(freq)
    st.session_state.load_hist.append(
        sum(a.current_load for a in st.session_state.agents)
    )
    st.session_state.tick_hist.append(st.session_state.tick)


# ── Derived metrics ───────────────────────────────────────────────────────

agents = st.session_state.agents
freq = st.session_state.freq
total_base = sum(a.base_load for a in agents)
total_load = sum(a.current_load for a in agents)
total_shed = total_base - total_load
n_online = sum(1 for a in agents if a.status == "Online")
n_shedding = sum(1 for a in agents if a.status == "Shedding")
n_offline = sum(1 for a in agents if a.status == "Offline")
n_ramping = sum(1 for a in agents if a.status == "Ramping")

if n_offline > 0:
    fleet_label, fleet_accent = "EMERGENCY", RED
    fleet_sub = f"{n_offline} offline · {n_shedding} shedding"
elif n_shedding > 0:
    fleet_label, fleet_accent = "RESPONDING", AMBER
    fleet_sub = f"{n_shedding} shedding · {n_online} online"
elif n_ramping > 0:
    fleet_label, fleet_accent = "RECOVERING", INDIGO
    fleet_sub = f"{n_ramping} ramping"
else:
    fleet_label, fleet_accent = "NOMINAL", GREEN
    fleet_sub = "All 50 online"

freq_delta = freq - FREQ_NOMINAL
freq_color = RED if freq < 59.95 else (AMBER if freq < 59.98 else TEXT_MUTED)
shed_pct = total_shed / total_base * 100


# ── KPI row (custom HTML cards) ──────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(_kpi("Grid Frequency", f"{freq:.3f} Hz",
                     f"{freq_delta:+.3f} Hz from nominal", freq_color),
                unsafe_allow_html=True)
with k2:
    st.markdown(_kpi("Load Shed", f"{total_shed:,.0f} kW",
                     f"{shed_pct:.1f}% of fleet capacity",
                     AMBER if shed_pct > 0 else TEXT_MUTED),
                unsafe_allow_html=True)
with k3:
    st.markdown(_kpi("Active Nodes",
                     f"{NUM_AGENTS - n_offline}/{NUM_AGENTS}",
                     f"{n_offline} offline" if n_offline else "All online",
                     RED if n_offline else TEXT_MUTED),
                unsafe_allow_html=True)
with k4:
    st.markdown(_kpi("Fleet Status", fleet_label, fleet_sub,
                     fleet_accent, accent=fleet_accent),
                unsafe_allow_html=True)

_gap(20)


# ── Main visualisation row ────────────────────────────────────────────────

col_grid, col_charts = st.columns([3, 2], gap="medium")

shed_lo = min(a.freq_shed for a in agents)
shed_hi = max(a.freq_shed for a in agents)
off_lo = min(a.freq_offline for a in agents)
off_hi = max(a.freq_offline for a in agents)

with col_grid:
    _section("Agent Network")
    st.plotly_chart(build_agent_grid(agents), use_container_width=True,
                    key="grid", config=_PLOTLY_CFG)

with col_charts:
    ticks = list(st.session_state.tick_hist)
    freqs = list(st.session_state.freq_hist)
    loads = list(st.session_state.load_hist)

    _section("Frequency")
    st.plotly_chart(build_freq_chart(ticks, freqs,
                                     shed_lo, shed_hi, off_lo, off_hi),
                    use_container_width=True, key="freq", config=_PLOTLY_CFG)
    _gap(4)
    _section("Aggregate Load")
    st.plotly_chart(build_load_chart(ticks, loads, total_base),
                    use_container_width=True, key="load", config=_PLOTLY_CFG)

_gap(8)


# ── Agent fleet table ─────────────────────────────────────────────────────

with st.expander("Agent Fleet Details", expanded=False):
    df = pd.DataFrame([
        {
            "Agent": f"HVAC-{a.agent_id:03d}",
            "Base kW": round(a.base_load, 1),
            "Current kW": round(a.current_load, 1),
            "Util %": round(a.current_load / a.base_load * 100, 1),
            "Status": a.status,
            "Shed Hz": round(a.freq_shed, 3),
            "Off Hz": round(a.freq_offline, 3),
        }
        for a in agents
    ])
    _CSS = {
        "Online":   f"background:#ecfdf5;color:{GREEN}",
        "Shedding": f"background:#fffbeb;color:{AMBER}",
        "Offline":  f"background:#fef2f2;color:{RED}",
        "Ramping":  f"background:#eef2ff;color:{INDIGO}",
    }
    st.dataframe(
        df.style.map(lambda v: _CSS.get(v, ""), subset=["Status"]),
        use_container_width=True, height=380, hide_index=True,
    )


# ── Auto-refresh ──────────────────────────────────────────────────────────

if st.session_state.mode == "auto" and not st.session_state.paused:
    time.sleep(interval)
    st.rerun()
