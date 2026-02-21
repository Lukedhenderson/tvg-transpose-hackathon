# Homeostatic Grid Edge

A decentralized Virtual Power Plant dashboard. 50 autonomous HVAC edge agents monitor grid frequency and shed load independently — no central cloud dispatch required.

## Problem

The US electric grid runs at 60 Hz. When demand exceeds supply, frequency drops. If it drops far enough, equipment trips, cascading outages follow, and the grid destabilizes.

Today's demand response is **centralized**: a cloud platform detects the frequency dip, decides which loads to curtail, and sends commands to each device. That round-trip takes time. For fast events like a generator tripping offline or a data center suddenly powering up its cooling systems, the latency can be the difference between a managed response and a blackout.

The core question: **what if the devices could decide for themselves?**

## Solution

**Homeostatic Grid Edge** moves the decision to the device. Each commercial HVAC system has a local frequency measurement and runs simple local logic:

- Frequency below my shed threshold → cut load by 50%
- Frequency below my offline threshold → shut down entirely
- Frequency recovers → ramp back up gradually

No cloud round-trip. No coordinator choosing who sheds. The grid stabilizes through many independent, local decisions — the way a biological system maintains homeostasis.

The key insight is **threshold diversity**. Every agent has a slightly different shed and offline frequency, drawn from a random distribution. A moderate dip (e.g. a data center surge pulling frequency to 59.94 Hz) only triggers the most sensitive ~30% of agents. A severe event (generator trip, 59.88 Hz) triggers most. A catastrophic cascade (59.83 Hz) triggers the entire fleet. This graduated response emerges naturally from diversity — no optimizer required.

## Implementation

### EdgeAgent

Each of the 50 agents is an instance of `EdgeAgent` with:

| Attribute | Description |
|-----------|-------------|
| `base_load` | Nominal power draw, 50–100 kW (random per agent) |
| `freq_shed` | Personal shed threshold, 59.92–59.98 Hz (random) |
| `freq_offline` | Personal offline threshold, 0.03–0.07 Hz below `freq_shed` |
| `ramp_rate` | Recovery speed, 4–12% of base load per tick (random) |

The `step(freq)` method implements all decision logic locally:

```
if freq < freq_offline  →  current_load = 0, status = "Offline"
if freq < freq_shed     →  current_load = 50% of base, status = "Shedding"
if freq ≥ freq_shed     →  ramp current_load toward base, status = "Ramping" → "Online"
```

Because thresholds are per-agent, the same frequency produces different responses across the fleet. Hover any node in the dashboard to see that agent's individual thresholds.

### Frequency Simulation

Live ERCOT API calls are too slow for a fluid 1 Hz dashboard, so frequency is simulated:

- **Mean-reverting random walk** around 60.0 Hz (drift coefficient 0.08)
- **~3% per-tick chance** of a sudden grid-stress dip (0.06–0.14 Hz drop)
- **Scenario mode**: four pre-built events pull frequency toward a target over a set duration, then release it to recover naturally

### Interaction Modes

| Mode | Behavior |
|------|----------|
| **Auto** | Simulation runs on a timer (0.3–3.0 s, configurable). Pause toggle available. |
| **Manual** | You drag a frequency slider (59.80–60.06 Hz). Agents respond instantly. Best for demos. |
| **Step-by-Step** | Advance one tick at a time to study state transitions. |

### Scenario Events

| Event | Target Frequency | Fleet Response |
|-------|-----------------|----------------|
| Data Center Surge | ~59.94 Hz | Partial — only the most sensitive agents shed |
| Generator Trip | ~59.88 Hz | Majority — most agents respond |
| Cascading Failure | ~59.83 Hz | Full fleet emergency |
| Restore Normal | 60.00 Hz | Graduated ramp-back at individual rates |

### Dashboard Layout

- **KPI cards** — Grid frequency, load shed (kW and % of capacity), active node count, fleet status with color-coded accent
- **Agent network** — 10×5 grid graph of all 50 agents. Node color encodes status (green/amber/red/indigo), node size encodes current load utilization. Connection lines show mesh topology. Hover for per-agent details.
- **Frequency chart** — Time-series with shaded threshold bands showing the shed and offline zones across the fleet
- **Load chart** — Aggregate VPP load, responds inversely to frequency drops
- **Fleet details** — Expandable table with per-agent base load, current load, utilization, status, and individual thresholds

### Tech Stack

- **Python 3** — simulation logic and agent model
- **Streamlit** — UI framework, session state, auto-refresh loop
- **Plotly** — charts and agent network graph
- **Pandas** — fleet detail table

No external APIs or grid data feeds required. Everything runs locally.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501**.

## Files

```
├── app.py              # All simulation + dashboard code
├── requirements.txt    # streamlit, plotly, pandas
└── README.md
```
