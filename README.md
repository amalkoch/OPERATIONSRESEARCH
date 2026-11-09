# 04 · Production Scheduling (Operations Research)

MILP case study models built with **PuLP** for the Falcon Die Casting (FDC) company, plus a SunOil transportation model.

**Source:** `~/op/`, `~/Desktop/or_project.py`
**Stack:** Python 3, `pulp`, `matplotlib`, `numpy`

## `falcon-die-casting/`
5 machines × 5 parts. Feasible pairs, demand, production rates and setup times are defined at the top of each file.
| File | Question |
|---|---|
| `Q2.py` (+ `or_project_Q2_draft.py`, `conscheck2.py`) | Q2: minimize total overtime (week 1) |
| `Q3.PY` (+ `conscheck3.py`) | Q3: minimize the maximum overtime (minimax) |
| `q4.py` | Q4: minimize overtime cost ($30/hr machine + $40/hr support) |
| `Q5.PY` | Q5: setup carryover from the initial machine setups |
| `q6.py` | Q6: week 11 shortfalls and penalties ($3 shortfall, $2 inventory) |
| `ProductionScheduler.py` | Class version that combines all models and draws charts |
| `*.png` | Output charts (schedules, overtime comparison, shortfall, weekly trend) |

## `sunoil-transport/`
`sunoil_transport.py` (originally named `hello`): a transportation and facility model with 2 supply types × 5 sources × 5 destinations.

## Run
```bash
pip install pulp matplotlib numpy
python falcon-die-casting/Q2.py
```

## Status (checked 2026-11-07)
**🟢 Complete.** I ran every model and each one solves to **Optimal**:
| Model | Result |
|---|---|
| Q2 total overtime | 119.76 hr |
| Q3 minimax | 135.98 hr total, all constraints verified ✓ |
| Q4 overtime cost | $5,158.08 |
| Q5 setup carryover | 87.13 hr (60 hr of setup saved) |
| Q6 week-11 shortfall | $13,324.77 |
| `ProductionScheduler.py` | Runs all of Q2–Q6, the sensitivity analysis and every chart |
- 🔧 **Fixed:** `sunoil_transport.py` had a syntax error (`]` where `)` belonged on the exclusivity constraint). It now solves with plants at locations 4 and 5.
- ℹ️ `Q1` (trial-and-error) has no file; it was presumably done by hand. `or_project_Q2_draft.py` and `conscheck2.py` are Q2 with extra comments.
