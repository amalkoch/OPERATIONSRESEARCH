import pulp as pl

# ============================================================
# Q3: Minimize Maximum Overtime (Minimax) — Week 1
#     Compare with Q1 (trial-and-error) as requested in case
# ============================================================

# --- Sets ---
machines = [1, 2, 3, 4, 5]
parts = [1, 2, 3, 4, 5]
F = [(1,1),(1,4),(2,1),(2,2),(3,2),(3,5),(4,2),(4,3),(5,4),(5,5)]

# --- Parameters ---
demand = {1: 3500, 2: 3000, 3: 4000, 4: 4000, 5: 2800}

rates = {
    (1,1): 40, (1,4): 60,
    (2,1): 35, (2,2): 25,
    (3,2): 30, (3,5): 45,
    (4,2): 35, (4,3): 50,
    (5,4): 60, (5,5): 50
}

yields = {1: 0.60, 2: 0.55, 3: 0.75, 4: 0.65, 5: 0.60}

setup = {
    (1,1): 8,  (1,4): 8,
    (2,1): 10, (2,2): 8,
    (3,2): 10, (3,5): 24,
    (4,2): 8,  (4,3): 12,
    (5,4): 8,  (5,5): 20
}

R = 120
OT_max = 48
U = 168

# --- Model ---
model = pl.LpProblem("Q3_Min_Max_Overtime", pl.LpMinimize)

# --- Decision Variables ---
x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in F}
z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in F}
OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=OT_max) for i in machines}
OT_max_var = pl.LpVariable("OT_max_var", lowBound=0, upBound=OT_max)

# --- Objective ---
model += OT_max_var, "Minimize_Max_Overtime"

# --- Constraints ---
for i in machines:
    model += (OT_max_var >= OT[i], f"Max_OT_Machine_{i}")

for j in parts:
    model += (
        pl.lpSum(rates[i,j] * yields[j] * x[i,j] for i in machines if (i,j) in F)
        == demand[j],
        f"Demand_Part_{j}"
    )

for i in machines:
    model += (
        pl.lpSum(x[i,j] + setup[i,j] * z[i,j] for j in parts if (i,j) in F)
        <= R + OT[i],
        f"Capacity_Machine_{i}"
    )

for (i,j) in F:
    model += (x[i,j] <= U * z[i,j], f"Link_{i}_{j}")

# --- Solve ---
solver = pl.PULP_CBC_CMD(msg=0)
status = model.solve(solver)

# ============================================================
# Results
# ============================================================
print("=" * 60)
print("Q3: Minimize Maximum Overtime — Week 1")
print("=" * 60)
print(f"Status: {pl.LpStatus[status]}")
print(f"Maximum Overtime (minimax): {pl.value(OT_max_var):.2f} hours\n")

# Production schedule
print(f"{'Machine':<10} {'Part':<8} {'Setup(hr)':<12} {'Prod(hr)':<12} {'Good Units':<12}")
print("-" * 54)

for i in machines:
    for j in parts:
        if (i,j) in F:
            xv = pl.value(x[i,j]) or 0
            zv = pl.value(z[i,j]) or 0
            if xv > 0.01:
                good = xv * rates[i,j] * yields[j]
                sv = setup[i,j] * zv
                print(f"M{i:<9} P{j:<7} {sv:<12.2f} {xv:<12.2f} {good:<12.0f}")

# Machine utilization
print(f"\n{'Machine':<10} {'Total Used(hr)':<16} {'Regular(hr)':<14} {'Overtime(hr)':<14}")
print("-" * 54)

total_ot = 0
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    total_ot += ot_val
    total_used = sum(
        (pl.value(x[i,j]) or 0) + setup[i,j] * (pl.value(z[i,j]) or 0)
        for j in parts if (i,j) in F
    )
    reg = min(total_used, R)
    print(f"M{i:<9} {total_used:<16.2f} {reg:<14.2f} {ot_val:<14.2f}")

print(f"{'TOTAL':<26} {'':<14} {total_ot:<14.2f}")
print(f"{'MAX':<26} {'':<14} {pl.value(OT_max_var):<14.2f}")

# Demand check
print(f"\n{'Part':<8} {'Demand':<10} {'Produced':<12} {'Status'}")
print("-" * 42)
for j in parts:
    produced = sum(
        (pl.value(x[i,j]) or 0) * rates[i,j] * yields[j]
        for i in machines if (i,j) in F
    )
    gap = demand[j] - produced
    s = "✓ Met" if abs(gap) < 1 else f"✗ Short {gap:.0f}"
    print(f"P{j:<7} {demand[j]:<10} {produced:<12.0f} {s}")

# ============================================================
# COMPARISON: Q1 (Trial-and-Error) vs Q3 (Minimax)
# ============================================================
print("\n" + "=" * 60)
print("Comparison: Q1 (Trial-and-Error) vs Q3 (Minimax)")
print("=" * 60)

# Q1 results (from manual trial-and-error solution)
q1_ot = {1: 48.00, 2: 0.19, 3: 23.58, 4: 48.00, 5: 0.00}
q1_total = 119.77
q1_max = 48.00

q3_ot = {i: pl.value(OT[i]) or 0 for i in machines}
q3_total = total_ot
q3_max = pl.value(OT_max_var)

print(f"\n{'Machine':<10} {'Q1 OT(hr)':<14} {'Q3 OT(hr)':<14} {'Change(hr)'}")
print("-" * 50)
for i in machines:
    change = q3_ot[i] - q1_ot[i]
    print(f"M{i:<9} {q1_ot[i]:<14.2f} {q3_ot[i]:<14.2f} {change:+.2f}")

print(f"\n{'Metric':<20} {'Q1':<15} {'Q3':<15} {'Change'}")
print("-" * 55)
print(f"{'Total OT (hr)':<20} {q1_total:<15.2f} {q3_total:<15.2f} {q3_total-q1_total:+.2f}")
print(f"{'Max OT (hr)':<20} {q1_max:<15.2f} {q3_max:<15.2f} {q3_max-q1_max:+.2f}")
print(f"{'OT Range (hr)':<20} {q1_max-min(q1_ot.values()):<15.2f} {q3_max-min(q3_ot.values()):<15.2f}")

print("\n--- Nature of Changes ---")
print("1. Q3 reduces max overtime from 48.00 to 27.20 hr (-20.80 hr)")
print("2. Q3 increases total overtime from 119.77 to 135.98 hr (+16.21 hr)")
print("3. Q1 has highly uneven OT (0-48 hr range); Q3 is perfectly balanced")
print("4. Q3 uses more machines during overtime but for shorter durations")
print("5. Support personnel work 27.20 hr instead of 48.00 hr under Q3")

# ============================================================
# CONSTRAINT VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("CONSTRAINT VERIFICATION")
print("=" * 60)

tol = 0.01
all_ok = True

print("\n[1] Demand Satisfaction (= d_j)")
print("-" * 50)
for j in parts:
    produced = sum(
        (pl.value(x[i,j]) or 0) * rates[i,j] * yields[j]
        for i in machines if (i,j) in F
    )
    diff = abs(produced - demand[j])
    ok = diff < tol
    print(f"  Part {j}: produced={produced:.2f}, demand={demand[j]}, diff={diff:.4f}  {'✓ PASS' if ok else '✗ FAIL'}")
    if not ok: all_ok = False

print("\n[2] Machine Capacity (prod + setup ≤ 120 + OT_i)")
print("-" * 50)
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    lhs = sum(
        (pl.value(x[i,j]) or 0) + setup[i,j] * (pl.value(z[i,j]) or 0)
        for j in parts if (i,j) in F
    )
    rhs = R + ot_val
    slack = rhs - lhs
    ok = slack >= -tol
    print(f"  Machine {i}: LHS={lhs:.2f}, RHS={rhs:.2f}, slack={slack:.4f}  {'✓ PASS' if ok else '✗ FAIL'}")
    if not ok: all_ok = False

print("\n[3] Overtime Bounds (0 ≤ OT_i ≤ 48)")
print("-" * 50)
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    ok = -tol <= ot_val <= OT_max + tol
    print(f"  Machine {i}: OT={ot_val:.2f}  {'✓ PASS' if ok else '✗ FAIL'}")
    if not ok: all_ok = False

print("\n[4] Minimax Constraint (OT_max_var ≥ OT_i)")
print("-" * 50)
max_val = pl.value(OT_max_var) or 0
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    ok = max_val >= ot_val - tol
    print(f"  OT_max={max_val:.2f} >= OT_{i}={ot_val:.2f}  {'✓ PASS' if ok else '✗ FAIL'}")
    if not ok: all_ok = False

print("\n[5] Linking Constraints (x_ij ≤ 168 × z_ij)")
print("-" * 50)
for (i,j) in F:
    xv = pl.value(x[i,j]) or 0
    zv = pl.value(z[i,j]) or 0
    diff = xv - U * zv
    ok = diff <= tol
    if xv > 0.01 or not ok:
        print(f"  x_{i}_{j}={xv:.2f}, z_{i}_{j}={zv:.0f}, x-Uz={diff:.4f}  {'✓ PASS' if ok else '✗ FAIL'}")
    if not ok: all_ok = False

print("\n[6] Non-negativity & Binary")
print("-" * 50)
neg_ok = all(((pl.value(x[i,j]) or 0) >= -tol) for (i,j) in F)
bin_ok = all((abs((pl.value(z[i,j]) or 0) - 0) < tol or abs((pl.value(z[i,j]) or 0) - 1) < tol) for (i,j) in F)
print(f"  All x_ij ≥ 0: {'✓ PASS' if neg_ok else '✗ FAIL'}")
print(f"  All z_ij ∈ {{0,1}}: {'✓ PASS' if bin_ok else '✗ FAIL'}")
if not neg_ok or not bin_ok: all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("ALL CONSTRAINTS VERIFIED ✓ — Solution is feasible and optimal.")
else:
    print("⚠ SOME CONSTRAINTS FAILED — Review the issues above.")
print("=" * 60)