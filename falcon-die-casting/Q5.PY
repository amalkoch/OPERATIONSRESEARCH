import pulp as pl

# ============================================================
# Q5: Setup Carryover — Week 1
#     Initial setups: M1→P1, M2→P2, M3→P5, M4→P3, M5→P4
#     These setups are free (no setup time needed)
#     Objective: Minimize total overtime (same as Q2)
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

# Initial setups (these pairs have FREE setup)
initial_setup = {(1,1), (2,2), (3,5), (4,3), (5,4)}

# Effective setup times: 0 if initial setup exists, else original
setup_eff = {}
for (i,j) in F:
    if (i,j) in initial_setup:
        setup_eff[i,j] = 0
    else:
        setup_eff[i,j] = setup[i,j]

R = 120
OT_max = 48
U = 168

# --- Model ---
model = pl.LpProblem("Q5_Setup_Carryover", pl.LpMinimize)

# --- Decision Variables ---
x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in F}
z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in F}
OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=OT_max) for i in machines}

# --- Objective: Minimize total overtime ---
model += pl.lpSum(OT[i] for i in machines), "Total_Overtime"

# --- Constraints ---

# Demand satisfaction
for j in parts:
    model += (
        pl.lpSum(rates[i,j] * yields[j] * x[i,j] for i in machines if (i,j) in F)
        == demand[j],
        f"Demand_Part_{j}"
    )

# Machine capacity — using EFFECTIVE setup times
for i in machines:
    model += (
        pl.lpSum(x[i,j] + setup_eff[i,j] * z[i,j] for j in parts if (i,j) in F)
        <= R + OT[i],
        f"Capacity_Machine_{i}"
    )

# Linking constraints
for (i,j) in F:
    model += (x[i,j] <= U * z[i,j], f"Link_{i}_{j}")

# --- Solve ---
solver = pl.PULP_CBC_CMD(msg=0)
status = model.solve(solver)

# ============================================================
# Results
# ============================================================
print("=" * 60)
print("Q5: Setup Carryover — Week 1")
print("=" * 60)
print(f"Status: {pl.LpStatus[status]}")
print(f"Optimal Total Overtime: {pl.value(model.objective):.2f} hours\n")

print("Initial Setups (FREE):")
for (i,j) in sorted(initial_setup):
    print(f"  M{i} → Part {j}  (setup saved: {setup[i,j]} hr)")
total_saved = sum(setup[i,j] for (i,j) in initial_setup if pl.value(z[i,j]) and pl.value(z[i,j]) > 0.5)
print(f"  Total setup time saved: {total_saved} hr\n")

# Production schedule
print(f"{'Machine':<10} {'Part':<8} {'Setup(hr)':<12} {'Prod(hr)':<12} {'Good Units':<12} {'Init?':<6}")
print("-" * 60)

for i in machines:
    for j in parts:
        if (i,j) in F:
            xv = pl.value(x[i,j]) or 0
            zv = pl.value(z[i,j]) or 0
            if xv > 0.01:
                good = xv * rates[i,j] * yields[j]
                sv = setup_eff[i,j] * zv
                init = "FREE" if (i,j) in initial_setup else ""
                print(f"M{i:<9} P{j:<7} {sv:<12.2f} {xv:<12.2f} {good:<12.0f} {init}")

# Machine utilization
print(f"\n{'Machine':<10} {'Total Used(hr)':<16} {'Regular(hr)':<14} {'Overtime(hr)':<14}")
print("-" * 54)

total_ot = 0
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    total_ot += ot_val
    total_used = sum(
        (pl.value(x[i,j]) or 0) + setup_eff[i,j] * (pl.value(z[i,j]) or 0)
        for j in parts if (i,j) in F
    )
    reg = min(total_used, R)
    print(f"M{i:<9} {total_used:<16.2f} {reg:<14.2f} {ot_val:<14.2f}")

print(f"{'TOTAL':<26} {'':<14} {total_ot:<14.2f}")

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

# Comparison with Q2
print("\n" + "=" * 60)
print("Comparison: Q2 vs Q5")
print("=" * 60)
print(f"{'Model':<8} {'Total OT(hr)':<15} {'Setup hrs used':<16} {'Note'}")
print("-" * 55)
q2_setup = 8+8+10+8+10+24+8+12+8+20  # all setups in Q2
q5_setup_used = sum(
    setup_eff[i,j] * (pl.value(z[i,j]) or 0)
    for (i,j) in F
)
print(f"{'Q2':<8} {'119.76':<15} {q2_setup:<16} All setups required")
print(f"{'Q5':<8} {total_ot:<15.2f} {q5_setup_used:<16.0f} Initial setups free")
print(f"\nOvertime saved by setup carryover: {119.76 - total_ot:.2f} hr")

# ============================================================
# CONSTRAINT VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("CONSTRAINT VERIFICATION")
print("=" * 60)

tol = 0.01
all_ok = True

# 1. Demand satisfaction
print("\n[1] Demand Satisfaction (= d_j)")
print("-" * 50)
for j in parts:
    produced = sum(
        (pl.value(x[i,j]) or 0) * rates[i,j] * yields[j]
        for i in machines if (i,j) in F
    )
    diff = abs(produced - demand[j])
    ok = diff < tol
    status_str = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Part {j}: produced={produced:.2f}, demand={demand[j]}, diff={diff:.4f}  {status_str}")
    if not ok:
        all_ok = False

# 2. Machine capacity (with effective setup times)
print("\n[2] Machine Capacity (prod + eff_setup ≤ 120 + OT_i)")
print("-" * 50)
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    lhs = sum(
        (pl.value(x[i,j]) or 0) + setup_eff[i,j] * (pl.value(z[i,j]) or 0)
        for j in parts if (i,j) in F
    )
    rhs = R + ot_val
    slack = rhs - lhs
    ok = slack >= -tol
    status_str = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Machine {i}: LHS={lhs:.2f}, RHS={rhs:.2f}, slack={slack:.4f}  {status_str}")
    if not ok:
        all_ok = False

# 3. Overtime bounds
print("\n[3] Overtime Bounds (0 ≤ OT_i ≤ 48)")
print("-" * 50)
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    ok = -tol <= ot_val <= OT_max + tol
    status_str = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Machine {i}: OT={ot_val:.2f}  {status_str}")
    if not ok:
        all_ok = False

# 4. Linking constraints
print("\n[4] Linking Constraints (x_ij ≤ 168 × z_ij)")
print("-" * 50)
for (i,j) in F:
    xv = pl.value(x[i,j]) or 0
    zv = pl.value(z[i,j]) or 0
    diff = xv - U * zv
    ok = diff <= tol
    status_str = "✓ PASS" if ok else "✗ FAIL"
    if xv > 0.01 or not ok:
        print(f"  x_{i}_{j}={xv:.2f}, z_{i}_{j}={zv:.0f}, x-Uz={diff:.4f}  {status_str}")
    if not ok:
        all_ok = False

# 5. Non-negativity
print("\n[5] Non-negativity (x_ij ≥ 0)")
print("-" * 50)
neg_found = False
for (i,j) in F:
    xv = pl.value(x[i,j]) or 0
    if xv < -tol:
        print(f"  x_{i}_{j}={xv:.4f}  ✗ FAIL")
        neg_found = True
        all_ok = False
if not neg_found:
    print("  All x_ij ≥ 0  ✓ PASS")

# 6. Binary check
print("\n[6] Binary Variables (z_ij ∈ {0, 1})")
print("-" * 50)
bin_ok = True
for (i,j) in F:
    zv = pl.value(z[i,j]) or 0
    ok = abs(zv - 0) < tol or abs(zv - 1) < tol
    if not ok:
        print(f"  z_{i}_{j}={zv:.4f}  ✗ FAIL")
        bin_ok = False
        all_ok = False
if bin_ok:
    print("  All z_ij ∈ {0, 1}  ✓ PASS")

# 7. Consistency
print("\n[7] Consistency (z_ij=1 ↔ x_ij>0)")
print("-" * 50)
cons_ok = True
for (i,j) in F:
    xv = pl.value(x[i,j]) or 0
    zv = pl.value(z[i,j]) or 0
    if xv > tol and zv < 0.5:
        print(f"  x_{i}_{j}={xv:.2f} but z_{i}_{j}=0  ✗ FAIL")
        cons_ok = False
        all_ok = False
    if xv < tol and zv > 0.5:
        print(f"  x_{i}_{j}=0 but z_{i}_{j}=1  ⚠ WARNING (unnecessary setup)")
if cons_ok:
    print("  All consistent  ✓ PASS")

# 8. Initial setup verification
print("\n[8] Initial Setup Verification (free setups correctly applied)")
print("-" * 50)
init_ok = True
for (i,j) in initial_setup:
    eff = setup_eff[i,j]
    orig = setup[i,j]
    ok = eff == 0
    status_str = "✓ PASS" if ok else "✗ FAIL"
    print(f"  ({i},{j}): original={orig}hr, effective={eff}hr  {status_str}")
    if not ok:
        init_ok = False
        all_ok = False

for (i,j) in F:
    if (i,j) not in initial_setup:
        eff = setup_eff[i,j]
        orig = setup[i,j]
        ok = eff == orig
        if not ok:
            print(f"  ({i},{j}): should be {orig}hr but is {eff}hr  ✗ FAIL")
            all_ok = False

if init_ok:
    print("  All initial setups correctly set to 0  ✓ PASS")

# Final verdict
print("\n" + "=" * 60)
if all_ok:
    print("ALL CONSTRAINTS VERIFIED ✓ — Solution is feasible and optimal.")
else:
    print("⚠ SOME CONSTRAINTS FAILED — Review the issues above.")
print("=" * 60)