import pulp as pl

# ============================================================
# Q6: Shortfalls & Penalties — Week 11
#     Demand may exceed capacity → shortfall allowed
#     Penalty: $3/unit for shortfall, $2/unit for inventory
#     OT costs: $30/hr machine, $40/hr support personnel
# ============================================================

# --- Sets ---
machines = [1, 2, 3, 4, 5]
parts = [1, 2, 3, 4, 5]
F = [(1,1),(1,4),(2,1),(2,2),(3,2),(3,5),(4,2),(4,3),(5,4),(5,5)]

# --- Parameters ---
# Week 11 demand (much higher than Week 1)
demand = {1: 4500, 2: 4000, 3: 5000, 4: 5000, 5: 3800}

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

# Cost parameters
C_machine = 30    # $/hr per machine overtime
C_support = 40    # $/hr support personnel (x max OT)
C_short = 3       # $/unit shortfall penalty
C_inv = 2         # $/unit/week inventory cost

# --- Model ---
model = pl.LpProblem("Q6_Shortfalls_Penalties", pl.LpMinimize)

# --- Decision Variables ---
x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in F}
z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in F}
OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=OT_max) for i in machines}
OT_max_var = pl.LpVariable("OT_max_var", lowBound=0, upBound=OT_max)

# NEW: shortfall and inventory variables
sh = {j: pl.LpVariable(f"sh_{j}", lowBound=0) for j in parts}
inv = {j: pl.LpVariable(f"inv_{j}", lowBound=0) for j in parts}

# --- Objective: Min total cost ---
model += (
    C_machine * pl.lpSum(OT[i] for i in machines)
    + C_support * OT_max_var
    + C_short * pl.lpSum(sh[j] for j in parts)
    + C_inv * pl.lpSum(inv[j] for j in parts),
    "Total_Cost"
)

# --- Constraints ---

# OT_max_var >= OT_i
for i in machines:
    model += (OT_max_var >= OT[i], f"Max_OT_Machine_{i}")

# Demand balance: production + shortfall - inventory = demand
for j in parts:
    model += (
        pl.lpSum(rates[i,j] * yields[j] * x[i,j] for i in machines if (i,j) in F)
        + sh[j] - inv[j] == demand[j],
        f"Demand_Part_{j}"
    )

# Machine capacity
for i in machines:
    model += (
        pl.lpSum(x[i,j] + setup[i,j] * z[i,j] for j in parts if (i,j) in F)
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
print("Q6: Shortfalls & Penalties — Week 11")
print("=" * 60)
print(f"Status: {pl.LpStatus[status]}")
print(f"Optimal Total Cost: ${pl.value(model.objective):.2f}\n")

total_ot = sum(pl.value(OT[i]) or 0 for i in machines)
max_ot = pl.value(OT_max_var) or 0
total_sh = sum(pl.value(sh[j]) or 0 for j in parts)
total_inv = sum(pl.value(inv[j]) or 0 for j in parts)

print(f"Machine OT Cost:    ${C_machine * total_ot:.2f}  ({total_ot:.2f} hr)")
print(f"Support Cost:       ${C_support * max_ot:.2f}  ({max_ot:.2f} hr)")
print(f"Shortfall Penalty:  ${C_short * total_sh:.2f}  ({total_sh:.0f} units)")
print(f"Inventory Cost:     ${C_inv * total_inv:.2f}  ({total_inv:.0f} units)")
print(f"Total Cost:         ${pl.value(model.objective):.2f}\n")

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

for i in machines:
    ot_val = pl.value(OT[i]) or 0
    total_used = sum(
        (pl.value(x[i,j]) or 0) + setup[i,j] * (pl.value(z[i,j]) or 0)
        for j in parts if (i,j) in F
    )
    reg = min(total_used, R)
    print(f"M{i:<9} {total_used:<16.2f} {reg:<14.2f} {ot_val:<14.2f}")

print(f"{'TOTAL':<26} {'':<14} {total_ot:<14.2f}")
print(f"{'MAX':<26} {'':<14} {max_ot:<14.2f}")

# Demand / Shortfall / Inventory
print(f"\n{'Part':<8} {'Demand':<10} {'Produced':<12} {'Shortfall':<12} {'Inventory':<12} {'Status'}")
print("-" * 66)
for j in parts:
    produced = sum(
        (pl.value(x[i,j]) or 0) * rates[i,j] * yields[j]
        for i in machines if (i,j) in F
    )
    sh_val = pl.value(sh[j]) or 0
    inv_val = pl.value(inv[j]) or 0
    if sh_val > 0.5:
        s = f"✗ Short {sh_val:.0f}"
    elif inv_val > 0.5:
        s = f"⚠ Excess {inv_val:.0f}"
    else:
        s = "✓ Met"
    print(f"P{j:<7} {demand[j]:<10} {produced:<12.0f} {sh_val:<12.0f} {inv_val:<12.0f} {s}")

# ============================================================
# CONSTRAINT VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("CONSTRAINT VERIFICATION")
print("=" * 60)

tol = 0.01
all_ok = True

# 1. Demand balance: production + shortfall - inventory = demand
print("\n[1] Demand Balance (prod + sh - inv = demand)")
print("-" * 50)
for j in parts:
    produced = sum(
        (pl.value(x[i,j]) or 0) * rates[i,j] * yields[j]
        for i in machines if (i,j) in F
    )
    sh_val = pl.value(sh[j]) or 0
    inv_val = pl.value(inv[j]) or 0
    lhs = produced + sh_val - inv_val
    diff = abs(lhs - demand[j])
    ok = diff < tol
    status_str = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Part {j}: prod={produced:.1f} + sh={sh_val:.1f} - inv={inv_val:.1f} = {lhs:.1f}, demand={demand[j]}  {status_str}")
    if not ok:
        all_ok = False

# 2. Machine capacity
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
print("\n[5] Non-negativity (x_ij ≥ 0, sh_j ≥ 0, inv_j ≥ 0)")
print("-" * 50)
neg_found = False
for (i,j) in F:
    xv = pl.value(x[i,j]) or 0
    if xv < -tol:
        print(f"  x_{i}_{j}={xv:.4f}  ✗ FAIL")
        neg_found = True
        all_ok = False
for j in parts:
    sv = pl.value(sh[j]) or 0
    iv = pl.value(inv[j]) or 0
    if sv < -tol:
        print(f"  sh_{j}={sv:.4f}  ✗ FAIL")
        neg_found = True
        all_ok = False
    if iv < -tol:
        print(f"  inv_{j}={iv:.4f}  ✗ FAIL")
        neg_found = True
        all_ok = False
if not neg_found:
    print("  All variables ≥ 0  ✓ PASS")

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

# 8. Minimax constraint
print("\n[8] Minimax Constraint (OT_max_var ≥ OT_i)")
print("-" * 50)
max_val = pl.value(OT_max_var) or 0
for i in machines:
    ot_val = pl.value(OT[i]) or 0
    ok = max_val >= ot_val - tol
    status_str = "✓ PASS" if ok else "✗ FAIL"
    print(f"  OT_max={max_val:.2f} >= OT_{i}={ot_val:.2f}  {status_str}")
    if not ok:
        all_ok = False

# 9. Cost verification
print("\n[9] Cost Calculation Verification")
print("-" * 50)
calc_cost = (C_machine * total_ot + C_support * max_ot
             + C_short * total_sh + C_inv * total_inv)
obj_cost = pl.value(model.objective)
cost_ok = abs(calc_cost - obj_cost) < 0.01
print(f"  30×{total_ot:.2f} + 40×{max_ot:.2f} + 3×{total_sh:.0f} + 2×{total_inv:.0f} = ${calc_cost:.2f}")
print(f"  Objective value = ${obj_cost:.2f}")
print(f"  Match: {'✓ PASS' if cost_ok else '✗ FAIL'}")
if not cost_ok:
    all_ok = False

# 10. Shortfall/Inventory logic check
print("\n[10] Shortfall/Inventory Logic (sh_j × inv_j = 0)")
print("-" * 50)
logic_ok = True
for j in parts:
    sh_val = pl.value(sh[j]) or 0
    inv_val = pl.value(inv[j]) or 0
    # A part should not have both shortfall AND inventory
    if sh_val > tol and inv_val > tol:
        print(f"  Part {j}: sh={sh_val:.1f} AND inv={inv_val:.1f}  ✗ FAIL (both nonzero)")
        logic_ok = False
        all_ok = False
if logic_ok:
    print("  No part has both shortfall and inventory  ✓ PASS")

# Final verdict
print("\n" + "=" * 60)
if all_ok:
    print("ALL CONSTRAINTS VERIFIED ✓ — Solution is feasible and optimal.")
else:
    print("⚠ SOME CONSTRAINTS FAILED — Review the issues above.")
print("=" * 60)