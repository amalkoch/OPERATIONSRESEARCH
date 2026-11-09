import pulp as pl

# ============================================================
# Q2: MILP Model - Minimize Total Overtime (Week 1)
# ============================================================

# --- Sets ---
machines = [1, 2, 3, 4, 5]
parts = [1, 2, 3, 4, 5]

# Feasible machine-part pairs
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

R = 120      # regular hours per machine per week
OT_max = 48  # max overtime per machine per week
U = 168      # big-M value (R + OT_max)

# --- Model ---
model = pl.LpProblem("Q2_Min_Total_Overtime", pl.LpMinimize)

# --- Decision Variables ---
x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in F}
z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in F}
OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=OT_max) for i in machines}

# --- Objective: Minimize total overtime ---
model += pl.lpSum(OT[i] for i in machines), "Total_Overtime"

# --- Constraints ---

# 6.1 Demand satisfaction: sum of good units from all feasible machines = demand
for j in parts:
    model += (
        pl.lpSum(rates[i,j] * yields[j] * x[i,j] for i in machines if (i,j) in F)
        == demand[j],
        f"Demand_Part_{j}"
    )

# 6.2 Machine capacity: production hours + setup hours <= regular + overtime
for i in machines:
    model += (
        pl.lpSum(x[i,j] + setup[i,j] * z[i,j] for j in parts if (i,j) in F)
        <= R + OT[i],
        f"Capacity_Machine_{i}"
    )

# 6.4 Linking: if x_ij > 0 then z_ij = 1
for (i,j) in F:
    model += (x[i,j] <= U * z[i,j], f"Link_{i}_{j}")

# --- Solve ---
solver = pl.PULP_CBC_CMD(msg=0)
status = model.solve(solver)

# ============================================================
# Results
# ============================================================
print("=" * 60)
print("Q2: Minimize Total Overtime — Week 1")
print("=" * 60)
print(f"Status: {pl.LpStatus[status]}")
print(f"Optimal Total Overtime: {pl.value(model.objective):.2f} hours\n")

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

total_ot = sum(pl.value(OT[i]) or 0 for i in machines)
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