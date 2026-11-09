import pulp

# Data from your IE4515_SunOil_data.json
data = {
    "k": 2, "n": 5, "m": 5,
    "Demands": [12, 8, 14, 16, 7],
    "AvailableAmount": [[10, 20], [10, 20], [10, 20], [10, 20], [10, 20]],
    "TransportationCost": [
        [81, 92, 101, 130, 115],
        [117, 77, 108, 98, 100],
        [102, 105, 95, 119, 111],
        [115, 125, 90, 59, 74],
        [142, 100, 103, 105, 71]
    ],
    "AnnualOperatingCost": [
        [6000, 4500, 6500, 4100, 4000], # Fixed Costs for Low (k=0)
        [9000, 6750, 9750, 6150, 6000]  # Fixed Costs for High (k=1)
    ]
}

# 1. Initialize Model
prob = pulp.LpProblem("SunOil_Optimization", pulp.LpMinimize)

# 2. Variables
# x[i, j, k]
unitsTransported = pulp.LpVariable.dicts("units", 
    ((i, j, k) for i in range(5) for j in range(5) for k in range(2)), 
    lowBound=0, cat='Continuous')

# y[i, k]
plantExists = pulp.LpVariable.dicts("open", 
    ((i, k) for i in range(5) for k in range(2)), 
    cat='Binary')

# 3. Objective Function
obj_transport = pulp.lpSum(data["TransportationCost"][i][j] * unitsTransported[(i, j, k)] 
                           for i in range(5) for j in range(5) for k in range(2))

obj_fixed = pulp.lpSum(data["AnnualOperatingCost"][k][i] * plantExists[(i, k)] 
                       for i in range(5) for k in range(2))

prob += obj_transport + obj_fixed

# 4. Constraints
# Demand
for j in range(5):
    prob += pulp.lpSum(unitsTransported[(i, j, k)] for i in range(5) for k in range(2)) >= data["Demands"][j]

# Capacity
for i in range(5):
    for k in range(2):
        prob += pulp.lpSum(unitsTransported[(i, j, k)] for j in range(5)) <= \
                data["AvailableAmount"][i][k] * plantExists[(i, k)]

# Exclusivity (One size per location)
for i in range(5):
    prob += pulp.lpSum(plantExists[(i, k)] for k in range(2)) <= 1

# 5. Solve
prob.solve()

print(f"Total Minimized Cost: ${pulp.value(prob.objective)}")
for v in prob.variables():
    if v.varValue > 0:
        print(f"{v.name} = {v.varValue}")