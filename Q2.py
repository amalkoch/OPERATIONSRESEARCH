import pulp
data = {
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
        [6000, 4500, 6500, 4100, 4000],   # low capacity
        [9000, 6750, 9750, 6150, 6000]    # high capacity
    ]
}
prob = pulp.LpProblem("SunOil_Optimization", pulp.LpMinimize)
unitsTransported = pulp.LpVariable.dicts(
    "units",
    ((i, j, k) for i in range(5) for j in range(5) for k in range(2)),
    lowBound=0
)

plantExists = pulp.LpVariable.dicts(
    "open",
    ((i, k) for i in range(5) for k in range(2)),
    cat="Binary"
)
prob += (
    pulp.lpSum(
        data["TransportationCost"][i][j] * unitsTransported[(i, j, k)]
        for i in range(5) for j in range(5) for k in range(2)
    )
    +
    pulp.lpSum(
        data["AnnualOperatingCost"][k][i] * plantExists[(i, k)]
        for i in range(5) for k in range(2)
    )
)

#demand constraints
for j in range(5):
    prob += pulp.lpSum(
        unitsTransported[(i, j, k)]
        for i in range(5) for k in range(2)
    ) >= data["Demands"][j]

# capacity constraints
for i in range(5):
    for k in range(2):
        prob += pulp.lpSum(
            unitsTransported[(i, j, k)]
            for j in range(5)
        ) <= data["AvailableAmount"][i][k] * plantExists[(i, k)]

# at most one plant size per location
for i in range(5):
    prob += pulp.lpSum(
        plantExists[(i, k)] for k in range(2)
    ) <= 1

# solve
prob.solve()

print("Total cost:", pulp.value(prob.objective))

for v in prob.variables():
    if v.varValue and v.varValue > 0:
        print(v.name, "=", v.varValue)
