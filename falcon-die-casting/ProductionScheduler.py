import pulp as pl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class ProductionScheduler:
    """
    Production scheduling optimizer for Falcon Die Casting Company (FDC).
    Solves single-week MILP models with various objectives and constraints.
    """

    def __init__(self):
        """Initialize with FDC problem data."""
        self.machines = [1, 2, 3, 4, 5]
        self.parts = [1, 2, 3, 4, 5]
        self.F = [(1,1),(1,4),(2,1),(2,2),(3,2),(3,5),(4,2),(4,3),(5,4),(5,5)]

        self.rates = {
            (1,1): 40, (1,4): 60,
            (2,1): 35, (2,2): 25,
            (3,2): 30, (3,5): 45,
            (4,2): 35, (4,3): 50,
            (5,4): 60, (5,5): 50
        }

        self.yields = {1: 0.60, 2: 0.55, 3: 0.75, 4: 0.65, 5: 0.60}

        self.setup = {
            (1,1): 8,  (1,4): 8,
            (2,1): 10, (2,2): 8,
            (3,2): 10, (3,5): 24,
            (4,2): 8,  (4,3): 12,
            (5,4): 8,  (5,5): 20
        }

        self.R = 120       # regular hours per machine per week
        self.OT_max = 48   # max overtime per machine per week
        self.U = 168        # big-M (R + OT_max)

        # Demand data for all 12 weeks
        self.all_demand = {
            1:  {1:3500, 2:3000, 3:4000, 4:4000, 5:2800},
            2:  {1:3000, 2:2800, 3:4000, 4:4300, 5:2800},
            3:  {1:3000, 2:2000, 3:4000, 4:3500, 5:3000},
            4:  {1:3000, 2:3000, 3:4000, 4:3800, 5:2800},
            5:  {1:3000, 2:3000, 3:4000, 4:4000, 5:2800},
            6:  {1:3500, 2:2500, 3:4000, 4:3800, 5:2500},
            7:  {1:3500, 2:2500, 3:3800, 4:4000, 5:2500},
            8:  {1:3300, 2:3400, 3:3700, 4:4200, 5:2500},
            9:  {1:3300, 2:3400, 3:0,    4:4500, 5:3000},
            10: {1:3200, 2:3000, 3:0,    4:4500, 5:3000},
            11: {1:4500, 2:4000, 3:5000, 4:5000, 5:3800},
            12: {1:3000, 2:2800, 3:4000, 4:4300, 5:2800},
        }

        # Store results for comparison
        self.results = {}

    # ==============================================================
    # INPUT VALIDATION
    # ==============================================================

    def _validate_demand(self, demand):
        """
        Validate demand input before solving.
        Raises ValueError with descriptive message if invalid.
        """
        if not isinstance(demand, dict):
            raise ValueError("Demand must be a dictionary {part: units}.")

        for j in self.parts:
            if j not in demand:
                raise ValueError(f"Missing demand for Part {j}.")
            if not isinstance(demand[j], (int, float)):
                raise ValueError(f"Demand for Part {j} must be a number, got {type(demand[j]).__name__}.")
            if demand[j] < 0:
                raise ValueError(f"Demand for Part {j} is negative ({demand[j]}). Must be ≥ 0.")

    def _validate_initial_setup(self, initial_setup):
        """
        Validate initial setup pairs for Q5.
        Raises ValueError if any pair is not a feasible machine-part combination.
        """
        if not isinstance(initial_setup, set):
            raise ValueError("Initial setup must be a set of (machine, part) tuples.")

        for (i, j) in initial_setup:
            if i not in self.machines:
                raise ValueError(f"Invalid machine {i} in initial setup. Must be in {self.machines}.")
            if j not in self.parts:
                raise ValueError(f"Invalid part {j} in initial setup. Must be in {self.parts}.")
            if (i, j) not in self.F:
                raise ValueError(f"Machine {i} cannot produce Part {j}. Not a feasible pair.")

    def _validate_costs(self, **costs):
        """
        Validate cost parameters.
        Raises ValueError if any cost is negative.
        """
        for name, val in costs.items():
            if not isinstance(val, (int, float)):
                raise ValueError(f"Cost '{name}' must be a number, got {type(val).__name__}.")
            if val < 0:
                raise ValueError(f"Cost '{name}' is negative ({val}). Must be ≥ 0.")

    # ==============================================================
    # CORE MODEL BUILDERS
    # ==============================================================

    def solve_q2(self, demand, label="Q2"):
        """
        Q2: Minimize total overtime.
        Args:
            demand: dict {part: units}
            label: string label for storing results
        Returns:
            dict with solution details
        """
        self._validate_demand(demand)

        model = pl.LpProblem(f"{label}_Min_Total_OT", pl.LpMinimize)

        x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in self.F}
        z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in self.F}
        OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=self.OT_max) for i in self.machines}

        model += pl.lpSum(OT[i] for i in self.machines)

        for j in self.parts:
            model += (
                pl.lpSum(self.rates[i,j] * self.yields[j] * x[i,j]
                         for i in self.machines if (i,j) in self.F)
                == demand[j], f"Demand_{j}"
            )

        for i in self.machines:
            model += (
                pl.lpSum(x[i,j] + self.setup[i,j] * z[i,j]
                         for j in self.parts if (i,j) in self.F)
                <= self.R + OT[i], f"Cap_{i}"
            )

        for (i,j) in self.F:
            model += (x[i,j] <= self.U * z[i,j], f"Link_{i}_{j}")

        model.solve(pl.PULP_CBC_CMD(msg=0))

        result = self._extract_results(model, x, z, OT, demand, label)
        self.results[label] = result
        return result

    def solve_q3(self, demand, label="Q3"):
        """
        Q3: Minimize maximum overtime (minimax).
        Args:
            demand: dict {part: units}
            label: string label for storing results
        Returns:
            dict with solution details
        """
        self._validate_demand(demand)

        model = pl.LpProblem(f"{label}_Min_Max_OT", pl.LpMinimize)

        x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in self.F}
        z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in self.F}
        OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=self.OT_max) for i in self.machines}
        OT_max_var = pl.LpVariable("OT_max", lowBound=0, upBound=self.OT_max)

        model += OT_max_var

        for i in self.machines:
            model += (OT_max_var >= OT[i], f"MaxOT_{i}")

        for j in self.parts:
            model += (
                pl.lpSum(self.rates[i,j] * self.yields[j] * x[i,j]
                         for i in self.machines if (i,j) in self.F)
                == demand[j], f"Demand_{j}"
            )

        for i in self.machines:
            model += (
                pl.lpSum(x[i,j] + self.setup[i,j] * z[i,j]
                         for j in self.parts if (i,j) in self.F)
                <= self.R + OT[i], f"Cap_{i}"
            )

        for (i,j) in self.F:
            model += (x[i,j] <= self.U * z[i,j], f"Link_{i}_{j}")

        model.solve(pl.PULP_CBC_CMD(msg=0))

        result = self._extract_results(model, x, z, OT, demand, label)
        result["max_ot"] = pl.value(OT_max_var)
        self.results[label] = result
        return result

    def solve_q4(self, demand, c_machine=30, c_support=40, label="Q4"):
        """
        Q4: Minimize total cost (machine OT + support personnel).
        Args:
            demand: dict {part: units}
            c_machine: cost per machine OT hour ($)
            c_support: cost per support personnel hour ($)
            label: string label for storing results
        Returns:
            dict with solution details
        """
        self._validate_demand(demand)
        self._validate_costs(c_machine=c_machine, c_support=c_support)

        model = pl.LpProblem(f"{label}_Min_Cost", pl.LpMinimize)

        x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in self.F}
        z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in self.F}
        OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=self.OT_max) for i in self.machines}
        OT_max_var = pl.LpVariable("OT_max", lowBound=0, upBound=self.OT_max)

        model += c_machine * pl.lpSum(OT[i] for i in self.machines) + c_support * OT_max_var

        for i in self.machines:
            model += (OT_max_var >= OT[i], f"MaxOT_{i}")

        for j in self.parts:
            model += (
                pl.lpSum(self.rates[i,j] * self.yields[j] * x[i,j]
                         for i in self.machines if (i,j) in self.F)
                == demand[j], f"Demand_{j}"
            )

        for i in self.machines:
            model += (
                pl.lpSum(x[i,j] + self.setup[i,j] * z[i,j]
                         for j in self.parts if (i,j) in self.F)
                <= self.R + OT[i], f"Cap_{i}"
            )

        for (i,j) in self.F:
            model += (x[i,j] <= self.U * z[i,j], f"Link_{i}_{j}")

        model.solve(pl.PULP_CBC_CMD(msg=0))

        result = self._extract_results(model, x, z, OT, demand, label)
        result["max_ot"] = pl.value(OT_max_var)
        result["cost_machine"] = c_machine * result["total_ot"]
        result["cost_support"] = c_support * result["max_ot"]
        result["total_cost"] = pl.value(model.objective)
        self.results[label] = result
        return result

    def solve_q5(self, demand, initial_setup=None, label="Q5"):
        """
        Q5: Setup carryover — initial setups are free.
        Args:
            demand: dict {part: units}
            initial_setup: set of (machine, part) tuples with free setup
            label: string label for storing results
        Returns:
            dict with solution details
        """
        if initial_setup is None:
            initial_setup = {(1,1), (2,2), (3,5), (4,3), (5,4)}

        self._validate_demand(demand)
        self._validate_initial_setup(initial_setup)

        setup_eff = {}
        for (i,j) in self.F:
            setup_eff[i,j] = 0 if (i,j) in initial_setup else self.setup[i,j]

        model = pl.LpProblem(f"{label}_Setup_Carryover", pl.LpMinimize)

        x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in self.F}
        z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in self.F}
        OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=self.OT_max) for i in self.machines}

        model += pl.lpSum(OT[i] for i in self.machines)

        for j in self.parts:
            model += (
                pl.lpSum(self.rates[i,j] * self.yields[j] * x[i,j]
                         for i in self.machines if (i,j) in self.F)
                == demand[j], f"Demand_{j}"
            )

        for i in self.machines:
            model += (
                pl.lpSum(x[i,j] + setup_eff[i,j] * z[i,j]
                         for j in self.parts if (i,j) in self.F)
                <= self.R + OT[i], f"Cap_{i}"
            )

        for (i,j) in self.F:
            model += (x[i,j] <= self.U * z[i,j], f"Link_{i}_{j}")

        model.solve(pl.PULP_CBC_CMD(msg=0))

        result = self._extract_results(model, x, z, OT, demand, label, setup_eff)
        result["setup_saved"] = sum(
            self.setup[i,j] for (i,j) in initial_setup
            if pl.value(z[i,j]) and pl.value(z[i,j]) > 0.5
        )
        self.results[label] = result
        return result

    def solve_q6(self, demand, c_machine=30, c_support=40, c_short=3, c_inv=2, label="Q6"):
        """
        Q6: Shortfalls and penalties — demand may not be fully met.
        Args:
            demand: dict {part: units}
            c_machine, c_support: OT costs ($)
            c_short: penalty per unit shortfall ($)
            c_inv: inventory carrying cost per unit ($)
            label: string label for storing results
        Returns:
            dict with solution details
        """
        self._validate_demand(demand)
        self._validate_costs(c_machine=c_machine, c_support=c_support, c_short=c_short, c_inv=c_inv)

        model = pl.LpProblem(f"{label}_Shortfalls", pl.LpMinimize)

        x = {(i,j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for (i,j) in self.F}
        z = {(i,j): pl.LpVariable(f"z_{i}_{j}", cat="Binary") for (i,j) in self.F}
        OT = {i: pl.LpVariable(f"OT_{i}", lowBound=0, upBound=self.OT_max) for i in self.machines}
        OT_max_var = pl.LpVariable("OT_max", lowBound=0, upBound=self.OT_max)
        sh = {j: pl.LpVariable(f"sh_{j}", lowBound=0) for j in self.parts}
        inv = {j: pl.LpVariable(f"inv_{j}", lowBound=0) for j in self.parts}

        model += (
            c_machine * pl.lpSum(OT[i] for i in self.machines)
            + c_support * OT_max_var
            + c_short * pl.lpSum(sh[j] for j in self.parts)
            + c_inv * pl.lpSum(inv[j] for j in self.parts)
        )

        for i in self.machines:
            model += (OT_max_var >= OT[i], f"MaxOT_{i}")

        for j in self.parts:
            model += (
                pl.lpSum(self.rates[i,j] * self.yields[j] * x[i,j]
                         for i in self.machines if (i,j) in self.F)
                + sh[j] - inv[j] == demand[j], f"Demand_{j}"
            )

        for i in self.machines:
            model += (
                pl.lpSum(x[i,j] + self.setup[i,j] * z[i,j]
                         for j in self.parts if (i,j) in self.F)
                <= self.R + OT[i], f"Cap_{i}"
            )

        for (i,j) in self.F:
            model += (x[i,j] <= self.U * z[i,j], f"Link_{i}_{j}")

        model.solve(pl.PULP_CBC_CMD(msg=0))

        result = self._extract_results(model, x, z, OT, demand, label)
        result["max_ot"] = pl.value(OT_max_var)
        result["shortfall"] = {j: pl.value(sh[j]) or 0 for j in self.parts}
        result["inventory"] = {j: pl.value(inv[j]) or 0 for j in self.parts}
        result["total_shortfall"] = sum(result["shortfall"].values())
        result["total_inventory"] = sum(result["inventory"].values())
        result["total_cost"] = pl.value(model.objective)
        result["cost_machine"] = c_machine * result["total_ot"]
        result["cost_support"] = c_support * result["max_ot"]
        result["cost_short"] = c_short * result["total_shortfall"]
        result["cost_inv"] = c_inv * result["total_inventory"]
        self.results[label] = result
        return result

    # ==============================================================
    # RESULT EXTRACTION
    # ==============================================================

    def _extract_results(self, model, x, z, OT, demand, label, setup_override=None):
        """Extract solution values into a structured dict."""
        s = setup_override if setup_override else self.setup

        schedule = []
        for i in self.machines:
            for j in self.parts:
                if (i,j) in self.F:
                    xv = pl.value(x[i,j]) or 0
                    zv = pl.value(z[i,j]) or 0
                    if xv > 0.01:
                        schedule.append({
                            "machine": i, "part": j,
                            "setup_hr": s[i,j] * zv,
                            "prod_hr": xv,
                            "good_units": xv * self.rates[i,j] * self.yields[j],
                            "z": int(round(zv))
                        })

        ot_vals = {i: pl.value(OT[i]) or 0 for i in self.machines}
        machine_util = {}
        for i in self.machines:
            total = sum(
                (pl.value(x[i,j]) or 0) + s[i,j] * (pl.value(z[i,j]) or 0)
                for j in self.parts if (i,j) in self.F
            )
            machine_util[i] = {
                "total_used": total,
                "regular": min(total, self.R),
                "overtime": ot_vals[i]
            }

        produced = {}
        for j in self.parts:
            produced[j] = sum(
                (pl.value(x[i,j]) or 0) * self.rates[i,j] * self.yields[j]
                for i in self.machines if (i,j) in self.F
            )

        return {
            "label": label,
            "status": pl.LpStatus[model.status],
            "objective": pl.value(model.objective),
            "total_ot": sum(ot_vals.values()),
            "ot_by_machine": ot_vals,
            "schedule": schedule,
            "machine_util": machine_util,
            "produced": produced,
            "demand": demand,
        }

    # ==============================================================
    # PRINTING
    # ==============================================================

    def print_results(self, label):
        """Print formatted results for a given model."""
        r = self.results[label]
        print("=" * 60)
        print(f"{label} Results")
        print("=" * 60)
        print(f"Status: {r['status']}")
        print(f"Objective: {r['objective']:.2f}")
        print(f"Total Overtime: {r['total_ot']:.2f} hr")
        if "max_ot" in r:
            print(f"Max Overtime: {r['max_ot']:.2f} hr")
        if "total_cost" in r:
            print(f"Total Cost: ${r['total_cost']:.2f}")
        if "total_shortfall" in r:
            print(f"Total Shortfall: {r['total_shortfall']:.0f} units")

        print(f"\n{'Machine':<10} {'Part':<8} {'Setup(hr)':<12} {'Prod(hr)':<12} {'Good Units':<12}")
        print("-" * 54)
        for s in r["schedule"]:
            print(f"M{s['machine']:<9} P{s['part']:<7} {s['setup_hr']:<12.2f} {s['prod_hr']:<12.2f} {s['good_units']:<12.0f}")

        print(f"\n{'Machine':<10} {'Total(hr)':<12} {'Regular(hr)':<14} {'Overtime(hr)':<14}")
        print("-" * 50)
        for i in self.machines:
            u = r["machine_util"][i]
            print(f"M{i:<9} {u['total_used']:<12.2f} {u['regular']:<14.2f} {u['overtime']:<14.2f}")
        print(f"{'TOTAL':<22} {'':<14} {r['total_ot']:<14.2f}")

        print(f"\n{'Part':<8} {'Demand':<10} {'Produced':<12} {'Status'}")
        print("-" * 42)
        for j in self.parts:
            d = r["demand"][j]
            p = r["produced"][j]
            sh = r.get("shortfall", {}).get(j, 0)
            if sh > 0.5:
                s = f"✗ Short {sh:.0f}"
            elif abs(p - d) < 1:
                s = "✓ Met"
            else:
                s = f"? Diff {d-p:.0f}"
            print(f"P{j:<7} {d:<10} {p:<12.0f} {s}")

    # ==============================================================
    # CONSTRAINT VERIFICATION
    # ==============================================================

    def verify_constraints(self, label):
        """
        Verify all constraints for a solved model.
        Checks: demand, capacity, OT bounds, linking, non-negativity,
                binary, consistency, and model-specific constraints.
        Args:
            label: result label to verify
        Returns:
            True if all constraints pass, False otherwise
        """
        r = self.results[label]
        tol = 0.01
        all_ok = True

        print("\n" + "=" * 60)
        print(f"CONSTRAINT VERIFICATION — {label}")
        print("=" * 60)

        # 1. Demand satisfaction
        print("\n[1] Demand Satisfaction")
        print("-" * 50)
        for j in self.parts:
            produced = r["produced"][j]
            d = r["demand"][j]
            sh = r.get("shortfall", {}).get(j, 0)
            inv = r.get("inventory", {}).get(j, 0)
            if "shortfall" in r:
                lhs = produced + sh - inv
                diff = abs(lhs - d)
                ok = diff < tol
                print(f"  P{j}: prod={produced:.1f} + sh={sh:.1f} - inv={inv:.1f} = {lhs:.1f}, demand={d}  {'✓' if ok else '✗'}")
            else:
                diff = abs(produced - d)
                ok = diff < tol
                print(f"  P{j}: produced={produced:.1f}, demand={d}, diff={diff:.4f}  {'✓' if ok else '✗'}")
            if not ok:
                all_ok = False

        # 2. Machine capacity
        print("\n[2] Machine Capacity (prod + setup ≤ 120 + OT_i)")
        print("-" * 50)
        for i in self.machines:
            u = r["machine_util"][i]
            ot = r["ot_by_machine"][i]
            rhs = self.R + ot
            slack = rhs - u["total_used"]
            ok = slack >= -tol
            print(f"  M{i}: used={u['total_used']:.2f}, capacity={rhs:.2f}, slack={slack:.4f}  {'✓' if ok else '✗'}")
            if not ok:
                all_ok = False

        # 3. Overtime bounds
        print("\n[3] Overtime Bounds (0 ≤ OT_i ≤ 48)")
        print("-" * 50)
        for i in self.machines:
            ot = r["ot_by_machine"][i]
            ok = -tol <= ot <= self.OT_max + tol
            print(f"  M{i}: OT={ot:.2f}  {'✓' if ok else '✗'}")
            if not ok:
                all_ok = False

        # 4. Linking & consistency (z=1 when x>0)
        print("\n[4] Linking & Consistency")
        print("-" * 50)
        link_ok = True
        for s in r["schedule"]:
            if s["prod_hr"] > tol and s["z"] < 1:
                print(f"  M{s['machine']}-P{s['part']}: prod={s['prod_hr']:.2f} but z=0  ✗")
                link_ok = False
                all_ok = False
        if link_ok:
            print("  All active productions have z=1  ✓")

        # 5. Non-negativity
        print("\n[5] Non-negativity")
        print("-" * 50)
        neg_ok = True
        for s in r["schedule"]:
            if s["prod_hr"] < -tol:
                print(f"  M{s['machine']}-P{s['part']}: prod={s['prod_hr']:.4f}  ✗")
                neg_ok = False
                all_ok = False
        if neg_ok:
            print("  All production hours ≥ 0  ✓")

        # 6. Model-specific checks
        if "max_ot" in r:
            print("\n[6] Minimax (OT_max ≥ all OT_i)")
            print("-" * 50)
            for i in self.machines:
                ot = r["ot_by_machine"][i]
                ok = r["max_ot"] >= ot - tol
                print(f"  OT_max={r['max_ot']:.2f} >= OT_{i}={ot:.2f}  {'✓' if ok else '✗'}")
                if not ok:
                    all_ok = False

        if "total_cost" in r:
            print("\n[7] Cost Verification")
            print("-" * 50)
            obj = r["total_cost"]
            calc = r.get("cost_machine", 0) + r.get("cost_support", 0) + r.get("cost_short", 0) + r.get("cost_inv", 0)
            ok = abs(obj - calc) < 0.01
            print(f"  Calculated=${calc:.2f}, Objective=${obj:.2f}  {'✓' if ok else '✗'}")
            if not ok:
                all_ok = False

        if "shortfall" in r:
            print("\n[8] Shortfall/Inventory Logic (no part has both)")
            print("-" * 50)
            logic_ok = True
            for j in self.parts:
                sh = r["shortfall"][j]
                inv = r["inventory"][j]
                if sh > tol and inv > tol:
                    print(f"  P{j}: sh={sh:.1f} AND inv={inv:.1f}  ✗")
                    logic_ok = False
                    all_ok = False
            if logic_ok:
                print("  No part has both shortfall and inventory  ✓")

        # Final verdict
        print("\n" + "=" * 60)
        if all_ok:
            print(f"{label}: ALL CONSTRAINTS VERIFIED ✓")
        else:
            print(f"{label}: ⚠ SOME CONSTRAINTS FAILED")
        print("=" * 60)
        return all_ok

    # ==============================================================
    # SENSITIVITY ANALYSIS
    # ==============================================================

    def sensitivity_yield(self, base_demand, question_func, label_prefix,
                          yield_changes=[0.05, 0.10, 0.15]):
        """
        Analyze impact of yield improvements on total overtime.
        Args:
            base_demand: demand dict
            question_func: method to call (e.g., self.solve_q2)
            label_prefix: e.g., "Q2"
            yield_changes: list of yield improvements to test
        Returns:
            dict of results
        """
        print("=" * 60)
        print(f"Sensitivity Analysis: Yield Improvement ({label_prefix})")
        print("=" * 60)

        original_yields = self.yields.copy()
        sens_results = {"base": self.results.get(label_prefix, question_func(base_demand, label_prefix))}

        print(f"{'Yield Change':<15} {'Total OT(hr)':<15} {'Change(hr)':<12} {'Change(%)'}")
        print("-" * 55)
        base_ot = sens_results["base"]["total_ot"]
        print(f"{'Baseline':<15} {base_ot:<15.2f} {'—':<12} {'—'}")

        for delta in yield_changes:
            self.yields = {j: min(1.0, original_yields[j] + delta) for j in self.parts}
            lbl = f"{label_prefix}_y+{int(delta*100)}"
            r = question_func(base_demand, lbl)
            sens_results[f"+{int(delta*100)}%"] = r
            change = r["total_ot"] - base_ot
            pct = (change / base_ot) * 100 if base_ot > 0 else 0
            print(f"+{int(delta*100)}%{'':<12} {r['total_ot']:<15.2f} {change:<12.2f} {pct:.1f}%")

        self.yields = original_yields
        return sens_results

    def sensitivity_demand(self, base_demand, question_func, label_prefix,
                           demand_changes=[-0.10, -0.05, 0.05, 0.10]):
        """
        Analyze impact of uniform demand changes on total overtime.
        Args:
            base_demand: demand dict
            question_func: method to call
            label_prefix: e.g., "Q2"
            demand_changes: list of demand multipliers
        Returns:
            dict of results
        """
        print("=" * 60)
        print(f"Sensitivity Analysis: Demand Change ({label_prefix})")
        print("=" * 60)

        sens_results = {"base": self.results.get(label_prefix, question_func(base_demand, label_prefix))}
        base_ot = sens_results["base"]["total_ot"]

        print(f"{'Demand Change':<15} {'Total OT(hr)':<15} {'Change(hr)':<12} {'Status'}")
        print("-" * 55)
        print(f"{'Baseline':<15} {base_ot:<15.2f} {'—':<12} {'Optimal'}")

        for delta in demand_changes:
            new_demand = {j: int(base_demand[j] * (1 + delta)) for j in self.parts}
            lbl = f"{label_prefix}_d{'+' if delta>=0 else ''}{int(delta*100)}"
            r = question_func(new_demand, lbl)
            change = r["total_ot"] - base_ot
            print(f"{'+' if delta>=0 else ''}{int(delta*100)}%{'':<12} {r['total_ot']:<15.2f} {change:<12.2f} {r['status']}")
            sens_results[f"{'+' if delta>=0 else ''}{int(delta*100)}%"] = r

        return sens_results

    def sensitivity_all_weeks(self):
        """
        Solve Q2 for all 12 weeks and show overtime trend.
        Returns:
            dict {week: result}
        """
        print("=" * 60)
        print("12-Week Overtime Trend (Q2: Min Total OT)")
        print("=" * 60)

        week_results = {}
        print(f"{'Week':<8} {'Total OT(hr)':<15} {'Max OT(hr)':<13} {'Status'}")
        print("-" * 45)

        for w in range(1, 13):
            d = self.all_demand[w]
            lbl = f"W{w}"
            # Try Q2 first; if infeasible, use Q6
            r = self.solve_q2(d, lbl)
            if r["status"] == "Optimal":
                max_ot = max(r["ot_by_machine"].values())
                print(f"W{w:<7} {r['total_ot']:<15.2f} {max_ot:<13.2f} {r['status']}")
            else:
                print(f"W{w:<7} {'—':<15} {'—':<13} {r['status']} (capacity exceeded)")
            week_results[w] = r

        return week_results

    # ==============================================================
    # VISUALIZATION
    # ==============================================================

    def plot_overtime_comparison(self, labels=None):
        """Bar chart comparing overtime by machine across models."""
        if labels is None:
            labels = [l for l in ["Q2", "Q3", "Q4", "Q5"] if l in self.results]

        fig, ax = plt.subplots(figsize=(10, 6))
        n = len(labels)
        width = 0.15
        x_pos = np.arange(len(self.machines))

        for idx, lbl in enumerate(labels):
            r = self.results[lbl]
            ot_vals = [r["ot_by_machine"][i] for i in self.machines]
            ax.bar(x_pos + idx * width, ot_vals, width, label=lbl)

        ax.set_xlabel("Machine")
        ax.set_ylabel("Overtime (hours)")
        ax.set_title("Overtime by Machine — Model Comparison")
        ax.set_xticks(x_pos + width * (n - 1) / 2)
        ax.set_xticklabels([f"M{i}" for i in self.machines])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("overtime_comparison.png", dpi=150)
        plt.close()
        print("Saved: overtime_comparison.png")

    def plot_production_schedule(self, label):
        """Stacked bar chart showing production hours per machine."""
        r = self.results[label]
        fig, ax = plt.subplots(figsize=(10, 6))

        machine_data = defaultdict(lambda: defaultdict(float))
        machine_setup = defaultdict(float)
        for s in r["schedule"]:
            machine_data[s["machine"]][s["part"]] = s["prod_hr"]
            machine_setup[s["machine"]] += s["setup_hr"]

        x_pos = np.arange(len(self.machines))
        bottoms = np.zeros(len(self.machines))
        colors = plt.cm.Set2(np.linspace(0, 1, 6))

        # Setup time bars
        setup_vals = [machine_setup.get(i, 0) for i in self.machines]
        ax.bar(x_pos, setup_vals, 0.6, label="Setup", color=colors[5], alpha=0.7)
        bottoms = np.array(setup_vals)

        # Production bars per part
        for j in self.parts:
            vals = [machine_data[i].get(j, 0) for i in self.machines]
            ax.bar(x_pos, vals, 0.6, bottom=bottoms, label=f"Part {j}", color=colors[j-1])
            bottoms += np.array(vals)

        ax.axhline(y=self.R, color="red", linestyle="--", alpha=0.7, label="Regular time (120hr)")
        ax.set_xlabel("Machine")
        ax.set_ylabel("Hours")
        ax.set_title(f"Production Schedule — {label}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"M{i}" for i in self.machines])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"schedule_{label}.png", dpi=150)
        plt.close()
        print(f"Saved: schedule_{label}.png")

    def plot_weekly_trend(self, week_results):
        """Line chart of overtime across 12 weeks."""
        weeks = sorted(week_results.keys())
        ot_vals = []
        for w in weeks:
            r = week_results[w]
            ot_vals.append(r["total_ot"] if r["status"] == "Optimal" else None)

        fig, ax = plt.subplots(figsize=(10, 5))
        valid = [(w, o) for w, o in zip(weeks, ot_vals) if o is not None]
        invalid = [(w, 0) for w, o in zip(weeks, ot_vals) if o is None]

        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid],
                    "b-o", linewidth=2, markersize=8, label="Total OT")
        if invalid:
            for w, _ in invalid:
                ax.axvline(x=w, color="red", linestyle="--", alpha=0.5)
            ax.plot([], [], "r--", label="Infeasible")

        ax.set_xlabel("Week")
        ax.set_ylabel("Total Overtime (hours)")
        ax.set_title("12-Week Overtime Trend")
        ax.set_xticks(weeks)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("weekly_trend.png", dpi=150)
        plt.close()
        print("Saved: weekly_trend.png")

    def plot_shortfall_analysis(self, label="Q6"):
        """Bar chart showing demand vs produced vs shortfall."""
        r = self.results[label]
        fig, ax = plt.subplots(figsize=(8, 5))

        x_pos = np.arange(len(self.parts))
        demand_vals = [r["demand"][j] for j in self.parts]
        produced_vals = [r["produced"][j] for j in self.parts]
        shortfall_vals = [r["shortfall"][j] for j in self.parts]

        ax.bar(x_pos - 0.2, demand_vals, 0.35, label="Demand", color="steelblue")
        ax.bar(x_pos + 0.2, produced_vals, 0.35, label="Produced", color="seagreen")

        for idx, sh in enumerate(shortfall_vals):
            if sh > 0.5:
                ax.annotate(f"Short: {sh:.0f}",
                           xy=(idx + 0.2, produced_vals[idx]),
                           xytext=(idx + 0.5, produced_vals[idx] + 200),
                           arrowprops=dict(arrowstyle="->", color="red"),
                           color="red", fontsize=9)

        ax.set_xlabel("Part")
        ax.set_ylabel("Units")
        ax.set_title(f"Demand vs Production — {label}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Part {j}" for j in self.parts])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"shortfall_{label}.png", dpi=150)
        plt.close()
        print(f"Saved: shortfall_{label}.png")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    fdc = ProductionScheduler()
    w1 = fdc.all_demand[1]
    w11 = fdc.all_demand[11]

    # --- Solve all models ---
    print("\n" + "=" * 60)
    print("SOLVING ALL MODELS")
    print("=" * 60)

    fdc.solve_q2(w1, "Q2")
    fdc.print_results("Q2")

    fdc.solve_q3(w1, "Q3")
    fdc.print_results("Q3")

    fdc.solve_q4(w1, label="Q4")
    fdc.print_results("Q4")

    fdc.solve_q5(w1, label="Q5")
    fdc.print_results("Q5")

    fdc.solve_q6(w11, label="Q6")
    fdc.print_results("Q6")

    # --- Verify all models ---
    print("\n" + "=" * 60)
    print("VERIFYING ALL MODELS")
    print("=" * 60)
    for lbl in ["Q2", "Q3", "Q4", "Q5", "Q6"]:
        fdc.verify_constraints(lbl)

    # --- Summary comparison ---
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<8} {'Total OT(hr)':<15} {'Max OT(hr)':<13} {'Cost($)':<12} {'Shortfall'}")
    print("-" * 60)
    for lbl in ["Q2", "Q3", "Q4", "Q5", "Q6"]:
        r = fdc.results[lbl]
        max_ot = r.get("max_ot", max(r["ot_by_machine"].values()))
        cost = r.get("total_cost", "—")
        sh = r.get("total_shortfall", 0)
        cost_str = f"${cost:.2f}" if isinstance(cost, float) else cost
        sh_str = f"{sh:.0f}" if sh > 0 else "0"
        print(f"{lbl:<8} {r['total_ot']:<15.2f} {max_ot:<13.2f} {cost_str:<12} {sh_str}")

    # --- Sensitivity Analysis ---
    print("\n")
    fdc.sensitivity_yield(w1, fdc.solve_q2, "Q2")
    print("\n")
    fdc.sensitivity_demand(w1, fdc.solve_q2, "Q2")
    print("\n")
    week_results = fdc.sensitivity_all_weeks()

    # --- Plots ---
    fdc.plot_overtime_comparison()
    fdc.plot_production_schedule("Q2")
    fdc.plot_production_schedule("Q5")
    fdc.plot_weekly_trend(week_results)
    fdc.plot_shortfall_analysis("Q6")