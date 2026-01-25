"""
Sensitivity Analysis for Transportation Problems
Corrected OAT Sensitivity using IT2 Fuzzy Sets

Fixes:
1. Enforced defuzzified cost shift (not only support widening)
2. Guarded sensitivity conclusion when changes are zero
"""

import json
import numpy as np
import time
from typing import Tuple, List

from transportation import Transportation
from vogels_approximation import VogelsApproximationMethod
from russels_approximation import RussellsApproximationMethod
from modi import MODI


# ========================================
# BASIC UTILITIES
# ========================================

def load_transportation_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return (
        np.array(data["costs"], dtype=float),
        np.array(data["supply"], dtype=float),
        np.array(data["demand"], dtype=float)
    )


def calculate_cost(costs, bfs):
    return sum(costs[int(r[1:])][int(c[1:])] * v for r, c, v in bfs)


def find_worst_cell(costs, bfs):
    worst = (-1, -1, 0.0)
    max_contrib = -1
    for r, c, v in bfs:
        i, j = int(r[1:]), int(c[1:])
        contrib = costs[i][j] * v
        if contrib > max_contrib:
            max_contrib = contrib
            worst = (i, j, contrib)
    return worst


# ========================================
# IT2 FUZZY CONVERSION (SHIFTED)
# ========================================

def crisp_to_it2_shifted(x, p):
    """
    IT2 trapezoid with asymmetric shift to force defuzzification change
    """
    shift = x * p
    width = 0.4 * x

    au = x - width / 2 + shift
    bu = x - width / 4 + shift
    cu = x + width / 4 + shift
    du = x + width / 2 + shift

    al = x - width / 3 + shift
    bl = x - width / 6 + shift
    cl = x + width / 6 + shift
    dl = x + width / 3 + shift

    return {
        "umf": [max(0.01, au), bu, cu, du, 1.0],
        "lmf": [max(0.01, al), bl, cl, dl, 0.7]
    }


def defuzzify(it2):
    au, bu, cu, du, _ = it2["umf"]
    al, bl, cl, dl, _ = it2["lmf"]
    return (au + bu + cu + du + al + bl + cl + dl) / 8


# ========================================
# OAT FILE CREATION (FIXED)
# ========================================

def create_oat_files(costs, supply, demand, cell, levels, tag):
    i, j = cell
    files = []

    for p in levels:
        perturbed = costs.copy()
        it2 = crisp_to_it2_shifted(costs[i, j], p)
        perturbed[i, j] = defuzzify(it2)

        fname = f"crisp_oat_{tag}_{int(p*100)}.json"
        with open(fname, "w") as f:
            json.dump({
                "costs": perturbed.tolist(),
                "supply": supply.tolist(),
                "demand": demand.tolist()
            }, f, indent=2)

        files.append(fname)
        print(f"✓ {fname}: {costs[i,j]:.2f} → {perturbed[i,j]:.2f}")

    return files


# ========================================
# METHOD RUNNER
# ========================================

def run(json_path, method):
    costs, supply, demand = load_transportation_data(json_path)
    tp = Transportation(costs, supply, demand)
    tp.setup_table(minimize=True)

    solver = VogelsApproximationMethod(tp) if method == "VAM" else RussellsApproximationMethod(tp)

    bfs = solver.solve(False)
    init_cost = calculate_cost(costs, bfs)

    modi = MODI(costs, bfs)
    _, opt_cost = modi.solve()

    worst = find_worst_cell(costs, bfs)

    return init_cost, opt_cost, worst


# ========================================
# MAIN
# ========================================

def main():
    BASE = "works100x100.json"
    LEVELS = [0.05, 0.10, 0.15]

    base_costs, base_supply, base_demand = load_transportation_data(BASE)

    vam_init, vam_opt, vam_worst = run(BASE, "VAM")
    ram_init, ram_opt, ram_worst = run(BASE, "RAM")

    print("\nVAM Worst Cell:", vam_worst)
    print("RAM Worst Cell:", ram_worst)

    vam_files = create_oat_files(base_costs, base_supply, base_demand, (vam_worst[0], vam_worst[1]), LEVELS, "vam")
    ram_files = create_oat_files(base_costs, base_supply, base_demand, (ram_worst[0], ram_worst[1]), LEVELS, "ram")

    vam_changes = []
    ram_changes = []

    for f in vam_files:
        _, opt, _ = run(f, "VAM")
        vam_changes.append(abs(opt - vam_opt))

    for f in ram_files:
        _, opt, _ = run(f, "RAM")
        ram_changes.append(abs(opt - ram_opt))

    vam_avg = np.mean(vam_changes)
    ram_avg = np.mean(ram_changes)

    print("\nAverage Cost Change VAM:", vam_avg)
    print("Average Cost Change RAM:", ram_avg)

    if vam_avg == 0 and ram_avg == 0:
        print("\nConclusion: BOTH methods are ROBUST (no sensitivity detected)")
    elif vam_avg > ram_avg:
        print("\nConclusion: VAM is MORE sensitive")
        print(f"Sensitivity Ratio: {vam_avg / ram_avg:.2f}x")
    else:
        print("\nConclusion: RAM is MORE sensitive")
        print(f"Sensitivity Ratio: {ram_avg / vam_avg:.2f}x")


if __name__ == "__main__":
    main()
