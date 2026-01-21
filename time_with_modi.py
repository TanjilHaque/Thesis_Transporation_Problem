import json
import time
import numpy as np

from transportation import Transportation
from vogels_approximation import VogelsApproximationMethod
from russels_approximation import RussellsApproximationMethod
from modi import MODI


def load_json_data(file_path):
    """Load transportation problem data from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    cost = np.array(data["costs"])
    supply = np.array(data["supply"])
    demand = np.array(data["demand"])
    return cost, supply, demand


def vam_with_modi(cost, supply, demand):
    # VAM
    trans_vam = Transportation(cost, supply, demand)
    trans_vam.setup_table(minimize=True)

    vam_method = VogelsApproximationMethod(trans_vam)

    start_vam = time.time()
    bfs_allocation = vam_method.solve(show_iter=False)
    end_vam = time.time()

    vam_time = end_vam - start_vam

    # MODI (using VAM BFS)
    modi = MODI(cost, bfs_allocation)

    start_modi = time.time()
    _ = modi.solve()
    end_modi = time.time()

    modi_time = end_modi - start_modi
    total_time = vam_time + modi_time

    return vam_time, modi_time, total_time


def ram_with_modi(cost, supply, demand):
    # RAM
    trans_ram = Transportation(cost, supply, demand)
    trans_ram.setup_table(minimize=True)

    ram_method = RussellsApproximationMethod(trans_ram)

    start_ram = time.time()
    bfs_allocation = ram_method.solve(show_iter=False)
    end_ram = time.time()

    ram_time = end_ram - start_ram

    # MODI (using RAM BFS)
    modi = MODI(cost, bfs_allocation)

    start_modi = time.time()
    _ = modi.solve()
    end_modi = time.time()

    modi_time = end_modi - start_modi
    total_time = ram_time + modi_time

    return ram_time, modi_time, total_time


def main():
    cost, supply, demand = load_json_data("ram_favorable_dataset.json")

    # VAM + MODI timing
    vam_time, vam_modi_time, vam_total = vam_with_modi(cost, supply, demand)
    print(f"VAM Time: {vam_time:.6f} seconds")
    print(f"VAM + MODI Time: {vam_total:.6f} seconds")
    print(f"MODI Time (after VAM): {vam_modi_time:.6f} seconds\n")

    # RAM + MODI timing
    ram_time, ram_modi_time, ram_total = ram_with_modi(cost, supply, demand)
    print(f"RAM Time: {ram_time:.6f} seconds")
    print(f"RAM + MODI Time: {ram_total:.6f} seconds")
    print(f"MODI Time (after RAM): {ram_modi_time:.6f} seconds\n")


if __name__ == "__main__":
    main()
