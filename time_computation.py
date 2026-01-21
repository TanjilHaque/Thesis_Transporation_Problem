import json
import numpy as np
import time

# Import existing classes
from transportation import Transportation
from vogels_approximation import VogelsApproximationMethod
from russels_approximation import RussellsApproximationMethod

def load_json_data(file_path):
    """Load transportation problem data from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    cost = np.array(data["costs"])
    supply = np.array(data["supply"])
    demand = np.array(data["demand"])
    return cost, supply, demand

def main():
    # Load JSON data
    #example1_from_reference_thesis_3x4
    cost, supply, demand = load_json_data("example10_works.json")

    # --- VOGEL'S METHOD ---
    trans_vam = Transportation(cost, supply, demand)
    trans_vam.setup_table(minimize=True)

    vam_method = VogelsApproximationMethod(trans_vam)

    start_vam = time.time()
    _ = vam_method.solve(show_iter=False)
    end_vam = time.time()
    vam_time = end_vam - start_vam

    # --- RUSSELL'S METHOD ---
    trans_ram = Transportation(cost, supply, demand)
    trans_ram.setup_table(minimize=True)

    ram_method = RussellsApproximationMethod(trans_ram)

    start_ram = time.time()
    _ = ram_method.solve(show_iter=False)
    end_ram = time.time()
    ram_time = end_ram - start_ram

    # --- Print time comparison ---
    print(f"Vogel's Approximation Method computation time: {vam_time:.6f} seconds")
    print(f"Russell's Approximation Method computation time: {ram_time:.6f} seconds")

if __name__ == "__main__":
    main()
