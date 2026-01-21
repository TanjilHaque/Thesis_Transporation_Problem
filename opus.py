import json
import numpy as np
import time

# Import existing classes
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

def main():
    # Load JSON data
    cost, supply, demand = load_json_data("ram_favorable_dataset.json")

    print("="*60)
    print("TIME COMPARISON: VAM+MODI vs RAM+MODI")
    print("="*60)

    # --- VOGEL'S METHOD + MODI ---
    trans_vam = Transportation(cost, supply, demand)
    trans_vam.setup_table(minimize=True)

    vam_method = VogelsApproximationMethod(trans_vam)

    start_vam_total = time.time()
    
    # VAM initial BFS
    start_vam = time.time()
    bfs_allocation_vam = vam_method.solve(show_iter=False)
    end_vam = time.time()
    vam_time = end_vam - start_vam

    # MODI optimization for VAM
    start_modi_vam = time.time()
    modi_vam = MODI(cost, bfs_allocation_vam)
    optimal_allocation_vam = modi_vam.solve()
    end_modi_vam = time.time()
    modi_vam_time = end_modi_vam - start_modi_vam
    
    end_vam_total = time.time()
    vam_total_time = end_vam_total - start_vam_total

    vam_initial_cost = sum(
        cost[int(r[1:])][int(c[1:])] * v
        for r, c, v in bfs_allocation_vam
    )
    vam_optimal_cost = modi_vam.cost_value()

    # --- RUSSELL'S METHOD + MODI ---
    trans_ram = Transportation(cost, supply, demand)
    trans_ram.setup_table(minimize=True)

    ram_method = RussellsApproximationMethod(trans_ram)

    start_ram_total = time.time()
    
    # RAM initial BFS
    start_ram = time.time()
    bfs_allocation_ram = ram_method.solve(show_iter=False)
    end_ram = time.time()
    ram_time = end_ram - start_ram

    # MODI optimization for RAM
    start_modi_ram = time.time()
    modi_ram = MODI(cost, bfs_allocation_ram)
    optimal_allocation_ram = modi_ram.solve()
    end_modi_ram = time.time()
    modi_ram_time = end_modi_ram - start_modi_ram
    
    end_ram_total = time.time()
    ram_total_time = end_ram_total - start_ram_total

    ram_initial_cost = sum(
        cost[int(r[1:])][int(c[1:])] * v
        for r, c, v in bfs_allocation_ram
    )
    ram_optimal_cost = modi_ram.cost_value()

    # --- Print Results ---
    print("\n" + "-"*60)
    print("VOGEL'S APPROXIMATION METHOD (VAM)")
    print("-"*60)
    print(f"VAM Initial BFS Time:        {vam_time:.6f} seconds")
    print(f"MODI Optimization Time:      {modi_vam_time:.6f} seconds")
    print(f"VAM + MODI Total Time:       {vam_total_time:.6f} seconds")
    print(f"Initial Cost (VAM):          {vam_initial_cost}")
    print(f"Optimal Cost (VAM+MODI):     {vam_optimal_cost}")

    print("\n" + "-"*60)
    print("RUSSELL'S APPROXIMATION METHOD (RAM)")
    print("-"*60)
    print(f"RAM Initial BFS Time:        {ram_time:.6f} seconds")
    print(f"MODI Optimization Time:      {modi_ram_time:.6f} seconds")
    print(f"RAM + MODI Total Time:       {ram_total_time:.6f} seconds")
    print(f"Initial Cost (RAM):          {ram_initial_cost}")
    print(f"Optimal Cost (RAM+MODI):     {ram_optimal_cost}")

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Time Difference (Total):     {abs(vam_total_time - ram_total_time):.6f} seconds")
    if vam_total_time < ram_total_time:
        print(f"Winner: VAM+MODI is {((ram_total_time - vam_total_time) / ram_total_time * 100):.2f}% faster")
    else:
        print(f"Winner: RAM+MODI is {((vam_total_time - ram_total_time) / vam_total_time * 100):.2f}% faster")
    
    print(f"\nBoth methods reach optimal cost: {vam_optimal_cost == ram_optimal_cost}")
    print("="*60)

if __name__ == "__main__":
    main()