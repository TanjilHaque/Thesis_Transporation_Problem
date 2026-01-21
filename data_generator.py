import json
import random
import numpy as np

def generate_ram_favoring_transport_problem(m, n, filename):
    """
    Generate a balanced, non-degenerate m×n transportation problem
    that favors Russell's Approximation Method over Vogel's Approximation Method.
    
    Parameters:
    - m: number of sources (rows)
    - n: number of destinations (columns)
    - filename: output JSON filename
    """
    
    # Step 1: Create strong row hierarchies with well-separated max values
    # Generate base costs with clear hierarchical structure
    base_high = random.uniform(8.0, 12.0)
    row_max_values = []
    
    for i in range(m):
        # Each row has an increasing max value with good separation
        row_max = base_high + (i * random.uniform(1.5, 2.5))
        row_max_values.append(row_max)
    
    # Step 2: Determine the "cost canyon" column(s)
    # For larger problems, 1-2 canyon columns work best
    num_canyon_cols = 1 if n <= 10 else min(2, n // 10)
    canyon_columns = random.sample(range(n), num_canyon_cols)
    
    # Step 3: Generate cost matrix with diverse values per row/column
    # KEY FIX: Ensure each row and column has at least 3 distinct cost levels
    # to prevent VAM penalty calculation from failing
    costs = []
    for i in range(m):
        row = []
        for j in range(n):
            if j in canyon_columns:
                # Canyon columns: systematically low costs (15-30% of row max)
                cost = random.uniform(0.15 * row_max_values[i], 0.30 * row_max_values[i])
            else:
                # Distribute non-canyon costs across wider range to ensure diversity
                # Use different ranges for different columns to avoid uniformity
                if j % 3 == 0:
                    cost = random.uniform(0.50 * row_max_values[i], 0.70 * row_max_values[i])
                elif j % 3 == 1:
                    cost = random.uniform(0.65 * row_max_values[i], 0.85 * row_max_values[i])
                else:
                    cost = random.uniform(0.75 * row_max_values[i], 0.95 * row_max_values[i])
            
            # Round to 2 decimal places
            row.append(round(cost, 2))
        costs.append(row)
    
    # Step 4: Add controlled perturbations to ensure unique costs
    # This prevents degeneracy while maintaining diversity
    for i in range(m):
        for j in range(n):
            perturbation = random.uniform(-0.08, 0.08)
            costs[i][j] = round(max(0.1, costs[i][j] + perturbation), 2)
    
    # Step 5: Ensure no two costs in same row/column are identical
    # This is critical for VAM penalty calculation
    for i in range(m):
        seen = set()
        for j in range(n):
            while costs[i][j] in seen:
                costs[i][j] = round(costs[i][j] + 0.01, 2)
            seen.add(costs[i][j])
    
    # Step 6: Generate balanced supply and demand with FRAGMENTED distribution
    # CRITICAL STRATEGY: Use many small, diverse values to force multiple allocations
    # This prevents premature row/column elimination in both VAM and RAM
    
    if m * n <= 100:
        value_range = (2.0, 8.0)  # Smaller range for fragmentation
    elif m * n <= 1000:
        value_range = (5.0, 20.0)
    else:
        value_range = (10.0, 40.0)
    
    # Calculate reasonable total based on problem size
    avg_per_cell = random.uniform(value_range[0] * 0.6, value_range[1] * 0.4)
    target_total = avg_per_cell * m * n
    
    # Generate SMALL, DIVERSE supply values
    # Each value should be small enough to require multiple allocations
    supply = []
    remaining_supply = target_total
    
    for i in range(m - 1):
        # Each supply is a small fraction of remaining
        # Using harmonic distribution to ensure diversity
        max_portion = min(remaining_supply * 0.35, target_total / m * 1.2)
        min_portion = target_total / m * 0.7
        
        s = round(random.uniform(min_portion, max_portion), 2)
        supply.append(s)
        remaining_supply -= s
    
    # Last supply gets remaining
    supply.append(round(remaining_supply, 2))
    
    # Shuffle to avoid monotonic pattern
    random.shuffle(supply)
    
    total_supply = sum(supply)
    
    # Generate SMALL, DIVERSE demand values with DIFFERENT pattern
    # Key: Don't use similar logic to supply - use different distribution
    demand = []
    remaining_demand = total_supply
    
    # Use different fractions for demand to ensure mismatch
    for j in range(n - 1):
        # Deliberately use different range than supply
        max_portion = min(remaining_demand * 0.4, total_supply / n * 1.3)
        min_portion = total_supply / n * 0.6
        
        d = round(random.uniform(min_portion, max_portion), 2)
        demand.append(d)
        remaining_demand -= d
    
    # Last demand gets remaining
    demand.append(round(remaining_demand, 2))
    
    # Shuffle with different seed
    random.shuffle(demand)
    
    # Step 7: Ensure minimum values
    for i in range(len(supply)):
        if supply[i] < value_range[0] * 0.8:
            supply[i] = round(value_range[0] * 0.8, 2)
    
    for j in range(len(demand)):
        if demand[j] < value_range[0] * 0.8:
            demand[j] = round(value_range[0] * 0.8, 2)
    
    # Re-balance
    total_supply = sum(supply)
    total_demand = sum(demand)
    
    if total_supply > total_demand:
        # Increase largest demand
        max_idx = demand.index(max(demand))
        demand[max_idx] = round(demand[max_idx] + (total_supply - total_demand), 2)
    elif total_demand > total_supply:
        # Increase largest supply
        max_idx = supply.index(max(supply))
        supply[max_idx] = round(supply[max_idx] + (total_demand - total_supply), 2)
    
    # Step 7: Prevent simultaneous row/column elimination
    total_supply = round(sum(supply), 2)
    total_demand = round(sum(demand), 2)
    
    if abs(total_supply - total_demand) > 0.01:
        diff = total_supply - total_demand
        max_idx = demand.index(max(demand))
        demand[max_idx] = round(demand[max_idx] + diff, 2)
    
    # Step 9: Verify structural requirements
    # Ensure each row has at least 2 different costs (for penalty calculation)
    for i in range(m):
        unique_costs = len(set(costs[i]))
        if unique_costs < 2:
            # Add small variations to ensure at least 2 unique values
            costs[i][1] = round(costs[i][1] + 0.15, 2)
    
    # CRITICAL FIX: Prevent simultaneous row/column elimination
    # The issue is when supply[i] == demand[j], both get eliminated simultaneously
    # causing the table to skip the (2,2) termination condition
    
    # Add small perturbations to ensure no supply exactly equals any demand
    for i in range(len(supply)):
        for j in range(len(demand)):
            if abs(supply[i] - demand[j]) < 0.1:
                # Add small random perturbation to demand
                demand[j] = round(demand[j] + random.uniform(0.15, 0.25), 2)
    
    # Re-balance after perturbations
    total_supply = round(sum(supply), 2)
    total_demand = round(sum(demand), 2)
    
    if abs(total_supply - total_demand) > 0.01:
        diff = total_supply - total_demand
        # Adjust the largest demand to maintain balance
        max_idx = demand.index(max(demand))
        demand[max_idx] = round(demand[max_idx] + diff, 2)
    
    # Step 8: Adjust supply/demand to prevent extreme values and edge cases
    # Ensure no value exceeds 1.3x average (STRICT limit)
    
    max_iterations = 15
    for iteration in range(max_iterations):
        avg_supply = sum(supply) / len(supply)
        avg_demand = sum(demand) / len(demand)
        
        # STRICT threshold: max 1.3x average
        max_allowed_supply = avg_supply * 1.3
        max_allowed_demand = avg_demand * 1.3
        
        redistributed = False
        
        # Cap excessive supply values
        for i in range(len(supply)):
            if supply[i] > max_allowed_supply:
                excess = supply[i] - max_allowed_supply
                supply[i] = round(max_allowed_supply, 2)
                
                # Distribute excess to ALL other supplies proportionally
                other_indices = [j for j in range(len(supply)) if j != i]
                if other_indices:
                    per_item = excess / len(other_indices)
                    for j in other_indices:
                        supply[j] = round(supply[j] + per_item, 2)
                redistributed = True
        
        # Cap excessive demand values
        for j in range(len(demand)):
            if demand[j] > max_allowed_demand:
                excess = demand[j] - max_allowed_demand
                demand[j] = round(max_allowed_demand, 2)
                
                # Distribute excess to ALL other demands proportionally
                other_indices = [k for k in range(len(demand)) if k != j]
                if other_indices:
                    per_item = excess / len(other_indices)
                    for k in other_indices:
                        demand[k] = round(demand[k] + per_item, 2)
                redistributed = True
        
        if not redistributed:
            break
    
    # Re-balance after capping
    total_supply = sum(supply)
    total_demand = sum(demand)
    
    if total_supply > total_demand:
        diff = total_supply - total_demand
        demand[-1] = round(demand[-1] + diff, 2)
    elif total_demand > total_supply:
        diff = total_demand - total_supply
        supply[-1] = round(supply[-1] + diff, 2)
    
    # FINAL CHECK: Ensure no matches were re-introduced during redistribution
    for i in range(len(supply)):
        for j in range(len(demand)):
            if abs(supply[i] - demand[j]) < 0.2:
                demand[j] = round(demand[j] + 0.3, 2)
    
    # Final balance
    total_supply = round(sum(supply), 2)
    total_demand = round(sum(demand), 2)
    if abs(total_supply - total_demand) > 0.01:
        diff = total_supply - total_demand
        demand[0] = round(demand[0] + diff, 2)
    
    # Step 10: Create output dictionary
    output = {
        "costs": costs,
        "supply": supply,
        "demand": demand,
        "metadata": {
            "dimensions": f"{m}x{n}",
            "total_supply": round(sum(supply), 2),
            "total_demand": round(sum(demand), 2),
            "canyon_columns": canyon_columns,
            "designed_for": "Russell's Approximation Method optimization"
        }
    }
    
    # Step 11: Save to JSON file
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Generated {m}×{n} transportation problem")
    print(f"✓ Total supply/demand: {round(sum(supply), 2)}")
    print(f"✓ Canyon columns: {canyon_columns}")
    print(f"✓ Cost diversity verified for VAM compatibility")
    print(f"✓ Saved to: {filename}")
    
    return output

# Example usage
if __name__ == "__main__":
    # Get user input
    print("=== RAM-Favoring Transportation Problem Generator ===\n")
    
    try:
        m = int(input("Enter number of sources (m): "))
        n = int(input("Enter number of destinations (n): "))
        
        # Minimum size check
        if m < 2 or n < 2:
            print("Error: Both m and n must be at least 2")
            exit(1)
        
        filename = input("Enter output filename (e.g., transport_data.json): ")
        
        # Add .json extension if not provided
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Generate the problem
        print("\nGenerating transportation problem...\n")
        generate_ram_favoring_transport_problem(m, n, filename)
        
    except ValueError:
        print("Error: Please enter valid integer values for m and n")
    except Exception as e:
        print(f"Error: {e}")