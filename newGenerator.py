import json
import random

def generate_ram_favorable_dataset(m, n, dominant_ratio=0.85):
    # Choose dominant columns (15% of total)
    k = max(1, int(n * dominant_ratio))
    dominant_cols = random.sample(range(n), k)

    # Cost settings
    low_cost_min, low_cost_max = 1, 5
    high_cost_min, high_cost_max = 60, 120

    costs = []
    for i in range(m):
        row = []
        for j in range(n):
            if j in dominant_cols:
                row.append(random.randint(low_cost_min, low_cost_max))
            else:
                row.append(random.randint(high_cost_min, high_cost_max))
        costs.append(row)

    # Generate supply and demand
    supply = [random.randint(10, 40) for _ in range(m)]
    total_supply = sum(supply)

    demand_raw = [random.randint(10, 40) for _ in range(n)]
    scale = total_supply / sum(demand_raw)
    demand = [max(1, int(round(d * scale))) for d in demand_raw]

    diff = total_supply - sum(demand)
    idx = 0
    while diff != 0:
        if diff > 0:
            demand[idx] += 1
            diff -= 1
        else:
            if demand[idx] > 1:
                demand[idx] -= 1
                diff += 1
        idx = (idx + 1) % n

    return {
        "costs": costs,
        "supply": supply,
        "demand": demand,
        "dominant_cols": dominant_cols
    }

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    random.seed(42)
    dataset = generate_ram_favorable_dataset(m=300, n=300, dominant_ratio=0.85)
    save_json("superior_ram_dataset.json", dataset)
    print("Saved: superior_ram_dataset.json")
    print("Dominant columns:", dataset["dominant_cols"])
