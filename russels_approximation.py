import numpy as np
import json
from transportation import Transportation
from modi import MODI


class RussellsApproximationMethod:
    """
    Russell's Approximation Method (RAM):
    Step-1: For each source row still under consideration, determine its Ui (largest cost in row i).
    Step-2: For each destination column still under consideration, determine its Vj (largest cost in column j).
    Step-3: For each variable, calculate Δij=cij-(Ui +  Vj).
    Step-4: Select the variable having the most negative Δ value, break ties arbitrarily.
    Step-5: Allocate as much as possible. Eliminate necessary cells from consideration. Return to Step-1.
    """

    def __init__(self, trans):
        self.trans = trans
        self.table = trans.table.copy()
        self.alloc = []

    def allocate(self, x, y):
        mins = min([self.table[x, -1], self.table[-1, y]])
        self.alloc.append([self.table[x, 0], self.table[0, y], mins])

        if self.table[x, -1] < self.table[-1, y]:
            self.table = np.delete(self.table, x, 0)
            self.table[-1, y] -= mins

        elif self.table[x, -1] > self.table[-1, y]:
            self.table = np.delete(self.table, y, 1)
            self.table[x, -1] -= mins

        else:
            self.table = np.delete(self.table, x, 0)
            self.table = np.delete(self.table, y, 1)

    def solve(self, show_iter=False):

        while self.table.shape != (2, 2):
            cost = self.table[1:-1, 1:-1]
            n, m = cost.shape

            # compute U and V
            U = np.max(cost, 1)
            V = np.max(cost, 0)

            # compute reduced cost
            for i in range(n):
                for j in range(m):
                    self.table[i + 1, j + 1] -= U[i] + V[j]

            # find the most negative
            mins = np.min(self.table[1:-1, 1:-1])
            x, y = np.argwhere(self.table[1:-1, 1:-1] == mins)[0]

            # allocate
            self.allocate(x + 1, y + 1)

            if show_iter:
                self.trans.print_frame(self.table)

        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":
    #example1_from_reference_thesis_3x4
    with open("ram_favorable_dataset.json", "r") as f:
        data = json.load(f)

    cost = np.array(data["costs"])
    supply = np.array(data["supply"])
    demand = np.array(data["demand"])

    # Initialize transportation problem
    trans = Transportation(cost, supply, demand)
    trans.setup_table(minimize=True)

    # --------------------
    # RAM (Initial BFS)
    # --------------------
    RAM = RussellsApproximationMethod(trans)
    bfs_allocation = RAM.solve(show_iter=False)

    ram_cost = sum(
        cost[int(r[1:])][int(c[1:])] * v
        for r, c, v in bfs_allocation
    )

    print(f"RAM Initial Cost: {ram_cost}")

    # --------------------
    # MODI Optimization
    # --------------------
    modi = MODI(cost, bfs_allocation)
    optimal_allocation = modi.solve()

    modi_cost = modi.cost_value()

    print(f"MODI Optimized Cost: {modi_cost}")
