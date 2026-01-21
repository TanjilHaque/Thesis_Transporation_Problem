import numpy as np
import json
from transportation import Transportation
from modi import MODI  # <-- UPDATED import


class VogelsApproximationMethod:
    """
    Vogel's Approximation Method (VAM) or penalty method
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

    def penalty(self, cost):
        gaps = np.zeros(cost.shape[0])
        for i, c in enumerate(cost):
            try:
                x, y = sorted(c)[:2]
            except ValueError:
                x, y = c[0], 0
            gaps[i] = abs(x - y)
        return gaps

    def solve(self, show_iter=False):

        while self.table.shape != (2, 2):

            cost = self.table[1:-1, 1:-1]
            supply = self.table[1:-1, -1]
            demand = self.table[-1, 1:-1]
            n = cost.shape[0]

            row_penalty = self.penalty(cost)
            col_penalty = self.penalty(cost.T)

            P = np.append(row_penalty, col_penalty)

            max_alloc = -np.inf
            for i in np.where(P == max(P))[0]:

                if i - n < 0:
                    r = i
                    L = cost[r]
                else:
                    c = i - n
                    L = cost[:, c]

                for j in np.where(L == min(L))[0]:
                    if i - n < 0:
                        c = j
                    else:
                        r = j

                    alloc = min([supply[r], demand[c]])
                    if alloc > max_alloc:
                        max_alloc = alloc
                        x, y = r, c

            self.allocate(x + 1, y + 1)

            if show_iter:
                self.trans.print_frame(self.table)

        return np.array(self.alloc, dtype=object)


# ==========================================================
# MAIN EXECUTION (VAM → MODI)
# ==========================================================
if __name__ == "__main__":

    with open("ram_favorable_dataset.json", "r") as f:
        data = json.load(f)

    cost = np.array(data["costs"])
    supply = np.array(data["supply"])
    demand = np.array(data["demand"])

    # Initialize transportation problem
    trans = Transportation(cost, supply, demand)
    trans.setup_table(minimize=True)

    # --------------------
    # VAM (Initial BFS)
    # --------------------
    VAM = VogelsApproximationMethod(trans)
    bfs_allocation = VAM.solve(show_iter=False)

    vam_cost = sum(
        cost[int(r[1:])][int(c[1:])] * v
        for r, c, v in bfs_allocation
    )

    print(f"VAM Initial Cost: {vam_cost}")

    # --------------------
    # MODI Optimization
    # --------------------
    modi = MODI(cost, bfs_allocation)  # <-- USING MODIMethod class
    optimal_allocation = modi.solve()

    modi_cost = modi.cost_value()

    print(f"MODI Optimized Cost: {modi_cost}")
