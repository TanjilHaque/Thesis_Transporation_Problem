import numpy as np
import json
from transportation import Transportation

class RussellsApproximationMethod:
    """
    Russell's Approximation Method (RAM):
    Step-1:	For each source row still under consideration, determine its Ui (largest cost in row i).
    Step-2:	For each destination column still under consideration, determine its Vj (largest cost in column j).
    Step-3:	For each variable, calculate Δij=cij-(Ui +  Vj).
    Step-4:	Select the variable having the most negative Δ value, break ties arbitrarily.
    Step-5:	Allocate as much as possible. Eliminate necessary cells from consideration. Return to Step-1.

    Source: https://cbom.atozmath.com/example/CBOM/Transportation.aspx?he=e&q=ram
    """

    def __init__(self, trans):
        self.trans = trans
        self.table = trans.table.copy()
        self.alloc = []

    def allocate(self, x, y):
        
        mins = min([self.table[x, -1], self.table[-1, y]])
        self.alloc.append([self.table[x, 0], self.table[0, y], mins])
        
        if self.table[x, -1] < self.table[-1, y]:
            #delete row and supply x then change value of demand y
            self.table = np.delete(self.table, x, 0)
            self.table[-1, y] -= mins
            
        elif self.table[x, -1] > self.table[-1, y]:
            #delete column and demand y then change value of supply x
            self.table = np.delete(self.table, y, 1)
            self.table[x, -1] -= mins
            
        else:
            #delete row and supply x, column and demand y
            self.table = np.delete(self.table, x, 0)
            self.table = np.delete(self.table, y, 1)

    def solve(self, show_iter=False):

        while self.table.shape != (2, 2):
            cost = self.table[1:-1, 1:-1]
            n, m = cost.shape

            #compute U and V
            U = np.max(cost, 1)
            V = np.max(cost, 0)

            #compute reduced cost
            for i in range(n):
                for j in range(m):
                    self.table[i + 1, j + 1] -= U[i] + V[j]
            
            #find the most negative
            mins = np.min(self.table[1:-1, 1:-1])
            x, y = np.argwhere(self.table[1:-1, 1:-1] == mins)[0]

            #allocated row x to column y or vice versa
            self.allocate(x + 1, y + 1)

            #print table
            if show_iter:
                self.trans.print_frame(self.table)
            
        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":

    with open("ram_favorable_dataset.json", "r") as f:
        data = json.load(f)

    cost = np.array(data["costs"])
    supply = np.array(data["supply"])
    demand = np.array(data["demand"])


    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    trans.setup_table(minimize=True)

    #initialize Russell's Approximation method with table that has been prepared before.
    RAM = RussellsApproximationMethod(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    allocation = RAM.solve(show_iter=False)

    # ---- ONLY TOTAL COST OUTPUT ----
    total_cost = 0
    for i, j, v in allocation:
        total_cost += cost[int(i[1:])][int(j[1:])] * v

    print(f"Total Cost: {total_cost}")
