import numpy as np
import json
from transportation import Transportation

class VogelsApproximationMethod:
    """
    Vogel's Approximation Method (VAM) or penalty method
    This method is preferred over the NWCM and VAM, because the initial basic feasible solution obtained by this method is either optimal solution or very nearer to the optimal solution.
    Vogel's Approximation Method (VAM) Steps (Rule)
    Step-1:	Find the cells having smallest and next to smallest cost in each row and write the difference (called penalty) along the side of the table in row penalty.
    Step-2:	Find the cells having smallest and next to smallest cost in each column and write the difference (called penalty) along the side of the table in each column penalty.
    Step-3:	Select the row or column with the maximum penalty and find cell that has least cost in selected row or column. Allocate as much as possible in this cell.
    If there is a tie in the values of penalties then select the cell where maximum allocation can be possible
    Step-4:	Adjust the supply & demand and cross out (strike out) the satisfied row or column.
    Step-5:	Repeact this steps until all supply and demand values are 0.

    Source: https://cbom.atozmath.com/example/CBOM/Transportation.aspx?he=e&q=vam
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

    def penalty(self, cost):
        #return gaps between two lowest cost in row/column
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

            #compute row and column penalties
            row_penalty = self.penalty(cost)
            col_penalty = self.penalty(cost.T)

            #check if maximum penalties value has a tie
            P = np.append(row_penalty, col_penalty)

            max_alloc = -np.inf
            for i in np.where(P == max(P))[0]:

                if i - n < 0:
                    r = i
                    L = cost[r]
                else:
                    c = i - n
                    L = cost[:, c]

                #check if minimum cost has a tie
                #in maximum row/columns penalties
                for j in np.where(L == min(L))[0]:
                    if i - n < 0:
                        c = j
                    else:
                        r = j

                    alloc = min([supply[r], demand[c]])
                    if alloc > max_alloc:
                        max_alloc = alloc
                        x, y = r, c

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

    #initialize Vogel's method with table that has been prepared before.
    VAM = VogelsApproximationMethod(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    allocation = VAM.solve(show_iter=False)

    # ---- ONLY TOTAL COST OUTPUT ----
    total_cost = 0
    for i, j, v in allocation:
        total_cost += cost[int(i[1:])][int(j[1:])] * v

    print(f"Total Cost: {total_cost}")
