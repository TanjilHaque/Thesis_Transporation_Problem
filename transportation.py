import numpy as np
import pandas as pd

class Transportation:

    def __init__(self, cost, supply, demand):

        self.n, self.m = cost.shape

        self.table = np.zeros((self.n + 2, self.m + 2), dtype=object)
        self.table[1:-1, 1:-1] = cost.copy()
        self.table[-1, 1:-1] = demand.copy()
        self.table[1:-1, -1] = supply.copy()
        self.table[0, 1::] = [f"C{i}" for i in range(self.m)] + ['Supply']
        self.table[1::, 0] = [f"R{i}" for i in range(self.n)] + ['Demand']

    def setup_table(self, minimize=True):
        
        if not minimize:
            #if problem is maximization then change to minimization
            #by substracting all cost from maximum cost
            cost = self.table[1:-1, 1:-1]
            self.table[1:-1, 1:-1] = np.max(cost) - cost
        
        #sum(supply) - sum(demand)
        gap = self.table[1:-1, -1].sum() - self.table[-1, 1:-1].sum()

        if gap > 0:
            #add dummy column
            dummy = np.array(['Dummy'] + ([0] * self.n) + [gap], dtype=object)
            self.table = np.insert(self.table, -1, dummy, axis=1)
        elif gap < 0:
            #add dummy row
            dummy = np.array(['Dummy'] + ([0] * self.m) + [-gap], dtype=object)
            self.table = np.insert(self.table, -1, dummy, axis=0)

        self.table[-1, -1] = self.table[1:-1, -1].sum()
        self.table = np.array(self.table,  dtype=object)

    def print_frame(self, table):
        df = pd.DataFrame(table[1:, 1:])
        df.columns = table[0, 1:]
        df.index = table[1:, 0]
        print(df, '\n')

    def print_table(self, allocation):
        alloc = [[i, j] for i, j, _ in allocation]
        
        cost, total = [], 0
        for i, x in enumerate(self.table[1:-1, 0]):
            temp = []
            for j, y in enumerate(self.table[0, 1:-1]):
                v = self.table[i + 1, j + 1]
                try:
                    z = alloc.index([x, y])
                    cell = f"{v}({allocation[z, -1]})"
                    total += v * allocation[z, -1]
                except ValueError:
                    cell = f"{v}"
                temp.append(cell)
            cost.append(temp)

        table = self.table.copy()
        table[1:-1, 1:-1] = cost

        self.print_frame(np.array(table))
        print("TOTAL COST: {}".format(total))