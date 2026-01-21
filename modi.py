import numpy as np

class MODI:
    def __init__(self, cost, bfs):
        self.cost = np.array(cost, dtype=float)
        self.n, self.m = self.cost.shape
        
        # We store the basis as a dictionary {(i, j): value}
        # Crucially, we keep cells in the basis even if their value is 0.0
        self.alloc = {}
        for r, c, v in bfs:
            i = int(r[1:]) if isinstance(r, str) else int(r)
            j = int(c[1:]) if isinstance(c, str) else int(c)
            self.alloc[(i, j)] = float(v)

        self._ensure_non_degenerate()

    def _ensure_non_degenerate(self):
        """Ensures exactly n + m - 1 cells are in the basis."""
        required = self.n + self.m - 1
        if len(self.alloc) == required:
            return

        for i in range(self.n):
            for j in range(self.m):
                if (i, j) not in self.alloc:
                    # Logic check: Adding this 0-cell must not create a loop
                    if not self._find_loop((i, j)):
                        self.alloc[(i, j)] = 0.0
                        if len(self.alloc) == required:
                            return

    def _compute_uv(self):
        u = [None] * self.n
        v = [None] * self.m
        u[0] = 0.0
        
        # Solving u_i + v_j = cost_ij for basic cells
        while None in u or None in v:
            changed = False
            for (i, j) in self.alloc.keys():
                if u[i] is not None and v[j] is None:
                    v[j] = self.cost[i, j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = self.cost[i, j] - v[j]
                    changed = True
            if not changed: break # Should not happen if non-degenerate
        return u, v

    def _find_loop(self, start_cell):
        """Finds a closed loop for the entering cell using a simplified path search."""
        basis = list(self.alloc.keys()) + [start_cell]
        
        def get_path(curr, path, search_row):
            if len(path) > 3 and curr == start_cell:
                return path[:-1]
            
            i, j = curr
            if search_row:
                # Look for cells in the same row
                candidates = [c for c in basis if c[0] == i and c != curr]
            else:
                # Look for cells in the same column
                candidates = [c for c in basis if c[1] == j and c != curr]
            
            for next_cell in candidates:
                if next_cell not in path or (next_cell == start_cell and len(path) > 2):
                    res = get_path(next_cell, path + [next_cell], not search_row)
                    if res: return res
            return None

        return get_path(start_cell, [start_cell], True)

    def _reallocate(self, loop):
        # Even indices: start_cell, then every 2nd (the + cells)
        # Odd indices: the cells we subtract from (the - cells)
        minus_cells = loop[1::2]
        theta = min(self.alloc[c] for c in minus_cells)

        # Update values
        for idx, cell in enumerate(loop):
            if idx % 2 == 0:
                self.alloc[cell] = self.alloc.get(cell, 0) + theta
            else:
                self.alloc[cell] -= theta

        # Remove EXACTLY ONE cell from the basis to maintain m+n-1
        # Even if multiple cells hit zero, we only drop the first one found.
        dropped = False
        for cell in minus_cells:
            if self.alloc[cell] == 0 and not dropped:
                del self.alloc[cell]
                dropped = True

    def solve(self):
        for _ in range(100): # Safety limit
            u, v = self._compute_uv()
            
            # Find the cell with the most negative opportunity cost (entering cell)
            min_reduced_cost = 0
            entering_cell = None
            
            for i in range(self.n):
                for j in range(self.m):
                    if (i, j) not in self.alloc:
                        # P_ij = u_i + v_j - cost_ij
                        # Optimization: we look for u_i + v_j > cost_ij
                        penalty = u[i] + v[j] - self.cost[i, j]
                        if penalty > min_reduced_cost:
                            min_reduced_cost = penalty
                            entering_cell = (i, j)

            if entering_cell is None:
                break # Optimal solution reached

            loop = self._find_loop(entering_cell)
            if loop:
                self._reallocate(loop)
            else:
                break
        
        return self.alloc, self.cost_value()

    def cost_value(self):
        return sum(self.cost[i, j] * v for (i, j), v in self.alloc.items())