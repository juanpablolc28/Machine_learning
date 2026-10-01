"""
Reinforcement Learning environment: a 10x10 grid world.

Cell legend
-----------
A = Agent start position
T = Target (goal)
o = Open / normal path
# = Wall (blocks movement)
D = Danger zone (passable, but penalized)

TODO (Johan Sebastian): this GRID_TEMPLATE is a placeholder layout, only
generated to satisfy the required counts (1 A, 1 T, 20 #, 10 D, 68 o) and to
guarantee a path exists between start and target. Please design your own
10x10 layout for the report (you can keep this one as a base and move walls
around) — the assignment explicitly asks each group to place its own walls
and danger zones. Keep the exact counts and keep it solvable: you can verify
both with `python rl_environment.py`, which prints the character counts and
runs a reachability check from A to T.
"""

from collections import deque

ROWS = 10
COLS = 10

AGENT = "A"
TARGET = "T"
OPEN = "o"
WALL = "#"
DANGER = "D"

GRID_TEMPLATE = [
    "Aooo#ooooo",
    "o#ooooDDoo",
    "oo#o#oDDoo",
    "oooo#oDooo",
    "##o####o##",
    "oooo#ooDoo",
    "o#oo#ooDDo",
    "oooo#o#oDo",
    "ooooooo#Do",
    "oooo#ooooT",
]


ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_DELTAS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

# Reward system -------------------------------------------------------
# TODO (Johan Sebastian / group): these values are a reasonable starting
# point but can be retuned; whatever you use, explain the choice in the
# report (how each value shapes the agent's behavior).
REWARD_NORMAL_MOVE = -0.5
REWARD_INVALID_MOVE = -3     # tried to move outside the 10x10 grid
REWARD_WALL_HIT = -6         # tried to move into a '#' cell
REWARD_DANGER_ZONE = -20     # moved into a 'D' cell
REWARD_GOAL = 500            # reached the target


class GridEnvironment:
    """A 10x10 grid world with walls and danger zones."""

    def __init__(self, layout=None):
        self.layout = layout if layout is not None else GRID_TEMPLATE
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        self.start = self._find(AGENT)
        self.target = self._find(TARGET)

    def _find(self, symbol):
        for r, row in enumerate(self.layout):
            for c, ch in enumerate(row):
                if ch == symbol:
                    return (r, c)
        raise ValueError(f"Symbol '{symbol}' not found in grid layout")

    def cell_type(self, state):
        r, c = state
        ch = self.layout[r][c]
        if ch == AGENT:
            return OPEN
        return ch

    def reset(self):
        return self.start

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, state, action):
        """Apply `action` from `state`.

        Returns (next_state, reward, done, cell_type) where `cell_type`
        describes the cell the agent ends up on (or attempted to enter).
        """
        r, c = state
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc

        if not self.in_bounds(nr, nc):
            return state, REWARD_INVALID_MOVE, False, "invalid"

        target_cell = self.layout[nr][nc]

        if target_cell == WALL:
            return state, REWARD_WALL_HIT, False, WALL

        next_state = (nr, nc)

        if target_cell == TARGET:
            return next_state, REWARD_GOAL, True, TARGET

        if target_cell == DANGER:
            return next_state, REWARD_DANGER_ZONE, False, DANGER

        return next_state, REWARD_NORMAL_MOVE, False, OPEN

    def render_rows(self, highlight=None):
        """Return the grid as a list of rows of dicts, for template rendering.

        `highlight` is an optional set of (r, c) tuples belonging to the
        learned path, so the template can style them without changing the
        underlying character.
        """
        highlight = highlight or set()
        rows = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append({
                    "symbol": self.layout[r][c],
                    "on_path": (r, c) in highlight,
                    "is_start": (r, c) == self.start,
                    "is_target": (r, c) == self.target,
                })
            rows.append(row)
        return rows


def is_solvable(layout):
    rows, cols = len(layout), len(layout[0])
    start = next((r, c) for r in range(rows) for c in range(cols) if layout[r][c] == AGENT)
    target = next((r, c) for r in range(rows) for c in range(cols) if layout[r][c] == TARGET)

    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == target:
            return True
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and layout[nr][nc] != WALL:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


if __name__ == "__main__":
    from collections import Counter

    flat = [ch for row in GRID_TEMPLATE for ch in row]
    print("Character counts:", Counter(flat))
    print("Solvable (A can reach T avoiding walls):", is_solvable(GRID_TEMPLATE))
