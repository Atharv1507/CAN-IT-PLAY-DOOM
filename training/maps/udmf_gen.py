"""Small-map generator (Person 2). Turns an ASCII grid into a playable UDMF map (.wad).

Why: Freedoom's own maps are huge (100-800 sectors, 50-250 monsters). A bot can only learn
"key -> locked door -> exit" if it finds the key often, so Stage 2/3 use small generated maps.

Grid legend (one character = one 128x128-unit cell):
    #  wall / void            .  floor
    P  player start           E  exit pad (walk onto it = level exit)
    X  exit switch (a wall cell; press USE on it = level exit)
    D  normal door            1 2 3  locked door: red / blue / yellow (card OR skull opens it)
    r b y  red / blue / yellow keycard      u v w  red / blue / yellow skull key
    z  zombieman   g  shotgun guy   i  imp   d  demon
    h  stimpack    H  medikit   a  ammo clip   s  shotgun   S  shells   A  armor

Doors must sit in a 1-cell-wide corridor: open cells on two opposite sides, walls on the other two.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from doom_env.wad import Lump, write_wad

CELL = 128
SOLID = {"#", "X"}
DOORS = {"D": 0, "1": 129, "2": 130, "3": 131}  # ZDoom lock numbers: 129/130/131 = any red/blue/yellow key
DOOR_TEX = {0: "BIGDOOR2", 129: "DOORRED", 130: "DOORBLU", 131: "DOORYEL"}
THINGS = {
    "P": 1,
    "r": 13, "b": 5, "y": 6, "u": 38, "v": 40, "w": 39,
    "z": 3004, "g": 9, "i": 3001, "d": 3002,
    "h": 2011, "H": 2012, "a": 2007, "s": 2001, "S": 2008, "A": 2018,
}
KEY_CHARS = {129: ("r", "u"), 130: ("b", "v"), 131: ("y", "w")}
WALL_TEX = ["STARTAN2", "STONE2", "BROWN1", "GRAY1", "BROWNGRN", "STARG3", "TEKGREN2", "BRICK7"]
FLOOR_TEX = ["FLOOR0_1", "FLOOR4_8", "FLOOR5_1", "FLAT5_4", "FLAT1", "FLOOR7_1"]
CEIL_TEX = ["CEIL1_1", "CEIL3_5", "FLAT1", "CEIL5_1"]


# ---------------------------------------------------------------------------- grid -> UDMF
def _cell_kind(ch: str) -> str:
    if ch in SOLID:
        return "solid"
    if ch in DOORS:
        return "door"
    if ch == "E":
        return "exit"
    return "floor"


def grid_to_udmf(grid: list[str], rng: np.random.Generator | None = None, player_angle: int = 0) -> str:
    rng = rng or np.random.default_rng(0)
    h = len(grid)
    w = max(len(r) for r in grid)
    g = [list(r.ljust(w, "#")) for r in grid]
    at = lambda r, c: g[r][c] if 0 <= r < h and 0 <= c < w else "#"

    # 1. sectors: connected floor areas (doors and exit pads get their own sectors)
    sector_of = [[-1] * w for _ in range(h)]
    sectors: list[dict] = []
    for r in range(h):
        for c in range(w):
            kind = _cell_kind(g[r][c])
            if kind == "solid" or sector_of[r][c] >= 0:
                continue
            sid = len(sectors)
            if kind == "door":
                lock = DOORS[g[r][c]]
                sectors.append({"kind": "door", "lock": lock, "floor": 0, "ceil": 0,
                                "light": 160, "ftex": "FLAT20", "ctex": "FLAT20", "wall": "DOORTRAK"})
                sector_of[r][c] = sid
                continue
            sectors.append({"kind": kind, "lock": -1, "floor": 0, "ceil": int(rng.choice([128, 160, 192])),
                            "light": int(rng.integers(128, 225)), "ftex": str(rng.choice(FLOOR_TEX)),
                            "ctex": str(rng.choice(CEIL_TEX)), "wall": str(rng.choice(WALL_TEX))})
            if kind == "exit":
                sectors[-1].update(ftex="GATE3", light=224)
            q = deque([(r, c)])
            sector_of[r][c] = sid
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                    if 0 <= nr < h and 0 <= nc < w and sector_of[nr][nc] < 0 and _cell_kind(g[nr][nc]) == kind:
                        sector_of[nr][nc] = sid
                        q.append((nr, nc))

    # 2. lines on every cell edge where something changes
    verts: dict[tuple[int, int], int] = {}
    vlist: list[tuple[int, int]] = []
    sides: list[str] = []
    lines: list[str] = []

    def vid(p):
        if p not in verts:
            verts[p] = len(vlist)
            vlist.append(p)
        return verts[p]

    def side(sec, top="-", mid="-", bot="-"):
        sides.append(f'sidedef {{ sector = {sec}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "{bot}"; }}')
        return len(sides) - 1

    def oriented(p, q, front_center):
        # UDMF: the front side is on the right of v1 -> v2
        cross = (q[0] - p[0]) * (front_center[1] - p[1]) - (q[1] - p[1]) * (front_center[0] - p[0])
        return (p, q) if cross < 0 else (q, p)

    center = lambda r, c: ((c + 0.5) * CELL, -(r + 0.5) * CELL)

    def edge(r1, c1, r2, c2):
        if r1 == r2:  # vertical edge between (r1,c1) and (r1,c1+1)
            x = (c1 + 1) * CELL
            return (x, -r1 * CELL), (x, -(r1 + 1) * CELL)
        y = -(r1 + 1) * CELL  # horizontal edge between (r1,c1) and (r1+1,c1)
        return (c1 * CELL, y), ((c1 + 1) * CELL, y)

    for r in range(-1, h):
        for c in range(-1, w):
            for r2, c2 in ((r, c + 1), (r + 1, c)):
                a, b = at(r, c), at(r2, c2)
                ka, kb = _cell_kind(a), _cell_kind(b)
                if ka == "solid" and kb == "solid":
                    continue
                p, q = edge(r, c, r2, c2)
                if ka == "solid" or kb == "solid":  # one-sided wall
                    (orr, oc), wall_ch = ((r2, c2), a) if ka == "solid" else ((r, c), b)
                    sec = sector_of[orr][oc]
                    v1, v2 = oriented(p, q, center(orr, oc))
                    if wall_ch == "X" and sectors[sec]["kind"] != "door":
                        sd = side(sec, mid="SW1EXIT")
                        lines.append(f"linedef {{ v1 = {vid(v1)}; v2 = {vid(v2)}; sidefront = {sd}; blocking = true; "
                                     f"special = 243; playeruse = true; }}")
                    else:
                        sd = side(sec, mid=sectors[sec]["wall"])
                        lines.append(f"linedef {{ v1 = {vid(v1)}; v2 = {vid(v2)}; sidefront = {sd}; blocking = true; dontpegbottom = true; }}")
                    continue
                sa, sb = sector_of[r][c], sector_of[r2][c2]
                if sa == sb:
                    continue
                if ka == "door" and kb == "door":
                    raise ValueError(f"two door cells touch at ({r},{c})")
                if kb == "door" or ka == "door":
                    (fr, fc, fs), ds = ((r, c, sa), sb) if kb == "door" else ((r2, c2, sb), sa)
                    lock = sectors[ds]["lock"]
                    v1, v2 = oriented(p, q, center(fr, fc))
                    front = side(fs, top=DOOR_TEX[lock])
                    back = side(ds)
                    if lock == 0:
                        spec = "special = 12; arg0 = 0; arg1 = 16; arg2 = 150; monsteruse = true;"
                    else:
                        spec = f"special = 13; arg0 = 0; arg1 = 16; arg2 = 150; arg3 = {lock};"
                    lines.append(f"linedef {{ v1 = {vid(v1)}; v2 = {vid(v2)}; sidefront = {front}; sideback = {back}; "
                                 f"twosided = true; {spec} playeruse = true; repeatspecial = true; }}")
                    continue
                v1, v2 = oriented(p, q, center(r, c))
                front, back = side(sa), side(sb)
                extra = " special = 243; playercross = true;" if "exit" in (ka, kb) else ""
                lines.append(f"linedef {{ v1 = {vid(v1)}; v2 = {vid(v2)}; sidefront = {front}; sideback = {back}; twosided = true;{extra} }}")

    # 3. things
    things = []
    for r in range(h):
        for c in range(w):
            ch = g[r][c]
            if ch in THINGS:
                x, y = center(r, c)
                ang = player_angle if ch == "P" else int(rng.choice([0, 90, 180, 270]))
                things.append(f"thing {{ x = {x:.1f}; y = {y:.1f}; angle = {ang}; type = {THINGS[ch]}; "
                              "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }")
    if sum(1 for row in g for ch in row if ch == "P") != 1:
        raise ValueError("grid needs exactly one P")

    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}" for x, y in vlist]
    for s in sectors:
        out.append(f'sector {{ texturefloor = "{s["ftex"]}"; textureceiling = "{s["ctex"]}"; '
                   f'heightfloor = {s["floor"]}; heightceiling = {s["ceil"]}; lightlevel = {s["light"]}; }}')
    out += sides + lines + things
    return "\n".join(out) + "\n"


def write_map(path: str | Path, grid: list[str], rng: np.random.Generator | None = None,
              player_angle: int = 0, map_name: str = "MAP01") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = grid_to_udmf(grid, rng, player_angle)
    write_wad(path, [Lump(map_name, b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
    return path


# ---------------------------------------------------------------------------- procedural layouts
SLOT = 9  # each room lives in a 9x9 slot; corridors run along the slot's middle row/column


def random_layout(rng: np.random.Generator, n_rooms: int, n_locks: int = 0,
                  monsters: tuple[int, int] = (0, 0), items: tuple[int, int] = (0, 2),
                  slots: tuple[int, int] = (3, 3), exit_kind: str | None = None,
                  room_size: tuple[int, int] = (3, 7)) -> tuple[list[str], int]:
    """Random tree of rooms joined by doors. Returns (grid, player_angle).

    n_locks locked doors are put on the path from the start room to the exit room; each key is
    placed in a room reachable without that lock, preferring side rooms (forces exploring and
    backtracking). exit_kind: "pad", "switch", "none" or None (random pad/switch)."""
    rows, cols = slots
    assert 1 <= n_rooms <= rows * cols
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    parent = {start: None}
    order = [start]
    while len(order) < n_rooms:
        a = order[int(rng.integers(len(order)))]
        nbrs = [(a[0] + dr, a[1] + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))]
        nbrs = [n for n in nbrs if 0 <= n[0] < rows and 0 <= n[1] < cols and n not in parent]
        if nbrs:
            b = nbrs[int(rng.integers(len(nbrs)))]
            parent[b] = a
            order.append(b)

    depth = {}
    for s in order:
        d, p = 0, s
        while parent[p] is not None:
            p, d = parent[p], d + 1
        depth[s] = d
    max_d = max(depth.values())
    exit_room = [s for s in order if depth[s] == max_d][int(rng.integers(sum(depth[s] == max_d for s in order)))]
    path = [exit_room]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()  # start ... exit
    n_locks = min(n_locks, len(path) - 1)

    H, W = rows * SLOT, cols * SLOT
    g = [["#"] * W for _ in range(H)]
    rect = {}
    for s in order:
        wd, ht = (int(rng.integers(room_size[0], room_size[1] + 1)) for _ in range(2))
        x0 = int(rng.integers(max(1, 5 - wd), min(4, 8 - wd) + 1))
        y0 = int(rng.integers(max(1, 5 - ht), min(4, 8 - ht) + 1))
        R0, C0 = s[0] * SLOT + y0, s[1] * SLOT + x0
        rect[s] = (R0, C0, R0 + ht - 1, C0 + wd - 1)
        for r in range(R0, R0 + ht):
            for c in range(C0, C0 + wd):
                g[r][c] = "."

    # locked edges = (child room) on the start->exit path
    lock_children = sorted(rng.choice(np.arange(1, len(path)), size=n_locks, replace=False).tolist()) if n_locks else []
    colors = rng.permutation([129, 130, 131]).tolist()
    lock_of_child = {path[i]: colors[k] for k, i in enumerate(lock_children)}

    for child in order[1:]:
        a, b = parent[child], child
        lock = lock_of_child.get(child)
        ch = DOORS_INV[lock] if lock else "D"
        if a[0] == b[0]:  # horizontal neighbours
            left, right = (a, b) if a[1] < b[1] else (b, a)
            row = a[0] * SLOT + 4
            cells = list(range(rect[left][3] + 1, rect[right][1]))
        else:
            top, bot = (a, b) if a[0] < b[0] else (b, a)
            col = a[1] * SLOT + 4
            cells = list(range(rect[top][2] + 1, rect[bot][0]))
        door_at = cells[int(rng.integers(len(cells)))]
        for k in cells:
            r, c = (row, k) if a[0] == b[0] else (k, col)
            g[r][c] = ch if k == door_at else "."

    def behind(room):  # locked children on the way from start to `room`
        out, p = set(), room
        while p is not None:
            if p in lock_of_child:
                out.add(p)
            p = parent[p]
        return out

    def free_cell(room):
        R0, C0, R1, C1 = rect[room]
        cells = [(r, c) for r in range(R0, R1 + 1) for c in range(C0, C1 + 1) if g[r][c] == "."]
        return cells[int(rng.integers(len(cells)))] if cells else None

    pr, pc = free_cell(start)
    g[pr][pc] = "P"

    on_path = set(path)
    for k, i in enumerate(lock_children):
        child = path[i]
        allowed = {path[j] for j in lock_children[:k]}
        cands = [s for s in order if behind(s) <= allowed and s not in (child,)]
        side_rooms = [s for s in cands if s not in on_path]
        pool = side_rooms or cands
        room = pool[int(rng.integers(len(pool)))]
        cell = free_cell(room)
        key_ch = KEY_CHARS[lock_of_child[child]][int(rng.random() < 0.3)]
        g[cell[0]][cell[1]] = key_ch

    kind = exit_kind or ("pad" if rng.random() < 0.5 else "switch")
    if kind == "pad":
        cell = free_cell(exit_room)
        g[cell[0]][cell[1]] = "E"
    elif kind == "switch":
        R0, C0, R1, C1 = rect[exit_room]
        walls = [(R0 - 1, c) for c in range(C0, C1 + 1)] + [(R1 + 1, c) for c in range(C0, C1 + 1)] + \
                [(r, C0 - 1) for r in range(R0, R1 + 1)] + [(r, C1 + 1) for r in range(R0, R1 + 1)]
        walls = [(r, c) for r, c in walls if 0 <= r < H and 0 <= c < W and g[r][c] == "#"]
        r, c = walls[int(rng.integers(len(walls)))]
        g[r][c] = "X"

    monster_rooms = [s for s in order if s != start] or [start]
    for _ in range(int(rng.integers(monsters[0], monsters[1] + 1))):
        cell = free_cell(monster_rooms[int(rng.integers(len(monster_rooms)))])
        if cell and abs(cell[0] - pr) + abs(cell[1] - pc) > 2:
            g[cell[0]][cell[1]] = str(rng.choice(["z", "z", "g", "i", "i", "d"]))
    for _ in range(int(rng.integers(items[0], items[1] + 1))):
        cell = free_cell(order[int(rng.integers(len(order)))])
        if cell:
            g[cell[0]][cell[1]] = str(rng.choice(["h", "H", "a", "S", "s", "A"]))

    # crop to the used area + 1-cell border
    used_r = [r for r in range(H) if any(ch != "#" for ch in g[r])]
    used_c = [c for c in range(W) if any(g[r][c] != "#" for r in range(H))]
    r0, r1 = max(0, used_r[0] - 1), min(H - 1, used_r[-1] + 1)
    c0, c1 = max(0, used_c[0] - 1), min(W - 1, used_c[-1] + 1)
    grid = ["".join(g[r][c0:c1 + 1]) for r in range(r0, r1 + 1)]
    return grid, int(rng.choice([0, 45, 90, 135, 180, 225, 270, 315]))


DOORS_INV = {v: k for k, v in DOORS.items() if v}
