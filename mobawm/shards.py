"""Readers for the files written by build/moba_collect.

Two formats, both mirroring structs in moba/moba_collect.c and moba/moba_state.h:

- **Game logs** (``.mbl``) -- the corpus. Per game: its seed, which
  controller drove each team, the 10 hero actions per tick, and a state hash.
  ~10 bytes/tick. Everything else is regenerated bit-exactly by
  ``moba_collect replay``.
- **Dense files** (``.mbs``) -- the full ``MobaState`` of every tick, in game
  and tick order, produced from a log by ``moba_collect replay --dense``.
  ~71 KB/tick raw, zlib-compressed per game (~13x); read one game at a time.

A game of n steps has n + 1 dense records. Record t < n is (s_t, a_t, r_t)
captured before the step, with ``done`` on the step that ends the game;
record n is the terminal state (``final`` = 1, no action). So within a game,
s_{t+1} is always record t+1.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOG_MAGIC = 0x4D424C33    # "MBL3"
DENSE_MAGIC = 0x4D425333  # "MBS3"
VERSION = 3

N_ENTITIES = 206
N_PLAYERS = 10
N_HEROES = 10                 # all 10 act: 0-4 radiant, 5-9 dire
OBS_DIM = 510
N_ACTIONS = 6
ACTION_SIZES = (7, 7, 3, 2, 2, 2)  # move_y, move_x, target mode, q, w, e
MAP = 128
SCAN_CAP = 64
NONE = 255                    # empty cell in pids / end of a target list
CONTROLLERS = {0: "net", 1: "script"}

# Per-entity fields that change during a game (moba_state.h). team, grid_id,
# lane, entity_type, basic_attack_cd and xp_on_kill are here because creep
# slots are reused across lanes and spawn_creep rewrites them every wave.
STATE_FIELDS = (
    "alive", "x", "y", "health", "max_health", "mana", "max_mana", "damage",
    "level", "xp", "move_modifier", "stun_timer", "move_timer", "q_timer",
    "w_timer", "e_timer", "basic_attack_timer", "is_hit", "waypoint",
    "target_pid", "team", "grid_id", "lane", "entity_type", "basic_attack_cd",
    "xp_on_kill",
)
F = {name: i for i, name in enumerate(STATE_FIELDS)}
REWARD_FIELDS = ("death", "xp", "distance", "tower")

# Entity index ranges.
PLAYERS = slice(0, 10)        # 0-4 radiant, 5-9 dire
CREEPS = slice(10, 110)
NEUTRALS = slice(110, 182)
TOWERS = slice(182, 206)      # 204 = dire ancient, 205 = radiant ancient

STATIC_DTYPE = np.dtype([      # per hero, constant for the game
    ("hero_type", "i1"), ("pad", "i1", (3,)), ("move_speed", "<f4"),
])

STATE_DTYPE = np.dtype([
    ("tick", "<i4"), ("creep_idx", "<i4"), ("rng_idx", "<i4"), ("pad", "<i4"),
    ("rng_state", "<u8"),                          # the env's random stream
    ("entity", "<f4", (N_ENTITIES, len(STATE_FIELDS))),
    ("reward", "<f4", (N_PLAYERS, 4)),
    ("grid", "u1", (MAP * MAP,)),                  # tile ids; not derivable from entities
    ("pids", "u1", (MAP * MAP,)),                  # entity index, NONE = empty
    ("scan", "u1", (N_ENTITIES, SCAN_CAP)),        # target lists, NONE-terminated
])

DENSE_HEADER_DTYPE = np.dtype([
    ("magic", "<u4"), ("version", "<u4"), ("record_bytes", "<u4"), ("n_records", "<u4"),
    ("n_games", "<u4"), ("state_bytes", "<u4"), ("vision_range", "<u4"), ("scan_cap", "<u4"),
    ("state_fields", "<u4"), ("reserved", "<u4"),
    ("statics", STATIC_DTYPE, (N_HEROES,)),
])

DENSE_GAME_DTYPE = np.dtype([("game_index", "<u4"), ("n_records", "<u4"), ("zbytes", "<u8")])

DENSE_RECORD_DTYPE = np.dtype([
    ("game_index", "<u4"), ("tick", "<i4"),
    ("done", "u1"), ("truncated", "u1"), ("winner", "u1"), ("final", "u1"),
    ("radiant_ctrl", "u1"), ("dire_ctrl", "u1"), ("pad", "u1", (2,)),
    ("actions", "u1", (N_HEROES, N_ACTIONS)),
    ("rewards", "<f4", (N_HEROES,)),
    ("obs", "u1", (N_HEROES, OBS_DIM)),            # the env's own obs, kept as a check
    ("state", STATE_DTYPE),
])

LOG_HEADER_DTYPE = np.dtype([
    ("magic", "<u4"), ("version", "<u4"), ("n_games", "<u4"), ("max_ticks", "<u4"),
    ("base_seed", "<u8"), ("first_game", "<u4"), ("vision_range", "<u4"),
    ("agent_speed", "<f4"), ("reward_death", "<f4"), ("reward_xp", "<f4"),
    ("reward_distance", "<f4"), ("reward_tower", "<f4"),
    ("epsilon", "<f4"), ("weights_hash", "<u8"), ("reserved", "<u4", (4,)),
])

GAME_HEADER_DTYPE = np.dtype([
    ("game_seed", "<u8"), ("game_index", "<u4"), ("n_ticks", "<u4"),
    ("winner", "u1"), ("truncated", "u1"), ("radiant_ctrl", "u1"), ("dire_ctrl", "u1"),
    ("zbytes", "<u4"), ("state_hash", "<u8"),
])


def unpack_actions(codes: np.ndarray) -> np.ndarray:
    """uint16 action codes (..., ) -> (..., 6) head indices."""
    codes = codes.astype(np.int64)
    out = np.empty(codes.shape + (N_ACTIONS,), np.uint8)
    for h in range(N_ACTIONS - 1, -1, -1):
        out[..., h] = codes % ACTION_SIZES[h]
        codes //= ACTION_SIZES[h]
    return out


@dataclass
class Game:
    seed: int
    index: int
    n_ticks: int
    winner: int           # 0 truncated, 1 radiant, 2 dire, 3 both ancients fell
    truncated: bool
    radiant_ctrl: str     # "net" | "script"
    dire_ctrl: str
    state_hash: int
    actions: np.ndarray   # (n_ticks, 10, 6) uint8


def read_log(path: str | Path):
    """Yield the games of a .mbl log in order. Returns (header, iterator)."""
    path = Path(path)
    data = path.read_bytes()
    header = np.frombuffer(data, LOG_HEADER_DTYPE, count=1)[0]
    if header["magic"] != LOG_MAGIC or header["version"] != VERSION:
        raise ValueError(f"{path}: not a v{VERSION} MOBA game log")

    def games():
        off = LOG_HEADER_DTYPE.itemsize
        for _ in range(int(header["n_games"])):
            gh = np.frombuffer(data, GAME_HEADER_DTYPE, count=1, offset=off)[0]
            off += GAME_HEADER_DTYPE.itemsize
            raw = zlib.decompress(data[off:off + int(gh["zbytes"])])
            off += int(gh["zbytes"])
            codes = np.frombuffer(raw, "<u2").reshape(int(gh["n_ticks"]), N_HEROES)
            yield Game(int(gh["game_seed"]), int(gh["game_index"]), int(gh["n_ticks"]),
                       int(gh["winner"]), bool(gh["truncated"]),
                       CONTROLLERS[int(gh["radiant_ctrl"])], CONTROLLERS[int(gh["dire_ctrl"])],
                       int(gh["state_hash"]), unpack_actions(codes))

    return header, games()


@dataclass
class Dense:
    path: Path
    header: np.void
    statics: np.ndarray   # (N_HEROES,) STATIC_DTYPE
    games: list           # (game_index, n_records, offset, zbytes) per game, in file order

    @property
    def n_records(self) -> int:
        return int(self.header["n_records"])

    def load_game(self, k: int) -> np.ndarray:
        """All records of the k-th game in the file, (n_ticks + 1,) DENSE_RECORD_DTYPE."""
        game_index, n, off, zb = self.games[k]
        with open(self.path, "rb") as f:
            f.seek(off)
            raw = zlib.decompress(f.read(zb))
        recs = np.frombuffer(raw, DENSE_RECORD_DTYPE)
        if len(recs) != n or recs[0]["game_index"] != game_index or not recs[-1]["final"]:
            raise ValueError(f"{self.path}: game {game_index} is malformed")
        return recs

    def iter_games(self):
        for k in range(len(self.games)):
            yield self.load_game(k)


def transitions(recs: np.ndarray):
    """(s_t, a_t, s_{t+1}) for one game's records: every step, terminal included."""
    return recs["state"][:-1], recs["actions"][:-1], recs["state"][1:]


def load_dense(path: str | Path) -> Dense:
    path = Path(path)
    header = np.fromfile(path, DENSE_HEADER_DTYPE, count=1)[0]
    if header["magic"] != DENSE_MAGIC or header["version"] != VERSION:
        raise ValueError(f"{path}: not a v{VERSION} MOBA dense file")
    for name, want in (("record_bytes", DENSE_RECORD_DTYPE.itemsize),
                       ("state_bytes", STATE_DTYPE.itemsize), ("scan_cap", SCAN_CAP),
                       ("state_fields", len(STATE_FIELDS))):
        if header[name] != want:
            raise ValueError(f"{path}: {name} {header[name]} != {want}; "
                             "moba_state.h / moba_collect.c and shards.py are out of sync")
    games, off = [], DENSE_HEADER_DTYPE.itemsize
    with open(path, "rb") as f:
        for _ in range(int(header["n_games"])):
            f.seek(off)
            g = np.frombuffer(f.read(DENSE_GAME_DTYPE.itemsize), DENSE_GAME_DTYPE)[0]
            off += DENSE_GAME_DTYPE.itemsize
            games.append((int(g["game_index"]), int(g["n_records"]), off, int(g["zbytes"])))
            off += int(g["zbytes"])
    return Dense(path, header, header["statics"].copy(), games)
