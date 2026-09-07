"""race.h and racewm/contract.py must not drift.

The Python side decides how every dataset column is interpreted; the C side
decides what actually gets written. If they disagree, every downstream number
is quietly wrong, so this is a hard failure rather than a warning.

    python3 tests/test_contract.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from racewm import contract, shards  # noqa: E402

FAILURES = []


def check(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main():
    header = ROOT / "race" / "race.h"

    print("== race.h vs contract.py ==")
    try:
        contract.check_against_header(header)
        check(True, "constants agree (STATE_VERSION, MAX_SECTORS, OBS_SIZE)")
    except AssertionError as e:
        check(False, "constants agree", str(e))

    h = contract.parse_header(header)
    print(f"    race.h: v{h['RACE_STATE_VERSION']}  sectors={h['RACE_MAX_SECTORS']}  obs={h['RACE_OBS_SIZE']}")

    print("== layout is well formed ==")
    check(len(contract.FIELDS) == contract.OBS_SIZE, "field count == OBS_SIZE")
    check([f.index for f in contract.FIELDS] == list(range(contract.OBS_SIZE)),
          "indices are dense and ordered")
    check(len(set(contract.NAMES)) == len(contract.NAMES), "names are unique")

    print("== schema hash is stable ==")
    check(contract.schema_hash() == contract.schema_hash(), "hash is deterministic")
    check(len(contract.schema_hash()) == 16, "hash is 16 hex chars")
    print(f"    schema_hash = {contract.schema_hash()}")

    print("== binary record layout ==")
    # 4 int32 + float + 4 uint8 + OBS_SIZE float32
    expected = 4 * 4 + 4 + 4 + contract.OBS_SIZE * 4
    check(shards.RECORD_DTYPE.itemsize == expected,
          f"Record is {expected} bytes", f"got {shards.RECORD_DTYPE.itemsize}")
    check(shards.RECORD_DTYPE["obs"].shape == (contract.OBS_SIZE,),
          "record obs width matches OBS_SIZE")

    print("== angle pairs ==")
    for a, b in contract.ANGLE_PAIRS:
        na, nb = contract.NAMES[a], contract.NAMES[b]
        check(na.startswith("sin_") and nb.startswith("cos_") and na[4:] == nb[4:],
              f"pair ({na}, {nb}) is a matched sin/cos")

    print("== round trip against a live shard, if one exists ==")
    probe = sorted((ROOT / "data").glob("*.bin")) if (ROOT / "data").exists() else []
    probe = [p for p in probe if not p.name.endswith(".tracks.bin")]
    if not probe:
        print("    skipped: no shards in data/ (run build/race_eval --dump first)")
    else:
        hdr = shards.read_header(probe[0])
        check(hdr["obs_dim"] == contract.OBS_SIZE, "shard obs_dim matches contract")
        check(hdr["state_version"] == contract.STATE_VERSION, "shard state_version matches")
        check(hdr["record_bytes"] == shards.RECORD_DTYPE.itemsize, "shard record size matches")
        print(f"    {probe[0].name}: v{hdr['state_version']} obs={hdr['obs_dim']} "
              f"{hdr['width']:.0f}x{hdr['height']:.0f} frameskip={hdr['frameskip']}")

    print()
    if FAILURES:
        print(f"contract: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("contract: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
