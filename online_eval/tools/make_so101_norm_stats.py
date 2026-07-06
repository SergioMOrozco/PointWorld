# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off script: create stats/so101/norm_stats.json by copying stats/droid/norm_stats.json
verbatim and renaming the "droid" domain key to "so101" in both top-level statistics dicts.

SO101 was never part of PointWorld's training data, so no real normalization statistics exist
for it. Aliasing droid's (the closest existing domain: real, single-arm, non-simulated) is an
explicit, labeled approximation -- see CLAUDE.md / online_eval/ plan notes. No numeric values
are changed here, only the domain key.
"""

import json
from pathlib import Path

SRC = Path("stats/droid/norm_stats.json")
DST = Path("stats/so101/norm_stats.json")


def main() -> None:
    with open(SRC, "r") as f:
        data = json.load(f)

    for top_key in ("statistics", "per_timestep_statistics"):
        assert top_key in data, f"expected top-level key {top_key!r} in {SRC}"
        assert "droid" in data[top_key], f"expected 'droid' domain under {top_key!r} in {SRC}"
        assert "so101" not in data[top_key], f"unexpected existing 'so101' domain under {top_key!r}"
        data[top_key]["so101"] = data[top_key].pop("droid")

    DST.parent.mkdir(parents=True, exist_ok=True)
    existing = [p for p in DST.parent.glob("*.json")]
    assert not existing, (
        f"{DST.parent} must contain exactly one norm_stats.json; found existing files: {existing}"
    )

    with open(DST, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Wrote {DST} (domain key 'droid' -> 'so101', values unchanged)")


if __name__ == "__main__":
    main()
