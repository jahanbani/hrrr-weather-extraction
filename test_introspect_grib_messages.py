import os
import sys
from pathlib import Path


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "hrrr_enhanced.py").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def list_grib_messages(
    grib_path: Path, target_short_names: set[str], summarize_targets: bool = True
) -> None:
    import pygrib

    print(f"\n=== File: {grib_path.name} ===")
    with pygrib.open(str(grib_path)) as grbs:
        count = 0
        target_hits = {}
        for i, g in enumerate(grbs, start=1):
            count += 1
            # Basic header line
            line = (
                f"#{i:03d} name={getattr(g, 'name', '?')} shortName={getattr(g, 'shortName', '?')} "
                f"levelType={getattr(g, 'typeOfLevel', '?')} level={getattr(g, 'level', '?')} "
                f"units={getattr(g, 'units', '?')} fcstHour={getattr(g, 'forecastTime', '?')}"
            )
            print(line)

            sn = getattr(g, "shortName", None)
            if sn and sn in target_short_names:
                target_hits.setdefault(sn, []).append(g)

        print(f"Total messages: {count}")

        if summarize_targets and target_hits:
            print("\n-- Selector summaries --")
            for sn, gs in sorted(target_hits.items()):
                # Summarize first occurrence to keep fast
                g = gs[0]
                try:
                    vals = g.values
                    vmin = float(vals.min())
                    vmax = float(vals.max())
                    vmean = float(vals.mean())
                    print(
                        f"{sn}: count={len(gs)} range=[{vmin:.3f}, {vmax:.3f}] mean={vmean:.3f}"
                    )
                except Exception as e:
                    print(f"{sn}: count={len(gs)} (failed to read data: {e})")


def main() -> None:
    # Ensure repo root on path for imports
    repo_root = find_repo_root()
    sys.path.insert(0, str(repo_root))

    try:
        from config_unified import HRRRConfig

        config = HRRRConfig()
        SELECTORS = config.SELECTORS
        DATADIR = "/home/alij/EE/Weather/data/hrrr"
    except Exception:
        print(
            "Warning: failed to import SELECTORS from prereise.gather.const; using defaults"
        )

    # Build target shortNames: configured selectors plus albedo variants
    target_short_names = set(SELECTORS.values()) | {"al", "albedo"}

    # Locate day folder next to hrrr_enhanced.py
    day_dir = repo_root / "20250101"
    if not day_dir.exists() and DATADIR:
        alt = Path(DATADIR) / "20250101"
        if alt.exists():
            day_dir = alt
    if not day_dir.exists():
        print(
            f"Error: expected folder {repo_root / '20250101'} or {DATADIR and (Path(DATADIR) / '20250101')} with GRIB files"
        )
        sys.exit(1)

    # Collect f00 and f01 files for 24 hours
    grib_files: list[Path] = []
    for hour in range(24):
        hh = f"{hour:02d}"
        for ff in ("00", "01"):
            fn = f"hrrr.t{hh}z.wrfsfcf{ff}.grib2"
            fp = day_dir / fn
            if fp.exists():
                grib_files.append(fp)

    if not grib_files:
        __import__("ipdb").set_trace()
        print(f"No GRIB files found in {day_dir}")
        sys.exit(1)

    # Runtime guard: set HRRR_LIST_MAX to limit how many files we summarize fully
    max_files_env = os.getenv("HRRR_LIST_MAX")
    max_files = int(max_files_env) if max_files_env else None

    # Iterate files and list all messages; summarize target selectors
    processed = 0
    for fp in grib_files:
        processed += 1
        summarize = True
        if max_files is not None and processed > max_files:
            summarize = False
        list_grib_messages(fp, target_short_names, summarize_targets=summarize)


if __name__ == "__main__":
    main()
