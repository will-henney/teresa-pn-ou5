#!/usr/bin/env python3
"""
Build a CSV of principal observables from *-global.yaml files.

- CLI: Typer (simple mode); discovery via Path.glob
- YAML structure: top-level keys are line IDs (e.g. "O  3 5006.84A"),
  each with fields including "Flux / H beta".

Included lines:
  He II 4685.68A -> HeII4686_Hb
  He I  5015.68A -> HeI5016_Hb
  O II  4651.00A -> OII4651_Hb
  O III 5006.84A -> OIII5007_Hb
  O III 4363 components and total:
    "O  3 4363.21A" (collisional; CO) -> OIII4363_CO_Hb
    "O 3R 4363.00A" (recombination; RR) -> OIII4363_RR_Hb
    "O 3C 4363.00A" (charge exchange; CX) -> OIII4363_CX_Hb
    Sum -> OIII4363_Hb
  And the ratio: OIII4363_over_OIII5007

All numeric outputs are rounded to 4 significant figures in the CSV.
"""

from pathlib import Path
from typing import Optional, Iterable, Dict, Any
import csv
import typer
import yaml
import math

app = typer.Typer(add_completion=False)

# ---- line IDs ----
HEII_4686    = "He  2 4685.68A"
HEI_5016     = "He  1 5015.68A"
OII_4651     = "O  2 4651.00A"
OIII_5007    = "O  3 5006.84A"
OIII4363_CO  = "O  3 4363.21A"   # collisional
OIII4363_RR  = "O 3R 4363.00A"   # recombination
OIII4363_CX  = "O 3C 4363.00A"   # charge exchange

# Ratio of 5007/4363 from pure recombination contribution
# (Pequignot+1991) to account for the fact that Cloudy does not
# calculate the recombination contribution to 5007
recomb_5007_4363 = 3.4
cx_5007_4363 = 3.4              # This needs checking!

# ---- helpers ----
def _iter_yaml_files(root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(root.glob(pattern))

def _parse_model_name(p: Path) -> Dict[str, Any]:
    stem = p.stem.replace("-global", "")
    toks = stem.split("-")
    out = {"model": stem, "EOS": None, "Z_amp": 0, "Z_lambda": "homog"}
    if len(toks) >= 4:
        eos = toks[3].lower()
        out["EOS"] = "const_n" if eos == "n" else ("const_P" if eos == "p" else None)
    for i, t in enumerate(toks):
        tl = t.lower()
        if tl.startswith("z") and "fluct" in tl:
            digits = "".join(ch for ch in t if ch.isdigit())
            if digits:
                out["Z_amp"] = int(digits)
            if i + 1 < len(toks) and toks[i + 1].lower() in ("short", "long"):
                out["Z_lambda"] = toks[i + 1].lower()
            break
    return out

def _norm_key(s: str) -> str:
    return " ".join(s.split())

def _get_flux_over_hb(data: Dict[str, Any], line_id: str) -> Optional[float]:
    """Return 'Flux / H beta' for the given line_id; tolerant to whitespace."""
    block = data.get(line_id)
    if not isinstance(block, dict):
        target = _norm_key(line_id)
        for k, v in data.items():
            if isinstance(v, dict) and _norm_key(str(k)) == target:
                block = v
                break
    if not isinstance(block, dict):
        return None
    val = block.get("Flux / H beta")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _fmt4(x: Optional[float]) -> Optional[str]:
    """Format to 4 significant figures; keep None as empty."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x):.4g}"
    except Exception:
        return ""

# ---- CLI ----
@app.command()
def main(
    in_dir: Path = typer.Argument(Path("."), help="Directory with *-global.yaml files"),
    pattern: str = typer.Option(
        "ou5a-T115-L660-n-lha*-global.yaml", "--pattern", "-p",
        help="Glob pattern (no regex) to find YAMLs"
    ),
    out_csv: Path = typer.Option(
        Path("ou5a-observables.csv"), "--out", "-o",
        help="Output CSV path"
    ),
):
    files = list(_iter_yaml_files(in_dir, pattern))
    if not files:
        typer.echo(f"No YAMLs matched pattern '{pattern}' in {in_dir}")
        raise typer.Exit(code=1)

    rows = []
    for yml in files:
        data = yaml.safe_load(yml.read_text())
        info = _parse_model_name(yml)

        he2_hb   = _get_flux_over_hb(data, HEII_4686)
        hei5016  = _get_flux_over_hb(data, HEI_5016)
        o2_hb    = _get_flux_over_hb(data, OII_4651)
        o3_5007  = _get_flux_over_hb(data, OIII_5007)

        o3_4363_co = _get_flux_over_hb(data, OIII4363_CO)
        o3_4363_rr = _get_flux_over_hb(data, OIII4363_RR)
        o3_4363_cx = _get_flux_over_hb(data, OIII4363_CX)
        # total 4363/Hb as the sum of available components
        o3_4363_total = sum(v for v in (o3_4363_co, o3_4363_rr, o3_4363_cx) if v is not None) \
                        if any(v is not None for v in (o3_4363_co, o3_4363_rr, o3_4363_cx)) else None
        # Approximately correct for missing recombination contribution to 5007
        o3_5007 += recomb_5007_4363 * o3_4363_rr + cx_5007_4363 * o3_4363_cx

        r4363_5007 = (o3_4363_total / o3_5007) if (o3_4363_total is not None and o3_5007 and o3_5007 > 0) else None

        rows.append({
            "model": info["model"],
            "EOS": info["EOS"],
            "Z_amp": info["Z_amp"],
            "Z_lambda": info["Z_lambda"],
            "HeII4686_Hb":  _fmt4(he2_hb),
            "HeI5016_Hb":   _fmt4(hei5016),
            "OII4651_Hb":   _fmt4(o2_hb),
            "OIII5007_Hb":  _fmt4(o3_5007),
            "OIII4363_CO_Hb": _fmt4(o3_4363_co),
            "OIII4363_RR_Hb": _fmt4(o3_4363_rr),
            "OIII4363_CX_Hb": _fmt4(o3_4363_cx),
            "OIII4363_Hb":    _fmt4(o3_4363_total),
            "OIII4363_over_OIII5007": _fmt4(r4363_5007),
        })

    # Sorting
    def _sort_key(r):
        eos_rank = {"const_n": 0, "const_P": 1, None: 2}.get(r["EOS"], 2)
        lam_rank = {"homog": 0, "short": 1, "long": 2}.get(r["Z_lambda"], 3)
        return (eos_rank, int(r["Z_amp"]), lam_rank, r["model"])
    rows.sort(key=_sort_key)

    cols = [
        "model","EOS","Z_amp","Z_lambda",
        "HeII4686_Hb","HeI5016_Hb","OII4651_Hb",
        "OIII5007_Hb",
        "OIII4363_CO_Hb","OIII4363_RR_Hb","OIII4363_CX_Hb",
        "OIII4363_Hb","OIII4363_over_OIII5007"
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    typer.echo(f"Wrote {out_csv} ({len(rows)} models)")

if __name__ == "__main__":
    app()
