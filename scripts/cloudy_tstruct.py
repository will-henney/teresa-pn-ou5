import numpy as np
import cloudytab
from pathlib import Path
import typer
import yaml

# Ions for which we want to extract the structure.
# Dict of {ION: (ELEMENT_COLUMN, ION_COLUMN), ...}
# ELEMENT_COLUMN is column name for element density in .abun file
# ION_COLUMN is column name for ion fraction in .ovr file
IONS = {
    "H+": ("abund H", "HII"),
    "He+": ("HELI", "HeII"),
    "He++": ("HELI", "HeIII"),
    "O+": ("OXYG", "O2"),
    "O++": ("OXYG", "O3"),
    "O+++": ("OXYG", "O4"),
}

def extract_tstruct_data(
        model: cloudytab.CloudyModel,
        ions: dict[tuple[str, str]] = IONS,
) -> dict[str, dict[str, float]]:
    """
    Loop over all emission lines and calculate some global quantities:

    * Total line luminosity (integral of emissivity over volume)
    * Emission-weighted mean Te, mean Ne, and fractional Te variance (Peimbert's t^2)
    * Flux ratio with respect to reference line (by default: H beta)

    Returns a dict with keys of line labels and values being another dict of quantities
    """
    r = model.data["rad"]["radius"]
    te = model.data["ovr"]["Te"]
    ne = model.data["ovr"]["eden"]
    dVdr = 4 * np.pi * r ** 2
    
    iondata = {}
    for ionid, (abun_col, ovr_col)  in ions.items():
        # ion density
        ni = model.data["ovr"][ovr_col] * 10 ** model.data["abun"][abun_col]
        # What we are weighting the temperature by
        weight = ne * ni * dVdr
        # volume emission measure
        vem = np.trapz(weight, r)
        if vem:
            t0 = np.trapz(te * weight, r) / vem
            t2 = np.trapz(
                ((te - t0) / t0) ** 2 * weight,
                r
            ) / vem
        else:
            t0 = t2 = np.nan
        iondata[ionid] = {
            "VEM": float(vem),
            "T_0": float(t0),
            "t^2": float(t2),
        }
    return iondata

def main(
        prefix_pattern: str,
        data_dir: Path = Path("."),
        save_id: str = "tstruct",
        verbose: bool = False,
) -> None:
    """
    Extract per-ion T_0 and t^2 from physical structure of Cloudy models
    """
    data_paths = data_dir.glob(prefix_pattern + ".in")
    for p in data_paths:
        model_id = p.stem
        data = extract_tstruct_data(cloudytab.CloudyModel(model_id))
        save_file = f"{model_id}-{save_id}.yaml"
        with open(save_file, "w") as f:
            yaml.dump(data, f)
            if verbose:
                print("Global per-ion T structure data saved to", save_file)

    
if __name__ == "__main__":
    typer.run(main)
