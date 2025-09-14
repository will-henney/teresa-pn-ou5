import numpy as np
import cloudytab
from pathlib import Path
import typer
import yaml

def extract_line_data(
        model: cloudytab.CloudyModel,
        reference_line: tuple[str, str] = ("H beta", "H  1 4861.32A"),
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
    linelist = list(model.data["ems"].columns[1:])
    linedata = {}
    for lineid in linelist:
        emis = model.data["ems"][lineid]
        lum = np.trapz(emis * dVdr, r)
        if lum:
            tmean = np.trapz(te * emis * dVdr, r) / lum
            nmean = np.trapz(ne * emis * dVdr, r) / lum
            t2 = np.trapz(
                ((te - tmean) / tmean) ** 2 * emis * dVdr,
                r
            ) / lum
        else:
            tmean = nmean = t2 = np.nan
        linedata[lineid] = {
            "Luminosity": float(lum),
            "Mean Te": float(tmean),
            "Mean Ne": float(nmean),
            "t^2": float(t2),
        }
    # Second pass to add flux relative to H beta
    if reference_line[1] in linelist:
        reference_lum = linedata[reference_line[1]]["Luminosity"]
        relative_label = f"Flux / {reference_line[0]}"
        for data in linedata.values():
            data[relative_label] = data["Luminosity"] / reference_lum
            
    return linedata

def main(
        prefix_pattern: str,
        data_dir: Path = Path("."),
        save_id: str = "global",
        verbose: bool = False,
) -> None:
    """
    Extract global parameters from emission line structure of Cloudy models
    (line luminosity, emission-weighted mean n_e, temperature, t^2)
    """
    data_paths = data_dir.glob(prefix_pattern + ".in")
    for p in data_paths:
        model_id = p.stem
        data = extract_line_data(cloudytab.CloudyModel(model_id))
        save_file = f"{model_id}-{save_id}.yaml"
        with open(save_file, "w") as f:
            yaml.dump(data, f)
            if verbose:
                print("Global line data saved to", save_file)

    
if __name__ == "__main__":
    typer.run(main)
