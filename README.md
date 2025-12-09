# Analysis of spectra of Ou 5 planetary nebula




## Layout of this repo

-   `data/` contains mainly FITS files
    -   `data/originals` is the full set of original files from Teresa.
        -   It is not included in the repo for space reasons but is available at [this Google Drive link.](https://drive.google.com/file/d/1xLLAS8lK4L31MmRzrAHUYBETDrKt5v0l/view?usp=drive_web)
    -   `data/first-look` is the initial partially reduced spectra from Teresa for reference
        -   Also includes a [PDF showing the slit positions](data/first-look/slit-spm-final-1.pdf).
-   `docs/` contains documentation
    - [pn-ou5-physics](docs/pn-ou5-physics.org) contains most of the work that is not in notebooks
    - [pn-ou5-pipeline](docs/pn-ou5-pipeline.org) describes all the steps in the initial data reduction
    - [pn-ou5-admin](docs/pn-ou5-admin.org) describes setting up the python virtual environment, etc
-   `notebooks/` contains lots of jupyter notebooks
-   `mes-longslit/` contains python source code for library functions that are used in the notebooks
-   [pyproject.toml](./pyproject.toml) specifies the dependencies for use with uv or other package manager
-   `cloudy/` contains input and output files from the Cloudy models
-   `cspn-tables/` contains third-party post-agb evolutionary tracks (from Miller Bertolami and from MIST)
