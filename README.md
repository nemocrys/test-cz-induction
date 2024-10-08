# test-cz-induction

[![DOI](https://zenodo.org/badge/448813510.svg)](https://doi.org/10.5281/zenodo.13890483)

Thermal simulation of Test-CZ Czochralski growth furnace with induction heating.

The project is developed and maintained by the [**Model experiments group**](https://www.ikz-berlin.de/en/research/materials-science/section-fundamental-description#c486) at the Leibniz Institute for Crystal Growth (IKZ).

### Referencing

If you use this code in your research, please cite our open-access publication:

> A. Enders-Seidlitz, J. Pal, and K. Dadzis, Development and validation of a thermal simulation for the Czochralski crystal growth process using model experiments *Journal of Crystal Growth*,  593 (2022) 126750. [https://doi.org/10.1016/j.jcrysgro.2022.126750](https://doi.org/10.1016/j.jcrysgro.2022.126750).

Further details can be found in:

> A. Wintzer, *Validation of multiphysical models for Czochralski crystal growth*. PhD thesis, Technische Universität Berlin, Berlin, 2024. [https://doi.org/10.14279/depositonce-20957](https://doi.org/10.14279/depositonce-20957)

## Overview

This project is used for 2D steady-state electromagnetism and heat transfer simulation of the NEMOCRYS Test-CZ Furnace.

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S002202482200238X-gr2_lrg.jpg" width=25%>


### Configuration

The configuration of the simulation is stored in yml-files. For normal simulations *config.yml* is used.

The files *config_geo.yml* (geometry configuration), *config_sim.yml* (simulation setup), *config_mat.yml* (material parameters) are usually left unchanged, to make changes to these parameters the section config_update in *config.yml* is used.

The configuration for the iterative crystal diameter computation can be found in *config_di.yml*.

### Geometry and simulation setup

The geometry and simulation setup is defined in the function *geometry* in *setup.py*.

### Execution

If the required python packages and elmer solver are installed simulations can be executed with the script "run.py". Just uncomment the required part!

Usage of docker is highly recommended, see next section.

## Computational setup (Docker)

The setup for the simulations is provided in form of a docker image, so just an installation of [Docker](https://docs.docker.com/get-docker/) is required on your system. The image nemocrys/opencgs:v0.2.1 is used (see [opencgs](https://github.com/nemocrys/opencgs) for more information), future versions may require changes to the code.

On Windows, the simulations can be executed with:
```
docker run --rm -v ${PWD}:/home/workdir nemocrys/opencgs:v0.2.1 python3 run.py
```

On Linux, the simulations can be executed with:
```
docker run -it --rm -v $PWD:/home/workdir -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) nemocrys/opencgs:v0.2.1 python3 run.py
```

## Acknowledgements

[This project](https://nemocrys.github.io/) has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 851768).

<img src="https://github.com/nemocrys/test-cz-induction/blob/main/EU-ERC.png">
