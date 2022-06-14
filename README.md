# test-cz-induction

Thermal simulation of Test-CZ Czochralski growth furnace with induction heating.

The project is developed and maintained by the [**Model experiments group**](https://www.ikz-berlin.de/en/research/materials-science/section-fundamental-description#c486) at the Leibniz Institute for Crystal Growth (IKZ).

### Referencing
If you use this code in your research, please cite our article (available with open access):

> A. Enders-Seidlitz, J. Pal, and K. Dadzis, Development and validation of a thermal simulation for the Czochralski crystal growth process using model experiments *Journal of Crystal Growth*, In Press. [https://doi.org/10.1016/j.jcrysgro.2022.126750](https://doi.org/10.1016/j.jcrysgro.2022.126750).

## Overview

This project is used for 2D steady-state electromagnetism and heat transfer simulation of the NEMOCRYS Test-CZ Furnace.

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

[This project](https://www.researchgate.net/project/NEMOCRYS-Next-Generation-Multiphysical-Models-for-Crystal-Growth-Processes) has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 851768).

<img src="https://github.com/nemocrys/test-cz-induction/blob/main/EU-ERC.png">
