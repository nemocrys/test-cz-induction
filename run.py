"""
Main script to execute the simulation.
"""

import opencgs.control as ctrl
from opencgs.sim import DiameterIteration, ParameterStudy, SteadyStateSim
from setup import geometry, simulation


if __name__ == "__main__":
    try:
        git_metadata = ctrl.get_git_metadata()
    except:
        git_metadata = "not available"  # no git on cluster nodes

    geo_config = ctrl.load_config("./config_geo.yml")
    sim_config = ctrl.load_config("./config_sim.yml")
    mat_config = ctrl.load_config("./config_mat.yml")
    config = ctrl.load_config("./config.yml")
    config.update({"metadata": git_metadata})

    # This is used to run steady state simulations / parameter studies
    sim = SteadyStateSim(geometry, geo_config, simulation, sim_config, mat_config, **config)
    # sim = ParameterStudy(SteadyStateSim, geometry, geo_config, simulation, sim_config, mat_config, **config)

    # This is used to run diameter iterations / diameter iteration parameter studies
    # config_di = ctrl.load_config("./config_di.yml")
    # config_di.update({"metadata": git_metadata})
    # sim = DiameterIteration(geometry, geo_config, simulation, sim_config, mat_config, **config_di)
    # sim = ParameterStudy(DiameterIteration, geometry, geo_config, simulation, sim_config, mat_config, **config_di)

    sim.execute()
