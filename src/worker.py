from mpi4py import MPI

import numpy as np
from ngsolve import *
from ngsolve import ngsglobals
import remo3d as rm


# Supress Netgen teminal output during mesh creation process
ngsglobals.msg_level = 0

# Connect to main process
try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
except:
    raise ValueError('The worker could not connect to main process')

## Collect data
# Collect information about shapes of broadcasted arrays
arrays_shape = list()
arrays_shape = comm.bcast(arrays_shape, root=0)

# Prepare empty variables
formation_parameters = np.empty(arrays_shape[0], dtype='float')
borehole_geometry = np.empty(arrays_shape[1], dtype='float')
mud_resistivities = np.empty(arrays_shape[2], dtype='float')
dip = float()
tools_parameters = dict()
simulation_depths = dict()
domain_radius = float()
mesh_generator = str()
preconditioner = str()
condense = bool()

# Fill variables with data
comm.Bcast([formation_parameters, MPI.FLOAT], root=0)
comm.Bcast([borehole_geometry, MPI.FLOAT], root=0)
comm.Bcast([mud_resistivities, MPI.FLOAT], root=0)
dip = comm.bcast(dip, root=0)
tools_parameters = comm.bcast(tools_parameters, root=0)
simulation_depths = comm.bcast(simulation_depths, root=0)
domain_radius =  comm.bcast(domain_radius, root=0)
mesh_generator = comm.bcast(mesh_generator, root=0)
preconditioner = comm.bcast(preconditioner, root=0)
condense = comm.bcast(condense, root=0)

## Wait for all workers to receive data
comm.barrier()

## Ask for tasks until receving stop sentinel
results = list()
for task in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
    try:
        tool = list(tools_parameters.keys())[task[1]]
        depth = task[0]
        tool_geometry = tools_parameters[tool][0,:3]
        source_terms = tools_parameters[tool][1,:3]
        geometric_factor = tools_parameters[tool][0,3]
        if mesh_generator=="gmsh":
            # Carve out suitable range of data
            local_borehole_geometry = rm.SelectGmshBoreholeDataRange(borehole_geometry, dip, simulation_depths[tool][depth], domain_radius)
            local_formation_geometry, local_formation_resistivity = rm.SelectGmshFormationDataRange(formation_parameters, dip, simulation_depths[tool][depth], domain_radius)
            # Create geometry and mesh
            if dip==0:
                mesh = rm.ConstructGmsh2dModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, local_borehole_geometry, rank, mesh_generator)
            else:
                mesh = rm.ConstructGmsh3dModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, dip, local_borehole_geometry, rank)
            sigma = CoefficientFunction([1/mud_resistivities[depth]] + list(1/local_formation_resistivity)) # Conductivity distribution within the model
            dirichlet_boundary = 'dirichlet_boundary'
        elif mesh_generator=="netgen":
        # Carve out suitable range of data
            local_formation_geometry, local_borehole_geometry, sigma = rm.SelectNetgenDataRange(borehole_geometry, formation_parameters, mud_resistivities[depth], simulation_depths[tool][depth], domain_radius)
            # Create geometry and mesh
            mesh = rm.ConstructNetgen2dModel(domain_radius, tool_geometry, local_formation_geometry, local_borehole_geometry, source_terms)
            dirichlet_boundary = [2]
        # Solve BVP
        fes, gfu = rm.SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense)
        # Compute measured resistivity
        measuring_electodes = tool_geometry[source_terms==0]
        if dip==0:
            if np.shape(measuring_electodes)[0] == 2:
                result = abs(geometric_factor * (gfu(mesh(0.0, measuring_electodes[1]))-gfu(mesh(0.0, measuring_electodes[0]))))
            elif np.shape(measuring_electodes)[0] == 1:
                result = abs(geometric_factor * gfu(mesh(0.0, measuring_electodes[0])))
        else:
            if np.shape(measuring_electodes)[0] == 2:
                result = abs(geometric_factor * (gfu(mesh(0.0, 0.0, measuring_electodes[1]))-gfu(mesh(0.0, 0.0, measuring_electodes[0]))))/2 # division by two because only halfsphere is present within the model
            elif np.shape(measuring_electodes)[0] == 1:
                result = abs(geometric_factor * gfu(mesh(0.0, 0.0, measuring_electodes[0])))/2 # division by two because only halfsphere is present within the model
        # Append result to results
        results.append([task[0], task[1], result])
    except:
        results.append([task[0], task[1], np.nan])

## Report results to master process
comm.gather(sendobj=results, root=0)

## Shutdown
comm.Disconnect()