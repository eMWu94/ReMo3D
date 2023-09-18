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
#mud_resistivities = np.empty(arrays_shape[2], dtype='float')
dip = float()
tools_parameters = dict()
simulation_depths = dict()
domain_radius = float()
mesh_generator = str()
output_folder = str()

# Fill variables with data
comm.Bcast([formation_parameters, MPI.FLOAT], root=0)
comm.Bcast([borehole_geometry, MPI.FLOAT], root=0)
#comm.Bcast([mud_resistivities, MPI.FLOAT], root=0)
dip = comm.bcast(dip, root=0)
tools_parameters = comm.bcast(tools_parameters, root=0)
simulation_depths = comm.bcast(simulation_depths, root=0)
domain_radius =  comm.bcast(domain_radius, root=0)
mesh_generator = comm.bcast(mesh_generator, root=0)
output_folder = comm.bcast(output_folder, root=0)

## Wait for all workers to receive data
comm.barrier()

## Ask for tasks until receving stop sentinel
results = list()
for task in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
    try:
        depth_index = task[0]
        tool = task[1]
        tool_geometry = tool[0,:]
        source_terms = tool[1,:]

        ## Generate mesh
        # Generate mesh using gmsh
        if mesh_generator=="gmsh":
            # Carve out suitable range of data
            local_borehole_geometry = rm.SelectGmshBoreholeDataRange(borehole_geometry, dip, simulation_depths[depth_index], domain_radius)
            local_formation_geometry, local_formation_resistivity = rm.SelectGmshFormationDataRange(formation_parameters, dip, simulation_depths[depth_index], domain_radius)          
            # Create geometry and mesh
            if dip==0:
                rm.ConstructGmsh2dModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, local_borehole_geometry, rank, mesh_generator,
                                        output_mode="file", output_folder=output_folder, file_number=depth_index)
            else:
                rm.ConstructGmsh3dModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, dip, local_borehole_geometry, rank,
                                        output_mode="file", output_folder=output_folder, file_number=depth_index)
        # Generate mesh using netgen
        elif mesh_generator=="netgen":
            # Carve out suitable range of data
            local_formation_geometry, local_borehole_geometry, local_formation_resistivity = rm.SelectNetgenDataRange(borehole_geometry, formation_parameters,
                                                                                                                      simulation_depths[depth_index], domain_radius)
            # Create geometry and mesh
            rm.ConstructNetgen2dModel(domain_radius, tool_geometry, local_formation_geometry, local_borehole_geometry, source_terms,
                                        output_mode="file", output_folder=output_folder, file_number=depth_index)
        results.append(True)
    except:
        results.append(False)

## Report results to master process
comm.gather(sendobj=results, root=0)

## Shutdown
comm.Disconnect()