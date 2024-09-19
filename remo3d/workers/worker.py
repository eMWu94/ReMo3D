# -*- coding: utf-8 -*-

from mpi4py import MPI

import numpy as np
import ngsolve as ngs

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gmsh_functions as gmf
import netgen_functions as ngf
import ngsolve_functions as ngsf

# Supress Netgen teminal output during mesh creation process
ngs.ngsglobals.msg_level = 0

# Connect to main process
try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
except:
    raise ValueError('The worker could not connect to main process')

## Collect and wait for all workers to receive data
solve_on = list()
solve_on = comm.bcast(solve_on, root=0)
comm.barrier()

# Import ngsolve functions
if solve_on[rank] == "CPU":
    import ngsolve_functions as ngsf
elif solve_on[rank] == "GPU":
    import ngsolve_functions_gpu as ngsf

## Ask for tasks until receving stop sentinel
for lvl_1_msg in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
    
    # Collect information about shapes of broadcasted arrays
    arrays_shape = list()
    arrays_shape = comm.bcast(arrays_shape, root=0)

    # Prepare empty variables
    formation_parameters = np.empty(arrays_shape[0], dtype='float')
    borehole_geometry = np.empty(arrays_shape[1], dtype='float')
    mud_resistivities = np.empty(arrays_shape[2], dtype='float')
    simulation_depths = np.empty(arrays_shape[3], dtype='float')
    dip = float()
    tools_parameters = dict()
    domain_radius = float()
    mesh_generator = str()
    preconditioner = str()
    condense = bool()
    task_list = list()

    # Fill variables with data
    comm.Bcast([formation_parameters, MPI.FLOAT], root=0)
    comm.Bcast([borehole_geometry, MPI.FLOAT], root=0)
    comm.Bcast([mud_resistivities, MPI.FLOAT], root=0)
    comm.Bcast([simulation_depths, MPI.FLOAT], root=0)
    dip = comm.bcast(dip, root=0)
    tools_parameters = comm.bcast(tools_parameters, root=0)
    domain_radius =  comm.bcast(domain_radius, root=0)
    mesh_generator = comm.bcast(mesh_generator, root=0)
    preconditioner = comm.bcast(preconditioner, root=0)
    condense = comm.bcast(condense, root=0)
    task_list = comm.bcast(task_list, root=0)

    ## Wait for all workers to receive data
    comm.barrier()

    results = list()
    for lvl_2_msg in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
        task = task_list[lvl_2_msg]
        try:
            depth_index = task[0]
            tool = task[1]
            tool_geometry = tool[0,:]
            source_terms = tool[1,:]

            if mesh_generator=="gmsh":
                # Carve out suitable range of data
                local_formation_geometry, local_borehole_geometry, sigma = gmf.SelectGmshDataRange(borehole_geometry, formation_parameters, dip, mud_resistivities[depth_index], simulation_depths[depth_index], domain_radius)
                # Create geometry and mesh
                if dip==0:
                    mesh = gmf.ConstructGmsh2dModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, local_borehole_geometry, rank, mesh_generator)
                else:
                    mesh = gmf.ConstructGmsh3dModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, dip, local_borehole_geometry, rank)
                dirichlet_boundary = 'dirichlet_boundary'
            # Generate mesh using netgen
            elif mesh_generator=="netgen":
                # Carve out suitable range of data
                local_formation_geometry, local_borehole_geometry, sigma = ngf.SelectNetgenDataRange(borehole_geometry, formation_parameters, mud_resistivities[depth_index], simulation_depths[depth_index], domain_radius)
                # Create geometry and mesh
                mesh = ngf.ConstructNetgen2dModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, local_borehole_geometry)
                dirichlet_boundary = [2]

            # Convert data to ngsolve format
            mesh = ngs.Mesh(mesh)
            sigma = ngs.CoefficientFunction(sigma)

            ## Compute measured resistivity
            for modelling_task in task[2]:
                tool = modelling_task[1]
                tool_geometry = tool[0,:]
                source_terms = tool[1,:]

                ## Solve BVP
                fes, gfu = ngsf.SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense)        

                # Compute resistivity values
                for rc_task in modelling_task[2]:
                    depth = rc_task[0]
                    tool = list(tools_parameters.keys())[rc_task[1]]
                    offset = rc_task[2]
                    tool_geometry = tools_parameters[tool][0,:3] + offset
                    source_terms = tools_parameters[tool][1,:3]
                    geometric_factor = tools_parameters[tool][0,3]
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
                    results.append([rc_task[0], rc_task[1], result])
        except:
            for modelling_task in task[2]:
                for rc_task in modelling_task[2]:
                    results.append([rc_task[0], rc_task[1], np.nan])

    # Report results to master process
    comm.barrier()
    comm.gather(sendobj=results, root=0)

## Shutdown
comm.Disconnect()