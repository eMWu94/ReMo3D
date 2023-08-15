from mpi4py import MPI

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib import ticker

import linecache as lc
import numpy as np
import scipy.interpolate as spi
import itertools
import datetime
import shutil
import sys
import os

import gmsh

import netgen.meshing as msh
from netgen.csg import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters, meshsize

from ngsolve import *


##  Main functions 

def SetToolsParameters(tools):
    """
    This function sets logging tools parameters based on their names.

    Parameters
    -------
    tools: list
        A list of tools. Names are strings and have to consist of symbols of 3 different electrods
        (A and/or B for current electrodes and M and/or N for measuring electrodes) listed from
        the top one to the bottom one and 2 numbers that specifies distances in meters between
        consecutive electrodes.
        Example: tools = ["N2.5M0.25A", "B5.7A0.4M"]

    Returns
    -------
    tools_parameters: dict
        A dictionary of numpy arrays that specify parameters of logging tools.
    """

    ### Check if data format is correct
    if type(tools)!=list or all(isinstance(s, str) for s in tools)==False:
        raise ValueError("Tools names have to be provided in the form of list of strings")

    ### Set tools parameters
    tools_parameters = dict()
    for tool in tools:
        
        # Extract information about tool geometry from the tool name
        tool_data = [str2float(item) for item in [''.join(group) for _, group in itertools.groupby(tool, str.isalpha)]]
        electrodes = tuple(x for x in tool_data if isinstance(x, str)) # symbols of eletrodes
        distances = [x for x in tool_data if isinstance(x, float)] # distances between electrodes
        ## Create tool parameters and add tool to dictionary
        tools_parameters[tool] = SetToolParameters(tool, electrodes, distances)         

    return tools_parameters

def SetModelParameters(formation_model_file, borehole_model_file, borehole_geometry='diameter', dip=0):
    """
    This function import formation and borehole parameters from txt files
    and checks if they are correct.

    Parameters
    -------
    formation_model_file: str
        A string that specifies path to file that stores parameters of the formation model.
    
    borehole_model_file: str
        A string that specifies path to file that stores parameters of the borehole model.
    
    borehole_geometry: str, optional
        A string that specifies type of borehole geometry. Available options: "diameter" and "radius".
        By default set to "diameter".

    dip: float, optional
        A value between 0 and 90. Describes dip of the layers in relation to the borehole axis.
        By default set to 0.

    Returns
    -------
    model_parameters: list
        A list of two numpy arrays that specify borehole and formation parameters.
    """

    # Allowed units conversion table
    units = {'M':1.0, 'DM':0.1, 'CM':0.01, 'MM':0.001, "IN":0.0254, 'FT':0.3048}

    ## Formation parameters
    formation_parameters = np.atleast_2d(np.loadtxt(formation_model_file, delimiter="\t", skiprows=2))
    # Formation geometry
    formation_units = lc.getline(formation_model_file, 2).split()[:-2]
    for i in range(len(formation_units)):
        if formation_units[i] in units.keys():
            formation_parameters[:,i] *= units[formation_units[i]]
        else:
            raise ValueError("{} unit in formation model file not recognized. Allowed units: M, DM, CM, MM, IN, FT".format(formation_units[i]))
    if (np.diff(formation_parameters[:,:2], axis=0)<=0.0).any()==True or (formation_parameters[1:,0]!=formation_parameters[:-1,1]).any()==True:
        raise ValueError("Uncorrect formation model geometry")
    if dip<0 or dip>=90:
        raise ValueError("Uncorrect dip angle")

    # Formation resistivity
    if np.nanmin(formation_parameters[:,[3,4]])<=0.0:
        raise ValueError('Formation resistivies have to be higher than 0 ohmm')

    ## Borehole parameters
    borehole_parameters = np.atleast_2d(np.loadtxt(borehole_model_file, delimiter="\t", skiprows=2))
    if np.shape(borehole_parameters)[0]<2:
        raise ValueError('Borehole paramaters have to be defined for at least two depths')
    
    # Borehole geometry
    borehole_units = lc.getline(borehole_model_file, 2).split()[:-1]
    for i in range(len(borehole_units)):
        if borehole_units[i] in units.keys():
            borehole_parameters[:,i] *= units[borehole_units[i]]
        else:
            raise ValueError("{} unit in borehole model file not recognized. Allowed units: M, DM, CM, MM, IN, FT".format(borehole_units[i]))
    if (np.diff(borehole_parameters[:,0], axis=0)<=0.0).any()==True or (borehole_parameters[:,1]<=0.0).any()==True:
        raise ValueError("Uncorrect borehole model geometry")
    
    if borehole_geometry=='diameter':
        borehole_parameters[:,1] /= 2
    elif borehole_geometry=='radius':
        pass
    else:
        raise ValueError("Uncorrect borehole geometry type - use 'diameter' or 'radius' to specify borehole geometry")

    # Borehole resistivity
    if np.nanmin(borehole_parameters[:,2])<=0.0:
        raise ValueError('Drilling mud resistivies have to be higher than 0 ohmm')

    # Geometry check
    for i in range(np.shape(formation_parameters)[0]):
        layer_extend = borehole_parameters[(borehole_parameters[:,0]>=formation_parameters[i,0]) & (borehole_parameters[:,0]<=formation_parameters[i,1]), 1]
        if np.any(layer_extend>=formation_parameters[i,2]):
            raise ValueError('Borehole radius have to be smaller than the extend of the filtration zone')

    model_parameters = [formation_parameters, borehole_parameters, dip]
    return model_parameters

def ComputeSyntheticLogs(tools_parameters, model_parameters, measurement_depths, force_single_electrode_configuration=False, domain_radius=50, processes=4, mesh_generator="gmsh", preconditioner="multigrid", condense=True):
    """
    This function computes syntetic logs.

    Parameters
    -------
    tools_parameters: dict
        A dictionary of numpy arrays created by the SetToolsParameters() function.

    model_parameters: list
        A list of numpy arrays created by the SetModelParameters() funcion.

    measurement_depths: array
        A 1D numpy array of depths of simulated measurements.
        Values have to be given in ascending order and corespond to depths of the model.

    force_single_electrode_configuration: str
        Specifies if two-electrode tool configurations will be changed to single-electrode tool configurations.
        Enables faster computations. Can be set to True or False.
        By default set to False.
        
    domain_radius: float, optional
        A radius of simulation domain in meters.
        By default set to 100.

    processes: int, optional
        Specify a number of processes. Minimal value that can be set is 2, the maximal value should not exceed the number of processes available on computing machine.
        By default set to 4.

    mesh_generator: string, optional
        Specify utiliezed mesh generator.
        For 2D models can be set to "gmsh" or "netgen", for 3D models it have to be set to "gmsh".
        By default set to "gmsh".
 
    preconditioner: string, optional
        Specify a type of utilized preconditioner. Available options: "local" and "multigrid".
        By default set to "multigrid".
    
    condense: bool, optional
        Specify if static condensation will be utilized to eliminate unknowns that are internal to elements from the global linear system.
        By default set to True.

    Returns
    -------
    logs: dict
        A dictionary of 1D numpy arrays with computed synthetic logs.
        If simulation for a certain depth and tool will fail for some reason, the NaN value will be inserted into log.
    """

    ### Start the clock
    start_time = datetime.datetime.now()

    ### Create temporary directory for mesh files
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    ### Unpack parameters and prepare data
    if force_single_electrode_configuration==False:
        pass
    elif force_single_electrode_configuration==True:
        for tool in tools_parameters.keys():
            if "A" in tool and "B" in tool:
                alternative_tool = tool.translate(str.maketrans("ABMN", "MNAB"))
                alternative_tool_data = [str2float(item) for item in [''.join(group) for _, group in itertools.groupby(alternative_tool, str.isalpha)]]
                alternative_electrodes = tuple(x for x in alternative_tool_data if isinstance(x, str)) # symbols of eletrodes
                alternative_distances = [x for x in alternative_tool_data if isinstance(x, float)] # distances between electrodes
                tools_parameters[tool] = SetToolParameters(alternative_tool, alternative_electrodes, alternative_distances)
    else:
        raise ValueError("The value of parameter force_single_electrode_configuration can be set only to True or False")
    
    formation_parameters = model_parameters[0]
    borehole_parameters = model_parameters[1]
    dip = model_parameters[2]*np.pi/180 #converted to radians

    simulation_depths = dict()
    domain_radius_alert = False
    for tool in tools_parameters.keys():
        tools_parameters[tool][0,:3] -= tools_parameters[tool][1,3] # center simulation around current electrodes
        simulation_depths[tool] = measurement_depths + tools_parameters[tool][1,3]
        if np.max(np.abs(tools_parameters[tool][0,:3])) > domain_radius:
            raise ValueError("Some electrodes are locate outside the simulation domain. Domain size have to be increased")
        elif np.max(np.abs(tools_parameters[tool][0,:3])) > 0.75*domain_radius:
            domain_radius_alert = True
    if domain_radius_alert == True:
        print("Some electrodes are located close to the boundary of the simulation domain. This may cause problems during simulation. Consider increase of the domain size")
    
    if dip!=0 and mesh_generator!="gmsh":
        raise ValueError("The only mesh generator supported in 3D models is gmsh")

    if dip!=0:
        # Create dense borehole geometry for the purpose of 3D mesh generation (necessary to avoid errors during meshing procedure)
        borehole_parameters = AddPointsToBorehole(borehole_parameters, 0.15)

    borehole_geometry = np.ascontiguousarray(borehole_parameters[:,:2])
    mud_resistivities = np.interp(measurement_depths, borehole_parameters[:,0], borehole_parameters[:,2])

    ### Parallel FEM computation
    ## Specify number of workers and tasks
    
    if type(processes) != int:
        raise ValueError("The number of processes have to be intager")
    if processes < 2:
        raise ValueError("Minimal number of processes is 2")
    
    n_workers = processes - 1 # one process is reserved for the master, the rest for the workers
    n_tasks = np.shape(measurement_depths)[0]

    ## Spawn workers
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker.py')], 
        maxprocs=n_workers)

    ## Broadcast data to workers
    # Broadcast information about shapes of broadcasted arrays
    arrays_shape = [np.shape(formation_parameters), np.shape(borehole_geometry), np.shape(mud_resistivities)]
    comm.bcast(arrays_shape, root=MPI.ROOT)

    # Broadcast data
    comm.Bcast([formation_parameters, MPI.FLOAT], root=MPI.ROOT)
    comm.Bcast([borehole_geometry, MPI.FLOAT], root=MPI.ROOT)
    comm.Bcast([mud_resistivities, MPI.FLOAT], root=MPI.ROOT)
    comm.bcast(dip, root=MPI.ROOT)
    comm.bcast(tools_parameters, root=MPI.ROOT)
    comm.bcast(simulation_depths, root=MPI.ROOT)
    comm.bcast(domain_radius, root=MPI.ROOT)
    comm.bcast(mesh_generator, root=MPI.ROOT)
    comm.bcast(preconditioner, root=MPI.ROOT)
    comm.bcast(condense, root=MPI.ROOT)

    ## Wait for all workers to receive data
    comm.barrier()

    ## Prepare tasks 
    task_list = []   
    tool_list = list(range(len(tools_parameters.keys())))
    measurement_depths_list = list(range(len(measurement_depths)))
    for task in itertools.product(measurement_depths_list, tool_list):
        task_list.append(list(task))
    msg_list = task_list + ([StopIteration] * n_workers) # Append stop sentinel for each worker

    ## Dispatch tasks to workers
    status = MPI.Status()
    for i in msg_list:
        if i != StopIteration:
            # Pass data to worker
            comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(obj=i, dest=status.Get_source())
            # Progress bar
            percent = ((i[0] + 1) * 100) // (n_tasks)
            sys.stdout.write('\rProgress: [%-50s] %3i%% ' % ('=' * (percent // 2), percent))
            sys.stdout.flush()
        else:
            # Tell worker to shutdown
            comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(obj=i, dest=status.Get_source())

    # Gather results from workers
    list_of_results = [item for sublist in comm.gather(None, root=MPI.ROOT) for item in sublist]
   
    ## Format and sort results
    results = np.empty([len(measurement_depths_list), len(tool_list)])
    for result in list_of_results:
        results[result[0], result[1]] = result[2]

    logs = dict()
    for i in range(len(tools_parameters.keys())):
        logs[list(tools_parameters.keys())[i]] = np.vstack([measurement_depths, results[:,i]]).T

    ### Shutdown MPI
    comm.Disconnect()

    ### Remove tmp folder with and mesh files
    shutil.rmtree("./tmp")

    ## Report time of computation
    print('\nProcessed in: ', datetime.datetime.now() - start_time)

    return logs


def SaveResults(model_parameters, measurement_results, measurement_to_save="auto", output_folder=None, plot_layout="auto", depth_lim="auto", rad_lim="auto", res_lim="auto", model_res_lim="auto", aspect_ratio = "auto", at_nan="break", interpolation_factor=1, colors="auto"):
    """
    This function saves results of modelling to txt file and produces raw visualization of the model and computed syntetic logs that is saved in PNG format.

    Parameters
    -------
    model_parameters: list
        A list of numpy arrays created by the SetModelParameters() funcion.

    measurement_results: dict
        A dictionary of 1D numpy arrays created by the ComputeSynteticLogs function.
        
    measurement_to_save: str or list
        A list of measurements to save.
        By default set to "auto" will save all measurements.

    output_folder: str
        A path to the folder where results will be saved.
        By default set to None will only show results without saving them.

    plot_layout: list, optional
        a list of sublists of tool names. Each sublist consist of tool names assigned to certain track.
        By default set to "auto" will plot all logs on a sigle track.

    depth_lim: list, optional
        A list of two floats that specify minimum and maximum depth of ploted data.
        By default set to "auto" will plot data for entire range of avilable depths.

    rad_lim: list, optional
        A list of two floats that specify minimum and maximum radius of ploted formation model.
        By default set to "auto" will plot data from borehole axis to radius 2 times as big as
        deepest filtration zone or 10 times as big as largest borehole radius if no filtration zone
        is present within the model.
    
    res_lim: list, optional
        A list of two floats that specify minimum and maximum value of ploted resistivity logs.
        By default set to "auto" will automaticly adjust range to show entire logs.
    
    aspect_ratio: float, optional
        A float that specify hight to widht ratio of the plot.
        By default set to "auto" will automaticly adjust the value of parameter based on number of tracks
        within layout and values of depth_lim and rad_lim parameters.

    at_nan: str, optional
        Specify if plot should be breaked or continued on Nan values.
        Available options: "break" and "continue".
        By default set to "break".
    
    interpolation_factor: float, optional
        Allows to smooth logs on vizualization. Have no inpact on output data.
        By default set to 1 (no interpolation).

    """
    
    if output_folder!=None:
        ### Create output folder
        output_subfolder = os.path.join(output_folder, "Results_{}/".format(str(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))))

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        ### Save data to txt files
        if measurement_to_save=="auto":
            measurement_to_save = list(measurement_results.keys())

        logs_to_save = measurement_to_save.copy()
        file_number = 1
        while len(logs_to_save)>0:
            logs = [logs_to_save[0]]
            for i in range(1, len(logs_to_save)):
                if np.shape(measurement_results[logs_to_save[0]][:,0])[0] == np.shape(measurement_results[logs_to_save[i]][:,0])[0]:
                    if np.all(np.isclose(measurement_results[logs_to_save[0]][:,0], measurement_results[logs_to_save[i]][:,0]), axis=0)==True:
                        logs.append(logs_to_save[i])

            for log in logs:
                logs_to_save.remove(log)

            results = measurement_results[logs[0]]
            for i in range(1, len(logs)):
                results = np.hstack([results, np.atleast_2d(measurement_results[logs[i]][:,1]).T])

            names = ['DEPTH'] + logs
            units = ['M'] + ['OHMM']*len(logs)
            header = '\t'.join([name for name in names]) + '\n' + '\t'.join([unit for unit in units])
            np.savetxt(output_subfolder + 'Results_{}.txt'.format(file_number), results,  fmt='%.4f', delimiter='\t', header=header, comments='')
            file_number += 1

    ### Formation model visualization
    
    ## Unpack parameters
    formation_parameters = model_parameters[0]
    borehole_parameters = model_parameters[1]
    dip = model_parameters[2]

    ## Smooth logs
    if interpolation_factor > 1:
        logs = list(measurement_results.keys())
        for log in logs:
            measurement_depths_interp = np.linspace(np.min(measurement_results[log][:,0]), np.max(measurement_results[log][:,0]), int(np.shape(measurement_results[log])[0]*interpolation_factor))
            interpolation = spi.interp1d(measurement_results[log][:,0], measurement_results[log][:,1], kind='cubic')
            measurement_results[log] = np.vstack([measurement_depths_interp, interpolation(measurement_depths_interp)]).T

    ## Prepare plot limits:
    if depth_lim == "auto":
        depth_lim = [np.nanmin(formation_parameters[:,:2]), np.nanmax(formation_parameters[:,:2])]
    if rad_lim == "auto":
        if np.all(np.isnan(formation_parameters[:,2])):
            rad_lim = [-10*np.nanmax(borehole_parameters[:,1]), 10*np.nanmax(borehole_parameters[:,1])]
        else:
            rad_lim = [-2*np.nanmax(formation_parameters[:,2]), 2*np.nanmax(formation_parameters[:,2])]
    if res_lim == "auto":
        res_max = 0
        for log in measurement_results.values():
            res_max = max(np.max(log), res_max)
        res_min = res_max
        for log in measurement_results.values():
            res_min = min(np.min(log), res_min)
        res_min = np.floor(res_min/10**np.floor(np.log10(res_max)-1)) * 10**np.floor(np.log10(res_max)-1)
        res_max = np.ceil(res_max/10**np.floor(np.log10(res_max)-1)) * 10**np.floor(np.log10(res_max)-1)
        res_lim = [res_min, res_max]
    if aspect_ratio == "auto":
        aspect_ratio = (depth_lim[1] - depth_lim[0])/25*1.25
    
    ## Prepare polygons
    patches = []

    # Formation
    a = np.tan(dip*np.pi/180)
    formation_parameters[0,0] -= a*rad_lim[1] # adjust model to fill the plot
    formation_parameters[-1,1] += a*rad_lim[1] # adjust model to fill the plot
    for i in range(np.shape(formation_parameters)[0]):
        if np.isnan(formation_parameters[i,2]) == True:
            vetrices = np.array([[rad_lim[0], formation_parameters[i,0]+a*rad_lim[0]],
                [rad_lim[0], formation_parameters[i,1]+a*rad_lim[0]],
                [rad_lim[1], formation_parameters[i,1]+a*rad_lim[1]],
                [rad_lim[1], formation_parameters[i,0]+a*rad_lim[1]]])
            polygon = Polygon(vetrices, closed=True)
            patches.append(polygon)
        else:
            vetrices = np.array([[rad_lim[0], formation_parameters[i,0]+a*rad_lim[0]],
                [rad_lim[0], formation_parameters[i,1]+a*rad_lim[0]],
                [rad_lim[1], formation_parameters[i,1]+a*rad_lim[1]],
                [rad_lim[1], formation_parameters[i,0]+a*rad_lim[1]]])
            polygon = Polygon(vetrices, closed=True)
            patches.append(polygon)
            vetrices = np.array([[-formation_parameters[i,2], formation_parameters[i,0]+a*-formation_parameters[i,2]],
                [-formation_parameters[i,2], formation_parameters[i,1]+a*-formation_parameters[i,2]],
                [formation_parameters[i,2], formation_parameters[i,1]+a*formation_parameters[i,2]],
                [formation_parameters[i,2], formation_parameters[i,0]+a*formation_parameters[i,2]]])
            polygon = Polygon(vetrices, closed=True)
            patches.append(polygon)
    resistivities = np.ndarray.flatten(np.flip(formation_parameters[:,3:], axis=1))

    # Borehole
    if borehole_parameters is not None:
        left_boundary = borehole_parameters[:,[1,0]]*[-1, 1]
        right_boundary = borehole_parameters[:,[1,0]]
        vetrices = np.vstack([left_boundary, np.flip(right_boundary, axis=0)])
        polygon = Polygon(vetrices, closed=True)
        patches.append(polygon)
        resistivities = np.hstack([resistivities, np.mean(borehole_parameters[:,2])])
    borehole_axis = Line2D([0, 0], depth_lim, color='black')

    ## Plot model and logs
    if plot_layout=="auto":
        tracks = 1
    else:
        tracks = len(plot_layout)

    fig_width = 5 + 5*tracks
    fig_hight = fig_width*aspect_ratio
    
    resistivities = resistivities[~np.isnan(resistivities)]
    collection = PatchCollection(patches, cmap=matplotlib.cm.viridis)
    collection.set_array(resistivities)

    if model_res_lim!="auto":
        collection.set_clim(model_res_lim)
        
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.titlepad': 14,
        "xtick.major.size": 10, "xtick.minor.size": 5, "ytick.major.size": 10, "ytick.minor.size": 5})

    fig, ax = plt.subplots(1, 1+tracks, sharey=True, figsize=[fig_width, fig_hight], facecolor="white")
    
    ax[0].add_collection(collection)
    ax[0].add_line(borehole_axis)
    ax[0].margins(x=0, y=0)
    ax[0].set_xlim(rad_lim)
    ax[0].set_ylim(depth_lim)
    ax[0].invert_yaxis()
    ax[0].minorticks_on()
    ax[0].set_title('Formation model\n'+ 'dip = ' + str(dip) + '\N{DEGREE SIGN}\n')
    ax[0].set_xlabel('Radial distance [m]', labelpad=10)
    ax[0].set_ylabel('Depth [m]', labelpad=10)
    ticks = ax[0].get_xticks()
    ax[0].xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax[0].set_xticklabels(["{0:.2f}".format(abs(tick)) for tick in ticks])
    ax[0].xaxis.set_ticks_position('top') 
    ax[0].xaxis.set_label_position('top') 
    ax[0].autoscale_view()
    
    for track in range(1, tracks+1):
            
        if colors=="auto":
            track_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            track_colors = colors[track-1]
        if plot_layout=="auto":
            logs = list(measurement_results.keys())
        else:
            logs = plot_layout[track-1]
        for i in range(len(logs)):
            if i==0:
                axis = ax[track]
            else:
                axis = ax[track].twiny()
            if at_nan=="break":
                axis.plot(measurement_results[logs[i]][:,1], measurement_results[logs[i]][:,0], color=track_colors[i]) ### ZMIANA NA OSTATNIĄ CHWILĘ color=track_colors[i%len(colors)]
            elif at_nan=="continue":
                nan_flag = ~np.isnan(measurement_results[logs[i]][:,1])
                axis.plot(measurement_results[logs[i]][nan_flag,1], measurement_results[logs[i]][nan_flag,0], color=track_colors[i]) ### ZMIANA NA OSTATNIĄ CHWILĘ color=track_colors[i%len(colors)]
            else:
                raise ValueError('at_nan paramater has to be set to "break" or "continue"')
            axis.set_xlabel(logs[i]+"\n[ohmm]", color=track_colors[i%len(track_colors)], labelpad=-8)
            axis.spines['top'].set_color(track_colors[i%len(track_colors)])
            axis.spines['top'].set_position(('outward', i*55+10))
            axis.set_xticks(res_lim)
            axis.tick_params(axis='x', color=track_colors[i%len(track_colors)])
            axis.set_xlim(res_lim)
        axis = ax[track].twiny().get_xaxis().set_visible(False)
    for track in range(1, tracks+1):
        ax[track].grid(True)
        ax[track].xaxis.set_label_position('top')
        ax[track].xaxis.set_ticks_position('top')
        ax[track].margins(x=0, y=0)
        ax[track].autoscale_view()
            
    colorbar = fig.colorbar(collection, ax=ax, location='bottom', orientation='horizontal', pad=0.05, label="Resistivity [ohmm]", shrink=min([1, aspect_ratio]))
    colorbar.ax.minorticks_on()

    ## Save plot to png file
    if output_folder!=None:
        plt.savefig(output_subfolder + 'Results_plot.png', bbox_inches='tight')


## Helper functions utilized within main funcions and workers

def str2float(item):
    """
    This function convert item from strings to floats (if possible).
    """
    try:
        return float(item)
    except ValueError:
        return item
        
def SetToolParameters(tool, electrodes, distances):
    """
    This function is constructing single tool parameters.
    """

    # Check if tool configuration is correct and set electrodes position in relation to measurment point position
    if len(electrodes)!=3 or np.shape(distances)[0]!=2 or min(distances)<=0:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))

    correct_configurations = itertools.permutations(["A", "B", "M", "N"], 3) # all correct electrodes configurations

    if electrodes in list(correct_configurations):
        # Calculate measurement point position in relation to the top electrode positon set to 0
        if distances[0] < distances[1]:
            z_mp =  distances[0]/2
        elif distances[0] > distances[1]:
            z_mp =  distances[0] + distances[1]/2
        else:
            raise ValueError("{} logging tool specification is uncorrect".format(tool))
        # Calculate electrodes positions in relation to measurment point position set to 0
        positons = np.array([0, 0+distances[0], 0+distances[0]+distances[1]]) # electrodes positions in relation to the top electrode positon set to 0
        z_a, z_b, z_m, z_n = np.NaN, np.NaN, np.NaN, np.NaN
        for i in range(3):
            if electrodes[i]=='A':
                z_a = positons[i] - z_mp
            elif electrodes[i]=='B':
                z_b = positons[i] - z_mp
            elif electrodes[i]=='M':
                z_m = positons[i] - z_mp
            elif electrodes[i]=='N':
                z_n = positons[i] - z_mp
    else:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))

    ## Calculate and arrange tool parameters into an array
    # alternate_source_terms - for 1 current electrode configuration (will not change the results of modelling)
    if np.isnan(z_a):
        BM = abs(z_b - z_m)
        BN = abs(z_b - z_n)
        geometric_factor = abs(4*np.pi*BM*BN/(BN-BM))
        depth_shift = z_b
        available_electrodes = np.array([z_b, z_m, z_n])
        source_terms = np.array([1, 0, 0])
    elif np.isnan(z_b):
        AM = abs(z_a - z_m)
        AN = abs(z_a - z_n)
        geometric_factor = abs(4*np.pi*AM*AN/(AN-AM))
        depth_shift = z_a
        available_electrodes = np.array([z_a, z_m, z_n])
        source_terms = np.array([1, 0, 0])
    elif np.isnan(z_m):
        AN = abs(z_a - z_n)
        BN = abs(z_b - z_n)
        geometric_factor = abs(4*np.pi*AN*BN/(AN-BN))
        depth_shift = (z_a+z_b)/2
        available_electrodes = np.array([z_a, z_b, z_n])
        source_terms = np.array([1, -1, 0])
    elif np.isnan(z_n):
        AM = abs(z_a - z_m)
        BM = abs(z_b - z_m)
        geometric_factor = abs(4*np.pi*AM*BM/(BM-AM))
        depth_shift = (z_a+z_b)/2
        available_electrodes = np.array([z_a, z_b, z_m])
        source_terms = np.array([1, -1, 0])

    sort_order = np.argsort([available_electrodes])
    tool_geometry = np.ndarray.flatten(available_electrodes[sort_order])
    source_terms = np.ndarray.flatten(source_terms[sort_order])

    # Merge tool parameters
    tool_parameters = np.hstack([np.vstack([tool_geometry, source_terms]), np.array([[geometric_factor], [depth_shift]])])

    return tool_parameters 



# GMSH functions

def AddPointsToBorehole(borehole_parameters, maximal_distance):

        ## Add additional points if borehole geometry is too sparse
        interpolated_depths = borehole_parameters[0,0]
        for i in range(1, np.shape(borehole_parameters)[0]):
            distance = borehole_parameters[i,0] - borehole_parameters[i-1,0]
            if distance > maximal_distance:
                additional_points = np.linspace(borehole_parameters[i-1,0], borehole_parameters[i,0], np.max([3, int(distance*10+1)]))
                interpolated_depths = np.hstack([interpolated_depths, additional_points[1:]])
            else:
                interpolated_depths = np.hstack([interpolated_depths, borehole_parameters[i,0]])

        ## Interpolate geometry and mud resistivities
        if np.shape(interpolated_depths)[0] > np.shape(borehole_parameters)[0]:
            geometry_interpolation = spi.interp1d(borehole_parameters[:,0], borehole_parameters[:,1], kind='linear')
            mud_resistivity_interpolation = spi.interp1d(borehole_parameters[:,0], borehole_parameters[:,2], kind='linear')

            borehole_parameters = np.vstack([interpolated_depths, geometry_interpolation(interpolated_depths), mud_resistivity_interpolation(interpolated_depths)]).T

        return borehole_parameters

def SelectGmshBoreholeDataRange(borehole_geometry, dip, simulation_depth, domain_radius):
    
    def domain_line_intersection (p1, p2, radius):
        x_1, y_1 = p1[1], p1[0]
        x_2, y_2 = p2[1], p2[0]
        d_x = x_2 - x_1
        d_y = y_2 - y_1
        d_r = np.sqrt(d_x**2 + d_y**2)
        D = x_1*y_2 - x_2*y_1
        delta = radius**2*d_r**2-D**2
        for sign in (-1, 1):
            x = (D*d_y+sign*np.sign(d_y)*d_x*np.sqrt(delta))/d_r**2
            y = (-D*d_x+sign*np.abs(d_y)*np.sqrt(delta))/d_r**2
            p = np.array([y, x])
            if np.dot(p1-p2, p1-p)>0 and np.dot(p1-p2, p1-p)<np.dot(p1-p2, p1-p2):
                return np.array([y, x])
        
    ### Borehole geometry
    ## Select data relevant to construct model geometry within simulation domain
    if np.shape(borehole_geometry)[0]==2:
        local_borehole_geometry = borehole_geometry.copy()    
    else:
        if dip==0:
            point_within_domain_mask = (borehole_geometry[:,0]-simulation_depth)**2+borehole_geometry[:,1]**2 < domain_radius**2
        else:
            point_within_domain_mask = np.abs(borehole_geometry[:,0]-simulation_depth) < domain_radius
        relevant_points_mask = np.convolve(point_within_domain_mask, np.array([True, True, True]), mode="same")
        local_borehole_geometry = borehole_geometry[relevant_points_mask,:]
    local_borehole_geometry[:,0] -= simulation_depth

    ## Adjust top point to the domain
    if dip==0:
        # do nothing if point is on domain boundary
        if local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  == domain_radius**2:
            pass
        # else if point is within the domain add additional point on domain boundary
        elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  < domain_radius**2:
            omega = np.arccos(local_borehole_geometry[0, 1]/domain_radius)
            local_borehole_geometry = np.vstack((np.array([-np.sin(omega)*domain_radius, local_borehole_geometry[0, 1]]), local_borehole_geometry))
        # else if point is outside the domain move it on the domain boundary
        elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  > domain_radius**2:
            local_borehole_geometry[0,:] = domain_line_intersection(local_borehole_geometry[0, :], local_borehole_geometry[1, :], domain_radius)
    else:
        # do nothing if point is on domain boundary
        if np.abs(local_borehole_geometry[0, 0]) == domain_radius:
            pass
        # else if point is within the domain add additional point on domain boundary
        elif np.abs(local_borehole_geometry[0, 0]) < domain_radius:
            local_borehole_geometry = np.vstack((np.array([-domain_radius, local_borehole_geometry[0, 1]]), local_borehole_geometry))
        # else if point is outside the domain move it on the domain boundary
        elif np.abs(local_borehole_geometry[0, 0]) > domain_radius:
            a = local_borehole_geometry[0,0]-(-domain_radius)
            b = -domain_radius-local_borehole_geometry[1,0]
            local_borehole_geometry[0,:] = [-domain_radius, (b*local_borehole_geometry[0,1]+a*local_borehole_geometry[1,1])/(a+b)]

    ## Adjust bottom point to the domain
    if dip==0:
        # do nothing if point is on domain boundary
        if local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2 == domain_radius**2:
            pass
        # else if point is within the domain add additional point on domain boundary
        elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2  < domain_radius**2:
            omega = np.arccos(local_borehole_geometry[-1, 1]/domain_radius)
            local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([np.sin(omega)*domain_radius, local_borehole_geometry[-1, 1]])))
        # else if point is outside the domain move it on the domain boundary
        elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2  > domain_radius**2:
            local_borehole_geometry[-1,:] = domain_line_intersection(local_borehole_geometry[-1, :], local_borehole_geometry[-2, :], domain_radius)
    else:
       # do nothing if point is on domain boundary
        if np.abs(local_borehole_geometry[-1, 0]) == domain_radius:
            pass
        # else if point is within the domain add additional point on domain boundary
        elif np.abs(local_borehole_geometry[-1, 0]) < domain_radius:
            local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([domain_radius, local_borehole_geometry[-1, 1]])))
        # else if point is outside the domain move it on the domain boundary
        elif np.abs(local_borehole_geometry[-1, 0]) > domain_radius:
            a = local_borehole_geometry[-1,0]-domain_radius
            b = domain_radius-local_borehole_geometry[-2,0]
            local_borehole_geometry[-1,:] = [domain_radius, (b*local_borehole_geometry[-1,1]+a*local_borehole_geometry[-2,1])/(a+b)]
    
    return (local_borehole_geometry)

def SelectGmshFormationDataRange(formation_parameters, dip, simulation_depth, domain_radius, active_geometry_window=0.99):
    
    ### Formation geometry
    ## Compute active geometry radius 
    # Active geometry prevents occurence of thin layers and small wedges at the very edge of simulation by ignoring occurence of small elements of model at the very edge of simulation domain
    active_geometry_radius = domain_radius * active_geometry_window 

    ## Select data relevant to construct model geometry within simulation domain
    # Select layers that are present within active geometry area
    local_formation_parameters = formation_parameters.copy()
    local_formation_parameters[:,:2] -= simulation_depth

    if dip==0:
        d = np.abs(local_formation_parameters[:,:2])
    else:
        a = np.tan(dip)
        b = 1
        c = local_formation_parameters[:,:2]
        d = np.abs(c)/(a**2 + b**2)**(1/2) # the equation is |a*x0+b*y0+c|/(a^2+b^2)^(1/2), since (x0, y0) = (0, 0) the form of equation is simplified
    relavant_layers = local_formation_parameters[np.any(d<active_geometry_radius, axis=1),:]

    # Remove undisturbed zones that are not present within active geometry area
    layers_to_check = ~np.isnan(relavant_layers[:,2])

    if dip==0:
        x_points = np.repeat(np.atleast_2d(relavant_layers[layers_to_check, 2]).T, 2, axis=1)
        y_points = relavant_layers[layers_to_check,:2]
    else:
        x_points = np.repeat(np.atleast_2d(relavant_layers[layers_to_check, 2]).T, 4, axis=1)
        x_points[:, :2] *= -1
        y_points = a*x_points + np.hstack([relavant_layers[layers_to_check,:2], relavant_layers[layers_to_check,:2]])
    d = (x_points**2 + y_points**2)**(1/2)

    zones_within = np.any(d<active_geometry_radius, axis=1)

    zones_to_remove = layers_to_check.copy()
    zones_to_remove[layers_to_check] = ~zones_within

    local_formation_model = relavant_layers.copy()
    if np.shape(formation_parameters)[1]==5:
        # If resistivity data are present modify information about model geometry and formation resistivity distribution
        local_formation_model[zones_to_remove, 4] = local_formation_model[zones_to_remove,3]
        local_formation_model[zones_to_remove, 2:4] = np.nan
    elif np.shape(formation_parameters)[1]==3:
        # If resistivity data are not present modify information about model geometry
        local_formation_model[zones_to_remove, 2] = np.nan

    ## Adjust bottom and top boundary to the domain
    # abs_c - value that will stretch bottom and top layers slightly outside simulation domain
    if dip==0:
        abs_c = domain_radius*1.01
    else:
        abs_c = domain_radius*(a**2 + b**2)**(1/2)*1.01 # Calculated from a simplified form of the equation |a*x0+b*y0+c|/(a^2+b^2)^(1/2)
    
    # Adjust top point to the domain
    if local_formation_model[0, 0] > -abs_c:
        local_formation_model[0, 0] = -abs_c
    
    # Adjust bottom point to the domain
    if local_formation_model[-1, 1] < abs_c:
        local_formation_model[-1, 1] = abs_c

    ### Return selected data
    if np.shape(formation_parameters)[1]==5:
        ## If resistivity data are present return information about model geometry and formation resistivity distribution
        # Set formation geometry
        local_formation_geometry = local_formation_model[:,:3]
        # Set formation resistivity distribution
        formation_resistivity_distribution = np.ndarray.flatten(local_formation_model[:,3:5])
        formation_resistivity_distribution = formation_resistivity_distribution[~np.isnan(formation_resistivity_distribution)]
        return (local_formation_geometry, formation_resistivity_distribution)
    elif np.shape(formation_parameters)[1]==3:
        ## If resistivity data are not present return information about model geometry
        return (local_formation_model)

def ReadGmsh(filename, mesh_dimensionality):
    """
    Utilized ReadGmsh function is a slightly modified version of the orginal ReadGmsh function from Netgen/NGSolve package
    Source: https://github.com/NGSolve/netgen/blob/master/python/read_gmsh.py
    """

    if not filename.endswith(".msh"):
        filename += ".msh"

    f = open(filename, 'r')
    mesh = msh.Mesh(dim=mesh_dimensionality)

    pointmap = {}
    facedescriptormap = {}
    namemap = { 0 : "default" }
    materialmap = {}
    bbcmap = {}

    segm = 1
    trig = 2
    quad = 3
    tet = 4
    hex = 5
    prism = 6
    pyramid = 7
    segm3 = 8      # 2nd order line
    trig6 = 9      # 2nd order trig
    tet10 = 11     # 2nd order tet
    point = 15
    quad8 = 16     # 2nd order quad
    hex20 = 17     # 2nd order hex
    prism15 = 18   # 2nd order prism
    pyramid13 = 19 # 2nd order pyramid
    segms = [segm, segm3]
    trigs = [trig, trig6]
    quads = [quad, quad8]
    tets = [tet, tet10]
    hexes = [hex, hex20]
    prisms = [prism, prism15]
    pyramids = [pyramid, pyramid13]
    elem0d = [point]
    elem1d = segms
    elem2d = trigs + quads
    elem3d = tets + hexes + prisms + pyramids

    num_nodes_map = { segm : 2,
                        trig : 3,
                        quad : 4,
                        tet : 4,
                        hex : 8,
                        prism : 6,
                        pyramid : 5,
                        segm3 : 3,
                        trig6 : 6,
                        tet10 : 10,
                        point : 1,
                        quad8 : 8,
                        hex20 : 20,
                        prism15 : 18,
                        pyramid13 : 19 }

    while True:
        line = f.readline()
        if line == "":
            break

        if line.split()[0] == "$PhysicalNames":
            #print('WARNING: Physical groups detected - Be sure to define them for every geometrical entity.')
            numnames = int(f.readline())
            for i in range(numnames):
                f.readline
                line = f.readline()
                namemap[int(line.split()[1])] = line.split()[2][1:-1]

        if line.split()[0] == "$Nodes":
            num = int(f.readline().split()[0])
            for i in range(num):
                line = f.readline()
                nodenum, x, y, z = line.split()[0:4]
                pnum = mesh.Add(msh.MeshPoint(msh.Pnt(float(x), float(y), float(z))))
                pointmap[int(nodenum)] = pnum

        if line.split()[0] == "$Elements":
            num = int(f.readline().split()[0])

            number_of_elements = num

            for i in range(num):
                line = f.readline().split()
                elmnum = int(line[0])
                elmtype = int(line[1])
                numtags = int(line[2])
                # the first tag is the physical group nr, the second tag is the group nr of the dim
                tags = [int(line[3 + k]) for k in range(numtags)]

                if elmtype not in num_nodes_map:
                    raise Exception("element type", elmtype, "not implemented")
                num_nodes = num_nodes_map[elmtype]

                nodenums = line[3 + numtags:3 + numtags + num_nodes]
                nodenums2 = [pointmap[int(nn)] for nn in nodenums]

                if elmtype in elem1d:
                    if mesh_dimensionality == 3:
                        if tags[1] in bbcmap:
                            index = bbcmap[tags[1]]
                        else:
                            index = len(bbcmap) + 1
                            if len(namemap):
                                mesh.SetCD2Name(index, namemap[tags[0]])
                            else:
                                mesh.SetCD2Name(index, "line" + str(tags[1]))
                            bbcmap[tags[1]] = index

                    elif mesh_dimensionality == 2:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = msh.FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[tags[0]]
                            else:
                                fd.bcname = 'line' + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[tags[0]])
                            else:
                                mesh.SetMaterial(index, "line" + str(tags[1]))
                            materialmap[tags[1]] = index

                    mesh.Add(msh.Element1D(index=index, vertices=nodenums2))

                if elmtype in elem2d:  # 2d elements
                    if mesh_dimensionality == 3:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = msh.FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[tags[0]]
                            else:
                                fd.bcname = "surf" + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[tags[0]])
                            else:
                                mesh.SetMaterial(index, "surf" + str(tags[1]))
                            materialmap[tags[1]] = index

                    if elmtype in trigs:
                        ordering = [i for i in range(3)]
                        if elmtype == trig6:
                            ordering += [4,5,3]
                    if elmtype in quads:
                        ordering = [i for i in range(4)]
                        if elmtype == quad8:
                            ordering += [4, 6, 7, 5]
                    mesh.Add(msh.Element2D(index, [nodenums2[i] for i in ordering]))

                if elmtype in elem3d:  # volume elements
                    if tags[1] in materialmap:
                        index = materialmap[tags[1]]
                    else:
                        index = len(materialmap) + 1
                        if len(namemap):
                            mesh.SetMaterial(index, namemap[tags[0]])
                        else:
                            mesh.SetMaterial(index, "vol" + str(tags[1]))
                        materialmap[tags[1]] = index

                    nodenums2 = [pointmap[int(nn)] for nn in nodenums]

                    if elmtype in tets:
                        ordering = [0,1,2,3]
                        if elmtype == tet10:
                            ordering += [4,6,7,5,9,8]
                    elif elmtype in hexes:
                        ordering = [0,1,5,4,3,2,6,7]
                        if elmtype == hex20:
                            ordering += [8,16,10,12,13,19,15,14,9,11,18,17]
                    elif elmtype in prisms:
                        ordering = [0,2,1,3,5,4]
                        if elmtype == prism15:
                            ordering += [7,6,9,8,11,10,13,12,14]
                    elif elmtype in pyramids:
                        ordering = [3,2,1,0,4]
                        if elmtype == pyramid13:
                            ordering += [10,5,6,8,12,11,9,7]
                    mesh.Add(msh.Element3D(index, [nodenums2[i] for i in ordering]))
    return mesh

def ConstructGmsh2dModel(domain_radius, tool_geometry, source_terms, formation_geometry, borehole_geometry, file_number, mesh_generator, output_folder_path="./tmp", output_mode="variable"):

    ### GMSH test
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("model_"+str(file_number))

    ## Add central point (removed from final geometry)
    gmsh.model.occ.addPoint(0, 0, 0, tag=1)

    ## Points at borehole axis (bottom -> top)
    gmsh.model.occ.addPoint(0, domain_radius, 0, tag=2)
    index_0D = 3
    for i in range(np.shape(tool_geometry)[0]):
        gmsh.model.occ.addPoint(0, np.flip(tool_geometry)[i], 0, tag=index_0D)
        index_0D +=1
    gmsh.model.occ.addPoint(0, -domain_radius, 0, tag=index_0D)
    index_0D +=1
    points_at_borehole_axis = list(np.arange(2, index_0D))

    ## Points at borehole boundary (top -> bottom)
    points_at_borehole_boundary = []
    for i in range(0, np.shape(borehole_geometry)[0]):
        gmsh.model.occ.addPoint(borehole_geometry[i,1], borehole_geometry[i,0], 0, tag=index_0D)
        points_at_borehole_boundary.append(index_0D)
        index_0D += 1

    # Add 2D borehole template
    index_1D = 1

    ## Add lines on borehole axis
    lines_at_borehole_axis = []
    for i in range(len(points_at_borehole_axis)-1):
        gmsh.model.occ.addLine(points_at_borehole_axis[i], points_at_borehole_axis[i+1], tag=index_1D)
        lines_at_borehole_axis.append(index_1D)
        index_1D += 1

    ## Add lines at borehole boundary
    lines_at_borehole_boundary = []
    for i in range(len(points_at_borehole_boundary)-1):
        gmsh.model.occ.addLine(points_at_borehole_boundary[i], points_at_borehole_boundary[i+1], tag=index_1D)
        lines_at_borehole_boundary.append(index_1D)
        index_1D += 1

    ## Add arc at top of the domain (axis -> boundary)
    gmsh.model.occ.addCircleArc(points_at_borehole_axis[-1], 1, points_at_borehole_boundary[0], tag=index_1D)
    top_arc = [index_1D]
    index_1D += 1

    ## Add arc at bottom of the domain (boundary -> axis)
    gmsh.model.occ.addCircleArc(points_at_borehole_boundary[-1], 1, points_at_borehole_axis[0], tag=index_1D)
    bottom_arc = [index_1D]
    index_1D += 1

    ## Add external domain boundary
    circle_arc = gmsh.model.occ.addCircleArc(points_at_borehole_boundary[0], 1, points_at_borehole_boundary[-1], tag=index_1D)
    domain_arc = [index_1D]
    index_1D += 1

    ## Add domain
    domain_loop = gmsh.model.occ.addCurveLoop(lines_at_borehole_boundary + domain_arc, tag=3)
    domain = gmsh.model.occ.addPlaneSurface([3], tag=3)

    ## Add borehole
    borehole_loop = gmsh.model.occ.addCurveLoop(lines_at_borehole_axis + top_arc + lines_at_borehole_boundary + bottom_arc, tag=4)
    borehole = gmsh.model.occ.addPlaneSurface([4], tag=4)

    ## Split formation into layers
    boundaries_z = np.sort(np.unique(formation_geometry[:,:2]))
    pg_index_list = [4]
    index_2D = 5
    if np.shape(boundaries_z)[0]==2:
        if np.isnan(formation_geometry[0, 2]):
            pg_index_list.append(6)
        else:
            filtration_template = gmsh.model.occ.addRectangle(0, boundaries_z[i-1], 0, formation_geometry[0, 2], boundaries_z[i]-boundaries_z[i-1], tag=1)
            filtration = gmsh.model.occ.intersect([(2, 1)], [(2,3)], removeTool = False, tag=index_2D)
            undisturbed_template = gmsh.model.occ.addRectangle(formation_geometry[0, 2], boundaries_z[i-1], 0, (domain_radius-formation_geometry[0, 2])*1.01, boundaries_z[i]-boundaries_z[i-1], tag=2)
            undisturbed = gmsh.model.occ.intersect([(2, 2)], [(2,3)], removeTool = False, tag=index_2D+1)
            pg_index_list.append(index_2D)
            pg_index_list.append(index_2D+1)
            index_2D += 2
    else:
        for i in range(1, np.shape(boundaries_z)[0]):
            if np.isnan(formation_geometry[i-1, 2]):
                layer_template = gmsh.model.occ.addRectangle(0, boundaries_z[i-1], 0, domain_radius*1.01, boundaries_z[i]-boundaries_z[i-1], tag=1)
                layer = gmsh.model.occ.intersect([(2, 1)], [(2,3)], removeTool = False, tag=index_2D)
                pg_index_list.append(index_2D)
                index_2D += 1
            else:
                filtration_template = gmsh.model.occ.addRectangle(0, boundaries_z[i-1], 0, formation_geometry[i-1, 2], boundaries_z[i]-boundaries_z[i-1], tag=1)
                filtration = gmsh.model.occ.intersect([(2, 1)], [(2,3)], removeTool = False, tag=index_2D)
                undisturbed_template = gmsh.model.occ.addRectangle(formation_geometry[i-1, 2], boundaries_z[i-1], 0, (domain_radius-formation_geometry[i-1, 2])*1.01, boundaries_z[i]-boundaries_z[i-1], tag=2)
                undisturbed = gmsh.model.occ.intersect([(2, 2)], [(2,3)], removeTool = False, tag=index_2D+1)
                pg_index_list.append(index_2D)
                pg_index_list.append(index_2D+1)
                index_2D += 2
        
        gmsh.model.occ.remove([(2,3)], recursive=True)

    gmsh.model.occ.removeAllDuplicates()        
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.field.add("MathEval", 1) # Horizontal, linear meshsize field
    gmsh.model.mesh.field.setString(1, "F", "x + 0.1")

    i = 2
    for electrode_position in tool_geometry[source_terms != 0]:
        gmsh.model.mesh.field.add("MathEval", i) # Meshsize field modification near current electrodes
        gmsh.model.mesh.field.setString(i, "F","(x^2 + (y+({}))^2)/2 + 0.01".format(electrode_position))
        i += 1

    if np.shape(tool_geometry[source_terms != 0])[0]==1:
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1,2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)
    else:
        gmsh.model.mesh.field.add("Min", 4)
        gmsh.model.mesh.field.setNumbers(4, "FieldsList", [1,2,3])
        gmsh.model.mesh.field.setAsBackgroundMesh(4)
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    dimention, lines = list(zip(*gmsh.model.occ.getEntities(1)))

    dirichlet_boundaries = []
    neumann_boundaries = []
    for i in range(len(lines)):
        nodes = gmsh.model.mesh.getNodes(dim=1, tag=lines[i], includeBoundary = True)
        coordinates = nodes[1].reshape((int(np.shape(nodes[1])[0]/3),3))
        R = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
        if np.allclose(R, domain_radius):
            dirichlet_boundaries.append(lines[i])
        else:
            neumann_boundaries.append(lines[i])

    db = gmsh.model.addPhysicalGroup(1, dirichlet_boundaries, 1)
    gmsh.model.setPhysicalName(1, db , "dirichlet_boundary")

    nb = gmsh.model.addPhysicalGroup(1, neumann_boundaries, 2)
    gmsh.model.setPhysicalName(2, nb , "neumann_boundary")

    i = 1
    for index in pg_index_list:
        surface = gmsh.model.addPhysicalGroup(2, [index], index)
        gmsh.model.setPhysicalName(2, surface, "surf_"+str(i))
        i += 1

    # Save file
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_folder_path + "/fm_"+str(file_number)+".msh")
    gmsh.finalize()
    
    if output_mode == "variable":
        # Read file
        mesh = ReadGmsh(output_folder_path + "/fm_"+str(file_number)+".msh", 2)
        mesh = Mesh(mesh)
        
        return mesh

def ConstructGmsh3dModel(domain_radius, tool_geometry, source_terms, formation_geometry, dip, borehole_geometry, file_number, output_folder_path="./tmp", output_mode="variable"):

    ### GMSH test
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("model_"+str(file_number))

    ## Points at borehole axis (bottom -> top)
    gmsh.model.occ.addPoint(0, 0, domain_radius, tag=1)
    for i in range(2,5):
        gmsh.model.occ.addPoint(0, 0, np.flip(tool_geometry)[i-2], tag=i)
    gmsh.model.occ.addPoint(0, 0, -domain_radius, tag=5)
    points_at_borehole_axis = [1,2,3,4,5]
    index_0D = 6

    ## Points at borehole boundary (top -> bottom)
    points_at_borehole_boundary = []
    for i in range(0, np.shape(borehole_geometry)[0]):
        gmsh.model.occ.addPoint(borehole_geometry[i,1], 0, borehole_geometry[i,0], tag=index_0D)
        points_at_borehole_boundary.append(index_0D)
        index_0D += 1

    # Add 2D borehole template
    borehole_points = points_at_borehole_axis + points_at_borehole_boundary
    index_1D = 1
    first_point = index_1D
    for i in range(len(borehole_points)-1):
        gmsh.model.occ.addLine(borehole_points[i], borehole_points[i+1], index_1D)
        index_1D += 1
    gmsh.model.occ.addLine(borehole_points[-1], borehole_points[0], index_1D)
    last_point = index_1D
    index_1D += 1

    borehole_loop = gmsh.model.occ.addCurveLoop(list(range(first_point, last_point+1)), 1)
    borehole_surface = gmsh.model.occ.addPlaneSurface([1], 1)

    ## Domain
    gmsh.model.occ.addSphere(0, 0, 0, domain_radius, tag=1, angle3 = np.pi)

    ## Add borehole template
    borehole_template = gmsh.model.occ.revolve([(2,1)], 0, 0, 0, 0, 0, 1, np.pi) # tag (3,2)

    # Split domain template into borehole and formation
    borehole = gmsh.model.occ.intersect([(3,2)], [(3,1)], tag = 3, removeObject= False, removeTool = False)
    formation = gmsh.model.occ.cut([(3,1)], [(3,2)], tag = 4)

    ## Formation
    boundaries_z = np.sort(np.unique(formation_geometry[:,:2]))
    pg_index_list = [3]
    index_3d = 5
    if np.shape(boundaries_z)[0]==2:
        if np.isnan(formation_geometry[0, 2]):
            pg_index_list.append(4)
            pass
        else:
            gmsh.model.occ.addCylinder(0, 0, -domain_radius*1.1, 0, 0, 2.2*domain_radius, formation_geometry[0, 2], tag = 1, angle = np.pi)
            gmsh.model.occ.intersect([(3, 4)], [(3,1)], removeObject= False, removeTool = False, tag=5)
            gmsh.model.occ.cut([(3, index_3d)], [(3,1)], tag=6)
            gmsh.model.occ.remove([(3,4)], recursive=True)
            pg_index_list.append(5)
            pg_index_list.append(6)
    else:
        for i in range(1, np.shape(boundaries_z)[0]):
            layer_height = (boundaries_z[i]-boundaries_z[i-1])*np.cos(dip) # true height of the dipping layer
            layer_middle = (boundaries_z[i]+boundaries_z[i-1])/2 # depth of the middle of the dipping layer
            if np.isnan(formation_geometry[i-1, 2]):
                box = gmsh.model.occ.addBox(-domain_radius*50, -domain_radius*50, layer_middle-layer_height/2, 100*domain_radius, 100*domain_radius, layer_height, tag = index_3d)
                gmsh.model.occ.rotate([(3, index_3d)], 0, 0, layer_middle, 0, 1, 0, dip)
                disc = gmsh.model.occ.intersect([(3, index_3d)], [(3, 4)], removeTool=False)
                pg_index_list.append(index_3d)
                index_3d += 1
            else:
                box = gmsh.model.occ.addBox(-domain_radius*50, -domain_radius*50, layer_middle-layer_height/2, 100*domain_radius, 100*domain_radius, layer_height, tag = index_3d)
                gmsh.model.occ.rotate([(3, index_3d)], 0, 0, layer_middle, 0, 1, 0, dip)
                disc = gmsh.model.occ.intersect([(3, index_3d)], [(3, 4)], removeTool=False)
                gmsh.model.occ.addCylinder(0, 0, -domain_radius*1.1, 0, 0, 2.2*domain_radius, formation_geometry[i-1, 2], tag = 1, angle = np.pi)
                gmsh.model.occ.intersect([(3, index_3d)], [(3,1)], removeObject= False, removeTool = False, tag=index_3d+1)
                gmsh.model.occ.cut([(3, index_3d)], [(3,1)], tag=index_3d+2)
                pg_index_list.append(index_3d+1)
                pg_index_list.append(index_3d+2)
                index_3d += 3
        gmsh.model.occ.remove([(3,4)], recursive=True)

    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.field.add("MathEval", 1) # Horizontal, linear meshsize field
    gmsh.model.mesh.field.setString(1, "F", "(x^2 + y^2)^0.5 + 0.1")

    i = 2
    for electrode_position in tool_geometry[source_terms != 0]:
        gmsh.model.mesh.field.add("MathEval", i) # Distance from the electrode
        gmsh.model.mesh.field.setString(i, "F","(x^2 + y^2 + (z+({}))^2)/2 + 0.01".format(electrode_position))
        i += 1

    if np.shape(tool_geometry[source_terms != 0])[0]==1:
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1,2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)
    else:
        gmsh.model.mesh.field.add("Min", 4)
        gmsh.model.mesh.field.setNumbers(4, "FieldsList", [1,2,3])
        gmsh.model.mesh.field.setAsBackgroundMesh(4)

    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(3)

    dimention, surfaces = list(zip(*gmsh.model.occ.getEntities(2)))
    dirichlet_boundaries = []
    neumann_boundaries = []
    for i in range(len(surfaces)):
        nodes = gmsh.model.mesh.getNodes(dim=2, tag=surfaces[i], includeBoundary = True)
        coordinates = nodes[1].reshape((int(np.shape(nodes[1])[0]/3),3))
        R = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2 + coordinates[:,2]**2)
        if np.allclose(R, domain_radius):
            dirichlet_boundaries.append(surfaces[i])
        else:
            neumann_boundaries.append(surfaces[i])

    i = 1
    for index in pg_index_list:
        volume = gmsh.model.addPhysicalGroup(3, [index], index)
        gmsh.model.setPhysicalName(3, volume, "vol_"+str(i))
        i += 1

    db = gmsh.model.addPhysicalGroup(2, dirichlet_boundaries, 1)
    gmsh.model.setPhysicalName(2, db , "dirichlet_boundary")

    nb = gmsh.model.addPhysicalGroup(2, neumann_boundaries, 2)
    gmsh.model.setPhysicalName(2, nb , "neumann_boundary")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_folder_path + "/fm_"+str(file_number)+".msh")
    gmsh.finalize()
    
    if output_mode == "variable":
        mesh = ReadGmsh(output_folder_path + "/fm_"+str(file_number)+".msh", 3)
        mesh = Mesh(mesh)
        
        return mesh


# Netgen functions

def SelectNetgenDataRange(borehole_geometry, formation_parameters, mud_resistivity, simulation_depth, domain_radius, active_geometry_window=0.99):

    def domain_line_intersection (p1, p2, radius, side):
        x_1, y_1 = p1[1], p1[0]
        x_2, y_2 = p2[1], p2[0]
        d_x = x_2 - x_1
        d_y = y_2 - y_1
        d_r = np.sqrt(d_x**2 + d_y**2)
        D = x_1*y_2 - x_2*y_1
        delta = radius**2*d_r**2-D**2
        for sign in (-1, 1):
            x = (D*d_y+sign*np.sign(d_y)*d_x*np.sqrt(delta))/d_r**2
            y = (-D*d_x+sign*np.abs(d_y)*np.sqrt(delta))/d_r**2
            p = np.array([y, x])
            if side=="top" and y<0 and np.dot(p1-p2, p1-p)> 0 and np.dot(p1-p2, p1-p)<np.dot(p1-p2, p1-p2):
                return np.array([y, x])
            elif side=="bottom" and y>0 and np.dot(p1-p2, p1-p)> 0 and np.dot(p1-p2, p1-p)<np.dot(p1-p2, p1-p2):
                return np.array([y, x])

    ### Borehole geometry
    ## Select data relevant to construct model geometry within simulation domain
    if np.shape(borehole_geometry)[0]==2:
        local_borehole_geometry = borehole_geometry.copy()
    else:
        point_within_domain_mask = (borehole_geometry[:,0]-simulation_depth)**2+borehole_geometry[:,1]**2 < domain_radius**2
        relevant_points_mask = np.convolve(point_within_domain_mask, np.array([True, True, True]), mode="same")
        local_borehole_geometry = borehole_geometry[relevant_points_mask,:]
    local_borehole_geometry[:,0] -= simulation_depth

    ## Adjust top point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1], domain_radius**2):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1] < domain_radius**2:
        omega = np.arccos(local_borehole_geometry[0, 1]/domain_radius)
        local_borehole_geometry = np.vstack((np.array([-np.sin(omega)*domain_radius, local_borehole_geometry[0, 1]]), local_borehole_geometry))
    # else if point is outside the domain move it on the domain boundary
    elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1] > domain_radius**2:
        local_borehole_geometry[0,:] = domain_line_intersection(local_borehole_geometry[0, :], local_borehole_geometry[1, :], domain_radius, side="top")
    
    ## Adjust bottom point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1], domain_radius**2):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1] < domain_radius**2:
        omega = np.arccos(local_borehole_geometry[-1, 1]/domain_radius)
        local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([np.sin(omega)*domain_radius, local_borehole_geometry[0, 1]])))
    # else if point is outside the domain move it on the domain boundary
    elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1] > domain_radius**2:
        local_borehole_geometry[-1,:] = domain_line_intersection(local_borehole_geometry[-1, :], local_borehole_geometry[-2, :], domain_radius, side="bottom")

    ### Formation geometry

    ## Compute active geometry radius 
    # Active geometry prevents occurence of thin layers and small wedges at the very edge of simulation by ignoring occurence of small elements of model at the very edge of simulation domain
    active_geometry_radius = domain_radius * active_geometry_window 

    ## Select data relevant to construct model geometry within simulation domain
    # Raw cut of relevant data\
    point_within = np.any((formation_parameters[:,:2]-simulation_depth)**2 <= active_geometry_radius**2, axis=1)
    line_across = np.all(np.vstack([np.all((formation_parameters[:,:2]-simulation_depth)**2 > active_geometry_radius**2, axis=1), formation_parameters[:,0]<simulation_depth, formation_parameters[:,1]>simulation_depth]), axis=0)
    local_formation_model = formation_parameters[np.any(np.vstack([point_within, line_across]) , axis=0),:]
    local_formation_model[:,:2] -= simulation_depth

    # Check if undisturbed zones in layers with filtration are located within simulation domain
    # Zones are located outside simulation domain if both characteristic points (top and bottom inside corners) 
    # and the line between them are located outside the domain
    filtration_zone_present_mask = ~np.isnan(local_formation_model[:,2])
    top_points_outside_mask = local_formation_model[:,0]**2 + local_formation_model[:,2]**2 >= active_geometry_radius**2
    bottom_points_outside_mask = local_formation_model[:,1]**2 + local_formation_model[:,2]**2 >= active_geometry_radius**2
    line_outside_mask = ~np.all(np.vstack([local_formation_model[:,0] < 0, local_formation_model[:,1] > 0, local_formation_model[:,2] < active_geometry_radius]), axis=0)
    zones_outside_mask = np.all(np.vstack([filtration_zone_present_mask, top_points_outside_mask, bottom_points_outside_mask, line_outside_mask]), axis=0)

    # Remove unrelevant zones from the model
    local_formation_model[zones_outside_mask, 2] = np.nan
    local_formation_model[zones_outside_mask, 4] = local_formation_model[zones_outside_mask, 3]
    local_formation_model[zones_outside_mask, 3] = np.nan

    ## Adjust top point to the domain
    if local_formation_model[0, 0] != local_borehole_geometry[0, 0]:
        local_formation_model[0, 0] = local_borehole_geometry[0, 0]
    
    ## Adjust bottom point to the domain
    if local_formation_model[-1, 1] != local_borehole_geometry[-1, 0]:
        local_formation_model[-1, 1] = local_borehole_geometry[-1, 0]

    ## Local formation grid: [region next to borehole wall, region at left side to filtration zone boundary (if present within simulation domain), region at right side of filtration zone boundary (if present within simulation domain), region next to simulation domain end]
    local_formation_grid = np.empty((np.shape(local_formation_model)[0], 2))
    region_index = 2
    for i in range(np.shape(local_formation_grid)[0]):
        if np.isnan(local_formation_model[i,3]) == True:
            local_formation_grid[i,:] = region_index
            region_index += 1
        else:
            local_formation_grid[i,0] = region_index
            local_formation_grid[i,1] = region_index + 1
            region_index += 2

    local_formation_geometry = np.hstack((local_formation_model[:,:3], local_formation_grid))

    formation_resistivity_distribution = np.ndarray.flatten(local_formation_model[:,3:5])
    formation_resistivity_distribution = formation_resistivity_distribution[~np.isnan(formation_resistivity_distribution)]

    local_conductivity_distribution = CoefficientFunction([1/mud_resistivity] + list(1/formation_resistivity_distribution)) # Conductivity within different parts of model

    return (local_formation_geometry, local_borehole_geometry, local_conductivity_distribution)

def ConstructNetgen2dModel(domain_radius, tool_geometry, formation_geometry, borehole_geometry, source_terms):

    mesh_size_min = 0.001
    mesh_size_max = 100
    mesh_density = "moderate"

    index_0D = 0 # points
    index_1D = 0 # lines

    ### Calculate z coordinates of layer boundaries
    boundaries_z =  np.sort(np.unique(formation_geometry[:,:2]))

    ### Add points [index, r, z]

    ## Add points at borehole axis
    n_electrodes = np.shape(tool_geometry)[0]
    points_at_borehole_axis = np.vstack([np.arange(n_electrodes+2) + index_0D, np.zeros(n_electrodes+2), np.hstack([-domain_radius, tool_geometry, domain_radius])]).T
    points = points_at_borehole_axis
    index_0D += n_electrodes+2

    ## Add points at borehole/rock interface
    # Calculate intersections of boundaries with the borehole wall
    boundaries_z_nr = np.atleast_2d(boundaries_z[~np.isin(boundaries_z, borehole_geometry[:,0])]).T # select layer boundaries from depths other than caliper points
    nr_intersections = np.concatenate([np.atleast_2d(np.interp(boundaries_z_nr[:,0], borehole_geometry[:,0], borehole_geometry[:,1])).T, boundaries_z_nr], axis=1) # intersections of boundaries from depths other than caliper points with the borehole wall

    # Merge intersections with caliper data
    cali_vertices = np.concatenate([np.flip(borehole_geometry, 1), nr_intersections], axis=0) # add intesetions to borehole geometry data
    cali_vertices = cali_vertices[np.lexsort([cali_vertices[:,1]])] # sort data in ascending z-value order
    boundary_points_at_borehole_wall_mask = np.isin(cali_vertices[:,1], boundaries_z) # mark intersections points of layer boundaries with borehole wall for later use

    # Add points at borehole/rock interface
    points_at_borehole_wall = np.hstack([np.atleast_2d(np.arange(np.shape(cali_vertices)[0])).T + index_0D, cali_vertices])
    points = np.vstack([points, points_at_borehole_wall])
    index_0D += np.shape(points_at_borehole_wall)[0]

    ## Add points at filtration/undisturbed zone interfaces
    points_at_filtration_boundaries = np.vstack((np.repeat(formation_geometry[:,2], 2), np.ndarray.flatten(formation_geometry[:,:2]))).T
    filtration_mask = np.unique(points_at_filtration_boundaries, axis=0, return_index=True)[1]

    map_points_at_filtration_to_boundaries = np.ndarray.flatten(np.vstack((np.arange(np.shape(formation_geometry)[0]), np.arange(np.shape(formation_geometry)[0])+1)).T)[np.sort(filtration_mask)] # prepared for lines

    points_at_filtration_boundaries = points_at_filtration_boundaries[np.sort(filtration_mask)]

    map_points_at_filtration_to_boundaries = map_points_at_filtration_to_boundaries[~np.isnan(points_at_filtration_boundaries).any(axis=1)] # prepared for lines
    points_at_filtration_boundaries = points_at_filtration_boundaries[~np.isnan(points_at_filtration_boundaries).any(axis=1)]
    points_at_filtration_boundaries = np.hstack([np.atleast_2d(np.arange(np.shape(points_at_filtration_boundaries)[0])).T + index_0D, points_at_filtration_boundaries])

    points_at_top_boundaries = formation_geometry[:,[2,0]][~np.isnan(formation_geometry[:,[2,0]]).any(axis=1)] # prepared for lines
    points_at_bottom_boundaries = formation_geometry[:,[2,1]][~np.isnan(formation_geometry[:,[2,1]]).any(axis=1)] # prepared for lines
    indices_of_points_at_top_boundaries = np.argwhere((points_at_top_boundaries[:, None] == points_at_filtration_boundaries[:,1:]).all(-1)==True)[:,1] + index_0D # prepared for lines
    indices_of_points_at_bottom_boundaries = np.argwhere((points_at_bottom_boundaries[:, None] == points_at_filtration_boundaries[:,1:]).all(-1)==True)[:,1] + index_0D # prepared for lines

    # Move points from outside to the domain boundary
    points_outside_domain_mask = points_at_filtration_boundaries[:,1]**2 + points_at_filtration_boundaries[:,2]**2 >= domain_radius**2
    omega = np.arccos(points_at_filtration_boundaries[points_outside_domain_mask, 1]/domain_radius)
    points_at_filtration_boundaries[points_outside_domain_mask, 2] = np.sign(points_at_filtration_boundaries[points_outside_domain_mask, 2])*np.sin(omega)*domain_radius

    # Add points at filtration/undisturbed zone interfaces
    points = np.vstack([points, points_at_filtration_boundaries])
    index_0D += np.shape(points_at_filtration_boundaries)[0]

    ## Add points at ends of layers boundaries
    omega = np.arcsin(boundaries_z[1:-1]/domain_radius)
    points_at_end = np.vstack([np.cos(omega)*domain_radius, boundaries_z[1:-1]]).T
    unique_points_at_end_mask = ~(points_at_end[:, None] == points_at_filtration_boundaries[:,1:]).all(-1).any(-1) # check if same points are already present in
    points_at_end = np.hstack([np.atleast_2d(np.arange(np.shape(points_at_end)[0])).T + index_0D, points_at_end[unique_points_at_end_mask,:]])

    points = np.vstack([points, points_at_end])
    index_0D += np.shape(points_at_end)[0]

    ## Add additional points at domain boundary to aproximate circular shape
    # Select existing points lying at the domain_boundary and convert them to polar coordinates
    existing_points_at_domain_boundary = points[np.isclose(points[:, 1]**2 + points[:, 2]**2, domain_radius**2),:]
    existing_points_at_domain_boundary = existing_points_at_domain_boundary[np.argsort(existing_points_at_domain_boundary[:,2]),:]

    existing_points_at_domain_boundary[0,2] = np.arctan(-np.inf)
    existing_points_at_domain_boundary[-1,2] = np.arctan(np.inf)
    existing_points_at_domain_boundary[1:-1,2] = np.arctan(existing_points_at_domain_boundary[1:-1,2]/existing_points_at_domain_boundary[1:-1,1])
    existing_points_at_domain_boundary[:,1] = domain_radius

    # Add points to follow circular shape of simulation domain
    angles_betweeen_existing_points = existing_points_at_domain_boundary[1:,2]-existing_points_at_domain_boundary[:-1,2]
    points_to_add = np.floor(angles_betweeen_existing_points/(9*np.pi/180)).astype(int) # number of points that will be added between existing points

    starting_index = index_0D
    points_at_domain_boundary = existing_points_at_domain_boundary[0,:]
    for i in range(np.shape(points_to_add)[0]):
        if points_to_add[i] > 0:
            index = np.array([0, points_to_add[i]+1])
            angle = np.array([existing_points_at_domain_boundary[i,2], existing_points_at_domain_boundary[i+1,2]])
            interpolated_angles = np.interp(np.arange(points_to_add[i])+1, index, angle)
            additional_points = np.vstack([np.arange(points_to_add[i]) + index_0D, np.full(points_to_add[i], domain_radius), interpolated_angles]).T
            points_at_domain_boundary = np.vstack([points_at_domain_boundary, additional_points, existing_points_at_domain_boundary[i+1,:]])
            index_0D += points_to_add[i]
        else:
            points_at_domain_boundary = np.vstack([points_at_domain_boundary, existing_points_at_domain_boundary[i+1,:]])
    
    points_at_domain_boundary[:,1], points_at_domain_boundary[:,2] = domain_radius*np.cos(points_at_domain_boundary[:,2]), domain_radius*np.sin(points_at_domain_boundary[:,2])
    new_points_at_domain_boundary = points_at_domain_boundary[points_at_domain_boundary[:,0]>=starting_index,:]
    points = np.vstack([points, new_points_at_domain_boundary])

    ### Add lines [index, start-point, end-point, boundary-condition, domain on the left side, domain on the right side]
    ## Add vertical lines
    # Add lines at borehole axis
    number_of_lines = np.shape(points_at_borehole_axis)[0]-1
    lines_at_borehole_axis = np.vstack([np.arange(number_of_lines), np.arange(number_of_lines), np.arange(number_of_lines)+1, np.ones(number_of_lines), np.zeros(number_of_lines), np.ones(number_of_lines)]).T
    lines = lines_at_borehole_axis
    index_1D += number_of_lines

    # Add lines at borehole/rock interface
    boundary_indices = np.argwhere(boundary_points_at_borehole_wall_mask).flatten()
    layers_extends = boundary_indices[1:] - boundary_indices[:-1]
    number_of_lines = np.shape(points_at_borehole_wall)[0]-1
    lines_at_borehole_wall = np.vstack([np.arange(number_of_lines) + index_1D, points_at_borehole_wall[:-1,0], points_at_borehole_wall[1:,0], np.ones(number_of_lines), np.ones(number_of_lines), np.repeat(formation_geometry[:,3], layers_extends)]).T
    lines = np.vstack([lines, lines_at_borehole_wall])
    index_1D += number_of_lines

    # Add lines at filtration/undisturbed zones interface
    number_of_lines = np.shape(points_at_top_boundaries)[0]
    lines_at_filtration_boundaries = np.vstack([np.arange(number_of_lines) + index_1D, indices_of_points_at_top_boundaries, indices_of_points_at_bottom_boundaries, np.ones(number_of_lines), formation_geometry[formation_geometry[:,3]!=formation_geometry[:,4],3:].T]).T
    lines = np.vstack([lines, lines_at_filtration_boundaries])
    index_1D += number_of_lines

    ## Add horizontal lines
    # Add lines at layers boundaries
    number_of_lines_at_boundary = np.empty_like(boundaries_z)
    for i in range(np.shape(number_of_lines_at_boundary)[0]):
        number_of_lines_at_boundary[i] = np.sum(np.all([points[:,2]==boundaries_z[i], points[:,1] > 0], axis=0)) - 1
    number_of_lines = int(np.sum(number_of_lines_at_boundary))

    areas_above_boundary = np.vstack((formation_geometry[:-1,3], np.full_like(formation_geometry[:-1,3], np.nan).T, formation_geometry[:-1,4])).T
    areas_below_boundary = np.vstack((formation_geometry[1:,3], np.full_like(formation_geometry[1:,3], np.nan).T, formation_geometry[1:,4])).T

    for i in range(np.shape(formation_geometry)[0]-1):
        if formation_geometry[i, 2] > formation_geometry[i+1, 2]:
            areas_above_boundary[i,1] = areas_above_boundary[i,0]
            areas_below_boundary[i,1] = areas_below_boundary[i,2]
        elif formation_geometry[i, 2] < formation_geometry[i+1, 2]:
            areas_above_boundary[i,1] = areas_above_boundary[i,2]
            areas_below_boundary[i,1] = areas_below_boundary[i,0]

    lines_at_boundaries = np.full((number_of_lines, 6), 1)
    lines_at_boundaries[:,0] = np.arange(number_of_lines) + index_1D

    j = 0
    for i in range(np.shape(boundaries_z[1:-1])[0]):
        points_at_ith_boundary = points[np.all([points[:,2]==boundaries_z[1:-1][i], points[:,1] > 0], axis=0), :]
        points_at_ith_boundary = points_at_ith_boundary[np.argsort(points_at_ith_boundary[:,1]),:]
        if np.shape(points_at_ith_boundary)[0] == 2:
            lines_at_boundaries[j,[1,2,4,5]] = [points_at_ith_boundary[0,0], points_at_ith_boundary[1,0], areas_below_boundary[i,2], areas_above_boundary[i,0]]
            j += 1
        elif np.shape(points_at_ith_boundary)[0] == 3:
            lines_at_boundaries[j:j+2,[1,2,4,5]] = np.vstack((points_at_ith_boundary[:2,0], points_at_ith_boundary[1:3,0], areas_below_boundary[i,[0,2]], areas_above_boundary[i,[0,2]])).T
            j += 2
        elif np.shape(points_at_ith_boundary)[0] == 4:
            lines_at_boundaries[j:j+3,[1,2,4,5]] = np.vstack((points_at_ith_boundary[:3,0], points_at_ith_boundary[1:4,0], areas_below_boundary[i,:], areas_above_boundary[i,:])).T
            j += 3

    lines = np.vstack([lines, lines_at_boundaries])
    index_1D += number_of_lines

    ## Add lines at domain boundary
    number_of_lines = np.shape(points_at_domain_boundary)[0]-1

    areas_next_to_domain_boundary = [1]
    for i in range(np.shape(formation_geometry)[0]):
        if formation_geometry[i,3] == formation_geometry[i,4]:
            areas_next_to_domain_boundary.append(formation_geometry[i,3])
        else:
            top_point_mask = formation_geometry[i,0]**2 + formation_geometry[i,2]**2 >= domain_radius**2
            bottom_point_mask = formation_geometry[i,1]**2 + formation_geometry[i,2]**2 >= domain_radius**2
            if top_point_mask==True and bottom_point_mask==True:
                areas_next_to_domain_boundary += [formation_geometry[i,3], formation_geometry[i,4], formation_geometry[i,3]]
            elif top_point_mask==True:
                areas_next_to_domain_boundary += [formation_geometry[i,3], formation_geometry[i,4]]
            elif bottom_point_mask==True:
                areas_next_to_domain_boundary += [formation_geometry[i,4], formation_geometry[i,3]]
            else:
                areas_next_to_domain_boundary.append(formation_geometry[i,4])
    areas_next_to_domain_boundary.append(1)

    areas_next_to_domain_boundary = np.repeat(areas_next_to_domain_boundary, points_to_add+1)
    lines_at_domain_boundary = np.vstack((np.arange(number_of_lines) + index_1D, points_at_domain_boundary[:-1,0], points_at_domain_boundary[1:,0], np.ones(np.shape(points_at_domain_boundary[1:,0]))+1, np.array(areas_next_to_domain_boundary), np.zeros(np.shape(points_at_domain_boundary[1:,0])))).T
    lines = np.vstack([lines, lines_at_domain_boundary])

    ### Convert arrays to lists
    points = list(map(list, points[:,1:]))
    lines = list(map(list, lines[:,1:].astype(int)))

    ### Add mesh size information
    for k in np.ndarray.flatten(np.argwhere(source_terms != 0))+1: # Set minimum mesh size around source points
        points[k].append(mesh_size_min)

    ### Create the model geometry
    model_geometry = SplineGeometry()

    pnums = [model_geometry.AppendPoint(*p) for p in points]

    for p1,p2,bc,left,right in lines:
        model_geometry.Append( ["line", pnums[p1], pnums[p2]], bc=bc, leftdomain=left, rightdomain=right)

    mesh = Mesh(model_geometry.GenerateMesh(eval("meshsize." + mesh_density), maxh=mesh_size_max))

    return mesh


# Ngsolve funtions

def AddPointSource(f, position, fac, model_dimensionality):
        spc = f.space
        if model_dimensionality==2:
            mp = spc.mesh(0,position)
        elif model_dimensionality==3:
            mp = spc.mesh(0,0,position)
        ei = ElementId(VOL, mp.nr)
        fel = spc.GetFE(ei)
        dnums = spc.GetDofNrs(ei)
        shape = fel.CalcShape(*mp.pnt)
        for d,s in zip(dnums, shape):
            f.vec[d] += fac*s

def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense):

    model_dimensionality = mesh.dim

    fes = H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = BilinearForm(fes, symmetric=False, condense=condense)

    if model_dimensionality==2:
        a += 2*np.pi*grad(u)*grad(v)*x*sigma*dx
    elif model_dimensionality==3:
        a += grad(u)*grad(v)*sigma*dx

    #start_time = datetime.datetime.now()  
    f = LinearForm(fes)
    f.Assemble()

    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource (f, tool_geometry[l], source_terms[l], model_dimensionality)

    c = Preconditioner(a, preconditioner)
    a.Assemble()
    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, c.mat, maxsteps=1000)
    if condense==True:
        f.vec.data += a.harmonic_extension_trans * f.vec
    gfu.vec.data = inv * f.vec

    if condense==True:
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec

    return(fes, gfu)



### Work in progress versions

def ComputeSyntheticLogs_v2(tools_parameters, model_parameters, measurement_depths, force_single_electrode_configuration=False, domain_radius=50, processes=4, mesh_generator="gmsh", preconditioner="multigrid", condense=True):

    def prepare_simulation_depths_and_tasks(tools_parameters, measurement_depths):

        current_electrode_depths = {}
        for tool in tools_parameters.keys():
            current_electrode_depths[tool] = measurement_depths + tools_parameters[tool][1,3] 
        simulation_depths = np.unique(np.hstack(list(current_electrode_depths.values())))

        tasks = []
        for simulation_depth_index in range(len(simulation_depths)):

            simulation_depth = simulation_depths[simulation_depth_index]
            modelling_tasks = []
            potential_electodes_depths = []
            for tool_index in range(len(list(tools_parameters.keys()))):
                tool = list(tools_parameters.keys())[tool_index]
                if np.any(np.isclose(current_electrode_depths[tool], simulation_depth)):
                    measurement_depth_index = np.argwhere(np.isclose(measurement_depths + tools_parameters[tool][1,3], simulation_depth))[0][0]
                    modelling_tasks.append([measurement_depth_index, tool_index])
                    tool_electrodes = tools_parameters[tool][:,:3].copy() 
                    tool_electrodes[0,:] -= tools_parameters[tool][1,3] 
                    potential_electodes_depths += list(tool_electrodes[0, tool_electrodes[1,:]==0])

            unique_potential_electodes_depths = np.unique(potential_electodes_depths)
            combined_tools = np.zeros((2, len(unique_potential_electodes_depths)+1))
            combined_tools[1,0] = 1
            combined_tools[0,1:] = unique_potential_electodes_depths
            combined_tools = combined_tools[:,combined_tools[0,:].argsort()]
            
            tasks.append([simulation_depth_index, combined_tools, modelling_tasks])

        return simulation_depths, tasks
    

    ### Start the clock
    start_time = datetime.datetime.now()

    ### Create temporary directory for mesh files
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    ### Unpack parameters and prepare data
    ## Tools
    # Convert tools to single-electrode configurations
    if force_single_electrode_configuration==False:
        pass
    elif force_single_electrode_configuration==True:
        for tool in tools_parameters.keys():
            if "A" in tool and "B" in tool:
                alternative_tool = tool.translate(str.maketrans("ABMN", "MNAB"))
                alternative_tool_data = [str2float(item) for item in [''.join(group) for _, group in itertools.groupby(alternative_tool, str.isalpha)]]
                alternative_electrodes = tuple(x for x in alternative_tool_data if isinstance(x, str)) # symbols of eletrodes
                alternative_distances = [x for x in alternative_tool_data if isinstance(x, float)] # distances between electrodes
                tools_parameters[tool] = SetToolParameters(alternative_tool, alternative_electrodes, alternative_distances)
    else:
        raise ValueError("The value of parameter force_single_electrode_configuration can be set only to True or False")
    
    # Check if all tools are in single-electode configuration
    single_electrode_computation_mode = True
    for tool in tools_parameters.keys():
        if np.isclose(np.sum(tools_parameters[tool][1,:3]), 0)==True:
            single_electrode_computation_mode = False
            
    ## Model
    formation_parameters = model_parameters[0]
    borehole_parameters = model_parameters[1]
    dip = model_parameters[2]*np.pi/180 #converted to radians
   
    ## Simulation domain
    domain_radius_alert = False
    for tool in tools_parameters.keys():
        tools_parameters[tool][0,:3] -= tools_parameters[tool][1,3] # center simulation around current electrodes
        if np.max(np.abs(tools_parameters[tool][0,:3])) > domain_radius:
            raise ValueError("Some electrodes are locate outside the simulation domain. Domain size have to be increased")
        elif np.max(np.abs(tools_parameters[tool][0,:3])) > 0.75*domain_radius:
            domain_radius_alert = True
    if domain_radius_alert == True:
        print("Some electrodes are located close to the boundary of the simulation domain. This may cause problems during simulation. Consider increase of the domain size")
    
    if dip!=0 and mesh_generator!="gmsh":
        raise ValueError("The only mesh generator supported in 3D models is gmsh")
    
    # Create dense borehole geometry for the purpose of 3D mesh generation (necessary to avoid errors during meshing procedure)
    if dip!=0:
        borehole_parameters = AddPointsToBorehole(borehole_parameters, 0.15)

    borehole_geometry = np.ascontiguousarray(borehole_parameters[:,:2])

    ### Parallel FEM computation
    
    tool_list = list(range(len(tools_parameters.keys())))
    measurement_depths_list = list(range(len(measurement_depths)))

    ## Compute simulation depths and prepare tasks
    if single_electrode_computation_mode==False:
        simulation_depths = dict()
        for tool in tools_parameters.keys():
            simulation_depths[tool] = measurement_depths + tools_parameters[tool][1,3]
        task_list = []   
        for task in itertools.product(measurement_depths_list, tool_list):
            task_list.append(list(task))
        n_tasks = np.shape(measurement_depths)[0]
        
    elif single_electrode_computation_mode==True:
        simulation_depths, task_list = prepare_simulation_depths_and_tasks(tools_parameters, measurement_depths)
        n_tasks = len(task_list)

    # Mud resistivities at simulation depths
    if single_electrode_computation_mode==False:
        mud_resistivities = np.interp(measurement_depths, borehole_parameters[:,0], borehole_parameters[:,2])   
    elif single_electrode_computation_mode==True:
        mud_resistivities = np.interp(simulation_depths, borehole_parameters[:,0], borehole_parameters[:,2])
    
    ## Specify number of workers
    if type(processes) != int:
        raise ValueError("The number of processes have to be intager")
    if processes < 2:
        raise ValueError("Minimal number of processes is 2")
    
    n_workers = processes - 1 # one process is reserved for the main
    
    ## Spawn workers
    
    if single_electrode_computation_mode==False:
        comm = MPI.COMM_WORLD.Spawn(
            sys.executable,
            args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker.py')], 
            maxprocs=n_workers)        
    elif single_electrode_computation_mode==True:
        comm = MPI.COMM_WORLD.Spawn(
            sys.executable,
            args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker_se.py')], 
            maxprocs=n_workers)

    ## Broadcast data to workers
    # Broadcast information about shapes of broadcasted arrays
    arrays_shape = [np.shape(formation_parameters), np.shape(borehole_geometry), np.shape(mud_resistivities)]
    comm.bcast(arrays_shape, root=MPI.ROOT)

    # Broadcast data
    comm.Bcast([formation_parameters, MPI.FLOAT], root=MPI.ROOT)
    comm.Bcast([borehole_geometry, MPI.FLOAT], root=MPI.ROOT)
    comm.Bcast([mud_resistivities, MPI.FLOAT], root=MPI.ROOT)
    comm.bcast(dip, root=MPI.ROOT)
    comm.bcast(tools_parameters, root=MPI.ROOT)
    comm.bcast(simulation_depths, root=MPI.ROOT)
    comm.bcast(domain_radius, root=MPI.ROOT)
    comm.bcast(mesh_generator, root=MPI.ROOT)
    comm.bcast(preconditioner, root=MPI.ROOT)
    comm.bcast(condense, root=MPI.ROOT)

    ## Wait for all workers to receive data
    comm.barrier()

    ## Convert tasks to mesages 
    msg_list = task_list + ([StopIteration] * n_workers) # Append stop sentinel for each worker

    ## Dispatch tasks to workers
    status = MPI.Status()
    for i in msg_list:
        if i != StopIteration:
            # Pass data to worker
            comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(obj=i, dest=status.Get_source())
            # Progress bar
            percent = ((i[0] + 1) * 100) // (n_tasks)
            sys.stdout.write('\rProgress: [%-50s] %3i%% ' % ('=' * (percent // 2), percent))
            sys.stdout.flush()
        else:
            # Tell worker to shutdown
            comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(obj=i, dest=status.Get_source())

    # Gather results from workers
    list_of_results = [item for sublist in comm.gather(None, root=MPI.ROOT) for item in sublist]
   
    ## Format and sort results
    results = np.empty([len(measurement_depths_list), len(tool_list)])
    for result in list_of_results:
        results[result[0], result[1]] = result[2]

    logs = dict()
    for i in range(len(tools_parameters.keys())):
        logs[list(tools_parameters.keys())[i]] = np.vstack([measurement_depths, results[:,i]]).T

    ### Shutdown MPI
    comm.Disconnect()

    ### Remove tmp folder and mesh files
    shutil.rmtree("./tmp")

    ## Report time of computation
    print('\nProcessed in: ', datetime.datetime.now() - start_time)

    return logs