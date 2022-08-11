from mpi4py import MPI

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

    def str2float(item):
        """
        This function convert item from strings to floats (if possible).
        """
        try:
            return float(item)
        except ValueError:
            return item

    ### Check if data format is correct
    if type(tools)!=list or all(isinstance(s, str) for s in tools)==False:
        raise ValueError("Tools names have to be provided in the form of list of strings")

    ### Set tools parameters
    tools_parameters = dict()
    for tool in tools:
        ## Extract information about tool geometry from the tool name
        tool_data = [str2float(item) for item in [''.join(group) for _, group in itertools.groupby(tool, str.isalpha)]]
        electrodes = tuple(x for x in tool_data if isinstance(x, str)) # symbols of eletrodes
        distances = [x for x in tool_data if isinstance(x, float)] # distances between electrodes

        ## Check if tool configuration is correct and set electrodes position in relation to measurment point position
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
        tool_parameters = np.hstack([np.vstack([tool_geometry, source_terms]), np.array([[geometric_factor],[depth_shift]])])

        ## Add tool to dictionary
        tools_parameters[tool] = tool_parameters

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

def ComputeSyntheticLogs(tools_parameters, model_parameters, measurement_depths, domain_radius=50, processes=4, preconditioner="multigrid", condense=True):
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

    domain_radius: float, optional
        A radius of simulation domain in meters.
        By default set to 100.

    processes: int, optional
        Specify a number of processes. Minimal value that can be set is 2, the maximal value should not exceed the number of processes available on computing machine.
        By default set to 4.
    
    preconditioner: 
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

    ### Start the clock
    start_time = datetime.datetime.now()

    ### Create temporary directory for mesh files
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    ### Unpack parameters and prepare data
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
        logs[list(tools_parameters.keys())[i]] = results[:,i]

    ### Shutdown MPI
    comm.Disconnect()

    ### Remove tmp folder with and mesh files
    shutil.rmtree("./tmp")

    ## Report time of computation
    print('\nProcessed in: ', datetime.datetime.now() - start_time)

    return logs

def SaveResults(model_parameters, measurement_depths, measurement_results, output_folder, plot_layout="auto", depth_lim="auto", rad_lim="auto", res_lim="auto", aspect_ratio = "auto", at_nan="break", interpolation=1):
    """
    This function saves results of modelling to txt file and produces raw visualization of the model and computed syntetic logs that is saved in PNG format.

    Parameters
    -------
    model_parameters: list
        A list of numpy arrays created by the SetModelParameters() funcion.
    
    measurement_depths: array
        A 1D numpy array of depths of simulated measurements.
        Values have to be given in ascending order and corespond to depths of the model.

    measurement_results: dict
        A dictionary of 1D numpy arrays created by the ComputeSynteticLogs function.

    output_folder: str
        A path to the folder where results will be saved.

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
    interpolation: float, optional
        Allows to smooth logs on vizualization.
        Have no inpact on output data.
        By default set to 1 (no interpolation).

    """
    
    ### Create output folder
    output_subfolder = os.path.join(output_folder, "Results_{}/".format(str(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))))

    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    ### Save data to txt file
    results = np.vstack([measurement_depths]+list(measurement_results.values())).T
    names = ['DEPTH'] + list(measurement_results.keys())
    units = ['M'] + ['OHMM']*len(list(measurement_results.keys()))
    header = '\t'.join([name for name in names]) + '\n' + '\t'.join([unit for unit in units])
    np.savetxt(output_subfolder + 'Results.txt', results,  fmt='%.4f', delimiter='\t', header=header, comments='')

    ### Formation model visualization
    
    ## Unpack parameters
    formation_parameters = model_parameters[0]
    borehole_parameters = model_parameters[1]
    dip = model_parameters[2]

    ## Smooth logs
    if interpolation > 1:
        measurement_depths_interp = np.linspace(np.min(measurement_depths), np.max(measurement_depths), int(np.shape(measurement_depths)[0]*interpolation))
        logs = list(measurement_results.keys())
        for log in logs:
            interpolation = spi.interp1d(measurement_depths, measurement_results[log], kind='cubic')
            measurement_results[log] = interpolation(measurement_depths_interp)
        measurement_depths = measurement_depths_interp

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
    left_boundary = borehole_parameters[:,[1,0]]*[-1, 1]
    right_boundary = borehole_parameters[:,[1,0]]
    vetrices = np.vstack([left_boundary, np.flip(right_boundary, axis=0)])
    polygon = Polygon(vetrices, closed=True)
    patches.append(polygon)
    resistivities = np.hstack([resistivities, np.mean(borehole_parameters[:,2])])

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
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.titlepad': 14,
        "xtick.major.size": 10, "xtick.minor.size": 5, "ytick.major.size": 10, "ytick.minor.size": 5})

    fig, ax = plt.subplots(1, 1+tracks, sharey=True, figsize=[fig_width, fig_hight])
    
    ax[0].add_collection(collection)
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
                axis.plot(measurement_results[logs[i]], measurement_depths, color=colors[i%len(colors)])
            elif at_nan=="continue":
                nan_flag = ~np.isnan(measurement_results[logs[i]])
                axis.plot(measurement_results[logs[i]][nan_flag], measurement_depths[nan_flag], color=colors[i%len(colors)])
            else:
                raise ValueError('at_nan paramater has to be set to "break" or "continue"')
            axis.set_xlabel(logs[i]+"\n[ohmm]", color=colors[i%len(colors)], labelpad=-8)
            axis.spines['top'].set_color(colors[i%len(colors)])
            axis.spines['top'].set_position(('outward', i*55+10))
            axis.set_xticks(res_lim)
            axis.tick_params(axis='x', color=colors[i%len(colors)])
            axis.set_xlim(res_lim)
        axis = ax[track].twiny().get_xaxis().set_visible(False)
    for track in range(1, tracks+1):
        ax[track].grid(True)
        ax[track].xaxis.set_label_position('top')
        ax[track].xaxis.set_ticks_position('top')
        ax[track].margins(x=0, y=0)
        ax[track].autoscale_view()
            
    colorbar = fig.colorbar(collection, ax=ax, location='bottom', orientation='horizontal', pad=0.05, label="Resistivity [ohmm]")
    colorbar.ax.minorticks_on()

    ## Save plot to png file
    plt.savefig(output_subfolder + 'Results_plot.png', bbox_inches='tight')
