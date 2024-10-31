# -*- coding: utf-8 -*-

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

##  Main classes and functions

class Model():

    # Allowed units conversion table
    conversion_table = {'M':1.0, 'DM':0.1, 'CM':0.01, 'MM':0.001, "IN":0.0254, 'FT':0.3048}
        
    def __init__(self, tools, force_single_electrode_configuration=True):
        """
        This function initializes modelling procedure for a specific set of tools.
        
        Attributes
        ----------        
        tools: list
            A list of tools. Names are strings and have to consist of symbols of 3 different electrods
            (A and/or B for current electrodes and M and/or N for measuring electrodes) listed from
            the top one to the bottom one and 2 numbers that specifies distances in meters between
            consecutive electrodes.
            Example: tools = ["N2.5M0.25A", "B5.7A0.4M"]
            
        force_single_electrode_configuration: bool, optional
            Specifies if two-electrode tool configurations will be changed to equivalent single-electrode tool configurations.
            Enables faster computations. Can be set to True or False.
            By default set to True.
        """
        # Set tools parameters
        self.tools, self.sec = self.set_tools_parameters(tools, force_single_electrode_configuration=force_single_electrode_configuration)

        # Initialize model atributes
        self.formation_model = None
        self.borehole_model = None
        self.dip_deg = None
        self.dip_rad = None
        
        # Initialize modelling procedure atributes
        self.workers = None
        self.cpu_workers = None
        self.gpu_workers = None
        self.comm = None
        
        # Initialize results atributes
        #self.meshes = None
        self.logs = None

    
    # Complete modelling procedure
    @classmethod
    def compute_synthetic_logs(
        cls,
        tools,
        measurement_depths,
        formation_model,
        borehole_model,
        force_single_electrode_configuration=True,
        formation_units=["M", "M", "M"],
        borehole_geometry_type='diameter',
        borehole_units=["M", "M"],
        dip=0,
        cpu_workers=4,
        gpu_workers=0, 
        domain_radius=50,
        batch_size=5,
        mesh_generator="auto",
        preconditioner="multigrid",
        condense=True):
        """
        This function performs complete moddeling procedure.
        
        Parameters
        ----------   
        tools: list
            A list of tools. Names are strings and have to consist of symbols of 3 different electrods
            (A and/or B for current electrodes and M and/or N for measuring electrodes) listed from
            the top one to the bottom one and 2 numbers that specifies distances in meters between
            consecutive electrodes.
            Example: tools = ["N2.5M0.25A", "B5.7A0.4M"]
            
        measurement_depths: array
            A 1D numpy array of depths of simulated measurements.
            Values have to be given in ascending order and corespond to depths of the model.
            
        formation_model: str or array
            A string that specifies path to file that stores parameters of the formation model
            or an array of formation model parameters (in this case dimentions have to be in meters).
            
        borehole_model: str or array
            A string that specifies path to file that stores parameters of the  borehole model
            or an array of borehole model parameters (in this case dimentions have to be in meters).
            
        force_single_electrode_configuration: bool, optional
            Specifies if two-electrode tool configurations will be changed to equivalent single-electrode tool configurations.
            Enables faster computations. Can be set to True or False.
            By default set to True.            
            
        formation_units: list, optional
            Three element list that specifies units of formation model diameters (tops and bottoms of boundaries and diameters of filtration zones).
            Example: ["M", "MM", "FT"]
            
        borehole_geometry_type: str, optional
            A string that specifies type of borehole geometry. Available options: "diameter" and "radius".
            By default set to "diameter". 
            
        borehole_units: list, optional
            Two element list that specifies units of borehole model diameters (depths and diameters or radii).
            Example: ["M", "MM"]        
            
        dip: float, optional
            A value between 0 and 90. Describes dip of the layers in relation to the borehole axis.
            By default set to 0.
            
        cpu_workers: int, optional
            Specify a number of processes that will solve the equations on cpu.
            Minimal value that can be set is 1. By default set to 4.
            
        gpu_workers: int, optional
            Specify a number of processes that will solve the equations on gpu.
            Minimal value that can be set is 0. By default set to 0.       
            
        domain_radius: float, optional
            A radius of simulation domain in meters.
            By default set to 50.

        batch_size: int, optional
            Specify a number of adjacent measurement points that are joined into a single mesh generation and simulation procedure to speed up the process.
            By default set to 5.    

        mesh_generator: string, optional
            Specify utiliezed mesh generator. Can be set to "gmsh" or "netgen" for 2D models and to "gmsh" for 3D models.
            By default set to "auto" and will chose "netgen" for 2D models and "gmsh" for 3D models.

        preconditioner: string, optional
            Specify a type of utilized preconditioner. Available options: "local" and "multigrid".
            By default set to "multigrid".

        condense: bool, optional
            Specify if static condensation will be utilized to eliminate unknowns that are internal to elements from the global linear system.
            By default set to True.            
            
        Returns
        -------
        model: Model
            An instance of the Model class containing all data related to performed modelling procedure.
        """
        model = cls(tools, force_single_electrode_configuration=force_single_electrode_configuration)

        model.set_model_parameters(formation_model, borehole_model, borehole_geometry_type=borehole_geometry_type, dip=dip)

        model.initialize_workers(cpu_workers=cpu_workers, gpu_workers=gpu_workers)

        model.simulate_logs(measurement_depths, domain_radius=domain_radius, batch_size=batch_size, mesh_generator=mesh_generator, preconditioner=preconditioner, condense=condense)

        model.shutdown_workers()
        
        return model

    
    # Functions associated with setting logging tools parameters  
    def set_tools_parameters(self, tools, force_single_electrode_configuration=True):
        """
        This function sets logging tools parameters based on their names.

        Parameters
        ----------   
        tools: list
            A list of tools. Names are strings and have to consist of symbols of 3 different electrods
            (A and/or B for current electrodes and M and/or N for measuring electrodes) listed from
            the top one to the bottom one and 2 numbers that specifies distances in meters between
            consecutive electrodes.
            Example: tools = ["N2.5M0.25A", "B5.7A0.4M"]
        force_single_electrode_configuration: bool, optional
            Specifies if two-electrode tool configurations will be changed to equivalent single-electrode tool configurations.
            Enables faster computations. Can be set to True or False.
            By default set to True.
        
        Returns
        -------
        tools_parameters: dict
            A dictionary of numpy arrays that specify parameters of logging tools.
        """
        ### Check if data format is correct
        if type(tools)!=list or all(isinstance(s, str) for s in tools)==False:
            raise ValueError("Tools names have to be provided in the form of list of strings")

        if type(force_single_electrode_configuration)!=bool:
            raise ValueError("The value of parameter force_single_electrode_configuration can be set only to True or False")   
            
        ### Set tools parameters
        tools_parameters = dict()
        for tool in tools:
            # Extract information about tool geometry from the tool name
            if force_single_electrode_configuration==True and "A" in tool and "B" in tool:
                # Convert tool to single-electrode configuration
                alternative_tool = tool.translate(str.maketrans("ABMN", "MNAB"))
                tool_data = [self._str2float(item) for item in [''.join(group) for _, group in itertools.groupby(alternative_tool, str.isalpha)]]
            else:
                tool_data = [self._str2float(item) for item in [''.join(group) for _, group in itertools.groupby(tool, str.isalpha)]]
            electrodes = tuple(x for x in tool_data if isinstance(x, str)) # symbols of eletrodes
            distances = [x for x in tool_data if isinstance(x, float)] # distances between electrodes
            ## Create tool parameters and add tool to dictionary
            tools_parameters[tool] = self._set_tool_parameters(tool, electrodes, distances)
        
        # Check if all tools are in single electrode configuration
        single_electrode_computation_mode = True
        for tool in tools_parameters.keys():
            if np.isclose(np.sum(tools_parameters[tool][1,:3]), 0)==True:
                single_electrode_computation_mode = False
                
        return tools_parameters, single_electrode_computation_mode
    
    
    def _set_tool_parameters(self, tool, electrodes, distances):
        """
        This function is constructing single tool parameters.
        
        Parameters
        ----------
        tool: str
        
        electrodes: tuple
            Tuple that contains symbols of electrodes.
            
        distances: list
            List that specifies distances between electrodes.

        Returns
        -------
        tool_parameters: numpy.ndarray
            An array containing all tool parameters.
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
        
        # Center tool geometry around current electrodes
        tool_parameters[0,:3] -= tool_parameters[1,3]

        return tool_parameters 

                
    def _str2float(self, item):
        """
        This function converts items from strings to floats (if possible).
        
        Parameters
        ----------
        item: str
            Item to convert
        
        Returns:
        item: float or str
            Item returned as float if conversion was possible or as str if conversion is not possible.
        """
        try:
            return float(item)
        except ValueError:
            return item
        
        
    # Functions associated with setting model parameters
    def set_model_parameters(self, formation_model, borehole_model, borehole_geometry_type='diameter', dip=0):
        """
        This function sets all class atributes associated with borehole and formation model.
        
        Parameters
        ----------
        formation_model: str or numpy.ndarray
            A string that specifies path to file that stores parameters of the formation model
            or an array of formation model parameters (in this case dimentions have to be in meters).
        borehole_model: str or numpy.ndarray
            A string that specifies path to file that stores parameters of the  borehole model
            or an array of borehole model parameters (in this case dimentions have to be in meters).
        borehole_geometry_type: str, optional
            A string that specifies type of borehole geometry. Available options: "diameter" and "radius".
            By default set to "diameter".            
        dip: float, optional
            A value between 0 and 90. Describes dip of the layers in relation to the borehole axis.
            By default set to 0.
        """
        # Load model data
        if isinstance(formation_model, str):
            self.formation_model = self.load_formation_parameters(formation_model)
        elif isinstance(formation_model, np.ndarray):
            self.formation_model = self.set_formation_parameters(formation_model)
        
        if isinstance(borehole_model, str):
            self.borehole_model = self.load_borehole_parameters(borehole_model, borehole_geometry_type)
        elif isinstance(borehole_model, np.ndarray):
            self.borehole_model = self.set_borehole_parameters(borehole_model, borehole_geometry_type)
            
        self.dip_deg, self.dip_rad = self.set_dip(dip)

        # Check model geometry
        self._check_model_geometry()    
         
            
    def load_formation_parameters(self, formation_model_file):
        """
        This function imports formation data from txt files and checks if they are correct.

        Parameters
        ----------
        formation_model_file: str
            A string that specifies path to file that stores parameters of the formation model.

        Returns
        -------
        formation_parameters: numpy.ndarray
            An array of formation parameters with dimentions converted to meters.
        """
        # Formation data
        formation_data = np.atleast_2d(np.loadtxt(formation_model_file, delimiter="\t", skiprows=2))
        
        # Formation units
        formation_units = lc.getline(formation_model_file, 2).split()[:-2]
        
        # Setup and basic checks
        formation_parameters = self.set_formation_parameters(formation_data, formation_units)
        
        return formation_parameters    
    
    
    def set_formation_parameters(self, formation_parameters, formation_units=["M", "M", "M"]):
        """
        This functions sets formation parameters, converts dimentions to meters and checks if all parameters are correct.
        
        Parameters
        ----------
        formation_parameters: numpy.ndarray
            An array of formation parameters.
            
        formation_units: list, optional
            Three element list that specifies units of formation model diameters (tops and bottoms of boundaries and diameters of filtration zones).
            Example: ["M", "MM", "FT"]
            
        Returns
        -------
        formation_parameters: numpy.ndarray
            An array of borehole parameters with dimentions converted to meters.
        """
        # Formation geometry
        for i in range(len(formation_units)):
            if formation_units[i] in self.conversion_table.keys():
                formation_parameters[:,i] *= self.conversion_table[formation_units[i]]
            else:
                raise ValueError("{} unit in formation model file not recognized. Allowed units: M, DM, CM, MM, IN, FT".format(formation_units[i]))
        if (np.diff(formation_parameters[:,:2], axis=0)<=0.0).any()==True or (formation_parameters[1:,0]!=formation_parameters[:-1,1]).any()==True:
            raise ValueError("Uncorrect formation model geometry")

        # Formation resistivity
        if np.nanmin(formation_parameters[:,[3,4]])<=0.0:
            raise ValueError('Formation resistivies have to be higher than 0 ohmm')

        return formation_parameters

    
    def load_borehole_parameters(self, borehole_model_file, borehole_geometry_type='diameter'):
        """
        This function imports borehole data from txt files and checks if they are correct.

        Parameters
        ----------
        borehole_model_file: str
            A string that specifies path to file that stores parameters of the  borehole model.
        
        borehole_geometry_type: str, optional
            A string that specifies type of borehole geometry. Available options: "diameter" and "radius".
            By default set to "diameter".
            
        Returns
        -------
        borehole_parameters: numpy.ndarray
            An array of borehole parameters with dimentions converted to meters.
        """
        # Borehole data
        borehole_data = np.atleast_2d(np.loadtxt(borehole_model_file, delimiter="\t", skiprows=2))

        # Borehole geometry
        borehole_units = lc.getline(borehole_model_file, 2).split()[:-1]

        # Setup and basic checks
        borehole_parameters = self.set_borehole_parameters(borehole_data, borehole_geometry_type=borehole_geometry_type, borehole_units=borehole_units)
                
        return borehole_parameters    
        
        
    def set_borehole_parameters(self, borehole_parameters, borehole_geometry_type='diameter', borehole_units=["M", "M"]):
        """
        This functions sets borehole parameters, converts dimentions to meters and checks if all parameters are correct.
        
        Parameters
        ----------
        borehole_parameters: numpy.ndarray
            An array of borehole parameters.
        
        borehole_geometry_type: str, optional
            A string that specifies type of borehole geometry. Available options: "diameter" and "radius".
            By default set to "diameter". 
            
        borehole_units: list, optional
            Two element list that specifies units of borehole model diameters (depths and diameters or radii).
            Example: ["M", "MM"] 
            
        Returns
        -------
        formation_parameters: numpy.ndarray
            An array of formation parameters with dimentions converted to meters.
        """
        if np.shape(borehole_parameters)[0]<2:
            raise ValueError('Borehole paramaters have to be defined for at least two depths')
            
        for i in range(len(borehole_units)):
            if borehole_units[i] in self.conversion_table.keys():
                borehole_parameters[:,i] *= self.conversion_table[borehole_units[i]]
            else:
                raise ValueError("{} unit in borehole model file not recognized. Allowed units: M, DM, CM, MM, IN, FT".format(borehole_units[i]))
        if (np.diff(borehole_parameters[:,0], axis=0)<=0.0).any()==True or (borehole_parameters[:,1]<=0.0).any()==True:
            raise ValueError("Uncorrect borehole model geometry")

        if borehole_geometry_type=='diameter':
            borehole_parameters[:,1] /= 2
        elif borehole_geometry_type=='radius':
            pass
        else:
            raise ValueError("Uncorrect borehole geometry type - use 'diameter' or 'radius' to specify borehole geometry")

        # Borehole resistivity
        if np.nanmin(borehole_parameters[:,2])<=0.0:
            raise ValueError('Drilling mud resistivies have to be higher than 0 ohmm')

        return borehole_parameters
    
    
    def set_dip(self, dip):
        """
        This function sets formation dip and checks if it is correct.

        Parameters
        ----------
        dip: float
            A value between 0 and 90. Describes dip of the layers in degrees in relation to the borehole axis.
            
        Returns
        -------
        dip: 
            A value between 0 and 90. Describes dip of the layers in degrees in relation to the borehole axis.
        dip_rad: 
            A value between of dip converted to radians.
        """
        if dip<0 or dip>=90:
            raise ValueError("Uncorrect dip angle")
        else:
            dip_rad = dip*np.pi/180
            return dip, dip_rad
    

    def _check_model_geometry(self):
        """
        This function checks if diameter of the borehole is not exceeding diameter of invasion zones.
        """
        # Currenty setup only for dip=0
        for i in range(np.shape(self.formation_model)[0]):
            layer_extend = self.borehole_model[(self.borehole_model[:,0]>=self.formation_model[i,0]) & (self.borehole_model[:,0]<=self.formation_model[i,1]), 1]
            if np.any(layer_extend>=self.formation_model[i,2]):
                raise ValueError('Borehole radius have to be smaller than the extend of the filtration zone')


    # Functions associated with modelling procedure
    def initialize_workers(self, cpu_workers=4, gpu_workers=0):
        """
        This function initializes workers that will be used during modelling procedure.
        
        Parameters
        ----------
        cpu_workers: int, optional
            Specify a number of processes that will solve the equations on cpu.
            Minimal value that can be set is 1. By default set to 4.
            
        gpu_workers: int, optional
            Specify a number of processes that will solve the equations on gpu.
            Minimal value that can be set is 0. By default set to 0.   
        """  
        ## Check GPU availability
        if gpu_workers > 0:
            try:
                import ngsolve.ngscuda
            except:
                print ("No CUDA library or device available. The number of gpu processes is set to 0")
                gpu_workers = 0    

        ## Specify number of workers
        if type(cpu_workers) != int or type(gpu_workers) != int:
            raise ValueError("The number of processes have to be an intager")
        if cpu_workers < 1:
            raise ValueError("Minimal number of cpu workers is 1")
        if gpu_workers < 0:
            raise ValueError("Minimal number of gpu workers is 0")    

        n_workers = cpu_workers + gpu_workers

        self.cpu_workers = cpu_workers
        self.gpu_workers = gpu_workers
        self.workers = cpu_workers + gpu_workers 
         
        solve_on = ["CPU"]*(cpu_workers) + ["GPU"]*gpu_workers

        ## Spawn workers
        self.comm = MPI.COMM_WORLD.Spawn(
                sys.executable,
                args=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workers/worker.py'), 
                maxprocs=n_workers)    
  
        ## Send and wait for all workers to receive data
        self.comm.bcast(solve_on, root=MPI.ROOT)
        self.comm.barrier()    
    
            
    def _prepare_simulation_depths_and_tasks(self, measurement_depths, batch_size):
        """
        This function prepares simulation tasks that later are dispatch to workers.
        """
        tools_simulation_depths = {}
        for tool in self.tools.keys():
            tools_simulation_depths[tool] = np.round(measurement_depths + self.tools[tool][1,3], decimals=4)
        
        if self.sec==True:
            simulation_depths = np.unique(np.hstack(list(tools_simulation_depths.values())))
        elif self.sec==False:
            simulation_depths = np.hstack(list(tools_simulation_depths.values()))
            simulated_tools_indices = [tool_index for tool_index in list(range(len(self.tools.keys()))) for _ in range(len(measurement_depths))]
            sort_order = np.argsort(simulation_depths)
            simulation_depths = simulation_depths[sort_order]
            simulated_tools_indices = [simulated_tools_indices[i] for i in sort_order]

        number_of_batches = int(np.ceil(simulation_depths.size/batch_size))
        simulation_depths = np.pad(simulation_depths.astype(float), (0, number_of_batches*batch_size - simulation_depths.size), mode='constant', constant_values=np.nan).reshape(number_of_batches, batch_size)
        combined_simulation_depths = np.round(np.nanmean(simulation_depths, axis=1), decimals=4)
        simulation_offsets = np.round(simulation_depths-combined_simulation_depths[:, None], decimals=4)

        tasks = []
        for batch_index in range(number_of_batches):
            # Close depths source model
            batch_potential_electodes_depths = []
            batch_current_electrodes_depths = []
            batch_modelling_tasks = []
            for depth_index in range(batch_size):
                simulation_depth_index = batch_index*batch_size+depth_index
                simulation_depth = simulation_depths[batch_index, depth_index]
                if np.isnan(simulation_depth):
                    break
                simulation_offset = simulation_offsets[batch_index, depth_index]
                modelling_tasks = []

                if self.sec==True:
                    potential_electodes_depths = []    
                    current_electrodes_depths = [] 
                    # Single model for all tools with same current electrode depth
                    for tool_index in range(len(list(self.tools.keys()))):
                        tool = list(self.tools.keys())[tool_index]
                        if np.any(np.isclose(tools_simulation_depths[tool], simulation_depth)):
                            measurement_depth_index = np.argwhere(np.isclose(measurement_depths + self.tools[tool][1,3], simulation_depth))[0][0]
                            modelling_tasks.append([measurement_depth_index, tool_index, simulation_offset])
                            tool_electrodes = self.tools[tool][:,:3].copy()
                            tool_electrodes[0,:3] += simulation_offset
                            tool_electrodes = np.round(tool_electrodes, 4)
                            current_electrodes_depths += list(tool_electrodes[0, tool_electrodes[1,:]!=0])
                            potential_electodes_depths += list(tool_electrodes[0, tool_electrodes[1,:]==0])
                            batch_current_electrodes_depths += list(tool_electrodes[0, tool_electrodes[1,:]!=0])
                            batch_potential_electodes_depths += list(tool_electrodes[0, tool_electrodes[1,:]==0])
                            
                    unique_current_electrodes_depths = np.unique(current_electrodes_depths)
                    unique_potential_electodes_depths = np.unique(potential_electodes_depths)
                    unique_potential_electodes_depths = unique_potential_electodes_depths[~np.isin(unique_potential_electodes_depths, unique_current_electrodes_depths)]
                                                                                                   
                    combined_tools = np.hstack([np.vstack([unique_potential_electodes_depths, np.zeros_like(unique_potential_electodes_depths)]),
                                                np.vstack([unique_current_electrodes_depths, np.ones_like(unique_current_electrodes_depths)])])
                
                if self.sec==False:
                    # Single model for every tool
                    tool_index = simulated_tools_indices[simulation_depth_index]
                    tool = list(self.tools.keys())[tool_index]
                    measurement_depth_index = np.argwhere(np.isclose(measurement_depths + self.tools[tool][1,3], simulation_depth))[0][0]
                    modelling_tasks.append([measurement_depth_index, tool_index, simulation_offset])
                    tool_electrodes = self.tools[tool][:,:3].copy()
                    tool_electrodes[0,:3] += simulation_offset
                    tool_electrodes = np.round(tool_electrodes, 4)

                    batch_current_electrodes_depths += list(tool_electrodes[0, tool_electrodes[1,:]!=0])
                    batch_potential_electodes_depths += list(tool_electrodes[0, tool_electrodes[1,:]==0])

                    combined_tools = tool_electrodes   
                    
                combined_tools = combined_tools[:,combined_tools[0,:].argsort()]

                batch_modelling_tasks.append([simulation_depth_index, combined_tools, modelling_tasks])

            unique_batch_current_electrodes_depths = np.unique(batch_current_electrodes_depths)
            unique_batch_potential_electodes_depths = np.unique(batch_potential_electodes_depths)
            unique_batch_potential_electodes_depths = unique_batch_potential_electodes_depths[~np.isin(unique_batch_potential_electodes_depths, unique_batch_current_electrodes_depths)]

            batch_combined_tools = np.hstack([np.vstack([unique_batch_potential_electodes_depths, np.zeros_like(unique_batch_potential_electodes_depths)]),
                                        np.vstack([unique_batch_current_electrodes_depths, np.ones_like(unique_batch_current_electrodes_depths)])])

            batch_combined_tools = batch_combined_tools[:,batch_combined_tools[0,:].argsort()]

            tasks.append([batch_index, batch_combined_tools, batch_modelling_tasks])
            
        return combined_simulation_depths, tasks      
    
    def _add_points_to_borehole(self, maximal_distance=0.15):
        """
        This function adds additional points if borehole geometry is too sparse for the purpose of avoiding errors during 3d meshing procedure.
        
        Parameters
        ----------
        maximal_distance: float, optional
            Distance above which additional points will be interpolated within borehole geometry.
        """
        ## Add additional points if borehole geometry is too sparse
        interpolated_depths = self.borehole_model[0,0]
        for i in range(1, np.shape(self.borehole_model)[0]):
            distance = self.borehole_model[i,0] - self.borehole_model[i-1,0]
            if distance > maximal_distance:
                additional_points = np.linspace(self.borehole_model[i-1,0], self.borehole_model[i,0], np.max([3, int(distance*10+1)]))
                interpolated_depths = np.hstack([interpolated_depths, additional_points[1:]])
            else:
                interpolated_depths = np.hstack([interpolated_depths, self.borehole_model[i,0]])

        ## Interpolate geometry and mud resistivities
        if np.shape(interpolated_depths)[0] > np.shape(self.borehole_model)[0]:
            geometry_interpolation = spi.interp1d(self.borehole_model[:,0], self.borehole_model[:,1], kind='linear')
            mud_resistivity_interpolation = spi.interp1d(self.borehole_model[:,0], self.borehole_model[:,2], kind='linear')

            borehole_parameters = np.vstack([interpolated_depths, geometry_interpolation(interpolated_depths), mud_resistivity_interpolation(interpolated_depths)]).T

        return borehole_parameters
    
    
    def simulate_logs(self, measurement_depths, domain_radius=50, batch_size=5, mesh_generator="auto", preconditioner="multigrid", condense=True):
        """
        This function prepares data, dispatches tasks to workers, gathers and assembles generated synthetic logs.
        
        Parameters
        ----------
        measurement_depths: np.ndarray
            A 1D array of depths of simulated measurements.
            Values have to be given in ascending order and corespond to depths of the model.

        domain_radius: float, optional
            A radius of simulation domain in meters.
            By default set to 50.

        batch_size: int, optional
            Specify a number of adjacent measurement points that are joined into a single mesh generation and simulation procedure to speed up the process.
            By default set to 5.    

        mesh_generator: string, optional
            Specify utiliezed mesh generator. Can be set to "gmsh" or "netgen" for 2D models and to "gmsh" for 3D models.
            By default set to "auto" and will chose "netgen" for 2D models and "gmsh" for 3D models.

        preconditioner: string, optional
            Specify a type of utilized preconditioner. Available options: "local" and "multigrid".
            By default set to "multigrid".

        condense: bool, optional
            Specify if static condensation will be utilized to eliminate unknowns that are internal to elements from the global linear system.
            By default set to True.
        """
        ### Start the clock
        start_time = datetime.datetime.now()

        ### Unpack parameters and prepare data

        ## Batch mode setup
        if batch_size>1:
            batch_mode = True
        else:
            batch_mode = False

        ## Model
        # Simulation domain
        domain_radius_alert = False
        for tool in self.tools.keys():
            if np.max(np.abs(self.tools[tool][0,:3])) > domain_radius:
                raise ValueError("Some electrodes are locate outside the simulation domain. Domain size have to be increased")
            elif np.max(np.abs(self.tools[tool][0,:3])) > 0.75*domain_radius:
                domain_radius_alert = True
        if domain_radius_alert == True:
            print("Some electrodes are located close to the boundary of the simulation domain. This may cause problems during simulation. Consider increase of the domain size")

        ## Mesh generator setup
        if mesh_generator=="auto":
            if np.isclose(self.dip_deg, 0):
                mesh_generator = "netgen"
            else:
                mesh_generator = "gmsh"

        # Check if mesh generator suports model geometry
        if ~np.isclose(self.dip_deg, 0) and mesh_generator!="gmsh":
            raise ValueError("The only mesh generator supported in 3D models is gmsh")

        # Create temporary directory for mesh files
        if mesh_generator=="gmsh" and not os.path.exists("./tmp"):
            os.makedirs("./tmp")

        # Create dense borehole geometry for the purpose of 3D mesh generation (necessary to avoid errors during meshing procedure)
        if self.dip_deg!=0:
            self.borehole_model = self._add_points_to_borehole()

        ### Parallel FEM computation

        tool_list = list(range(len(self.tools.keys())))
        measurement_depths_list = list(range(len(measurement_depths)))

        ## Compute simulation depths and prepare tasks
        simulation_depths, task_list = self._prepare_simulation_depths_and_tasks(measurement_depths, batch_size)

        n_tasks = len(task_list)

        # Borehole geometry and resistivities at simulation depths
        borehole_geometry = np.ascontiguousarray(self.borehole_model[:,:2])
        mud_resistivities = np.interp(simulation_depths, self.borehole_model[:,0], self.borehole_model[:,2])

        # Setup workers
        msg_list = ["Setup"]*(self.cpu_workers + self.gpu_workers)
        
        status = MPI.Status()
        i = 0
        for msg in msg_list:
            if msg != StopIteration:
                i += 1
                # Pass data to worker
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
        
        ## Broadcast data to workers
        # Broadcast information about shapes of broadcasted arrays
        arrays_shape = [np.shape(self.formation_model), np.shape(borehole_geometry), np.shape(mud_resistivities), np.shape(simulation_depths)]
        self.comm.bcast(arrays_shape, root=MPI.ROOT)

        # Broadcast data
        self.comm.Bcast([self.formation_model, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([borehole_geometry, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([mud_resistivities, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([simulation_depths, MPI.FLOAT], root=MPI.ROOT)
        self.comm.bcast(self.dip_rad, root=MPI.ROOT)
        self.comm.bcast(self.tools, root=MPI.ROOT)
        self.comm.bcast(domain_radius, root=MPI.ROOT)
        self.comm.bcast(mesh_generator, root=MPI.ROOT)
        self.comm.bcast(preconditioner, root=MPI.ROOT)
        self.comm.bcast(condense, root=MPI.ROOT)
        self.comm.bcast(task_list, root=MPI.ROOT)

        ## Wait for all workers to receive data
        self.comm.barrier()
        
        ## Convert tasks to mesages 
        print("{} simulation tasks prepared".format(n_tasks))
        msg_list = list(np.arange(n_tasks)) + ([StopIteration] * (self.cpu_workers + self.gpu_workers)) # Append stop sentinel for each worker
        
        ## Dispatch tasks to workers
        status = MPI.Status()
        i = 0
        for msg in msg_list:
            if msg != StopIteration:
                i += 1
                # Pass data to worker
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
                # Progress bar
                percent = ((i) * 100) // (n_tasks)
                sys.stdout.write('\rProgress: [%-50s] %3i%% ' % ('=' * (percent // 2), percent))
                sys.stdout.flush()
            else:
                # Tell worker to shutdown
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
        
        self.comm.barrier()
        
        # Gather results from workers
        list_of_results = [item for sublist in self.comm.gather(None, root=MPI.ROOT) for item in sublist]

        ## Format and sort results
        results = np.empty([len(measurement_depths_list), len(tool_list)])
        for result in list_of_results:
            results[result[0], result[1]] = result[2]

        logs = dict()
        for i in range(len(self.tools.keys())):
            logs[list(self.tools.keys())[i]] = np.vstack([measurement_depths, results[:,i]]).T

        ### Remove tmp folder and mesh files
        if mesh_generator=="gmsh":
            shutil.rmtree("./tmp")

        ### Report time of computation
        print('\nProcessed in: ', datetime.datetime.now() - start_time)

        # Save logs
        self.logs = logs
        

    def shutdown_workers(self):
        """
        This function shuts workers that were used during modelling procedure.
        """
        msg_list = [StopIteration] * self.workers # Append stop sentinel for each worker
        
        ## Dispatch tasks to workers 
        status = MPI.Status()
        i = 0
        for msg in msg_list:
            # Tell worker to shutdown
            self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            self.comm.send(obj=msg, dest=status.Get_source())                             

        
    def save_results(self, output_folder=None, measurements_to_save="auto", plot_layout="auto", plot_depth_lim="auto", plot_aspect_ratio="auto", model_rad_lim="auto",
                     model_res_lim="auto", logs_res_lim="auto", logs_at_nan="break", logs_interpolation_factor=1, logs_colours="auto"):
        """
        This function saves results of modelling to txt file and produces raw visualization of the model and computed syntetic logs that is saved in PNG format.

        Parameters
        ----------
        output_folder: str
            A path to the folder where results will be saved.
            By default set to None will only show raw visualization of the model and computed syntetic logs without saving them to txt files (works only in Jupyter Notebooks).

        measurements_to_save: str or list, optional
            A list of measurements to save to txt files.
            By default set to "auto" will save all measurements.

        plot_layout: list, optional
            A list of sublists of tool names. Each sublist consist of tool names assigned to certain track.
            By default set to "auto" will plot all logs on a sigle track.

        plot_depth_lim: list, optional
            A list of two floats that specify minimum and maximum depth of ploted data.
            By default set to "auto" will plot data for entire range of avilable depths.

        plot_aspect_ratio: float, optional
            A float that specify hight to widht ratio of the plot.
            By default set to "auto" will automaticly adjust the value of parameter based on number of tracks
            within layout and values of plot_depth_lim and  model_rad_lim parameters.

        model_rad_lim: list, optional
            A list of two floats that specify minimum and maximum radius of ploted formation model.
            By default set to "auto" will plot data from borehole axis to radius 2 times as big as
            deepest filtration zone or 10 times as big as largest borehole radius if no filtration zone
            is present within the model.

        model_res_lim: list, optional
            A list of two floats that specify minimum and maximum value of model resistivity.
            By default set to "auto" will automaticly adjust range to model resistivities.

        logs_res_lim: list, optional
            A list of two floats that specify minimum and maximum value of ploted resistivity logs.
            By default set to "auto" will automaticly adjust range to show entire logs.

        logs_at_nan: str, optional
            Specify if plot should be breaked or continued on Nan values.
            Available options: "break" and "continue".
            By default set to "break".

        logs_interpolation_factor: float, optional
            Allows to smooth logs on vizualization. Have no inpact on output data.
            By default set to 1 (no interpolation).

        logs_colours: list, optional
            A list of sublists of logs colours. Each sublist consist of logs colours assigned to certain track.
            Have to have the same structure as list pased in plot_layout parameter.
            By default set to "auto" will automaticly assign different colours to logs.
        """
        
        ### Create folder structure and save results to txt files
        if output_folder!=None:
            ## Create output folder
            output_subfolder = os.path.join(output_folder, "Results_{}/".format(str(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))))

            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            ## Save data to txt files
            if measurements_to_save=="auto":
                measurements_to_save = list(self.logs.keys())

            logs_to_save = measurements_to_save.copy()
            file_number = 1
            while len(logs_to_save)>0:
                logs = [logs_to_save[0]]
                for i in range(1, len(logs_to_save)):
                    if np.shape(self.logs[logs_to_save[0]][:,0])[0] == np.shape(self.logs[logs_to_save[i]][:,0])[0]:
                        if np.all(np.isclose(self.logs[logs_to_save[0]][:,0], self.logs[logs_to_save[i]][:,0]), axis=0)==True:
                            logs.append(logs_to_save[i])

                for log in logs:
                    logs_to_save.remove(log)

                results = self.logs[logs[0]]
                for i in range(1, len(logs)):
                    results = np.hstack([results, np.atleast_2d(self.logs[logs[i]][:,1]).T])

                names = ['DEPTH'] + logs
                units = ['M'] + ['OHMM']*len(logs)
                header = '\t'.join([name for name in names]) + '\n' + '\t'.join([unit for unit in units])
                np.savetxt(output_subfolder + 'Results_{}.txt'.format(file_number), results,  fmt='%.4f', delimiter='\t', header=header, comments='')
                file_number += 1

        ### Formation model visualization
        ## Unpack parameters
        formation_parameters = self.formation_model
        borehole_parameters = self.borehole_model
        dip = self.dip_deg

        ## Smooth logs
        if logs_interpolation_factor > 1:
            logs = list(self.logs.keys())
            for log in logs:
                measurement_depths_interp = np.linspace(np.min(self.logs[log][:,0]), np.max(self.logs[log][:,0]), int(np.shape(self.logs[log])[0]*logs_interpolation_factor))
                interpolation = spi.interp1d(self.logs[log][:,0], self.logs[log][:,1], kind='cubic')
                self.logs[log] = np.vstack([measurement_depths_interp, interpolation(measurement_depths_interp)]).T

        ## Prepare plot limits:
        if plot_depth_lim == "auto":
            plot_depth_lim = [np.nanmin(formation_parameters[:,:2]), np.nanmax(formation_parameters[:,:2])]
        if  model_rad_lim == "auto":
            if np.all(np.isnan(formation_parameters[:,2])):
                 model_rad_lim = [-10*np.nanmax(borehole_parameters[:,1]), 10*np.nanmax(borehole_parameters[:,1])]
            else:
                 model_rad_lim = [-2*np.nanmax(formation_parameters[:,2]), 2*np.nanmax(formation_parameters[:,2])]
        if logs_res_lim == "auto":
            res_max = 0
            for log in self.logs.values():
                res_max = max(np.max(log), res_max)
            res_min = res_max
            for log in self.logs.values():
                res_min = min(np.min(log), res_min)
            res_min = np.floor(res_min/10**np.floor(np.log10(res_max)-1)) * 10**np.floor(np.log10(res_max)-1)
            res_max = np.ceil(res_max/10**np.floor(np.log10(res_max)-1)) * 10**np.floor(np.log10(res_max)-1)
            logs_res_lim = [res_min, res_max]
        if plot_aspect_ratio == "auto":
            plot_aspect_ratio = (plot_depth_lim[1] - plot_depth_lim[0])/25*1.25

        ## Prepare polygons
        patches = []

        # Formation
        a = np.tan(dip*np.pi/180)
        formation_parameters[0,0] -= a* model_rad_lim[1] # adjust model to fill the plot
        formation_parameters[-1,1] += a* model_rad_lim[1] # adjust model to fill the plot
        for i in range(np.shape(formation_parameters)[0]):
            if np.isnan(formation_parameters[i,2]) == True:
                vetrices = np.array([[ model_rad_lim[0], formation_parameters[i,0]+a* model_rad_lim[0]],
                    [model_rad_lim[0], formation_parameters[i,1]+a* model_rad_lim[0]],
                    [model_rad_lim[1], formation_parameters[i,1]+a* model_rad_lim[1]],
                    [model_rad_lim[1], formation_parameters[i,0]+a* model_rad_lim[1]]])
                polygon = Polygon(vetrices, closed=True)
                patches.append(polygon)
            else:
                vetrices = np.array([[ model_rad_lim[0], formation_parameters[i,0]+a* model_rad_lim[0]],
                    [model_rad_lim[0], formation_parameters[i,1]+a* model_rad_lim[0]],
                    [model_rad_lim[1], formation_parameters[i,1]+a* model_rad_lim[1]],
                    [model_rad_lim[1], formation_parameters[i,0]+a* model_rad_lim[1]]])
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
        borehole_axis = Line2D([0, 0], plot_depth_lim, color='black')

        ## Plot model and logs
        if plot_layout=="auto":
            tracks = 1
        else:
            tracks = len(plot_layout)

        fig_width = 5 + 5*tracks
        fig_hight = fig_width*plot_aspect_ratio

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
        ax[0].set_xlim( model_rad_lim)
        ax[0].set_ylim(plot_depth_lim)
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

            if logs_colours=="auto":
                track_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            else:
                track_colours = logs_colours[track-1]
            if plot_layout=="auto":
                logs = list(self.logs.keys())
            else:
                logs = plot_layout[track-1]
            for i in range(len(logs)):
                if i==0:
                    axis = ax[track]
                else:
                    axis = ax[track].twiny()
                if logs_at_nan=="break":
                    axis.plot(self.logs[logs[i]][:,1], self.logs[logs[i]][:,0], color=track_colours[i]) # change from: color=track_colours[i%len(logs_colours)]
                elif logs_at_nan=="continue":
                    nan_flag = ~np.isnan(self.logs[logs[i]][:,1])
                    axis.plot(self.logs[logs[i]][nan_flag,1], self.logs[logs[i]][nan_flag,0], color=track_colours[i]) # change from: color=track_colours[i%len(logs_colours)]
                else:
                    raise ValueError('logs_at_nan paramater has to be set to "break" or "continue"')
                axis.set_xlabel(logs[i]+"\n[ohmm]", color=track_colours[i%len(track_colours)], labelpad=-8)
                axis.spines['top'].set_color(track_colours[i%len(track_colours)])
                axis.spines['top'].set_position(('outward', i*55+10))
                axis.set_xticks(logs_res_lim)
                axis.tick_params(axis='x', color=track_colours[i%len(track_colours)])
                axis.set_xlim(logs_res_lim)
            axis = ax[track].twiny().get_xaxis().set_visible(False)
        for track in range(1, tracks+1):
            ax[track].grid(True)
            ax[track].xaxis.set_label_position('top')
            ax[track].xaxis.set_ticks_position('top')
            ax[track].margins(x=0, y=0)
            ax[track].autoscale_view()

        colorbar = fig.colorbar(collection, ax=ax, location='bottom', orientation='horizontal', pad=0.05, label="Resistivity [ohmm]", shrink=min([1, plot_aspect_ratio]))
        colorbar.ax.minorticks_on()

        ## Save plot to png file
        if output_folder!=None:
            plt.savefig(output_subfolder + 'Results_plot.png', bbox_inches='tight')



    ########################
    ### WORK IN PROGRESS ###
    ######################## 



    # Functions associated with modelling procedure
    def initialize_meshing_workers(self, workers=4):
        """
        This function initializes workers that will be used during meshing procedure.
        
        Parameters
        ----------
        workers: int, optional
            Specify a number of processes that will generate meshes.
            Minimal value that can be set is 1. By default set to 4.
        """  
        ## Specify number of workers
        if type(workers) != int:
            raise ValueError("The number of processes have to be an intager")
        if workers < 1:
            raise ValueError("Minimal number of workers is 1")

        self.workers = workers
        
        ## Spawn workers
        self.comm = MPI.COMM_WORLD.Spawn(
                sys.executable,
                args=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workers/meshing_worker.py'), 
                maxprocs=self.workers)    

    
    def initialize_simulation_workers(self, cpu_workers=4, gpu_workers=0):
        """
        This function initializes workers that will be used during simulation procedure.
        
        Parameters
        ----------
        cpu_workers: int, optional
            Specify a number of processes that will solve the equations on cpu.
            Minimal value that can be set is 1. By default set to 4.
            
        gpu_workers: int, optional
            Specify a number of processes that will solve the equations on gpu.
            Minimal value that can be set is 0. By default set to 0.   
        """  
        ## Check GPU availability
        if gpu_workers > 0:
            try:
                import ngsolve.ngscuda
            except:
                print ("No CUDA library or device available. The number of gpu processes is set to 0")
                gpu_workers = 0    

        ## Specify number of workers
        if type(cpu_workers) != int or type(gpu_workers) != int:
            raise ValueError("The number of processes have to be an intager")
        if cpu_workers < 1:
            raise ValueError("Minimal number of cpu workers is 1")
        if gpu_workers < 0:
            raise ValueError("Minimal number of gpu workers is 0")    

        n_workers = cpu_workers + gpu_workers

        self.cpu_workers = cpu_workers
        self.gpu_workers = gpu_workers
        self.workers = cpu_workers + gpu_workers 
         
        solve_on = ["CPU"]*(cpu_workers) + ["GPU"]*gpu_workers

        ## Spawn workers
        self.comm = MPI.COMM_WORLD.Spawn(
                sys.executable,
                args=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workers/simulation_worker.py'), 
                maxprocs=n_workers)    
  
        ## Send and wait for all workers to receive data
        self.comm.bcast(solve_on, root=MPI.ROOT)
        self.comm.barrier()   
        
        
    def generate_meshes(self, measurement_depths, domain_radius=50, batch_size=5, mesh_generator="auto"):
        """
        This function prepares data, dispatches tasks to workers, gathers and assembles generated meshes.
        
        Parameters
        ----------
        measurement_depths: np.ndarray
            A 1D array of depths of simulated measurements.
            Values have to be given in ascending order and corespond to depths of the model.

        domain_radius: float, optional
            A radius of simulation domain in meters.
            By default set to 50.

        batch_size: int, optional
            Specify a number of adjacent measurement points that are joined into a single mesh generation and simulation procedure to speed up the process.
            By default set to 5.    

        mesh_generator: string, optional
            Specify utiliezed mesh generator. Can be set to "gmsh" or "netgen" for 2D models and to "gmsh" for 3D models.
            By default set to "auto" and will chose "netgen" for 2D models and "gmsh" for 3D models.
        """
        ### Start the clock
        start_time = datetime.datetime.now()

        ### Unpack parameters and prepare data

        ## Batch mode setup
        if batch_size>1:
            batch_mode = True
        else:
            batch_mode = False

        ## Model
        # Simulation domain
        domain_radius_alert = False
        for tool in self.tools.keys():
            if np.max(np.abs(self.tools[tool][0,:3])) > domain_radius:
                raise ValueError("Some electrodes are locate outside the simulation domain. Domain size have to be increased")
            elif np.max(np.abs(self.tools[tool][0,:3])) > 0.75*domain_radius:
                domain_radius_alert = True
        if domain_radius_alert == True:
            print("Some electrodes are located close to the boundary of the simulation domain. This may cause problems during simulation. Consider increase of the domain size")

        ## Mesh generator setup
        if mesh_generator=="auto":
            if np.isclose(self.dip_deg, 0):
                mesh_generator = "netgen"
            else:
                mesh_generator = "gmsh"

        # Check if mesh generator suports model geometry
        if ~np.isclose(self.dip_deg, 0) and mesh_generator!="gmsh":
            raise ValueError("The only mesh generator supported in 3D models is gmsh")

        # Create temporary directory for gmsh mesh files
        if mesh_generator=="gmsh" and not os.path.exists("./tmp"):
            os.makedirs("./tmp")

        # Create temporary directory for netgen mesh files
        if not os.path.exists("./meshfiles"):
            os.makedirs("./meshfiles")

        # Create dense borehole geometry for the purpose of 3D mesh generation (necessary to avoid errors during meshing procedure)
        if self.dip_deg!=0:
            self.borehole_model = self._add_points_to_borehole()

        ### Parallel FEM computation

        tool_list = list(range(len(self.tools.keys())))
        measurement_depths_list = list(range(len(measurement_depths)))

        ## Compute simulation depths and prepare tasks
        simulation_depths, task_list = self._prepare_simulation_depths_and_tasks(measurement_depths, batch_size)

        n_tasks = len(task_list)

        # Borehole geometry and resistivities at simulation depths
        borehole_geometry = np.ascontiguousarray(self.borehole_model[:,:2])
        mud_resistivities = np.interp(simulation_depths, self.borehole_model[:,0], self.borehole_model[:,2])

        # Setup workers
        msg_list = ["Setup"]*(self.workers)
        
        status = MPI.Status()
        i = 0
        for msg in msg_list:
            if msg != StopIteration:
                i += 1
                # Pass data to worker
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
        
        ## Broadcast data to workers
        # Broadcast information about shapes of broadcasted arrays
        arrays_shape = [np.shape(self.formation_model), np.shape(borehole_geometry), np.shape(mud_resistivities), np.shape(simulation_depths)]
        self.comm.bcast(arrays_shape, root=MPI.ROOT)

        # Broadcast data
        self.comm.Bcast([self.formation_model, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([borehole_geometry, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([mud_resistivities, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([simulation_depths, MPI.FLOAT], root=MPI.ROOT)
        self.comm.bcast(self.dip_rad, root=MPI.ROOT)
        self.comm.bcast(self.tools, root=MPI.ROOT)
        self.comm.bcast(domain_radius, root=MPI.ROOT)
        self.comm.bcast(mesh_generator, root=MPI.ROOT)
        self.comm.bcast(task_list, root=MPI.ROOT)
        
        ## Wait for all workers to receive data
        self.comm.barrier()
        
        ## Convert tasks to mesages 
        print("{} meshing tasks prepared".format(n_tasks))
        msg_list = list(np.arange(n_tasks)) + ([StopIteration] * (self.workers)) # Append stop sentinel for each worker
        ## Dispatch tasks to workers
        status = MPI.Status()
        i = 0
        for msg in msg_list:
            if msg != StopIteration:
                i += 1
                # Pass data to worker
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
                # Progress bar
                percent = ((i) * 100) // (n_tasks)
                sys.stdout.write('\rProgress: [%-50s] %3i%% ' % ('=' * (percent // 2), percent))
                sys.stdout.flush()
            else:
                # Tell worker to shutdown
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
        
        self.comm.barrier()

        ### Remove tmp folder and mesh files
        if mesh_generator=="gmsh":
            shutil.rmtree("./tmp")

        ### Report time of computation
        print('\nProcessed in: ', datetime.datetime.now() - start_time)

        
    def create_logs(self, measurement_depths, domain_radius=50, batch_size=5, mesh_generator="auto", preconditioner="multigrid", condense=True):
        """
        This function prepares data, dispatches tasks to workers, gathers and assembles generated synthetic logs.
        
        Parameters
        ----------
        measurement_depths: np.ndarray
            A 1D array of depths of simulated measurements.
            Values have to be given in ascending order and corespond to depths of the model.

        domain_radius: float, optional
            A radius of simulation domain in meters.
            By default set to 50.

        batch_size: int, optional
            Specify a number of adjacent measurement points that are joined into a single mesh generation and simulation procedure to speed up the process.
            By default set to 5.    

        mesh_generator: string, optional
            Specify utiliezed mesh generator. Can be set to "gmsh" or "netgen" for 2D models and to "gmsh" for 3D models.
            By default set to "auto" and will chose "netgen" for 2D models and "gmsh" for 3D models.

        preconditioner: string, optional
            Specify a type of utilized preconditioner. Available options: "local" and "multigrid".
            By default set to "multigrid".

        condense: bool, optional
            Specify if static condensation will be utilized to eliminate unknowns that are internal to elements from the global linear system.
            By default set to True.
        """
        ### Start the clock
        start_time = datetime.datetime.now()

        ### Unpack parameters and prepare data

        ## Batch mode setup
        if batch_size>1:
            batch_mode = True
        else:
            batch_mode = False

        ## Model
        # Simulation domain
        domain_radius_alert = False
        for tool in self.tools.keys():
            if np.max(np.abs(self.tools[tool][0,:3])) > domain_radius:
                raise ValueError("Some electrodes are locate outside the simulation domain. Domain size have to be increased")
            elif np.max(np.abs(self.tools[tool][0,:3])) > 0.75*domain_radius:
                domain_radius_alert = True
        if domain_radius_alert == True:
            print("Some electrodes are located close to the boundary of the simulation domain. This may cause problems during simulation. Consider increase of the domain size")

        ## Mesh generator setup
        if mesh_generator=="auto":
            if np.isclose(self.dip_deg, 0):
                mesh_generator = "netgen"
            else:
                mesh_generator = "gmsh"

        # Check if mesh generator suports model geometry
        if ~np.isclose(self.dip_deg, 0) and mesh_generator!="gmsh":
            raise ValueError("The only mesh generator supported in 3D models is gmsh")

        # Create temporary directory for mesh files
        if mesh_generator=="gmsh" and not os.path.exists("./tmp"):
            os.makedirs("./tmp")

        # Create dense borehole geometry for the purpose of 3D mesh generation (necessary to avoid errors during meshing procedure)
        if self.dip_deg!=0:
            self.borehole_model = self._add_points_to_borehole()

        ### Parallel FEM computation

        tool_list = list(range(len(self.tools.keys())))
        measurement_depths_list = list(range(len(measurement_depths)))

        ## Compute simulation depths and prepare tasks
        simulation_depths, task_list = self._prepare_simulation_depths_and_tasks(measurement_depths, batch_size)

        n_tasks = len(task_list)

        # Borehole geometry and resistivities at simulation depths
        borehole_geometry = np.ascontiguousarray(self.borehole_model[:,:2])
        mud_resistivities = np.interp(simulation_depths, self.borehole_model[:,0], self.borehole_model[:,2])

        # Setup workers
        msg_list = ["Setup"]*(self.cpu_workers + self.gpu_workers)
        
        status = MPI.Status()
        i = 0
        for msg in msg_list:
            if msg != StopIteration:
                i += 1
                # Pass data to worker
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
        
        ## Broadcast data to workers
        # Broadcast information about shapes of broadcasted arrays
        arrays_shape = [np.shape(self.formation_model), np.shape(borehole_geometry), np.shape(mud_resistivities), np.shape(simulation_depths)]
        self.comm.bcast(arrays_shape, root=MPI.ROOT)

        # Broadcast data
        self.comm.Bcast([self.formation_model, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([borehole_geometry, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([mud_resistivities, MPI.FLOAT], root=MPI.ROOT)
        self.comm.Bcast([simulation_depths, MPI.FLOAT], root=MPI.ROOT)
        self.comm.bcast(self.dip_rad, root=MPI.ROOT)
        self.comm.bcast(self.tools, root=MPI.ROOT)
        self.comm.bcast(domain_radius, root=MPI.ROOT)
        self.comm.bcast(mesh_generator, root=MPI.ROOT)
        self.comm.bcast(preconditioner, root=MPI.ROOT)
        self.comm.bcast(condense, root=MPI.ROOT)
        self.comm.bcast(task_list, root=MPI.ROOT)

        ## Wait for all workers to receive data
        self.comm.barrier()
        
        ## Convert tasks to mesages 
        print("{} simulation tasks prepared".format(n_tasks))
        msg_list = list(np.arange(n_tasks)) + ([StopIteration] * (self.cpu_workers + self.gpu_workers)) # Append stop sentinel for each worker
        
        ## Dispatch tasks to workers
        status = MPI.Status()
        i = 0
        for msg in msg_list:
            if msg != StopIteration:
                i += 1
                # Pass data to worker
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
                # Progress bar
                percent = ((i) * 100) // (n_tasks)
                sys.stdout.write('\rProgress: [%-50s] %3i%% ' % ('=' * (percent // 2), percent))
                sys.stdout.flush()
            else:
                # Tell worker to shutdown
                self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                self.comm.send(obj=msg, dest=status.Get_source())
        
        self.comm.barrier()
        
        # Gather results from workers
        list_of_results = [item for sublist in self.comm.gather(None, root=MPI.ROOT) for item in sublist]

        ## Format and sort results
        results = np.empty([len(measurement_depths_list), len(tool_list)])
        for result in list_of_results:
            results[result[0], result[1]] = result[2]

        logs = dict()
        for i in range(len(self.tools.keys())):
            logs[list(self.tools.keys())[i]] = np.vstack([measurement_depths, results[:,i]]).T

        ### Remove tmp folder and mesh files
        if mesh_generator=="gmsh":
            shutil.rmtree("./tmp")

        ### Report time of computation
        print('\nProcessed in: ', datetime.datetime.now() - start_time)

        # Save logs
        self.logs = logs