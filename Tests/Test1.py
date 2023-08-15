"""
   How to run:
   mpiexec python3 Test1.py
"""

import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import src.remo3d as rm
import numpy as np

np.set_printoptions(suppress = True)

# Specify input data
tools = ["B5.7A0.4M", "M5.7N0.4A", "M2.0A0.5B", "A2.0M0.5N"] # logging tools
formation_model_file = "./Input/Formation.txt" # path to file with formation parameters
borehole_model_file = "./Input/Borehole.txt" # path to file with borehole parameters
measurement_depths = np.arange(0, 25.1, 0.25) # measurement points

# Set tools parameters
tools_parameters = rm.SetToolsParameters(tools)

# Set model parameters
model_parameters = rm.SetModelParameters(formation_model_file, borehole_model_file)

# Compute synthetic logs
logs = rm.ComputeSyntheticLogs(tools_parameters, model_parameters, measurement_depths, tools_version="alternative",
                               domain_radius=25, processes=12, mesh_generator="gmsh")

# Save results
output_folder = "./Output" # path to output folder
rm.SaveResults(model_parameters, logs, output_folder=output_folder, 
               plot_layout=[["B5.7A0.4M", "M5.7N0.4A"], ["M2.0A0.5B", "A2.0M0.5N"]],
               depth_lim=[0,25], rad_lim=[-1,1], res_lim=[0,30], aspect_ratio=1.25, at_nan="continue")