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
# tools = ["B5.7A0.4M", "M5.7N0.4A",
#          "M0.4A5.7B", "A0.4N5.7M",
#          "M2.0A0.5B", "A2.0M0.5N",
#          "B0.5A2.0M", "N0.5M2.0A"] # logging tools
tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"] # logging tools
formation_model_file = "./Input/Example/Formation.txt" # path to file with formation parameters
borehole_model_file = "./Input/Example/Borehole.txt" # path to file with borehole parameters
measurement_depths = np.arange(0, 25.1, 0.25) # measurement points
measurement_depths_mask = np.all(np.vstack([measurement_depths>=2, measurement_depths<=23]).T, axis=1)

# Set tools parameters
tools_parameters = rm.SetToolsParameters(tools)

# Set model parameters
model_parameters = rm.SetModelParameters(formation_model_file, borehole_model_file)

# Compute synthetic logs
measurement_depths_mask = np.all(np.vstack([measurement_depths>=3, measurement_depths<=22]), axis=0)

results = rm.CreateMeshFiles(tools_parameters, model_parameters, measurement_depths, measurement_depths_mask=measurement_depths_mask, force_single_electrode_configuration=True,
                             domain_radius=50, processes=12, mesh_generator="netgen", output_folder="./meshfiles")

logs = rm.ComputeSyntheticLogs(tools_parameters, model_parameters, measurement_depths, measurement_depths_mask=measurement_depths_mask, force_single_electrode_configuration=True,
                               domain_radius=50, processes=12, mesh_source="./meshfiles", mesh_generator="netgen")

# Save results
output_folder = "./Output" # path to output folder

# rm.SaveResults(model_parameters, logs, output_folder=output_folder, 
# #                plot_layout=[["B5.7A0.4M", "M5.7N0.4A"], ["M0.4A5.7B", "A0.4N5.7M"], ["M2.0A0.5B", "A2.0M0.5N"] ,["B0.5A2.0M", "N0.5M2.0A"]],
#               plot_layout=[["B5.7A0.4M", "B4.48A1.62M"], ["M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]],
#                depth_lim=[0,25], rad_lim=[-1,1], res_lim=[0,30], aspect_ratio=0.75, at_nan="continue")

rm.SaveResults(model_parameters,
                  logs,
                  output_folder="./Output", 
                  #measurements_to_save=["B5.7A0.4M", "B4.48A1.62M"],
                  plot_layout=[["B5.7A0.4M", "B4.48A1.62M"], ["M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]],
                  plot_depth_lim=[0, 25],
                  plot_aspect_ratio=1.25,
                  model_rad_lim=[-1, 1],
                  model_res_lim=[0, 30],
                  logs_res_lim=[0, 30],
                  logs_at_nan="break",
                  logs_interpolation_factor=5,
                  logs_colours=[["blue", "red"], ["orange", "green", "grey", "purple"]])