"""
   Example 2
   The script presents a more advanced use of the package.
   Required and optional parameters are used.

   How to run:
   mpiexec python3 Test_1.py
"""
import sys
import os

# Add the parent directory of both subdirectories to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'remo3d_gpu_3'))

import remo3d as rm
import numpy as np

# Specify input data
tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"] # logging tools
formation_model_file = "./Input/Formation_3.txt" # path to file with formation parameters
borehole_model_file = "./Input/Borehole_2.txt" # path to file with borehole parameters
measurement_depths = np.arange(0, 50.1, 0.25) # measurement points

# Set tools parameters
tools_parameters = rm.SetToolsParameters(tools)

# Set model parameters
model_parameters = rm.SetModelParameters(formation_model_file, borehole_model_file,
                                         borehole_geometry='diameter', dip=0)

# # Compute synthetic logs
logs = rm.ComputeSyntheticLogs(tools_parameters, model_parameters, measurement_depths,
                               domain_radius=25, cpu_processes=12, mesh_generator="netgen", condense=True,
                               force_single_electrode_configuration=True, batch_size=10)

# Save results
output_folder = "./Output" # path to output folder
rm.SaveResults(model_parameters, logs, output_folder=output_folder,
               plot_layout=[["B5.7A0.4M", "B4.48A1.62M"], ["M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]],
               plot_depth_lim=[0, 25], plot_aspect_ratio=1.25,
               model_rad_lim=[-1, 1], model_res_lim=[0, 20],
               logs_colours = [["red", "blue"], ["green", "orange", "purple", "deepskyblue"]],
               logs_res_lim=[0, 30], logs_at_nan="break")

