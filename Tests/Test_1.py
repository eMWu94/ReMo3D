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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'remo3d_test_1'))

from remo3d import Model
import numpy as np

# Specify input data
tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"] # logging tools
formation_model_file = "./Input/Formation.txt" # path to file with formation parameters
borehole_model_file = "./Input/Borehole.txt" # path to file with borehole parameters
measurement_depths = np.arange(0, 25.1, 0.1) # measurement points

# Create model and simulate logs
model = Model.compute_synthetic_logs(tools, measurement_depths, formation_model_file, borehole_model_file, borehole_geometry_type='diameter', dip=0,
                                     cpu_workers=11, gpu_workers=0, mesh_generator="netgen", domain_radius=25, batch_size=5)

# # Save results
# model.save_results(output_folder="./Output",
#                plot_layout=[["B5.7A0.4M", "B4.48A1.62M"], ["M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]],
#                plot_depth_lim=[0, 25], plot_aspect_ratio=1.25,
#                model_rad_lim=[-1, 1], model_res_lim=[0, 20],
#                logs_colours = [["red", "blue"], ["green", "orange", "purple", "deepskyblue"]],
#                logs_res_lim=[0, 30], logs_at_nan="break")

