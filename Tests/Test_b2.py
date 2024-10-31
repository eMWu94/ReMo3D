"""
   Example 2
   The script presents a more advanced use of the package.
   Required and optional parameters are used.

   How to run:
   mpiexec python3 Test_b2.py
"""
import sys
import os

# Add the parent directory of both subdirectories to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'remo3d_test_2'))

from remo3d import Model
import numpy as np

# Specify input data
tools = ["A0.4M6.0N", "A1.62M6.0N"] # logging tools
formation_model_file = "./Input/Formation_test.txt" # path to file with formation parameters
borehole_model_file = "./Input/Borehole_test.txt" # path to file with borehole parameters
measurement_depths = np.arange(3755, 3790.1, 0.25) # measurement points

# Create model and simulate logs
model = Model.compute_synthetic_logs(tools, measurement_depths, formation_model_file, borehole_model_file, borehole_geometry_type='diameter', dip=0,
                                     cpu_workers=11, gpu_workers=0, mesh_generator="netgen", domain_radius=25, batch_size=10)

# Save results
model.save_results(output_folder="./Output",
               plot_layout=[["A0.4M6.0N", "A1.62M6.0N"]],
               plot_depth_lim=[3755, 3790], plot_aspect_ratio=1.25,
               model_rad_lim=[-1, 1], model_res_lim=[0, 25],
               logs_colours = [["green", "orange"]],
               logs_res_lim=[0, 25], logs_at_nan="break")

