"""
   Example 2
   The script presents a more advanced use of the package.
   Required and optional parameters are used.

   How to run:
   mpiexec python3 Test_2.py
"""
import sys
import os
import datetime

# Add the parent directory of both subdirectories to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'remo3d'))

from remo3d import Model
import numpy as np

# Specify input data
tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"] # logging tools
formation_model_file = "./Input/Formation_J4.txt" # path to file with formation parameters
borehole_model_file = "./Input/Borehole_J4.txt" # path to file with borehole parameters
measurement_depths = np.arange(1562, 1477.1, 0.25) # measurement points

# Create model and simulate logs
### Start the clock
start_time = datetime.datetime.now()

model = Model(tools, force_single_electrode_configuration=True)
model.initialize_workers(cpu_workers=11, gpu_workers=0)

model.set_model_parameters(formation_model_file, borehole_model_file, borehole_geometry_type='diameter', dip=0)

for i in range(10):
    model.simulate_logs(measurement_depths, domain_radius=25, batch_size=10, mesh_generator="netgen")

model.shutdown_workers()

### Report time of computation
print('\nTotal time: ', datetime.datetime.now() - start_time)

# Save results
model.save_results(output_folder="./Output",
               plot_layout=[["B5.7A0.4M", "B4.48A1.62M"], ["M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]],
               plot_depth_lim=[0, 25], plot_aspect_ratio=1.25,
               model_rad_lim=[-1, 1], model_res_lim=[0, 20],
               logs_colours = [["red", "blue"], ["green", "orange", "purple", "deepskyblue"]],
               logs_res_lim=[0, 30], logs_at_nan="break")

