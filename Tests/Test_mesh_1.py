"""
   Example 2
   The script presents a more advanced use of the package.
   Required and optional parameters are used.

   How to run:
   mpiexec python3 Test_mesh_1.py
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
model = Model(tools, force_single_electrode_configuration=True)

model.set_model_parameters(formation_model_file, borehole_model_file, borehole_geometry_type="diameter", dip=0)

model.initialize_meshing_workers(workers=12)

model.generate_meshes(measurement_depths, domain_radius=25, batch_size=10, mesh_generator="netgen")

model.shutdown_workers()

# model.initialize_simulation_workers(cpu_workers=12, gpu_workers=0)

# model.simulate_logs(measurement_depths, domain_radius=25, batch_size=10, mesh_generator="netgen")

# model.shutdown_workers()

print(model.meshes)
print(model.logs)