"""
   Example 1
   The script presents a basic use of the package.
   Only required parameters are used.

   How to run:
   mpiexec python3 Example_01.py
"""
import remo3d as rm
import numpy as np

# Specify input data
tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"] # logging tools
formation_model_file = "./Input/Formation.txt" # path to file with formation parameters
borehole_model_file = "./Input/Borehole.txt" # path to file with borehole parameters
measurement_depths = np.arange(0, 25.1, 0.1) # measurement points

# Set tools parameters
tools_parameters = rm.SetToolsParameters(tools)

# Set model parameters
model_parameters = rm.SetModelParameters(formation_model_file, borehole_model_file)

# Compute synthetic logs
logs = rm.ComputeSyntheticLogs(tools_parameters, model_parameters, measurement_depths)

# Save results
output_folder = "./Output" # path to output folder
rm.SaveResults(model_parameters, logs, output_folder=output_folder)

