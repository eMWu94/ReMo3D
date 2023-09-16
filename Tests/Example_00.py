"""
   Example 0
   The script allows to print descriptions of all main functions from the package.

   How to run:
   mpiexec python3 Example_00.py
"""

#import remo3d as rm
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import src.remo3d as rm

help(rm.SetToolsParameters)

help(rm.SetModelParameters)

help(rm.ComputeSyntheticLogs)

help(rm.SaveResults)
