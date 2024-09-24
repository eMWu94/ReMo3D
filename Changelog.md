# 1.4.0
 -
 - Corrections within docstrings. 

# 1.3.0
 - Conversion of the main part of the code to classes.
 - Spliting initialization, log computation and shutdown of workers into separate functions (done with the intention to better adapt the code to the purpuse of use within inversion algorithms where simulations are performed multiple times within a procedure).
 
# 1.2.0
 - Conversion of data into ngsolve format is now done within the worker, not within netgen and gmsh functions.
 - Output format of netgen and gmsh functions is now standardized.
 
# 1.1.0
 - Package restructuring - splitting remo3d.py file into remo3d.py, gmsh_functions.py, netgen_functions.py and ngsolve_functions.py files.
 - If all simulated tools are in one current electrode configuration all measurements, where the current electrode is located at the same point are computed simultaneously in a a single mesh generation and simulation procedure to speed up the process.
 - Addition of the batch mode, where multiple adjacent measurement points are joined into a single mesh generation and simulation procedure to speed up the process.
 - Addition of Changelog.md file.

# 1.0.0
 - Basic version of the package as described in the publication Wilkosz, M. (2022). ReMo3D – an open-source Python package for 2D and 3D simulation of normal and lateral resistivity logs. Geology, Geophysics and Environment, 48(2), 195–211. https://doi.org/10.7494/geol.2022.48.2.195


