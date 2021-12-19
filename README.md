# ReMo3D
ReMo3D is a Python package that allows to generate synthetic normal and lateral logs for complex 2D and 3d models. The package is built around a high-performance multiphysics finite element software Netgen/NGSolve and supports distributed-memory parallel computation.

## Installation

The following software is required to use ReMo3D package:
- MPI implementation (recommended Mpich for Linux and Microsoft MPI for Windows),
- Gmsh,
- Netgen/NGSolve,
- Python 3.7 or above.

After all prerequisites are succesufully installed run one of the following commands to install ReMo3D:

Linux:
```
pip3 install git+https://github.com/eMWu94/ReMo3D.git#egg=remo3d
```

Windows:
```
pip install git+https://github.com/eMWu94/ReMo3D.git#egg=remo3d
```

The package was tested on Ubuntu 18.04, Ubuntu 20.04 and Windows 10 Pro.
