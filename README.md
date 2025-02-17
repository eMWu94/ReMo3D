# ReMo3D
ReMo3D is a Python package that allows to generate synthetic normal and lateral logs for complex 2D and 3D models. The package is built around a finite element mesh generator Gmsh and a high-performance multiphysics finite element software Netgen/NGSolve and supports distributed-memory parallel computation.

## Installation
The following software is required to use ReMo3D package:
- MPI implementation (recommended Mpich for Linux and Microsoft MPI for Windows),
- Gmsh,
- Netgen/NGSolve,
- Python 3.7 or above.

After all prerequisites are succesufully installed run one of the following commands to install ReMo3D:

Linux:
```
pip3 install git+https://github.com/eMWu94/ReMo3D.git
```

Windows:
```
pip install git+https://github.com/eMWu94/ReMo3D.git
```

The package was tested on Ubuntu 18.04, Ubuntu 20.04, Windows 10 Pro and Windows 11 Pro.

## Expected computation times
On a default settings simulation of 100 measurement points of a single logging tool on a PC equipped with AMD Ryzen 2600 CPU takes around 15-30 seconds in case of a 2D model of moderate complexity and around 15-30 minutes in case of a 3D model of moderate complexity.

## Funding
The  research  was  funded  by  the  National  Sci-ence  Centre,  Poland,  grant  number  2020/37/N/ST10/03230.
