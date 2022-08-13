from setuptools import setup

setup(
    name='remo3d',
    version='1.0.0',
    author='Michał Wilkosz',
    author_email='michal.m.wilkosz@gmail.com',
    description='ReMo3D is a Python package that allows to generate synthetic normal and lateral logs for 2D and 3D models',
    py_modules=['remo3d', 'worker'],
    package_dir={'': 'src'},
    install_requires= ['numpy', 'scipy', 'matplotlib', 'mpi4py', 'gmsh'],
    license='LGPL-2.1 License'
)
