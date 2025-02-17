from setuptools import setup, find_packages

setup(
    name='remo3d',
    version='1.4.0',
    author='Michał Wilkosz',
    author_email='michal.m.wilkosz@gmail.com',
    description='ReMo3D is a Python package that allows to generate synthetic normal and lateral logs for 2D and 3D models',
    py_modules=['remo3d', 'worker'],
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires= ['numpy', 'scipy', 'matplotlib', 'mpi4py', 'gmsh'],
    license='LGPL-2.1 License'
)
