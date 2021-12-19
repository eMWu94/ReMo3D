from setuptools import setup

setup(
    name='remo3d',
    version='1.0.0',
    author='Michał Wilkosz',
    author_email='michal.m.wilkosz@gmail.com',
    description='ReMo3D is a Python package that allows to generate synthetic normal and lateral logs for complex 2D and 3D models',
    py_modules=['remo3d', 'worker_2D', 'worker_3D'],
    package_dir={'': 'src'},
    install_requires= ['numpy', 'matplotlib', 'mpi4py'],
    license='LGPL-2.1 License'
)
