# -*- coding: utf-8 -*-

import numpy as np
import ngsolve as ngs
    
ngs.ngsglobals.msg_level=0

from ngsolve_functions import AddPointSource

from ngsolve.ngscuda import *


# Ngsolve gpu funtions

def SolveBVP(mesh, sigma, tool_geometry_list, source_terms_list, offsets, dirichlet_boundary, preconditioner, condense):

    model_dimensionality = mesh.dim

    fes = ngs.H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = ngs.BilinearForm(fes, symmetric=False, condense=condense)

    if model_dimensionality==2:
        a += 2*np.pi*ngs.grad(u)*ngs.grad(v)*ngs.x*sigma*ngs.dx
    elif model_dimensionality==3:
        a += ngs.grad(u)*ngs.grad(v)*sigma*ngs.dx

    #start_time = datetime.datetime.now()  
    f = ngs.LinearForm(fes)
    f.Assemble()

    for i in range(len(tool_geometry_list)):
        tool_geometry = tool_geometry_list[i] + offsets[i]
        source_terms = source_terms_list[i]
        for l in range(np.shape(source_terms)[0]):
            if source_terms[l] != 0.0:
                AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)
    c = ngs.Preconditioner(a, preconditioner)
    a.Assemble()
    gfu = ngs.GridFunction(fes)

    adev = a.mat.CreateDeviceMatrix()
    cdev = c.mat.CreateDeviceMatrix()
    fdev = f.vec.CreateDeviceVector(copy=True)

    inv = ngs.CGSolver(adev, cdev, maxsteps=1000, printrates=False)

    gfu.vec.data = inv * fdev
        
    if condense==True:
        f.vec.data += a.harmonic_extension_trans * f.vec
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec
        
    return fes, gfu