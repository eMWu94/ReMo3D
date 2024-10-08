# -*- coding: utf-8 -*-

import numpy as np
import ngsolve as ngs

#ngs.ngsglobals.msg_level=0

# Ngsolve funtions

def AddPointSource(f, position, fac, model_dimensionality):
        spc = f.space
        if model_dimensionality==2:
            mp = spc.mesh(0,position)
        elif model_dimensionality==3:
            mp = spc.mesh(0,0,position)
        ei = ngs.ElementId(ngs.VOL, mp.nr)
        fel = spc.GetFE(ei)
        dnums = spc.GetDofNrs(ei)
        shape = fel.CalcShape(*mp.pnt)
        for d,s in zip(dnums, shape):
            f.vec[d] += fac*s

def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense):

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

    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)

    c = ngs.Preconditioner(a, preconditioner)
    a.Assemble()
    gfu = ngs.GridFunction(fes)
    
    inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
    gfu.vec.data = inv * f.vec

    if condense==True:
        f.vec.data += a.harmonic_extension_trans * f.vec
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec
        
    return fes, gfu