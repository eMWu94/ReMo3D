from mpi4py import MPI

from ngsolve import *
from ngsolve import ngsglobals

from netgen.read_gmsh import ReadGmsh

import gmsh

import numpy as np
import scipy.interpolate as spi

def SelectDataRange(borehole_geometry, formation_parameters, dip, mud_resistivity, simulation_depth, domain_radius, active_geometry_window=0.99):

    ### Borehole geometry
    ## Select data relevant to construct model geometry within simulation domain
    if np.shape(borehole_geometry)[0]==2:
        local_borehole_geometry = borehole_geometry.copy()
    else:
        point_within_domain_mask = np.abs(borehole_geometry[:,0]-simulation_depth) < domain_radius
        relevant_points_mask = np.convolve(point_within_domain_mask, np.array([True, True, True]), mode="same")
        local_borehole_geometry = borehole_geometry[relevant_points_mask,:]
    local_borehole_geometry[:,0] -= simulation_depth

    ## Add additional points if borehole geometry is too sparse
    interpolated_depths = local_borehole_geometry[0,0]
    for i in range(1, np.shape(local_borehole_geometry)[0]):
        distance = local_borehole_geometry[i,0] - local_borehole_geometry[i-1,0]
        if distance > 0.15:
            additional_points = np.linspace(local_borehole_geometry[i-1,0], local_borehole_geometry[i,0], np.max([3, int(distance*10+1)]))
            interpolated_depths = np.hstack([interpolated_depths, additional_points[1:]])
        else:
            interpolated_depths = np.hstack([interpolated_depths, local_borehole_geometry[i,0]])

    if np.shape(interpolated_depths)[0] > np.shape(local_borehole_geometry)[0]:
        interpolation = spi.interp1d(local_borehole_geometry[:,0], local_borehole_geometry[:,1], kind='linear')
        local_borehole_geometry = np.vstack([interpolated_depths, interpolation(interpolated_depths)]).T

    ## Adjust top point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(np.abs(local_borehole_geometry[0, 0]), domain_radius):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif np.abs(local_borehole_geometry[0, 0]) < domain_radius:
        local_borehole_geometry = np.vstack((np.array([-domain_radius, local_borehole_geometry[0, 1]]), local_borehole_geometry))
    # else if point is outside the domain move it on the domain boundary
    elif np.abs(local_borehole_geometry[0, 0]) > domain_radius:
        a = local_borehole_geometry[0,0]-(-domain_radius)
        b = -domain_radius-local_borehole_geometry[1,0]
        local_borehole_geometry[0,:] = [-domain_radius, (b*local_borehole_geometry[0,1]+a*local_borehole_geometry[1,1])/(a+b)]

    ## Adjust bottom point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(np.abs(local_borehole_geometry[-1, 0]), domain_radius):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif np.abs(local_borehole_geometry[-1, 0]) < domain_radius:
        local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([domain_radius, local_borehole_geometry[-1, 1]])))
    # else if point is outside the domain move it on the domain boundary
    elif np.abs(local_borehole_geometry[-1, 0]) > domain_radius:
        a = local_borehole_geometry[-1,0]-domain_radius
        b = domain_radius-local_borehole_geometry[-2,0]
        local_borehole_geometry[-1,:] = [domain_radius, (b*local_borehole_geometry[-1,1]+a*local_borehole_geometry[-2,1])/(a+b)]

    ### Formation geometry
    ## Compute active geometry radius 
    # Active geometry prevents occurence of thin layers and small wedges at the very edge of simulation by ignoring occurence of small elements of model at the very edge of simulation domain
    active_geometry_radius = domain_radius * active_geometry_window 

    ## Select data relevant to construct model geometry within simulation domain
    # Select layers that are present within active geometry area
    local_formation_parameters = formation_parameters.copy()
    local_formation_parameters[:,:2] -= simulation_depth

    a = np.tan(dip)
    b = 1
    c = local_formation_parameters[:,:2]
    
    d = np.abs(c)/(a**2 + b**2)**(1/2) # the equation is |a*x0+b*y0+c|/(a^2+b^2)^(1/2), since (x0, y0) = (0, 0) the form of equation is simplified
    
    relavant_layers = local_formation_parameters[np.any(d<active_geometry_radius, axis=1),:]

    # Remove undisturbed zones that are not present within active geometry area
    layers_to_check = ~np.isnan(relavant_layers[:,2])

    x_points = np.repeat(np.atleast_2d(relavant_layers[layers_to_check, 2]).T, 4, axis=1)
    x_points[:, :2] *= -1

    y_points = a*x_points + np.hstack([relavant_layers[layers_to_check,:2], relavant_layers[layers_to_check,:2]])
    d = (x_points**2 + y_points**2)**(1/2)
    zones_within = np.any(d<active_geometry_radius, axis=1)

    zones_to_remove = layers_to_check.copy()
    zones_to_remove[layers_to_check] = ~zones_within

    local_formation_model = relavant_layers.copy()
    local_formation_model[zones_to_remove, 4] = local_formation_model[zones_to_remove,3]
    local_formation_model[zones_to_remove, 2:4] = np.nan

    ## Adjust bottom and top boundary to the domain
    abs_c = domain_radius*(a**2 + b**2)**(1/2)*1.01 # value that will stretch bottom and top layers slightly outside simulation domain. Calculated from a simplified form of the equation |a*x0+b*y0+c|/(a^2+b^2)^(1/2)
    
    # Adjust top point to the domain
    if local_formation_model[0, 0] > -abs_c:
        local_formation_model[0, 0] = -abs_c
    
    # Adjust bottom point to the domain
    if local_formation_model[-1, 1] < abs_c:
        local_formation_model[-1, 1] = abs_c

    ## Set formation geometry
    local_formation_geometry = local_formation_model[:,:3]

    ## Set formation conductivity distribution
    formation_resistivity_distribution = np.ndarray.flatten(local_formation_model[:,3:5])
    formation_resistivity_distribution = formation_resistivity_distribution[~np.isnan(formation_resistivity_distribution)]

    local_conductivity_distribution = CoefficientFunction([1/mud_resistivity] + list(1/formation_resistivity_distribution)) #Conductivity within different parts of model

    return (local_formation_geometry, local_borehole_geometry, local_conductivity_distribution)

def ConstructModel(domain_radius, tool_geometry, source_terms, formation_geometry, dip, borehole_geometry, rank):

    ### GMSH test
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("model_"+str(rank))

    ## Points at borehole axis (bottom -> top)
    gmsh.model.occ.addPoint(0, 0, domain_radius, tag=1)
    for i in range(2,5):
        gmsh.model.occ.addPoint(0, 0, np.flip(tool_geometry)[i-2], tag=i)
    gmsh.model.occ.addPoint(0, 0, -domain_radius, tag=5)
    points_at_borehole_axis = [1,2,3,4,5]
    index_0D = 6

    ## Points at borehole boundary (top -> bottom)
    points_at_borehole_boundary = []
    for i in range(0, np.shape(borehole_geometry)[0]):
        gmsh.model.occ.addPoint(borehole_geometry[i,1], 0, borehole_geometry[i,0], tag=index_0D)
        points_at_borehole_boundary.append(index_0D)
        index_0D += 1

    # Add 2D borehole template
    borehole_points = points_at_borehole_axis + points_at_borehole_boundary
    index_1D = 1
    first_point = index_1D
    for i in range(len(borehole_points)-1):
        gmsh.model.occ.addLine(borehole_points[i], borehole_points[i+1], index_1D)
        index_1D += 1
    gmsh.model.occ.addLine(borehole_points[-1], borehole_points[0], index_1D)
    last_point = index_1D
    index_1D += 1

    borehole_loop = gmsh.model.occ.addCurveLoop(list(range(first_point, last_point+1)), 1)
    borehole_surface = gmsh.model.occ.addPlaneSurface([1], 1)

    ## Domain
    gmsh.model.occ.addSphere(0, 0, 0, domain_radius, tag=1, angle3 = np.pi)

    ## Add borehole template
    borehole_template = gmsh.model.occ.revolve([(2,1)], 0, 0, 0, 0, 0, 1, np.pi) # tag (3,2)

    # Split domain template into borehole and formation
    borehole = gmsh.model.occ.intersect([(3,2)], [(3,1)], tag = 3, removeObject= False, removeTool = False)
    formation = gmsh.model.occ.cut([(3,1)], [(3,2)], tag = 4)

    ## Formation
    boundaries_z = np.sort(np.unique(formation_geometry[:,:2]))
    pg_index_list = [3]
    index_3d = 5
    if np.shape(boundaries_z)[0]==2:
        if np.isnan(formation_geometry[0, 2]):
            pg_index_list.append(4)
            pass
        else:
            gmsh.model.occ.addCylinder(0, 0, -domain_radius*1.1, 0, 0, 2.2*domain_radius, formation_geometry[0, 2], tag = 1, angle = np.pi)
            gmsh.model.occ.intersect([(3, 4)], [(3,1)], removeObject= False, removeTool = False, tag=5)
            gmsh.model.occ.cut([(3, index_3d)], [(3,1)], tag=6)
            gmsh.model.occ.remove([(3,4)], recursive=True)
            pg_index_list.append(5)
            pg_index_list.append(6)
    else:
        for i in range(1, np.shape(boundaries_z)[0]):
            layer_height = (boundaries_z[i]-boundaries_z[i-1])*np.cos(dip) # true height of the dipping layer
            layer_middle = (boundaries_z[i]+boundaries_z[i-1])/2 # depth of the middle of the dipping layer
            if np.isnan(formation_geometry[i-1, 2]):
                box = gmsh.model.occ.addBox(-domain_radius*50, -domain_radius*50, layer_middle-layer_height/2, 100*domain_radius, 100*domain_radius, layer_height, tag = index_3d)
                gmsh.model.occ.rotate([(3, index_3d)], 0, 0, layer_middle, 0, 1, 0, dip)
                disc = gmsh.model.occ.intersect([(3, index_3d)], [(3, 4)], removeTool=False)
                pg_index_list.append(index_3d)
                index_3d += 1
            else:
                box = gmsh.model.occ.addBox(-domain_radius*50, -domain_radius*50, layer_middle-layer_height/2, 100*domain_radius, 100*domain_radius, layer_height, tag = index_3d)
                gmsh.model.occ.rotate([(3, index_3d)], 0, 0, layer_middle, 0, 1, 0, dip)
                disc = gmsh.model.occ.intersect([(3, index_3d)], [(3, 4)], removeTool=False)
                gmsh.model.occ.addCylinder(0, 0, -domain_radius*1.1, 0, 0, 2.2*domain_radius, formation_geometry[i-1, 2], tag = 1, angle = np.pi)
                gmsh.model.occ.intersect([(3, index_3d)], [(3,1)], removeObject= False, removeTool = False, tag=index_3d+1)
                gmsh.model.occ.cut([(3, index_3d)], [(3,1)], tag=index_3d+2)
                pg_index_list.append(index_3d+1)
                pg_index_list.append(index_3d+2)
                index_3d += 3
        gmsh.model.occ.remove([(3,4)], recursive=True)

    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.field.add("MathEval", 1) # Horizontal, linear meshsize field
    gmsh.model.mesh.field.setString(1, "F", "(x^2 + y^2)^0.5 + 0.1")

    i = 2
    for electrode_position in tool_geometry[source_terms != 0]:
        gmsh.model.mesh.field.add("MathEval", i) # Distance from the electrode
        gmsh.model.mesh.field.setString(i, "F","(x^2 + y^2 + (z+({}))^2)^0.5".format(electrode_position))

        gmsh.model.mesh.field.add("MathEval", i+1) # Meshsize field modification
        gmsh.model.mesh.field.setString(i+1, "F", "((F{}^2)/2) + 0.01".format(i))

        i += 2

    if np.shape(tool_geometry[source_terms != 0])[0]==1:
        gmsh.model.mesh.field.add("Min", 4)
        gmsh.model.mesh.field.setNumbers(4, "FieldsList", [1, 3])
        gmsh.model.mesh.field.setAsBackgroundMesh(4)
    else:
        gmsh.model.mesh.field.add("Min", 6)
        gmsh.model.mesh.field.setNumbers(6, "FieldsList", [1, 3, 5])

        gmsh.model.mesh.field.setAsBackgroundMesh(6)

    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(3)

    dimention, surfaces = list(zip(*gmsh.model.occ.getEntities(2)))
    dirichlet_boundaries = []
    neumann_boundaries = []
    for i in range(len(surfaces)):
        nodes = gmsh.model.mesh.getNodes(dim=2, tag=surfaces[i], includeBoundary = True)
        coordinates = nodes[1].reshape((int(np.shape(nodes[1])[0]/3),3))
        R = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2 + coordinates[:,2]**2)
        if np.allclose(R, domain_radius):
            dirichlet_boundaries.append(surfaces[i])
        else:
            neumann_boundaries.append(surfaces[i])

    i = 1
    for index in pg_index_list:
        volume = gmsh.model.addPhysicalGroup(3, [index], index)
        gmsh.model.setPhysicalName(3, volume, "vol_"+str(i))
        i += 1

    db = gmsh.model.addPhysicalGroup(2, dirichlet_boundaries, 1)
    gmsh.model.setPhysicalName(2, db , "dirichlet_boundary")

    nb = gmsh.model.addPhysicalGroup(2, neumann_boundaries, 2)
    gmsh.model.setPhysicalName(2, nb , "neumann_boundary")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write("./tmp/fm_"+str(rank)+".msh")
    gmsh.finalize()

    mesh = ReadGmsh("./tmp/fm_"+str(rank)+".msh")
    mesh = Mesh(mesh)
    
    return mesh, dirichlet_boundaries

def AddPointSource(f, x, y, z, fac):
    spc = f.space
    mp = spc.mesh(x,y,z)
    ei = ElementId(VOL, mp.nr)
    fel = spc.GetFE(ei)
    dnums = spc.GetDofNrs(ei)
    shape = fel.CalcShape(*mp.pnt)
    for d,s in zip(dnums, shape):
        f.vec[d] += fac*s

def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundaries, preconditioner, condense):
    
    fes = H1(mesh, order=3, dirichlet='dirichlet_boundary', autoupdate=True)

    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = BilinearForm(fes, symmetric=False, condense=condense)
    a += grad(u)*grad(v)*sigma*dx
    
    f = LinearForm(fes)
    f.Assemble()

    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource (f, 0.0, 0.0, tool_geometry[l], source_terms[l])
    c = Preconditioner(a, preconditioner)
    a.Assemble()

    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, c.mat, maxsteps=1000)
    
    if condense==True:
        f.vec.data += a.harmonic_extension_trans * f.vec

    gfu.vec.data = inv * f.vec

    if condense==True:
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec

    return(fes, gfu)

# Supress Netgen teminal output during mesh creation process
ngsglobals.msg_level = 0

# Connect to main process
try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
except:
    raise ValueError('The worker could not connect to main process')

## Collect data
# Collect information about shapes of broadcasted arrays
arrays_shape = list()
arrays_shape = comm.bcast(arrays_shape, root=0)

# Prepare empty variables
formation_parameters = np.empty(arrays_shape[0], dtype='float')
borehole_geometry = np.empty(arrays_shape[1], dtype='float')
mud_resistivities = np.empty(arrays_shape[2], dtype='float')

tools_parameters = dict()
simulation_depths = dict()
domain_radius = float()
preconditioner = str()
condense = bool()
dip = float()

# Fill variables with data
comm.Bcast([formation_parameters, MPI.FLOAT], root=0)
comm.Bcast([borehole_geometry, MPI.FLOAT], root=0)
comm.Bcast([mud_resistivities, MPI.FLOAT], root=0)

tools_parameters = comm.bcast(tools_parameters, root=0)
simulation_depths = comm.bcast(simulation_depths, root=0)
domain_radius =  comm.bcast(domain_radius, root=0)
preconditioner = comm.bcast(preconditioner, root=0)
condense = comm.bcast(condense, root=0)
dip = comm.bcast(dip, root=0)

## Wait for all workers to receive data
comm.barrier()

## Ask for tasks until receving stop sentinel
results = list()
for task in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
    try:
        tool = list(tools_parameters.keys())[task[1]]
        depth = task[0]
        tool_geometry = tools_parameters[tool][0,:3]
        source_terms = tools_parameters[tool][1,:3]
        geometric_factor = tools_parameters[tool][0,3]
        # Carve out suitable range of data
        local_formation_geometry, local_borehole_geometry, sigma = SelectDataRange(borehole_geometry, formation_parameters, dip, mud_resistivities[depth], simulation_depths[tool][depth], domain_radius)
        # Create geometry and mesh
        mesh, dirichlet_boundaries = ConstructModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, dip, local_borehole_geometry, rank)
        # Solve BVP
        fes, gfu = SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundaries, preconditioner, condense=True)
        measuring_electodes = tool_geometry[source_terms==0]
        if np.shape(measuring_electodes)[0] == 2:
            result = abs(geometric_factor * (gfu(mesh(0.0, 0.0, measuring_electodes[1]))-gfu(mesh(0.0, 0.0, measuring_electodes[0]))))/2 # division by two because only halfsphere is present within the model
        elif np.shape(measuring_electodes)[0] == 1:
            result = abs(geometric_factor * gfu(mesh(0.0, 0.0, measuring_electodes[0])))/2 # division by two because only halfsphere is present within the model
        results.append([task[0], task[1], result])
    except:
        results.append([task[0], task[1], np.nan])

## Report results to master process
comm.gather(sendobj=results, root=0)

## Shutdown
comm.Disconnect()
