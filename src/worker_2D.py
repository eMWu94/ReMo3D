from mpi4py import MPI

from ngsolve import *
from ngsolve import ngsglobals
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters, meshsize

from netgen.read_gmsh import ReadGmsh

import gmsh

import numpy as np

def SelectDataRange(borehole_geometry, formation_parameters, mud_resistivity, simulation_depth, domain_radius, active_geometry_window=0.99):

    def domain_line_intersection (p1, p2, radius):
        x_1, y_1 = p1[1], p1[0]
        x_2, y_2 = p2[1], p2[0]
        d_x = x_2 - x_1
        d_y = y_2 - y_1
        d_r = np.sqrt(d_x**2 + d_y**2)
        D = x_1*y_2 - x_2*y_1
        delta = radius**2*d_r**2-D**2
        for sign in (-1, 1):
            x = (D*d_y+sign*np.sign(d_y)*d_x*np.sqrt(delta))/d_r**2
            y = (-D*d_x+sign*np.abs(d_y)*np.sqrt(delta))/d_r**2
            p = np.array([y, x])
            if np.dot(p1-p2, p1-p)> 0 and np.dot(p1-p2, p1-p)<np.dot(p1-p2, p1-p2):
                return np.array([y, x])
    
    ### Borehole geometry
    ## Select data relevant to construct model geometry within simulation domain
    if np.shape(borehole_geometry)[0]==2:
        local_borehole_geometry = borehole_geometry.copy()
    else:
        point_within_domain_mask = (borehole_geometry[:,0]-simulation_depth)**2+borehole_geometry[:,1]**2 < domain_radius**2
        relevant_points_mask = np.convolve(point_within_domain_mask, np.array([True, True, True]), mode="same")
        local_borehole_geometry = borehole_geometry[relevant_points_mask,:].copy()
    local_borehole_geometry[:,0] -= simulation_depth

    ## Adjust top point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2 , domain_radius**2):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  < domain_radius**2:
        omega = np.arccos(local_borehole_geometry[0, 1]/domain_radius)
        local_borehole_geometry = np.vstack((np.array([-np.sin(omega)*domain_radius, local_borehole_geometry[0, 1]]), local_borehole_geometry))
    # else if point is outside the domain move it on the domain boundary
    elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  > domain_radius**2:
        local_borehole_geometry[0,:] = domain_line_intersection(local_borehole_geometry[0, :], local_borehole_geometry[1, :], domain_radius)
    
    ## Adjust bottom point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2 , domain_radius**2):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2  < domain_radius**2:
        omega = np.arccos(local_borehole_geometry[-1, 1]/domain_radius)
        local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([np.sin(omega)*domain_radius, local_borehole_geometry[-1, 1]])))
    # else if point is outside the domain move it on the domain boundary
    elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2  > domain_radius**2:
        local_borehole_geometry[-1,:] = domain_line_intersection(local_borehole_geometry[-1, :], local_borehole_geometry[-2, :], domain_radius)

    ### Formation geometry
    ## Compute active geometry radius 
    # Active geometry prevents occurence of thin layers and small wedges at the very edge of simulation by ignoring occurence of small elements of model at the very edge of simulation domain
    active_geometry_radius = domain_radius * active_geometry_window 

    ## Select data relevant to construct model geometry within simulation domain
    # Select layers that are present within active geometry area
    local_formation_parameters = formation_parameters.copy()
    local_formation_parameters[:,:2] -= simulation_depth

    relavant_layers = local_formation_parameters[np.any(np.abs(local_formation_parameters[:,:2])<active_geometry_radius, axis=1),:]

    # Remove undisturbed zones that are not present within active geometry area
    layers_to_check = ~np.isnan(relavant_layers[:,2])

    x_points = np.repeat(np.atleast_2d(relavant_layers[layers_to_check, 2]).T, 2, axis=1)

    y_points = relavant_layers[layers_to_check,:2]

    d = (x_points**2 + y_points**2)**(1/2)
    zones_within = np.any(d<active_geometry_radius, axis=1)

    zones_to_remove = layers_to_check.copy()
    zones_to_remove[layers_to_check] = ~zones_within

    local_formation_model = relavant_layers.copy()
    local_formation_model[zones_to_remove, 4] = local_formation_model[zones_to_remove,3]
    local_formation_model[zones_to_remove, 2:4] = np.nan

    ## Adjust bottom and top boundary to the domain
    abs_c = domain_radius*1.01 # value that will stretch bottom and top layers slightly outside simulation domain

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

def ConstructModel(domain_radius, tool_geometry, source_terms, formation_geometry, borehole_geometry, rank):

    ### GMSH test
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("model_"+str(rank))

    ## Add central point (removed from final geometry)
    gmsh.model.occ.addPoint(0, 0, 0, tag=1)

    ## Points at borehole axis (bottom -> top)
    gmsh.model.occ.addPoint(0, domain_radius, 0, tag=2)
    for i in range(3,6):
        gmsh.model.occ.addPoint(0, np.flip(tool_geometry)[i-3], 0, tag=i)
    gmsh.model.occ.addPoint(0, -domain_radius, 0, tag=6)
    points_at_borehole_axis = [2,3,4,5,6]
    index_0D = 7

    ## Points at borehole boundary (top -> bottom)
    points_at_borehole_boundary = []
    for i in range(0, np.shape(borehole_geometry)[0]):
        gmsh.model.occ.addPoint(borehole_geometry[i,1], borehole_geometry[i,0], 0, tag=index_0D)
        points_at_borehole_boundary.append(index_0D)
        index_0D += 1

    # Add 2D borehole template
    index_1D = 1

    ## Add lines on borehole axis
    lines_at_borehole_axis = []
    for i in range(len(points_at_borehole_axis)-1):
        gmsh.model.occ.addLine(points_at_borehole_axis[i], points_at_borehole_axis[i+1], tag=index_1D)
        lines_at_borehole_axis.append(index_1D)
        index_1D += 1

    ## Add lines at borehole boundary
    lines_at_borehole_boundary = []
    for i in range(len(points_at_borehole_boundary)-1):
        gmsh.model.occ.addLine(points_at_borehole_boundary[i], points_at_borehole_boundary[i+1], tag=index_1D)
        lines_at_borehole_boundary.append(index_1D)
        index_1D += 1

    ## Add arc at top of the domain (axis -> boundary)
    gmsh.model.occ.addCircleArc(points_at_borehole_axis[-1], 1, points_at_borehole_boundary[0], tag=index_1D)
    top_arc = [index_1D]
    index_1D += 1

    ## Add arc at bottom of the domain (boundary -> axis)
    gmsh.model.occ.addCircleArc(points_at_borehole_boundary[-1], 1, points_at_borehole_axis[0], tag=index_1D)
    bottom_arc = [index_1D]
    index_1D += 1

    ## Add external domain boundary
    circle_arc = gmsh.model.occ.addCircleArc(points_at_borehole_boundary[0], 1, points_at_borehole_boundary[-1], tag=index_1D)
    domain_arc = [index_1D]
    index_1D += 1

    ## Add domain
    domain_loop = gmsh.model.occ.addCurveLoop(lines_at_borehole_boundary + domain_arc, tag=3)
    domain = gmsh.model.occ.addPlaneSurface([3], tag=3)

    ## Add borehole
    borehole_loop = gmsh.model.occ.addCurveLoop(lines_at_borehole_axis + top_arc + lines_at_borehole_boundary + bottom_arc, tag=4)
    borehole = gmsh.model.occ.addPlaneSurface([4], tag=4)

    ## Split formation into layers
    boundaries_z = np.sort(np.unique(formation_geometry[:,:2]))
    pg_index_list = [4]
    index_2D = 5
    if np.shape(boundaries_z)[0]==2:
        if np.isnan(formation_geometry[0, 2]):
            pg_index_list.append(6)
        else:
            filtration_template = gmsh.model.occ.addRectangle(0, boundaries_z[i-1], 0, formation_geometry[0, 2], boundaries_z[i]-boundaries_z[i-1], tag=1)
            filtration = gmsh.model.occ.intersect([(2, 1)], [(2,3)], removeTool = False, tag=index_2D)
            undisturbed_template = gmsh.model.occ.addRectangle(formation_geometry[0, 2], boundaries_z[i-1], 0, (domain_radius-formation_geometry[0, 2])*1.01, boundaries_z[i]-boundaries_z[i-1], tag=2)
            undisturbed = gmsh.model.occ.intersect([(2, 2)], [(2,3)], removeTool = False, tag=index_2D+1)
            pg_index_list.append(index_2D)
            pg_index_list.append(index_2D+1)
            index_2D += 2
    else:
        for i in range(1, np.shape(boundaries_z)[0]):
            if np.isnan(formation_geometry[i-1, 2]):
                layer_template = gmsh.model.occ.addRectangle(0, boundaries_z[i-1], 0, domain_radius*1.01, boundaries_z[i]-boundaries_z[i-1], tag=1)
                layer = gmsh.model.occ.intersect([(2, 1)], [(2,3)], removeTool = False, tag=index_2D)
                pg_index_list.append(index_2D)
                index_2D += 1
            else:
                filtration_template = gmsh.model.occ.addRectangle(0, boundaries_z[i-1], 0, formation_geometry[i-1, 2], boundaries_z[i]-boundaries_z[i-1], tag=1)
                filtration = gmsh.model.occ.intersect([(2, 1)], [(2,3)], removeTool = False, tag=index_2D)
                undisturbed_template = gmsh.model.occ.addRectangle(formation_geometry[i-1, 2], boundaries_z[i-1], 0, (domain_radius-formation_geometry[i-1, 2])*1.01, boundaries_z[i]-boundaries_z[i-1], tag=2)
                undisturbed = gmsh.model.occ.intersect([(2, 2)], [(2,3)], removeTool = False, tag=index_2D+1)
                pg_index_list.append(index_2D)
                pg_index_list.append(index_2D+1)
                index_2D += 2
        
        gmsh.model.occ.remove([(2,3)], recursive=True)

    gmsh.model.occ.removeAllDuplicates()        
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.field.add("MathEval", 1) # Horizontal, linear meshsize field
    gmsh.model.mesh.field.setString(1, "F", "x + 0.1")

    i = 2
    for electrode_position in tool_geometry[source_terms != 0]:
        gmsh.model.mesh.field.add("MathEval", i) # Meshsize field modification near current electrodes
        gmsh.model.mesh.field.setString(i, "F","(x^2 + (y+({}))^2)/2 + 0.01".format(electrode_position))
        i += 1

    if np.shape(tool_geometry[source_terms != 0])[0]==1:
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1,2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)
    else:
        gmsh.model.mesh.field.add("Min", 4)
        gmsh.model.mesh.field.setNumbers(4, "FieldsList", [1,2,3])
        gmsh.model.mesh.field.setAsBackgroundMesh(4)
    
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)

    dimention, lines = list(zip(*gmsh.model.occ.getEntities(1)))

    dirichlet_boundaries = []
    neumann_boundaries = []
    for i in range(len(lines)):
        nodes = gmsh.model.mesh.getNodes(dim=1, tag=lines[i], includeBoundary = True)
        coordinates = nodes[1].reshape((int(np.shape(nodes[1])[0]/3),3))
        R = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
        if np.allclose(R, domain_radius):
            dirichlet_boundaries.append(lines[i])
        else:
            neumann_boundaries.append(lines[i])

    i = 1
    for index in pg_index_list:
        surface = gmsh.model.addPhysicalGroup(2, [index], index)
        gmsh.model.setPhysicalName(2, surface, "surf_"+str(i))
        i += 1

    db = gmsh.model.addPhysicalGroup(1, dirichlet_boundaries, 1)
    gmsh.model.setPhysicalName(1, db , "dirichlet_boundary")

    nb = gmsh.model.addPhysicalGroup(1, neumann_boundaries, 2)
    gmsh.model.setPhysicalName(2, nb , "neumann_boundary")

    # Save file
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write("./tmp/fm_"+str(rank)+".msh")
    gmsh.finalize()

    # Read file
    mesh = ReadGmsh("./tmp/fm_"+str(rank)+".msh")
    mesh = Mesh(mesh)
    
    return mesh, dirichlet_boundaries

def AddPointSource(f, x, y, fac):
    spc = f.space
    mp = spc.mesh(x,y)
    ei = ElementId(VOL, mp.nr)
    fel = spc.GetFE(ei)
    dnums = spc.GetDofNrs(ei)
    shape = fel.CalcShape(*mp.pnt)
    for d,s in zip(dnums, shape):
        f.vec[d] += fac*s

def SolveBVP(mesh, sigma, tool_geometry, source_terms, preconditioner, condense):
    
    fes = H1(mesh, order=3, dirichlet='dirichlet_boundary', autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = BilinearForm(fes, symmetric=False, condense=condense)
    a += 2*np.pi*grad(u)*grad(v)*x*sigma*dx
    
    f = LinearForm(fes)
    f.Assemble()

    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource (f, 0.0, tool_geometry[l], source_terms[l])

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

# Fill variables with data
comm.Bcast([formation_parameters, MPI.FLOAT], root=0)
comm.Bcast([borehole_geometry, MPI.FLOAT], root=0)
comm.Bcast([mud_resistivities, MPI.FLOAT], root=0)

tools_parameters = comm.bcast(tools_parameters, root=0)
simulation_depths = comm.bcast(simulation_depths, root=0)
domain_radius =  comm.bcast(domain_radius, root=0)
preconditioner = comm.bcast(preconditioner, root=0)
condense = comm.bcast(condense, root=0)

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
        local_formation_geometry, local_borehole_geometry, sigma = SelectDataRange(borehole_geometry, formation_parameters, mud_resistivities[depth], simulation_depths[tool][depth], domain_radius)
        # Create geometry and mesh
        mesh, dirichlet_boundaries = ConstructModel(domain_radius, tool_geometry, source_terms, local_formation_geometry, local_borehole_geometry, rank)
        # Solve BVP
        fes, gfu = SolveBVP(mesh, sigma, tool_geometry, source_terms, preconditioner, condense)
        # Compute measured resistivity
        measuring_electodes = tool_geometry[source_terms==0]
        if np.shape(measuring_electodes)[0] == 2:
            result = abs(geometric_factor * (gfu(mesh(0.0, measuring_electodes[1]))-gfu(mesh(0.0, measuring_electodes[0]))))
        elif np.shape(measuring_electodes)[0] == 1:
            result = abs(geometric_factor * gfu(mesh(0.0, measuring_electodes[0])))
        # Append result to results
        results.append([task[0], task[1], result])
    except:
        results.append([task[0], task[1], np.nan])

## Report results to master process
comm.gather(sendobj=results, root=0)

## Shutdown
comm.Disconnect()
