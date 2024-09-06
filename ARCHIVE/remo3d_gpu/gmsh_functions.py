# -*- coding: utf-8 -*-

import gmsh
import numpy as np
import scipy.interpolate as spi
import netgen.meshing as msh

# GMSH functions

def SelectGmshBoreholeDataRange(borehole_geometry, dip, simulation_depth, domain_radius):
    
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
            if np.dot(p1-p2, p1-p)>0 and np.dot(p1-p2, p1-p)<np.dot(p1-p2, p1-p2):
                return np.array([y, x])
        
    ### Borehole geometry
    ## Select data relevant to construct model geometry within simulation domain
    if np.shape(borehole_geometry)[0]==2:
        local_borehole_geometry = borehole_geometry.copy()    
    else:
        if dip==0:
            point_within_domain_mask = (borehole_geometry[:,0]-simulation_depth)**2+borehole_geometry[:,1]**2 < domain_radius**2
        else:
            point_within_domain_mask = np.abs(borehole_geometry[:,0]-simulation_depth) < domain_radius
        relevant_points_mask = np.convolve(point_within_domain_mask, np.array([True, True, True]), mode="same")
        local_borehole_geometry = borehole_geometry[relevant_points_mask,:]
    local_borehole_geometry[:,0] -= simulation_depth

    ## Adjust top point to the domain
    if dip==0:
        # do nothing if point is on domain boundary
        if local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  == domain_radius**2:
            pass
        # else if point is within the domain add additional point on domain boundary
        elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  < domain_radius**2:
            omega = np.arccos(local_borehole_geometry[0, 1]/domain_radius)
            local_borehole_geometry = np.vstack((np.array([-np.sin(omega)*domain_radius, local_borehole_geometry[0, 1]]), local_borehole_geometry))
        # else if point is outside the domain move it on the domain boundary
        elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1]**2  > domain_radius**2:
            local_borehole_geometry[0,:] = domain_line_intersection(local_borehole_geometry[0, :], local_borehole_geometry[1, :], domain_radius)
    else:
        # do nothing if point is on domain boundary
        if np.abs(local_borehole_geometry[0, 0]) == domain_radius:
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
    if dip==0:
        # do nothing if point is on domain boundary
        if local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2 == domain_radius**2:
            pass
        # else if point is within the domain add additional point on domain boundary
        elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2  < domain_radius**2:
            omega = np.arccos(local_borehole_geometry[-1, 1]/domain_radius)
            local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([np.sin(omega)*domain_radius, local_borehole_geometry[-1, 1]])))
        # else if point is outside the domain move it on the domain boundary
        elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1]**2  > domain_radius**2:
            local_borehole_geometry[-1,:] = domain_line_intersection(local_borehole_geometry[-1, :], local_borehole_geometry[-2, :], domain_radius)
    else:
       # do nothing if point is on domain boundary
        if np.abs(local_borehole_geometry[-1, 0]) == domain_radius:
            pass
        # else if point is within the domain add additional point on domain boundary
        elif np.abs(local_borehole_geometry[-1, 0]) < domain_radius:
            local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([domain_radius, local_borehole_geometry[-1, 1]])))
        # else if point is outside the domain move it on the domain boundary
        elif np.abs(local_borehole_geometry[-1, 0]) > domain_radius:
            a = local_borehole_geometry[-1,0]-domain_radius
            b = domain_radius-local_borehole_geometry[-2,0]
            local_borehole_geometry[-1,:] = [domain_radius, (b*local_borehole_geometry[-1,1]+a*local_borehole_geometry[-2,1])/(a+b)]
    
    return (local_borehole_geometry)

def SelectGmshFormationDataRange(formation_parameters, dip, simulation_depth, domain_radius, active_geometry_window=0.99):
    
    ### Formation geometry
    ## Compute active geometry radius 
    # Active geometry prevents occurence of thin layers and small wedges at the very edge of simulation by ignoring occurence of small elements of model at the very edge of simulation domain
    active_geometry_radius = domain_radius * active_geometry_window 

    ## Select data relevant to construct model geometry within simulation domain
    # Select layers that are present within active geometry area
    local_formation_parameters = formation_parameters.copy()
    local_formation_parameters[:,:2] -= simulation_depth

    if dip==0:
        d = np.abs(local_formation_parameters[:,:2])
    else:
        a = np.tan(dip)
        b = 1
        c = local_formation_parameters[:,:2]
        d = np.abs(c)/(a**2 + b**2)**(1/2) # the equation is |a*x0+b*y0+c|/(a^2+b^2)^(1/2), since (x0, y0) = (0, 0) the form of equation is simplified
    relavant_layers = local_formation_parameters[np.any(d<active_geometry_radius, axis=1),:]

    # Remove undisturbed zones that are not present within active geometry area
    layers_to_check = ~np.isnan(relavant_layers[:,2])

    if dip==0:
        x_points = np.repeat(np.atleast_2d(relavant_layers[layers_to_check, 2]).T, 2, axis=1)
        y_points = relavant_layers[layers_to_check,:2]
    else:
        x_points = np.repeat(np.atleast_2d(relavant_layers[layers_to_check, 2]).T, 4, axis=1)
        x_points[:, :2] *= -1
        y_points = a*x_points + np.hstack([relavant_layers[layers_to_check,:2], relavant_layers[layers_to_check,:2]])
    d = (x_points**2 + y_points**2)**(1/2)

    zones_within = np.any(d<active_geometry_radius, axis=1)

    zones_to_remove = layers_to_check.copy()
    zones_to_remove[layers_to_check] = ~zones_within

    local_formation_model = relavant_layers.copy()
    if np.shape(formation_parameters)[1]==5:
        # If resistivity data are present modify information about model geometry and formation resistivity distribution
        local_formation_model[zones_to_remove, 4] = local_formation_model[zones_to_remove,3]
        local_formation_model[zones_to_remove, 2:4] = np.nan
    elif np.shape(formation_parameters)[1]==3:
        # If resistivity data are not present modify information about model geometry
        local_formation_model[zones_to_remove, 2] = np.nan

    ## Adjust bottom and top boundary to the domain
    # abs_c - value that will stretch bottom and top layers slightly outside simulation domain
    if dip==0:
        abs_c = domain_radius*1.01
    else:
        abs_c = domain_radius*(a**2 + b**2)**(1/2)*1.01 # Calculated from a simplified form of the equation |a*x0+b*y0+c|/(a^2+b^2)^(1/2)
    
    # Adjust top point to the domain
    if local_formation_model[0, 0] > -abs_c:
        local_formation_model[0, 0] = -abs_c
    
    # Adjust bottom point to the domain
    if local_formation_model[-1, 1] < abs_c:
        local_formation_model[-1, 1] = abs_c

    ### Return selected data
    if np.shape(formation_parameters)[1]==5:
        ## If resistivity data are present return information about model geometry and formation resistivity distribution
        # Set formation geometry
        local_formation_geometry = local_formation_model[:,:3]
        # Set formation resistivity distribution
        formation_resistivity_distribution = np.ndarray.flatten(local_formation_model[:,3:5])
        formation_resistivity_distribution = formation_resistivity_distribution[~np.isnan(formation_resistivity_distribution)]
        return (local_formation_geometry, formation_resistivity_distribution)
    elif np.shape(formation_parameters)[1]==3:
        ## If resistivity data are not present return information about model geometry
        return (local_formation_model)

    
def SelectGmshDataRange(borehole_geometry, formation_parameters, dip, mud_resistivity, simulation_depth, domain_radius, active_geometry_window=0.99):

    local_borehole_geometry = SelectGmshBoreholeDataRange(borehole_geometry, dip, simulation_depth, domain_radius)
    local_formation_geometry, local_formation_resistivity = SelectGmshFormationDataRange(formation_parameters, dip, simulation_depth, domain_radius, active_geometry_window)          
    sigma = [1/mud_resistivity] + list(1/local_formation_resistivity)
    
    return local_formation_geometry, local_borehole_geometry, sigma


def ReadGmsh(filename, mesh_dimensionality):
    """
    Utilized ReadGmsh function is a slightly modified version of the orginal ReadGmsh function from Netgen/NGSolve package
    Source: https://github.com/NGSolve/netgen/blob/master/python/read_gmsh.py
    """

    if not filename.endswith(".msh"):
        filename += ".msh"

    f = open(filename, 'r')
    mesh = msh.Mesh(dim=mesh_dimensionality)

    pointmap = {}
    facedescriptormap = {}
    namemap = { 0 : "default" }
    materialmap = {}
    bbcmap = {}

    segm = 1
    trig = 2
    quad = 3
    tet = 4
    hex = 5
    prism = 6
    pyramid = 7
    segm3 = 8      # 2nd order line
    trig6 = 9      # 2nd order trig
    tet10 = 11     # 2nd order tet
    point = 15
    quad8 = 16     # 2nd order quad
    hex20 = 17     # 2nd order hex
    prism15 = 18   # 2nd order prism
    pyramid13 = 19 # 2nd order pyramid
    segms = [segm, segm3]
    trigs = [trig, trig6]
    quads = [quad, quad8]
    tets = [tet, tet10]
    hexes = [hex, hex20]
    prisms = [prism, prism15]
    pyramids = [pyramid, pyramid13]
    elem0d = [point]
    elem1d = segms
    elem2d = trigs + quads
    elem3d = tets + hexes + prisms + pyramids

    num_nodes_map = { segm : 2,
                        trig : 3,
                        quad : 4,
                        tet : 4,
                        hex : 8,
                        prism : 6,
                        pyramid : 5,
                        segm3 : 3,
                        trig6 : 6,
                        tet10 : 10,
                        point : 1,
                        quad8 : 8,
                        hex20 : 20,
                        prism15 : 18,
                        pyramid13 : 19 }

    while True:
        line = f.readline()
        if line == "":
            break

        if line.split()[0] == "$PhysicalNames":
            #print('WARNING: Physical groups detected - Be sure to define them for every geometrical entity.')
            numnames = int(f.readline())
            for i in range(numnames):
                f.readline
                line = f.readline()
                namemap[int(line.split()[1])] = line.split()[2][1:-1]

        if line.split()[0] == "$Nodes":
            num = int(f.readline().split()[0])
            for i in range(num):
                line = f.readline()
                nodenum, x, y, z = line.split()[0:4]
                pnum = mesh.Add(msh.MeshPoint(msh.Pnt(float(x), float(y), float(z))))
                pointmap[int(nodenum)] = pnum

        if line.split()[0] == "$Elements":
            num = int(f.readline().split()[0])

            number_of_elements = num

            for i in range(num):
                line = f.readline().split()
                elmnum = int(line[0])
                elmtype = int(line[1])
                numtags = int(line[2])
                # the first tag is the physical group nr, the second tag is the group nr of the dim
                tags = [int(line[3 + k]) for k in range(numtags)]

                if elmtype not in num_nodes_map:
                    raise Exception("element type", elmtype, "not implemented")
                num_nodes = num_nodes_map[elmtype]

                nodenums = line[3 + numtags:3 + numtags + num_nodes]
                nodenums2 = [pointmap[int(nn)] for nn in nodenums]

                if elmtype in elem1d:
                    if mesh_dimensionality == 3:
                        if tags[1] in bbcmap:
                            index = bbcmap[tags[1]]
                        else:
                            index = len(bbcmap) + 1
                            if len(namemap):
                                mesh.SetCD2Name(index, namemap[tags[0]])
                            else:
                                mesh.SetCD2Name(index, "line" + str(tags[1]))
                            bbcmap[tags[1]] = index

                    elif mesh_dimensionality == 2:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = msh.FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[tags[0]]
                            else:
                                fd.bcname = 'line' + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[tags[0]])
                            else:
                                mesh.SetMaterial(index, "line" + str(tags[1]))
                            materialmap[tags[1]] = index

                    mesh.Add(msh.Element1D(index=index, vertices=nodenums2))

                if elmtype in elem2d:  # 2d elements
                    if mesh_dimensionality == 3:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = msh.FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[tags[0]]
                            else:
                                fd.bcname = "surf" + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[tags[0]])
                            else:
                                mesh.SetMaterial(index, "surf" + str(tags[1]))
                            materialmap[tags[1]] = index

                    if elmtype in trigs:
                        ordering = [i for i in range(3)]
                        if elmtype == trig6:
                            ordering += [4,5,3]
                    if elmtype in quads:
                        ordering = [i for i in range(4)]
                        if elmtype == quad8:
                            ordering += [4, 6, 7, 5]
                    mesh.Add(msh.Element2D(index, [nodenums2[i] for i in ordering]))

                if elmtype in elem3d:  # volume elements
                    if tags[1] in materialmap:
                        index = materialmap[tags[1]]
                    else:
                        index = len(materialmap) + 1
                        if len(namemap):
                            mesh.SetMaterial(index, namemap[tags[0]])
                        else:
                            mesh.SetMaterial(index, "vol" + str(tags[1]))
                        materialmap[tags[1]] = index

                    nodenums2 = [pointmap[int(nn)] for nn in nodenums]

                    if elmtype in tets:
                        ordering = [0,1,2,3]
                        if elmtype == tet10:
                            ordering += [4,6,7,5,9,8]
                    elif elmtype in hexes:
                        ordering = [0,1,5,4,3,2,6,7]
                        if elmtype == hex20:
                            ordering += [8,16,10,12,13,19,15,14,9,11,18,17]
                    elif elmtype in prisms:
                        ordering = [0,2,1,3,5,4]
                        if elmtype == prism15:
                            ordering += [7,6,9,8,11,10,13,12,14]
                    elif elmtype in pyramids:
                        ordering = [3,2,1,0,4]
                        if elmtype == pyramid13:
                            ordering += [10,5,6,8,12,11,9,7]
                    mesh.Add(msh.Element3D(index, [nodenums2[i] for i in ordering]))
    return mesh

def ConstructGmsh2dModel(domain_radius, tool_geometry, source_terms, formation_geometry, borehole_geometry, file_number, mesh_generator, output_folder_path="./tmp", output_mode="variable"):

    ### GMSH test
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("model_"+str(file_number))

    ## Add central point (removed from final geometry)
    gmsh.model.occ.addPoint(0, 0, 0, tag=1)

    ## Points at borehole axis (bottom -> top)
    gmsh.model.occ.addPoint(0, domain_radius, 0, tag=2)
    index_0D = 3
    for i in range(np.shape(tool_geometry)[0]):
        gmsh.model.occ.addPoint(0, np.flip(tool_geometry)[i], 0, tag=index_0D)
        index_0D +=1
    gmsh.model.occ.addPoint(0, -domain_radius, 0, tag=index_0D)
    index_0D +=1
    points_at_borehole_axis = list(np.arange(2, index_0D))

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
    fields = [1]
    
    i = 2
    for electrode_position in tool_geometry[source_terms != 0]:
        gmsh.model.mesh.field.add("MathEval", i) # Meshsize field modification near current electrodes
        gmsh.model.mesh.field.setString(i, "F","(x^2 + (y+({}))^2)/2 + 0.01".format(electrode_position))
        fields.append(i)
        i += 1
        
    gmsh.model.mesh.field.add("Min", i)
    gmsh.model.mesh.field.setNumbers(i, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(i)
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)
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

    db = gmsh.model.addPhysicalGroup(1, dirichlet_boundaries, 1)
    gmsh.model.setPhysicalName(1, db , "dirichlet_boundary")

    nb = gmsh.model.addPhysicalGroup(1, neumann_boundaries, 2)
    gmsh.model.setPhysicalName(2, nb , "neumann_boundary")

    i = 1
    for index in pg_index_list:
        surface = gmsh.model.addPhysicalGroup(2, [index], index)
        gmsh.model.setPhysicalName(2, surface, "surf_"+str(i))
        i += 1

    # Save file
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_folder_path + "/fm_"+str(file_number)+".msh")
    gmsh.finalize()
    
    if output_mode == "variable":
        # Read file
        mesh = ReadGmsh(output_folder_path + "/fm_"+str(file_number)+".msh", 2)
        
        return mesh

def ConstructGmsh3dModel(domain_radius, tool_geometry, source_terms, formation_geometry, dip, borehole_geometry, file_number, output_folder_path="./tmp", output_mode="variable"):

    ### GMSH test
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("model_"+str(file_number))

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
    fields = [1]
    
    i = 2
    for electrode_position in tool_geometry[source_terms != 0]:
        gmsh.model.mesh.field.add("MathEval", i) # Meshsize field modification near current electrodes
        gmsh.model.mesh.field.setString(i, "F","(x^2 + (y+({}))^2)/2 + 0.01".format(electrode_position))
        fields.append(i)
        i += 1
        
    gmsh.model.mesh.field.add("Min", i)
    gmsh.model.mesh.field.setNumbers(i, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(i)

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
    gmsh.write(output_folder_path + "/fm_"+str(file_number)+".msh")
    gmsh.finalize()
    
    if output_mode == "variable":
        mesh = ReadGmsh(output_folder_path + "/fm_"+str(file_number)+".msh", 3)
        
        return mesh