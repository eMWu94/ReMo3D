# -*- coding: utf-8 -*-

import numpy as np

import netgen.meshing as msh
from netgen.csg import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters, meshsize

# # Netgen functions

def SelectNetgenDataRange(borehole_geometry, formation_parameters, mud_resistivity, simulation_depth, domain_radius, active_geometry_window=0.99):

    def domain_line_intersection (p1, p2, radius, side):
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
            if side=="top" and y<0 and np.dot(p1-p2, p1-p)> 0 and np.dot(p1-p2, p1-p)<np.dot(p1-p2, p1-p2):
                return np.array([y, x])
            elif side=="bottom" and y>0 and np.dot(p1-p2, p1-p)> 0 and np.dot(p1-p2, p1-p)<np.dot(p1-p2, p1-p2):
                return np.array([y, x])

    ### Borehole geometry
    ## Select data relevant to construct model geometry within simulation domain
    if np.shape(borehole_geometry)[0]==2:
        local_borehole_geometry = borehole_geometry.copy()
    else:
        point_within_domain_mask = (borehole_geometry[:,0]-simulation_depth)**2+borehole_geometry[:,1]**2 < domain_radius**2
        relevant_points_mask = np.convolve(point_within_domain_mask, np.array([True, True, True]), mode="same")
        local_borehole_geometry = borehole_geometry[relevant_points_mask,:]
    local_borehole_geometry[:,0] -= simulation_depth

    ## Adjust top point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1], domain_radius**2):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1] < domain_radius**2:
        omega = np.arccos(local_borehole_geometry[0, 1]/domain_radius)
        local_borehole_geometry = np.vstack((np.array([-np.sin(omega)*domain_radius, local_borehole_geometry[0, 1]]), local_borehole_geometry))
    # else if point is outside the domain move it on the domain boundary
    elif local_borehole_geometry[0, 0]**2 + local_borehole_geometry[0, 1] > domain_radius**2:
        local_borehole_geometry[0,:] = domain_line_intersection(local_borehole_geometry[0, :], local_borehole_geometry[1, :], domain_radius, side="top")
    
    ## Adjust bottom point to the domain
    # do nothing if point is on domain boundary
    if np.isclose(local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1], domain_radius**2):
        pass
    # else if point is within the domain add additional point on domain boundary
    elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1] < domain_radius**2:
        omega = np.arccos(local_borehole_geometry[-1, 1]/domain_radius)
        local_borehole_geometry = np.vstack((local_borehole_geometry, np.array([np.sin(omega)*domain_radius, local_borehole_geometry[0, 1]])))
    # else if point is outside the domain move it on the domain boundary
    elif local_borehole_geometry[-1, 0]**2 + local_borehole_geometry[-1, 1] > domain_radius**2:
        local_borehole_geometry[-1,:] = domain_line_intersection(local_borehole_geometry[-1, :], local_borehole_geometry[-2, :], domain_radius, side="bottom")

    ### Formation geometry

    ## Compute active geometry radius 
    # Active geometry prevents occurence of thin layers and small wedges at the very edge of simulation by ignoring occurence of small elements of model at the very edge of simulation domain
    active_geometry_radius = domain_radius * active_geometry_window 

    ## Select data relevant to construct model geometry within simulation domain
    # Raw cut of relevant data\
    point_within = np.any((formation_parameters[:,:2]-simulation_depth)**2 <= active_geometry_radius**2, axis=1)
    line_across = np.all(np.vstack([np.all((formation_parameters[:,:2]-simulation_depth)**2 > active_geometry_radius**2, axis=1), formation_parameters[:,0]<simulation_depth, formation_parameters[:,1]>simulation_depth]), axis=0)
    local_formation_model = formation_parameters[np.any(np.vstack([point_within, line_across]) , axis=0),:]
    local_formation_model[:,:2] -= simulation_depth

    # Check if undisturbed zones in layers with filtration are located within simulation domain
    # Zones are located outside simulation domain if both characteristic points (top and bottom inside corners) 
    # and the line between them are located outside the domain
    filtration_zone_present_mask = ~np.isnan(local_formation_model[:,2])
    top_points_outside_mask = local_formation_model[:,0]**2 + local_formation_model[:,2]**2 >= active_geometry_radius**2
    bottom_points_outside_mask = local_formation_model[:,1]**2 + local_formation_model[:,2]**2 >= active_geometry_radius**2
    line_outside_mask = ~np.all(np.vstack([local_formation_model[:,0] < 0, local_formation_model[:,1] > 0, local_formation_model[:,2] < active_geometry_radius]), axis=0)
    zones_outside_mask = np.all(np.vstack([filtration_zone_present_mask, top_points_outside_mask, bottom_points_outside_mask, line_outside_mask]), axis=0)

    # Remove unrelevant zones from the model
    local_formation_model[zones_outside_mask, 2] = np.nan
    local_formation_model[zones_outside_mask, 4] = local_formation_model[zones_outside_mask, 3]
    local_formation_model[zones_outside_mask, 3] = np.nan

    ## Adjust top point to the domain
    if local_formation_model[0, 0] != local_borehole_geometry[0, 0]:
        local_formation_model[0, 0] = local_borehole_geometry[0, 0]
    
    ## Adjust bottom point to the domain
    if local_formation_model[-1, 1] != local_borehole_geometry[-1, 0]:
        local_formation_model[-1, 1] = local_borehole_geometry[-1, 0]

    ## Local formation grid: [region next to borehole wall, region at left side to filtration zone boundary (if present within simulation domain), region at right side of filtration zone boundary (if present within simulation domain), region next to simulation domain end]
    local_formation_grid = np.empty((np.shape(local_formation_model)[0], 2))
    region_index = 2
    for i in range(np.shape(local_formation_grid)[0]):
        if np.isnan(local_formation_model[i,3]) == True:
            local_formation_grid[i,:] = region_index
            region_index += 1
        else:
            local_formation_grid[i,0] = region_index
            local_formation_grid[i,1] = region_index + 1
            region_index += 2

    local_formation_geometry = np.hstack((local_formation_model[:,:3], local_formation_grid))

    formation_resistivity_distribution = np.ndarray.flatten(local_formation_model[:,3:5])
    formation_resistivity_distribution = formation_resistivity_distribution[~np.isnan(formation_resistivity_distribution)]

    local_conductivity_distribution = [1/mud_resistivity] + list(1/formation_resistivity_distribution) # Conductivity within different parts of model

    return (local_formation_geometry, local_borehole_geometry, local_conductivity_distribution)


def PrepareNetgen2dModelGeometry(domain_radius, tool_geometry, formation_geometry, borehole_geometry, source_terms, mesh_size_min):

    index_0D = 0 # points
    index_1D = 0 # lines

    ### Calculate z coordinates of layer boundaries
    boundaries_z =  np.sort(np.unique(formation_geometry[:,:2]))

    ### Add points [index, r, z]

    ## Add points at borehole axis
    n_electrodes = np.shape(tool_geometry)[0]
    points_at_borehole_axis = np.vstack([np.arange(n_electrodes+2) + index_0D, np.zeros(n_electrodes+2), np.hstack([-domain_radius, tool_geometry, domain_radius])]).T
    points = points_at_borehole_axis
    index_0D += n_electrodes+2

    ## Add points at borehole/rock interface
    # Calculate intersections of boundaries with the borehole wall
    boundaries_z_nr = np.atleast_2d(boundaries_z[~np.isin(boundaries_z, borehole_geometry[:,0])]).T # select layer boundaries from depths other than caliper points
    nr_intersections = np.concatenate([np.atleast_2d(np.interp(boundaries_z_nr[:,0], borehole_geometry[:,0], borehole_geometry[:,1])).T, boundaries_z_nr], axis=1) # intersections of boundaries from depths other than caliper points with the borehole wall

    # Merge intersections with caliper data
    cali_vertices = np.concatenate([np.flip(borehole_geometry, 1), nr_intersections], axis=0) # add intesetions to borehole geometry data
    cali_vertices = cali_vertices[np.lexsort([cali_vertices[:,1]])] # sort data in ascending z-value order
    boundary_points_at_borehole_wall_mask = np.isin(cali_vertices[:,1], boundaries_z) # mark intersections points of layer boundaries with borehole wall for later use

    # Add points at borehole/rock interface
    points_at_borehole_wall = np.hstack([np.atleast_2d(np.arange(np.shape(cali_vertices)[0])).T + index_0D, cali_vertices])
    points = np.vstack([points, points_at_borehole_wall])
    index_0D += np.shape(points_at_borehole_wall)[0]

    ## Add points at filtration/undisturbed zone interfaces
    points_at_filtration_boundaries = np.vstack((np.repeat(formation_geometry[:,2], 2), np.ndarray.flatten(formation_geometry[:,:2]))).T
    filtration_mask = np.unique(points_at_filtration_boundaries, axis=0, return_index=True)[1]

    map_points_at_filtration_to_boundaries = np.ndarray.flatten(np.vstack((np.arange(np.shape(formation_geometry)[0]), np.arange(np.shape(formation_geometry)[0])+1)).T)[np.sort(filtration_mask)] # prepared for lines

    points_at_filtration_boundaries = points_at_filtration_boundaries[np.sort(filtration_mask)]

    map_points_at_filtration_to_boundaries = map_points_at_filtration_to_boundaries[~np.isnan(points_at_filtration_boundaries).any(axis=1)] # prepared for lines
    points_at_filtration_boundaries = points_at_filtration_boundaries[~np.isnan(points_at_filtration_boundaries).any(axis=1)]
    points_at_filtration_boundaries = np.hstack([np.atleast_2d(np.arange(np.shape(points_at_filtration_boundaries)[0])).T + index_0D, points_at_filtration_boundaries])

    points_at_top_boundaries = formation_geometry[:,[2,0]][~np.isnan(formation_geometry[:,[2,0]]).any(axis=1)] # prepared for lines
    points_at_bottom_boundaries = formation_geometry[:,[2,1]][~np.isnan(formation_geometry[:,[2,1]]).any(axis=1)] # prepared for lines
    indices_of_points_at_top_boundaries = np.argwhere((points_at_top_boundaries[:, None] == points_at_filtration_boundaries[:,1:]).all(-1)==True)[:,1] + index_0D # prepared for lines
    indices_of_points_at_bottom_boundaries = np.argwhere((points_at_bottom_boundaries[:, None] == points_at_filtration_boundaries[:,1:]).all(-1)==True)[:,1] + index_0D # prepared for lines

    # Move points from outside to the domain boundary
    points_outside_domain_mask = points_at_filtration_boundaries[:,1]**2 + points_at_filtration_boundaries[:,2]**2 >= domain_radius**2
    omega = np.arccos(points_at_filtration_boundaries[points_outside_domain_mask, 1]/domain_radius)
    points_at_filtration_boundaries[points_outside_domain_mask, 2] = np.sign(points_at_filtration_boundaries[points_outside_domain_mask, 2])*np.sin(omega)*domain_radius

    # Add points at filtration/undisturbed zone interfaces
    points = np.vstack([points, points_at_filtration_boundaries])
    index_0D += np.shape(points_at_filtration_boundaries)[0]

    ## Add points at ends of layers boundaries
    omega = np.arcsin(boundaries_z[1:-1]/domain_radius)
    points_at_end = np.vstack([np.cos(omega)*domain_radius, boundaries_z[1:-1]]).T
    unique_points_at_end_mask = ~(points_at_end[:, None] == points_at_filtration_boundaries[:,1:]).all(-1).any(-1) # check if same points are already present in
    points_at_end = np.hstack([np.atleast_2d(np.arange(np.shape(points_at_end)[0])).T + index_0D, points_at_end[unique_points_at_end_mask,:]])

    points = np.vstack([points, points_at_end])
    index_0D += np.shape(points_at_end)[0]

    ## Add additional points at domain boundary to aproximate circular shape
    # Select existing points lying at the domain_boundary and convert them to polar coordinates
    existing_points_at_domain_boundary = points[np.isclose(points[:, 1]**2 + points[:, 2]**2, domain_radius**2),:]
    existing_points_at_domain_boundary = existing_points_at_domain_boundary[np.argsort(existing_points_at_domain_boundary[:,2]),:]

    existing_points_at_domain_boundary[0,2] = np.arctan(-np.inf)
    existing_points_at_domain_boundary[-1,2] = np.arctan(np.inf)
    existing_points_at_domain_boundary[1:-1,2] = np.arctan(existing_points_at_domain_boundary[1:-1,2]/existing_points_at_domain_boundary[1:-1,1])
    existing_points_at_domain_boundary[:,1] = domain_radius

    # Add points to follow circular shape of simulation domain
    angles_betweeen_existing_points = existing_points_at_domain_boundary[1:,2]-existing_points_at_domain_boundary[:-1,2]
    points_to_add = np.floor(angles_betweeen_existing_points/(9*np.pi/180)).astype(int) # number of points that will be added between existing points

    starting_index = index_0D
    points_at_domain_boundary = existing_points_at_domain_boundary[0,:]
    for i in range(np.shape(points_to_add)[0]):
        if points_to_add[i] > 0:
            index = np.array([0, points_to_add[i]+1])
            angle = np.array([existing_points_at_domain_boundary[i,2], existing_points_at_domain_boundary[i+1,2]])
            interpolated_angles = np.interp(np.arange(points_to_add[i])+1, index, angle)
            additional_points = np.vstack([np.arange(points_to_add[i]) + index_0D, np.full(points_to_add[i], domain_radius), interpolated_angles]).T
            points_at_domain_boundary = np.vstack([points_at_domain_boundary, additional_points, existing_points_at_domain_boundary[i+1,:]])
            index_0D += points_to_add[i]
        else:
            points_at_domain_boundary = np.vstack([points_at_domain_boundary, existing_points_at_domain_boundary[i+1,:]])
    
    points_at_domain_boundary[:,1], points_at_domain_boundary[:,2] = domain_radius*np.cos(points_at_domain_boundary[:,2]), domain_radius*np.sin(points_at_domain_boundary[:,2])
    new_points_at_domain_boundary = points_at_domain_boundary[points_at_domain_boundary[:,0]>=starting_index,:]
    points = np.vstack([points, new_points_at_domain_boundary])

    ### Add lines [index, start-point, end-point, boundary-condition, domain on the left side, domain on the right side]
    ## Add vertical lines
    # Add lines at borehole axis
    number_of_lines = np.shape(points_at_borehole_axis)[0]-1
    lines_at_borehole_axis = np.vstack([np.arange(number_of_lines), np.arange(number_of_lines), np.arange(number_of_lines)+1, np.ones(number_of_lines), np.zeros(number_of_lines), np.ones(number_of_lines)]).T
    lines = lines_at_borehole_axis
    index_1D += number_of_lines

    # Add lines at borehole/rock interface
    boundary_indices = np.argwhere(boundary_points_at_borehole_wall_mask).flatten()
    layers_extends = boundary_indices[1:] - boundary_indices[:-1]
    number_of_lines = np.shape(points_at_borehole_wall)[0]-1
    lines_at_borehole_wall = np.vstack([np.arange(number_of_lines) + index_1D, points_at_borehole_wall[:-1,0], points_at_borehole_wall[1:,0], np.ones(number_of_lines), np.ones(number_of_lines), np.repeat(formation_geometry[:,3], layers_extends)]).T
    lines = np.vstack([lines, lines_at_borehole_wall])
    index_1D += number_of_lines

    # Add lines at filtration/undisturbed zones interface
    number_of_lines = np.shape(points_at_top_boundaries)[0]
    lines_at_filtration_boundaries = np.vstack([np.arange(number_of_lines) + index_1D, indices_of_points_at_top_boundaries, indices_of_points_at_bottom_boundaries, np.ones(number_of_lines), formation_geometry[formation_geometry[:,3]!=formation_geometry[:,4],3:].T]).T
    lines = np.vstack([lines, lines_at_filtration_boundaries])
    index_1D += number_of_lines

    ## Add horizontal lines
    # Add lines at layers boundaries
    number_of_lines_at_boundary = np.empty_like(boundaries_z)
    for i in range(np.shape(number_of_lines_at_boundary)[0]):
        number_of_lines_at_boundary[i] = np.sum(np.all([points[:,2]==boundaries_z[i], points[:,1] > 0], axis=0)) - 1
    number_of_lines = int(np.sum(number_of_lines_at_boundary))

    areas_above_boundary = np.vstack((formation_geometry[:-1,3], np.full_like(formation_geometry[:-1,3], np.nan).T, formation_geometry[:-1,4])).T
    areas_below_boundary = np.vstack((formation_geometry[1:,3], np.full_like(formation_geometry[1:,3], np.nan).T, formation_geometry[1:,4])).T

    for i in range(np.shape(formation_geometry)[0]-1):
        if formation_geometry[i, 2] > formation_geometry[i+1, 2]:
            areas_above_boundary[i,1] = areas_above_boundary[i,0]
            areas_below_boundary[i,1] = areas_below_boundary[i,2]
        elif formation_geometry[i, 2] < formation_geometry[i+1, 2]:
            areas_above_boundary[i,1] = areas_above_boundary[i,2]
            areas_below_boundary[i,1] = areas_below_boundary[i,0]

    lines_at_boundaries = np.full((number_of_lines, 6), 1)
    lines_at_boundaries[:,0] = np.arange(number_of_lines) + index_1D

    j = 0
    for i in range(np.shape(boundaries_z[1:-1])[0]):
        points_at_ith_boundary = points[np.all([points[:,2]==boundaries_z[1:-1][i], points[:,1] > 0], axis=0), :]
        points_at_ith_boundary = points_at_ith_boundary[np.argsort(points_at_ith_boundary[:,1]),:]
        if np.shape(points_at_ith_boundary)[0] == 2:
            lines_at_boundaries[j,[1,2,4,5]] = [points_at_ith_boundary[0,0], points_at_ith_boundary[1,0], areas_below_boundary[i,2], areas_above_boundary[i,0]]
            j += 1
        elif np.shape(points_at_ith_boundary)[0] == 3:
            lines_at_boundaries[j:j+2,[1,2,4,5]] = np.vstack((points_at_ith_boundary[:2,0], points_at_ith_boundary[1:3,0], areas_below_boundary[i,[0,2]], areas_above_boundary[i,[0,2]])).T
            j += 2
        elif np.shape(points_at_ith_boundary)[0] == 4:
            lines_at_boundaries[j:j+3,[1,2,4,5]] = np.vstack((points_at_ith_boundary[:3,0], points_at_ith_boundary[1:4,0], areas_below_boundary[i,:], areas_above_boundary[i,:])).T
            j += 3

    lines = np.vstack([lines, lines_at_boundaries])
    index_1D += number_of_lines

    ## Add lines at domain boundary
    number_of_lines = np.shape(points_at_domain_boundary)[0]-1

    areas_next_to_domain_boundary = [1]
    for i in range(np.shape(formation_geometry)[0]):
        if formation_geometry[i,3] == formation_geometry[i,4]:
            areas_next_to_domain_boundary.append(formation_geometry[i,3])
        else:
            top_point_mask = formation_geometry[i,0]**2 + formation_geometry[i,2]**2 >= domain_radius**2
            bottom_point_mask = formation_geometry[i,1]**2 + formation_geometry[i,2]**2 >= domain_radius**2
            if top_point_mask==True and bottom_point_mask==True:
                areas_next_to_domain_boundary += [formation_geometry[i,3], formation_geometry[i,4], formation_geometry[i,3]]
            elif top_point_mask==True:
                areas_next_to_domain_boundary += [formation_geometry[i,3], formation_geometry[i,4]]
            elif bottom_point_mask==True:
                areas_next_to_domain_boundary += [formation_geometry[i,4], formation_geometry[i,3]]
            else:
                areas_next_to_domain_boundary.append(formation_geometry[i,4])
    areas_next_to_domain_boundary.append(1)

    areas_next_to_domain_boundary = np.repeat(areas_next_to_domain_boundary, points_to_add+1)
    lines_at_domain_boundary = np.vstack((np.arange(number_of_lines) + index_1D, points_at_domain_boundary[:-1,0], points_at_domain_boundary[1:,0], np.ones(np.shape(points_at_domain_boundary[1:,0]))+1, np.array(areas_next_to_domain_boundary), np.zeros(np.shape(points_at_domain_boundary[1:,0])))).T
    lines = np.vstack([lines, lines_at_domain_boundary])

    return points, lines


def ConstructNetgen2dModel(domain_radius, tool_geometry_list, formation_geometry_list, borehole_geometry_list, source_terms_list, points_y_offsets):

    mesh_size_min = 0.001
    mesh_size_max = 10
    mesh_density = "moderate"
    
    all_points = []
    all_lines = []
    points_numbering_offset = 0
    lines_numbering_offset = 0
    region_numbering_offset = 0
    
    for model_index in range(len(tool_geometry_list)):

        tool_geometry = tool_geometry_list[model_index]
        formation_geometry = formation_geometry_list[model_index]
        borehole_geometry = borehole_geometry_list[model_index]
        source_terms = source_terms_list[model_index]
        
        ### 
        points, lines = PrepareNetgen2dModelGeometry(domain_radius, tool_geometry, formation_geometry, borehole_geometry, source_terms, mesh_size_min)

        ### Add and update offsets
        points[:,2] += points_y_offsets[model_index] 
        points[:,0] += points_numbering_offset + 1
        lines[:,1:3] += points_numbering_offset + 1
        lines[:,0] += lines_numbering_offset + 1
        
        lines[:,-2:][lines[:,-2:]!=0] += region_numbering_offset
        
        points_numbering_offset = np.max(lines[:,1:3])
        lines_numbering_offset = np.max(lines[:,0])
        region_numbering_offset = np.max(lines[:,-2:])

        ### Convert arrays to lists
        points = list(map(list, points[:,1:]))
        lines = list(map(list, lines[:,1:].astype(int)))
        
        ### Add mesh size information
        for k in np.ndarray.flatten(np.argwhere(source_terms != 0))+1: # Set minimum mesh size around source points
            points[k].append(mesh_size_min)

        ### Merge data
        all_points += points
        all_lines += lines

    ### Create the model geometry
    model_geometry = SplineGeometry()

    pnums = [model_geometry.AppendPoint(*p) for p in all_points]

    for p1,p2,bc,left,right in all_lines:
        model_geometry.Append( ["line", pnums[p1-1], pnums[p2-1]], bc=bc, leftdomain=left, rightdomain=right)

    mesh = model_geometry.GenerateMesh(eval("meshsize." + mesh_density), maxh=mesh_size_max)

    return mesh