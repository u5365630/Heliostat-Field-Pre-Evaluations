#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heliostat Field Performance Analysis Tool

This program provides rapid pre-evaluation of heliostat field performance for 
Concentrated Solar Power (CSP) systems. It calculates various optical efficiency 
losses and generates visualizations to assess field layout effectiveness.

The tool analyzes six primary loss mechanisms:
1. Cosine losses - Angular effects between sun and heliostat normals
2. Shading losses - Heliostats blocking incoming solar radiation
3. Blocking losses - Heliostats obstructing reflected beams to receiver
4. Spillage losses - Reflected radiation missing the receiver aperture
5. Reflectivity losses - Mirror surface efficiency degradation
6. Attenuation losses - Atmospheric absorption of reflected beams

Key Features:
- Field-wide efficiency calculations with spatial distribution analysis
- Color-coded visualization plots showing loss factor distributions
- CSV export of efficiency values and heliostat performance rankings
- Statistical analysis of field performance characteristics

Usage:
- Load heliostat position data from CSV file
- Define solar conditions (azimuth, zenith angles)
- Execute analysis to generate efficiency maps and performance statistics
- Export results for further analysis or optimization studies

Based on methodologies from SOLSTICE ray-tracing software and adapted for
rapid preliminary field assessment.

Author: hutchizzle
Created: May 9, 2025

=============================================================================
USER INPUT REQUIREMENTS - MODIFY THESE LINES BEFORE RUNNING:
=============================================================================
Line ~752: Update file path for input CSV: pos_and_aiming.csv
Line ~774: Choose case study: ACTIVE_CASE = 'C1.1' or 'C1.2'
Line ~1047: Update file path for efficiencies CSV export
Line ~1048: Update file path for rankings CSV export
=============================================================================
"""

import math as m
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy import integrate
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd
import os
from shapely.geometry import Polygon
from scipy.special import erf
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, spearmanr

start_time = time.time()

# Extract from https://github.com/anustg/solstice-scripts/blob/master/solsticepy/cal_field.py

class FieldPF:
    """
    Heliostat Field Performance Calculator
    
    Performs preliminary calculation of heliostat field optical efficiency.
    Calculates cosine factors, shading/blocking losses, spillage effects,
    and other optical losses for rapid field assessment.
    
    Note: Angle conventions used in this program:
      * azimuth: solar azimuth angle, from South toward West (degrees)
      * zenith: solar zenith angle, 0 from vertical (degrees)
    
    Example:
        >>> field = FieldPF(receiver_norm=np.r_[0,1,0])
        >>> sun_vec = field.get_solar_vector(azimuth=0, zenith=12)
        >>> norms = field.get_normals(towerheight=62, hstpos=positions, sun_vec=sun_vec)
        >>> cos_eff = field.get_cosine(hst_norms=norms, sun_vec=sun_vec)
    """

    def __init__(self, receiver_norm=np.r_[0,1,0], heliostat_width=10, heliostat_height=10):
        """
        Initialize the field performance calculator.
        
        Args:
            receiver_norm (numpy array): Unit normal vector of receiver aperture (default: [0,1,0])
            heliostat_width (float): Width of individual heliostats in meters (default: 10)
            heliostat_height (float): Height of individual heliostats in meters (default: 10)
        """
        self.rec_norm = receiver_norm.reshape(3,1)
        self.heliostat_width = heliostat_width
        self.heliostat_height = heliostat_height
        self.heliostat_area = heliostat_width * heliostat_height

    # =========================================================================
    # SOLAR GEOMETRY AND HELIOSTAT ORIENTATION FUNCTIONS
    # =========================================================================

    def get_solar_vector(self, azimuth, zenith):
        """
        Calculate normalized solar vector from azimuth and zenith angles.
        
        Args:
            azimuth (float): Solar azimuth angle in degrees (South=0, West=+90)
            zenith (float): Solar zenith angle in degrees (0=vertical)
            
        Returns:
            numpy array: Normalized 3D solar vector [x, y, z]
        """
        azimuth = np.radians(azimuth)
        zenith = np.radians(zenith)
        
        sun_z = np.cos(zenith)
        sun_y = -np.sin(zenith) * np.cos(azimuth)
        sun_x = -np.sin(zenith) * np.sin(azimuth)
        
        sun_vec = np.array([sun_x, sun_y, sun_z])
        return sun_vec
    
    def get_normals(self, towerheight, hstpos, sun_vec):
        """
        Calculate normal vectors for each heliostat using ideal tracking.
        
        Each heliostat normal bisects the angle between the incoming solar
        vector and the vector toward the receiver (tower).
        
        Args:
            towerheight (float): Height of receiver tower in meters
            hstpos (numpy array): Heliostat positions as nx3 array [x, y, z]
            sun_vec (numpy array): Solar vector as 3-element array
            
        Returns:
            numpy array: Normal vectors for each heliostat as nx3 array
        """
        # Calculate unit vectors from heliostats to receiver
        tower_vec = -hstpos
        tower_vec[:, -1] += towerheight
        tower_vec /= np.sqrt(np.sum(tower_vec**2, axis=1)[:, None])
        
        # Normal vector bisects sun and tower vectors
        hst_norms = sun_vec + tower_vec
        hst_norms /= np.sqrt(np.sum(hst_norms**2, axis=1)[:, None])
        return hst_norms

    def get_rec_view(self, towerheight, hstpos):
        """
        Calculate viewing angles from receiver to each heliostat.
        
        Determines which heliostats are geometrically visible from
        the receiver aperture based on angle thresholds.
        
        Args:
            towerheight (float): Height of receiver tower in meters
            hstpos (numpy array): Heliostat positions as nx3 array
            
        Returns:
            numpy array: Viewing angles in radians for each heliostat
        """
        # Calculate unit vectors from receiver to heliostats
        tower_vec = -hstpos
        tower_vec[:, -1] += towerheight
        tower_vec /= np.sqrt(np.sum(tower_vec**2, axis=1)[:, None])
        
        # Calculate angle between receiver normal and heliostat direction
        view = np.arccos(np.dot(-tower_vec, self.rec_norm))
        return view.flatten()

    # =========================================================================
    # OPTICAL EFFICIENCY LOSS CALCULATIONS
    # =========================================================================

    def get_cosine(self, hst_norms, sun_vec):
        """
        Calculate cosine efficiency for each heliostat.
        
        Cosine factor represents the projected area of the heliostat
        as seen from the sun direction.
        
        Args:
            hst_norms (numpy array): Heliostat normal vectors as nx3 array
            sun_vec (numpy array): Solar vector as 3-element array
            
        Returns:
            numpy array: Cosine efficiency for each heliostat [0-1]
        """
        cos_factor = np.sum(hst_norms * sun_vec, axis=1)
        return cos_factor

    def get_shading(self, hstpos, sun_vec, hst_norms):
        """
        Calculate shading efficiency using 2D projection method.
        
        Projects all heliostats onto a plane perpendicular to solar rays
        and calculates shadow overlap areas to determine shading losses.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            sun_vec (numpy array): Solar vector as 3-element array
            hst_norms (numpy array): Heliostat normal vectors as nx3 array
            
        Returns:
            numpy array: Shading efficiency for each heliostat [0-1]
        """
        num_hst = len(hstpos)
        shading_efficiency = np.ones(num_hst)
        
        # Normalize sun vector
        sun_ray = sun_vec / np.linalg.norm(sun_vec)
        
        # Create projection plane perpendicular to sun ray
        if abs(sun_ray[2]) < 0.9:
            plane_x = np.cross(sun_ray, [0, 0, 1])
        else:
            plane_x = np.cross(sun_ray, [1, 0, 0])
        plane_x = plane_x / np.linalg.norm(plane_x)
        plane_y = np.cross(sun_ray, plane_x)
        
        # Project heliostat positions onto plane
        projected_positions = []
        for pos in hstpos:
            x_proj = np.dot(pos, plane_x)
            y_proj = np.dot(pos, plane_y)
            depth = np.dot(pos, sun_ray)
            projected_positions.append((x_proj, y_proj, depth))
        
        # Process heliostats from closest to sun to farthest
        sorted_indices = np.argsort([p[2] for p in projected_positions])
        
        for idx, i in enumerate(sorted_indices):
            # Skip heliostats not facing the sun
            if np.dot(hst_norms[i], sun_ray) <= 0:
                continue
            
            x_i, y_i, depth_i = projected_positions[i]
            
            # Check for shadows from closer heliostats
            for j_idx in range(idx):
                j = sorted_indices[j_idx]
                x_j, y_j, depth_j = projected_positions[j]
                
                if depth_j >= depth_i:
                    continue
                
                # Calculate overlap between projected squares
                dx = x_i - x_j
                dy = y_i - y_j
                half_width = self.heliostat_width / 2
                
                if abs(dx) < half_width*2 and abs(dy) < half_width*2:
                    overlap_x = 2*half_width - abs(dx)
                    overlap_y = 2*half_width - abs(dy)
                    overlap_area = overlap_x * overlap_y
                    
                    overlap_fraction = min(1.0, overlap_area / self.heliostat_area)
                    shading_efficiency[i] -= overlap_fraction
        
        return np.clip(shading_efficiency, 0, 1)

    def get_reflectivity(self, num_heliostats):
        """
        Calculate mirror reflectivity efficiency.
        
        Accounts for mirror surface quality and cleanliness effects
        on reflected radiation.
        
        Args:
            num_heliostats (int): Number of heliostats in the field
            
        Returns:
            numpy array: Reflectivity efficiency for each heliostat [0-1]
        """
        initial_reflectivity = 0.95
        cleanliness_factor = 1.0
        reflectivity_factor = initial_reflectivity * cleanliness_factor
        return np.full(num_heliostats, reflectivity_factor)
    
    def project_point_onto_plane(self, p, p0, n):
        """
        Project a point onto a plane defined by point and normal vector.
        
        Args:
            p (numpy array): Point to project
            p0 (numpy array): Point on the plane
            n (numpy array): Plane normal vector
            
        Returns:
            numpy array: Projected point coordinates
        """
        return p - np.dot(p - p0, n) * n

    def get_blocking(self, hstpos, receiver_pos, sun_vec, search_radius=75, angle_thresh_deg=30):
        """
        Calculate blocking efficiency using Beam Plane Projection Method.
        
        Determines how much reflected radiation from each heliostat is
        blocked by other heliostats on its path to the receiver.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            receiver_pos (numpy array): Receiver position as 3-element array
            sun_vec (numpy array): Solar vector as 3-element array
            search_radius (float): Search radius for potential blockers in meters
            angle_thresh_deg (float): Angular threshold for blocking consideration
            
        Returns:
            numpy array: Blocking efficiency for each heliostat [0-1]
        """
        from shapely.geometry import Polygon
        from scipy.spatial import cKDTree
        
        # Ensure receiver_pos is numpy array
        if isinstance(receiver_pos, list):
            receiver_pos = np.array(receiver_pos)
        
        # Calculate heliostat normal vectors
        norms = self.get_normals(receiver_pos[2], hstpos, sun_vec)
        
        # Build heliostat geometry data
        heliostats = []
        for i in range(len(hstpos)):
            center = hstpos[i]
            normal = norms[i]
            
            # Create local coordinate system
            z = normal
            x = np.cross([0, 0, 1], z)
            if np.linalg.norm(x) < 1e-6:
                x = np.array([1, 0, 0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.column_stack((x, y, z))
            
            # Define heliostat corners in local coordinates
            corners_local = np.array([
                [-self.heliostat_width/2, -self.heliostat_height/2, 0],
                [self.heliostat_width/2, -self.heliostat_height/2, 0],
                [self.heliostat_width/2, self.heliostat_height/2, 0],
                [-self.heliostat_width/2, self.heliostat_height/2, 0]
            ])
            
            # Transform to world coordinates
            corners_world = np.dot(corners_local, R.T) + center
            
            heliostats.append({
                "pos": center,
                "normal": normal,
                "corners": corners_world
            })

        # Setup for beam direction calculations
        receiver_pos_far = receiver_pos + 100 * sun_vec
        angle_thresh_cos = np.cos(np.radians(angle_thresh_deg))
        
        # Build spatial index for efficient neighbor searching
        positions_2d = np.array([hst['pos'][:2] for hst in heliostats])
        tree = cKDTree(positions_2d)
        
        blocking_efficiency = np.ones(len(hstpos))
        
        for idx, hst in enumerate(heliostats):
            # Calculate beam direction (heliostat to receiver)
            r_vec = receiver_pos_far - hst['pos']
            r_vec /= np.linalg.norm(r_vec)
            
            # Project heliostat corners onto plane perpendicular to beam
            corners_proj = [self.project_point_onto_plane(pt, hst['pos'], r_vec) 
                           for pt in hst['corners']]
            
            # Create 2D coordinate system in projection plane
            z = r_vec
            x = np.cross([0, 0, 1], z)
            if np.linalg.norm(x) < 1e-6:
                x = np.array([1, 0, 0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            R_plane = np.vstack((x, y)).T
            
            # Convert projected corners to 2D coordinates
            beam_poly_2d = [np.dot(R_plane.T, pt - hst['pos']) for pt in corners_proj]
            beam_polygon = Polygon(beam_poly_2d)
            
            # Find neighboring heliostats within search radius
            neighbor_ids = tree.query_ball_point(hst['pos'][:2], r=search_radius)
            total_overlap_area = 0
            
            for j in neighbor_ids:
                if j == idx:
                    continue
                    
                blocker = heliostats[j]
                
                # Check if blocker is in direction of receiver
                b_vec = blocker['pos'] - hst['pos']
                b_vec /= np.linalg.norm(b_vec)
                
                if np.dot(b_vec, r_vec) > angle_thresh_cos:
                    # Project blocker corners onto same plane
                    corners_blocker = [self.project_point_onto_plane(pt, hst['pos'], r_vec) 
                                     for pt in blocker['corners']]
                    
                    # Convert to 2D coordinates
                    blocker_poly_2d = [np.dot(R_plane.T, pt - hst['pos']) 
                                     for pt in corners_blocker]
                    blocker_polygon = Polygon(blocker_poly_2d)
                    
                    # Calculate overlap area
                    try:
                        overlap = beam_polygon.intersection(blocker_polygon)
                        total_overlap_area += overlap.area
                    except:
                        continue
            
            # Calculate blocking efficiency
            if beam_polygon.area > 0:
                blocking_fraction = min(total_overlap_area / beam_polygon.area, 1.0)
                blocking_efficiency[idx] = 1.0 - blocking_fraction
            else:
                blocking_efficiency[idx] = 1.0
        
        return np.clip(blocking_efficiency, 0, 1)
    


    def get_attenuation(self, towerheight, focul_length, num_heliostats):
        """
        Calculate atmospheric attenuation efficiency.
        
        Accounts for atmospheric absorption of reflected radiation
        over the path from heliostat to receiver.
        
        Args:
            towerheight (float): Height of receiver tower in meters
            focul_length (numpy array): Focal lengths for each heliostat
            num_heliostats (int): Number of heliostats in field
            
        Returns:
            numpy array: Attenuation efficiency for each heliostat
        """
        attenuation = 1.0
        return np.full(num_heliostats, attenuation)
        
    def hflcal_spillage_efficiency(self, hst_pos, hst_norms, focal_lengths, rec_size, 
                                   o_sun, o_slope, o_track):
        """
        Calculate spillage efficiency using HFLCAL method with receiver incidence correction.
        
        Determines fraction of reflected radiation that successfully reaches
        the receiver aperture, accounting for beam spreading and optical errors.
        
        Args:
            hst_pos (numpy array): Heliostat positions as nx3 array
            hst_norms (numpy array): Heliostat normal vectors as nx3 array
            focal_lengths (numpy array): Focal length for each heliostat
            rec_size (tuple): Receiver dimensions (width, height) in meters
            o_sun (float): Sun shape error in radians
            o_slope (float): Slope error in radians
            o_track (float): Tracking error in radians
            
        Returns:
            numpy array: Spillage efficiency for each heliostat [0-1]
        """
        d = 10  # heliostat side length [m]
        w, h = rec_size  # receiver width and height
        receiver_normal = np.array([0, 0, 1])
        receiver_pos = np.array([0, 0, 62.0])
    
        N = hst_pos.shape[0]
        eta_spillage = np.zeros(N)
    
        for i in range(N):
            D = np.linalg.norm(hst_pos[i] - receiver_pos)
            n_i = hst_norms[i]
            cos_rec = np.abs(n_i[2])
    
            # Receiver incidence angle correction
            cos_phi_rec = np.clip(np.dot(n_i, receiver_normal) / 
                                (np.linalg.norm(n_i) * np.linalg.norm(receiver_normal)), 0, 1)
            correction_factor = cos_phi_rec ** 0.3044
    
            # Calculate beam quality and astigmatic errors
            o_bq = np.sqrt(2) * o_slope
            
            f = focal_lengths[i]
            w_t = d * np.abs(D / f - 1)
            w_s = d * np.abs(D / f - 1)
            w_t2 = w_t ** 2
            w_s2 = w_s ** 2
            o_ast = np.sqrt(0.5 * (w_t2 + w_s2)) / (4 * D)
    
            # Calculate total Gaussian beam width
            o_total = D * np.sqrt(o_sun**2 + o_bq**2 + o_ast**2 + o_track**2) / cos_rec
    
            # Integrate Gaussian distribution over receiver area
            def integrand(y, x):
                return np.exp(-(x**2 + y**2) / (2 * o_total**2))
    
            integral, _ = dblquad(integrand, -w / 2, w / 2, 
                                lambda x: -h / 2, lambda x: h / 2)
            eta = integral / (2 * np.pi * o_total**2)
    
            eta_spillage[i] = eta
    
        return eta_spillage

    def get_receiver_reflectivity(self, num_heliostats):
        """
        Calculate receiver surface reflectivity losses.
        
        Accounts for radiation reflected away from receiver surface
        rather than being absorbed.
        
        Args:
            num_heliostats (int): Number of heliostats in field
            
        Returns:
            float: Receiver reflectivity efficiency (constant)
        """
        reflectivity = 0.9
        return reflectivity

    # =========================================================================
    # VISUALIZATION AND PLOTTING FUNCTIONS
    # =========================================================================

    def plot_cosine(self, hstpos, cosine_factor, sun_vec, case_info=None):
        """
        Plot cosine efficiency distribution across the heliostat field.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            cosine_factor (numpy array): Cosine efficiency values
            sun_vec (numpy array): Solar vector for arrow visualization
            case_info (dict): Case study information for title
        """
        x = hstpos[:, 0]
        y = hstpos[:, 1]
        av = (view < np.pi/2.)
        
        plt.figure(1)
        cm = plt.colormaps.get_cmap('rainbow')
        cs = plt.scatter(x[av], y[av], c=cosine_factor[av], cmap=cm, s=25, vmin=0.4, vmax=1)
        plt.colorbar(cs)
        
        # Add sun vector arrow
        field_pos_x, field_pos_y = 300, 50
        arrow_scale = 100
        arrow_x = -sun_vec[0] * arrow_scale
        arrow_y = -sun_vec[1] * arrow_scale
        
        plt.arrow(field_pos_x, field_pos_y, arrow_x, arrow_y, 
                  width=10, head_width=25, head_length=30, fc='gold', ec='black', 
                  label='(-) Sun Vec')
        
        # Create title following specified structure
        if case_info:
            title = f"Pre-Evaluation Plot of Heliostat Cosine \n Efficiencies for {case_info['name']} Conditions in {case_info['id']}"
        else:
            title = 'Pre-Evaluation Plot of Heliostat Cosine \n Efficiencies'
            
        plt.title(title)
        plt.xlabel('X Position (m) (West-East)')
        plt.ylabel('Y-Position (m) (South-North)')
        plt.legend(loc='lower left', prop={'size': 8})
        plt.show()

    def plot_attenuation(self, hstpos, attenuation, sun_vec, case_info=None):
        """
        Plot atmospheric attenuation efficiency across the field.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            attenuation (numpy array): Attenuation efficiency values
            sun_vec (numpy array): Solar vector for arrow visualization
            case_info (dict): Case study information for title
        """
        x = hstpos[:, 0]
        y = hstpos[:, 1]
        
        plt.figure(1)
        fig, ax1 = plt.subplots()
        cm = plt.colormaps.get_cmap('rainbow')
        
        av = (view < 1.5)
        cs = plt.scatter(x[av], y[av], c=attenuation[av], cmap=cm)
        plt.colorbar(cs)
        
        # Add sun vector arrow
        field_pos_x, field_pos_y = 300, 50
        arrow_scale = 100
        arrow_x = -sun_vec[0] * arrow_scale
        arrow_y = -sun_vec[1] * arrow_scale
        
        plt.arrow(field_pos_x, field_pos_y, arrow_x, arrow_y, 
                  width=10, head_width=25, head_length=30, fc='gold', ec='black', 
                  label='(-) Sun Vec')
        
        nv = (view >= 1.5)
        plt.scatter(x[nv], y[nv], c='gray')
        
        # Create title following specified structure
        if case_info:
            title = f"Pre-Evaluation Plot of Heliostat Attenuation \n Efficiencies for {case_info['name']} Conditions in {case_info['id']}"
        else:
            title = 'Pre-Evaluation Plot of Heliostat Attenuation \n Efficiencies'
            
        plt.title(title)
        plt.xlabel('X Position (m) (West-East)')
        plt.ylabel('Y-Position (m) (South-North)')
        plt.legend(loc='lower left', prop={'size': 8})
        plt.show()

    def plot_shading_loss(self, hstpos, shading_efficiency, sun_vec, case_info=None):
        """
        Plot shading efficiency distribution across the field.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            shading_efficiency (numpy array): Shading efficiency values
            sun_vec (numpy array): Solar vector for arrow visualization
            case_info (dict): Case study information for title
        """
        x = hstpos[:, 0]
        y = hstpos[:, 1]
        av = (view < np.pi/2.)
        
        plt.figure(1)
        cm = plt.colormaps.get_cmap('rainbow')
        cs = plt.scatter(x[av], y[av], c=shading_efficiency[av], cmap=cm, s=25, vmin=0.6, vmax=1)
        plt.colorbar(cs)
        
        # Add sun vector arrow
        field_pos_x, field_pos_y = 300, 50
        arrow_scale = 100
        arrow_x = -sun_vec[0] * arrow_scale
        arrow_y = -sun_vec[1] * arrow_scale
        
        plt.arrow(field_pos_x, field_pos_y, arrow_x, arrow_y, 
                  width=10, head_width=25, head_length=30, fc='gold', ec='black', 
                  label='(-) Sun Vec')
        
        # Create title following specified structure
        if case_info:
            title = f"Pre-Evaluation Plot of Heliostat Shading \n Efficiencies for {case_info['name']} Conditions in {case_info['id']}"
        else:
            title = 'Pre-Evaluation Plot of Heliostat Shading \n Efficiencies'
            
        plt.title(title)
        plt.xlabel('X Position (m) (West-East)')
        plt.ylabel('Y-Position (m) (South-North)')
        plt.axis('equal')
        plt.legend(loc='lower left', prop={'size': 8})
        plt.show()

    def plot_blocking_loss(self, hstpos, blocking_efficiency, sun_vec, case_info=None):
        """
        Plot blocking efficiency distribution across the field.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            blocking_efficiency (numpy array): Blocking efficiency values
            sun_vec (numpy array): Solar vector for arrow visualization
            case_info (dict): Case study information for title
        """
        x = hstpos[:, 0]
        y = hstpos[:, 1]
        av = (view < np.pi/2.)
        
        plt.figure(1)
        cm = plt.colormaps.get_cmap('rainbow')
        cs = plt.scatter(x[av], y[av], c=blocking_efficiency[av], cmap=cm, s=25, vmin=0.6, vmax=1)
        plt.colorbar(cs)
        
        # Add sun vector arrow
        field_pos_x, field_pos_y = 300, 50
        arrow_scale = 100
        arrow_x = -sun_vec[0] * arrow_scale
        arrow_y = -sun_vec[1] * arrow_scale
        
        plt.arrow(field_pos_x, field_pos_y, arrow_x, arrow_y, 
                  width=10, head_width=25, head_length=30, fc='gold', ec='black', 
                  label='(-) Sun Vec')
        
        # Create title following specified structure
        if case_info:
            title = f"Pre-Evaluation Plot of Heliostat Blocking \n Efficiencies for {case_info['name']} Conditions in {case_info['id']}"
        else:
            title = 'Pre-Evaluation Plot of Heliostat Blocking \n Efficiencies'
            
        plt.title(title)
        plt.xlabel('X Position (m) (West-East)')
        plt.ylabel('Y-Position (m) (South-North)')
        plt.axis('equal')
        plt.legend(loc='lower left', prop={'size': 7})
        plt.show()

    def plot_spillage(self, hstpos, spillage, sun_vec, case_info=None):
        """
        Plot spillage efficiency distribution across the field.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            spillage (numpy array): Spillage efficiency values
            sun_vec (numpy array): Solar vector for arrow visualization
            case_info (dict): Case study information for title
        """
        x = hstpos[:, 0]
        y = hstpos[:, 1]
        av = (view < np.pi/2.)
        
        plt.figure(1)
        cm = plt.colormaps.get_cmap('rainbow')
        cs = plt.scatter(x[av], y[av], c=spillage[av], cmap=cm, s=25, vmin=0.4, vmax=1)
        plt.colorbar(cs)
        
        # Add sun vector arrow
        field_pos_x, field_pos_y = 300, 50
        arrow_scale = 100
        arrow_x = -sun_vec[0] * arrow_scale
        arrow_y = -sun_vec[1] * arrow_scale
        
        plt.arrow(field_pos_x, field_pos_y, arrow_x, arrow_y, 
                  width=10, head_width=25, head_length=30, fc='gold', ec='black', 
                  label='(-) Sun Vec')
        
        # Create title following specified structure
        if case_info:
            title = f"Pre-Evaluation Plot of Heliostat Spillage \n Efficiencies for {case_info['name']} Conditions in {case_info['id']}"
        else:
            title = 'Pre-Evaluation Plot of Heliostat Spillage \n Efficiencies'
            
        plt.title(title)
        plt.xlabel('X Position (m) (West-East)')
        plt.ylabel('Y-Position (m) (South-North)')
        plt.legend(loc='lower left', prop={'size': 8})
        plt.show()

    def plot_efficiency(self, hstpos, efficiency, sun_vec, case_info=None):
        """
        Plot overall field efficiency distribution.
        
        Args:
            hstpos (numpy array): Heliostat positions as nx3 array
            efficiency (numpy array): Combined efficiency values
            sun_vec (numpy array): Solar vector for arrow visualization
            case_info (dict): Case study information for title
        """
        x = hstpos[:, 0]
        y = hstpos[:, 1]
        
        plt.figure(1)
        fig, ax1 = plt.subplots()
        cm = plt.colormaps.get_cmap('rainbow')
        
        av = (view < 1.5)
        cs = plt.scatter(x[av], y[av], c=efficiency[av], cmap=cm, s=25, vmin=0.1, vmax=0.9)
        plt.colorbar(cs)
        
        # Add sun vector arrow
        field_pos_x, field_pos_y = 300, 50
        arrow_scale = 100
        arrow_x = -sun_vec[0] * arrow_scale
        arrow_y = -sun_vec[1] * arrow_scale
        
        plt.arrow(field_pos_x, field_pos_y, arrow_x, arrow_y, 
                  width=10, head_width=25, head_length=30, fc='gold', ec='black', 
                  label='(-) Sun Vec')
        
        nv = (view >= 1.5)
        plt.scatter(x[nv], y[nv], c='gray')
        
        # Create title following specified structure
        if case_info:
            title = f"Pre-Evaluation Plot of Heliostat Efficiencies Across \n Field for {case_info['name']} Conditions in {case_info['id']}"
        else:
            title = 'Pre-Evaluation Plot of Heliostat Efficiencies Across \n Field'
            
        plt.title(title)
        plt.xlabel('X Position (m) (West-East)')
        plt.ylabel('Y-Position (m) (South-North)')
        plt.legend(loc='lower left', prop={'size': 8})
        plt.show()


# =============================================================================
# MAIN EXECUTION SECTION
# =============================================================================

if __name__ == '__main__':
    
    # =========================================================================
    # FIELD CONFIGURATION PARAMETERS
    # =========================================================================
    
    # Load heliostat field layout data
    # File format: CSV with columns [x, y, z, focal_length, aim_x, aim_y, aim_z]
    pos_and_aiming = np.loadtxt('/"INPUT DIRECTORY"/pos_and_aiming.csv', 
                               delimiter=',', skiprows=2)
    pos = pos_and_aiming[:, :3]  # Heliostat positions [x, y, z]
    aim = pos_and_aiming[:, 4:]  # Aiming points
    focul_length = pos_and_aiming[:, 3]  # Focal lengths
    
    # Initialize field performance calculator
    field = FieldPF(np.r_[0, 1, 0])
    
    # Field geometry parameters
    towerheight = 62  # meters
    receiver_pos = [0, 0, towerheight]
    helio_width = 10  # meters
    helio_height = 10  # meters
    rec_width = 8  # meters
    rec_height = 6  # meters
    rec_size = (8, 6)
    rec_pos = np.array([0, 0, 62.0])
    
    # =========================================================================
    # CASE STUDY CONFIGURATION
    # =========================================================================
    
    # Define case study parameters
    # Note: Angles extracted from SOLSTICE sun position calculations
    # - Date: June 21st (Summer Solstice)
    # - Location: PS10 field, Spain (37°26'35"N 6°15'00"W)
    # - Azimuth normalized for program orientation (South=0, West=+90)
    
    case_studies = {
        'C1.1': {
            'name': 'Solar Noon',
            'description': 'Solar noon conditions - highest sun elevation',
            'azimuth': 0.0,
            'zenith': 13.987953925483858,
            'time': 'Solar Noon'
        },
        'C1.2': {
            'name': 'Morning',
            'description': 'Two hours after sunrise - low sun angle',
            'azimuth': -103.31434583806747,
            'zenith': 67.91774797592434,
            'time': '2 hours after sunrise'
        }
    }
    
    # SELECT CASE STUDY TO RUN
    ACTIVE_CASE = 'C1.1'  # Change to 'C1.2' for morning conditions
    
    # =========================================================================
    # COMPARATIVE ANALYSIS MODE (Optional)
    # =========================================================================
    
    RUN_COMPARISON = False  # Set to True to run both cases and compare
    
    if RUN_COMPARISON:
        print("Running comparative analysis of both case studies...")
        
        comparison_results = {}
        
        for case_id, case_config in case_studies.items():
            print(f"\nAnalyzing Case {case_id}: {case_config['name']}")
            
            # Set solar conditions for this case
            azimuth = case_config['azimuth']
            zenith = case_config['zenith']
            
            # Calculate solar geometry
            sun_vec = field.get_solar_vector(azimuth, zenith)
            norms = field.get_normals(towerheight, hstpos=pos, sun_vec=sun_vec)
            
            # Calculate efficiencies
            cos_eff = field.get_cosine(hst_norms=norms, sun_vec=sun_vec)
            shad_eff = field.get_shading(pos, sun_vec, norms)
            bloc_eff = field.get_blocking(pos, receiver_pos, sun_vec)
            spil_eff = field.hflcal_spillage_efficiency(pos, norms, focul_length, rec_size, 
                                                       o_sun, o_slope, o_track)
            helio_eff = cos_eff * shad_eff * ref_eff * bloc_eff * att_eff * spil_eff * rec_ref_eff
            
            # Store results
            comparison_results[case_id] = {
                'config': case_config,
                'efficiencies': {
                    'cosine': cos_eff,
                    'shading': shad_eff,
                    'blocking': bloc_eff,
                    'spillage': spil_eff,
                    'overall': helio_eff
                },
                'power_output': np.sum((incid_pow * helio_eff)) / 1e6  # MW
            }
        
        # Print comparison summary
        print(f"\n{'='*80}")
        print("CASE STUDY COMPARISON RESULTS")
        print(f"{'='*80}")
        
        for case_id, results in comparison_results.items():
            config = results['config']
            effs = results['efficiencies']
            print(f"\nCase {case_id}: {config['name']} ({config['time']})")
            print(f"  Solar Position: Az={config['azimuth']:.1f}°, Ze={config['zenith']:.1f}°")
            print(f"  Average Cosine Efficiency: {np.mean(effs['cosine']):.4f}")
            print(f"  Average Shading Efficiency: {np.mean(effs['shading']):.4f}")
            print(f"  Average Blocking Efficiency: {np.mean(effs['blocking']):.4f}")
            print(f"  Average Spillage Efficiency: {np.mean(effs['spillage']):.4f}")
            print(f"  Average Overall Efficiency: {np.mean(effs['overall']):.4f}")
            print(f"  Total Power Output: {results['power_output']:.2f} MW")
        
        # Calculate differences
        c11_power = comparison_results['C1.1']['power_output']
        c12_power = comparison_results['C1.2']['power_output']
        power_diff = ((c11_power - c12_power) / c12_power) * 100
        
        print(f"\nComparison Summary:")
        print(f"  Solar noon produces {power_diff:.1f}% more power than morning conditions")
        
        exit(0)  # Exit after comparison
    
    else:
        # Single case analysis
        current_case = case_studies[ACTIVE_CASE]
        azimuth = current_case['azimuth']
        zenith = current_case['zenith']
        
        print(f"Running Case Study {ACTIVE_CASE}: {current_case['name']}")
        print(f"Description: {current_case['description']}")
        print(f"Solar Position: Azimuth {azimuth:.1f}°, Zenith {zenith:.1f}°")
    
    # Optical error parameters (Gaussian standard deviations)
    o_sun = 2.24e-3    # Sun shape error (radians)
    o_slope = 2.5e-3   # Slope error (radians)
    o_track = 0.63e-3  # Tracking error (radians)
    
    num_heliostats = len(pos)
    
    # =========================================================================
    # SOLAR GEOMETRY CALCULATIONS
    # =========================================================================
    
    view = field.get_rec_view(towerheight, hstpos=pos)
    sun_vec = field.get_solar_vector(azimuth, zenith)
    norms = field.get_normals(towerheight, hstpos=pos, sun_vec=sun_vec)
    
    # =========================================================================
    # OPTICAL EFFICIENCY CALCULATIONS
    # =========================================================================
    
    print("Calculating optical efficiencies...")
    
    cos_eff = field.get_cosine(hst_norms=norms, sun_vec=sun_vec)
    shad_eff = field.get_shading(pos, sun_vec, norms)
    ref_eff = field.get_reflectivity(num_heliostats=len(pos))
    bloc_eff = field.get_blocking(pos, receiver_pos, sun_vec)
    att_eff = field.get_attenuation(towerheight=towerheight, focul_length=focul_length, num_heliostats=num_heliostats)
    spil_eff = field.hflcal_spillage_efficiency(pos, norms, focul_length, rec_size, 
                                               o_sun, o_slope, o_track)
    rec_ref_eff = field.get_reflectivity(num_heliostats=len(pos))
    
    # Calculate combined heliostat efficiency
    helio_eff = cos_eff * shad_eff * ref_eff * bloc_eff * att_eff * spil_eff * rec_ref_eff
    
    # =========================================================================
    # LOSS FACTOR CALCULATIONS
    # =========================================================================
    
    cos_loss = (1 - cos_eff)
    shad_loss = (1 - shad_eff)
    ref_loss = (1 - ref_eff)
    bloc_loss = (1 - bloc_eff)
    att_loss = (1 - att_eff)
    spil_loss = (1 - spil_eff)
    rec_ref_loss = (1 - rec_ref_eff)
    
    # =========================================================================
    # POWER CALCULATIONS
    # =========================================================================
    
    # Incident solar power parameters
    DNI = 1000  # Direct normal irradiance (W/m²)
    surface_area = helio_height * helio_width  # Heliostat surface area (m²)
    incid_pow = DNI * surface_area  # Incident power per heliostat (W)
    
    # Calculate cumulative power losses through optical chain
    cos_pow_loss = incid_pow * cos_loss
    shad_pow_loss = (incid_pow - cos_pow_loss) * shad_loss
    ref_pow_loss = (incid_pow - cos_pow_loss - shad_pow_loss) * ref_loss
    bloc_pow_loss = (incid_pow - cos_pow_loss - shad_pow_loss - ref_pow_loss) * bloc_loss
    att_pow_loss = (incid_pow - cos_pow_loss - shad_pow_loss - ref_pow_loss - bloc_pow_loss) * att_loss
    spil_pow_loss = (incid_pow - cos_pow_loss - shad_pow_loss - ref_pow_loss - 
                     bloc_pow_loss - att_pow_loss) * spil_loss
    rec_ref_pow_loss = (incid_pow - cos_pow_loss - shad_pow_loss - ref_pow_loss - 
                        bloc_pow_loss - att_pow_loss - spil_pow_loss) * rec_ref_loss
    
    # Calculate net power absorbed at receiver
    rec_abs_power = (incid_pow - cos_pow_loss - shad_pow_loss - ref_pow_loss - 
                     bloc_pow_loss - att_pow_loss - spil_pow_loss - rec_ref_pow_loss)
    
    # =========================================================================
    # PERFORMANCE RANKING CALCULATIONS
    # =========================================================================
    
    # Calculate performance rankings (1 = best performance)
    helio_eff_ranks = np.argsort(np.argsort(-helio_eff)) + 1
    helio_pow_ranks = np.argsort(np.argsort(-rec_abs_power)) + 1
    cos_eff_ranks = np.argsort(np.argsort(-cos_eff)) + 1
    shad_eff_ranks = np.argsort(np.argsort(-shad_eff)) + 1
    bloc_eff_ranks = np.argsort(np.argsort(-bloc_eff)) + 1
    spil_eff_ranks = np.argsort(np.argsort(-spil_eff)) + 1
    
    # =========================================================================
    # VISUALIZATION PLOTS
    # =========================================================================
    
    print("Generating visualization plots...")
    
    # Prepare case information for plot titles
    case_info = {
        'id': ACTIVE_CASE,
        'name': current_case['name']
    }
    
    cosine_plot = field.plot_cosine(hstpos=pos, cosine_factor=cos_eff, sun_vec=sun_vec, case_info=case_info)
    shading_plot = field.plot_shading_loss(pos, shad_eff, sun_vec=sun_vec, case_info=case_info)
    blocking_plot = field.plot_blocking_loss(pos, bloc_eff, sun_vec=sun_vec, case_info=case_info)
    attenuation_plot = field.plot_attenuation(hstpos=pos, attenuation=att_eff, sun_vec=sun_vec, case_info=case_info)
    spillage_plot = field.plot_spillage(hstpos=pos, spillage=spil_eff, sun_vec=sun_vec, case_info=case_info)
    overall_efficiency_plot = field.plot_efficiency(hstpos=pos, efficiency=helio_eff, sun_vec=sun_vec, case_info=case_info)
    
    # =========================================================================
    # CSV DATA EXPORT
    # =========================================================================
    
    print("Exporting results to CSV files...")
    
    # Define CSV file paths
    efficiencies_csv_path = '/"INPUT DIRECTORY"/Heliostat_Efficiencies.csv'
    rankings_csv_path = '/"INPUT DIRECTORY"Heliostat_Rankings.csv'
    
    # Export efficiency values
    with open(efficiencies_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        
        # Write header
        header = [
            'Heliostat_index', 'x_coordinate', 'y_coordinate',
            'cos_eff', 'shad_eff', 'bloc_eff', 'spil_eff', 'helio_eff'
        ]
        csv_writer.writerow(header)
        
        # Write data rows
        for i in range(num_heliostats):
            row = [
                i, pos[i, 0], pos[i, 1],
                cos_eff[i], shad_eff[i], bloc_eff[i], spil_eff[i], helio_eff[i]
            ]
            csv_writer.writerow(row)
    
    # Export performance rankings
    with open(rankings_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        
        # Write header
        header = [
            'Heliostat_index', 'x_coordinate', 'y_coordinate',
            'cos_eff_rank', 'shad_eff_rank', 'bloc_eff_rank', 'spil_eff_rank', 'helio_eff_rank'
        ]
        csv_writer.writerow(header)
        
        # Write data rows
        for i in range(num_heliostats):
            row = [
                i, pos[i, 0], pos[i, 1],
                cos_eff_ranks[i], shad_eff_ranks[i], bloc_eff_ranks[i], 
                spil_eff_ranks[i], helio_eff_ranks[i]
            ]
            csv_writer.writerow(row)
    
    # =========================================================================
    # PERFORMANCE STATISTICS SUMMARY
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("HELIOSTAT FIELD PERFORMANCE ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print(f"\nField Configuration:")
    print(f"  Total heliostats: {num_heliostats}")
    print(f"  Tower height: {towerheight} m")
    print(f"  Heliostat size: {helio_width} × {helio_height} m")
    print(f"  Receiver size: {rec_width} × {rec_height} m")
    print(f"  Solar conditions: Azimuth {azimuth:.1f}°, Zenith {zenith:.1f}°")
    
    print(f"\nField-Wide Efficiency Statistics:")
    print(f"  Average Cosine Efficiency: {np.mean(cos_eff):.4f} ± {np.std(cos_eff):.4f}")
    print(f"  Average Shading Efficiency: {np.mean(shad_eff):.4f} ± {np.std(shad_eff):.4f}")
    print(f"  Average Blocking Efficiency: {np.mean(bloc_eff):.4f} ± {np.std(bloc_eff):.4f}")
    print(f"  Average Spillage Efficiency: {np.mean(spil_eff):.4f} ± {np.std(spil_eff):.4f}")
    print(f"  Average Overall Efficiency: {np.mean(helio_eff):.4f} ± {np.std(helio_eff):.4f}")
    print(f"  Total Field Power Output: {np.sum(rec_abs_power)/1e6:.2f} MW")
    
    # =========================================================================
    # ALTERNATIVE Functions
    # =========================================================================
    
    '''
    def get_attenuation(self, towerheight, focul_length, num_heliostats):
        """
        Calculate atmospheric attenuation efficiency.
        
        Accounts for atmospheric absorption of reflected radiation
        over the path from heliostat to receiver.
        
        Args:
            towerheight (float): Height of receiver tower in meters
            focul_length (numpy array): Focal lengths for each heliostat
            num_heliostats (int): Number of heliostats in field
            
        Returns:
            numpy array: Attenuation efficiency for each heliostat
        """
        attenuation = 0.99321 - 0.0001176*focul_length+1.97*10**(-8)*focul_length**2
        #not relvant for current field as heliostats lie within 1000m
        return attenuation
        
        elif focul_length > 1000:
            attenuation = np.exp(-0.0001106*focul_length)
          
        
        attenuation = 1.0
        return np.full(num_heliostats, attenuation)
    def get_corners(self, center, normal, width=None, height=None):
         """
         Calculate the corner positions of a heliostat given its center and normal vector.
         
         Arguments:
             center (numpy array): Center position of the heliostat (3,)
             normal (numpy array): Normal vector of the heliostat (3,)
             width (float): Width of the heliostat (default: self.heliostat_width)
             height (float): Height of the heliostat (default: self.heliostat_height)
             
         Returns:
             numpy array: Corner positions of the heliostat (4x3)
         """
         if width is None:
             width = self.heliostat_width
         if height is None:
             height = self.heliostat_height
             
         z = normal
         x = np.cross([0, 0, 1], z)
         if np.linalg.norm(x) < 1e-6:
             x = np.array([1, 0, 0])
         x /= np.linalg.norm(x)
         y = np.cross(z, x)
         R = np.column_stack((x, y, z))
         local = np.array([
             [-width/2, -height/2, 0],
             [ width/2, -height/2, 0],
             [ width/2,  height/2, 0],
             [-width/2,  height/2, 0]
         ])
         return (R @ local.T).T + center

    def to_local_2d(self, points, origin, normal):
         """
         Convert 3D points to local 2D coordinates on a plane defined by origin and normal.
         
         Arguments:
             points (numpy array): 3D points to convert (nx3)
             origin (numpy array): Origin of the local coordinate system (3,)
             normal (numpy array): Normal vector of the plane (3,)
             
         Returns:
             numpy array: 2D coordinates in the local plane (nx2)
         """
         x_axis = np.cross([0, 0, 1], normal)
         if np.linalg.norm(x_axis) < 1e-6:
             x_axis = np.array([1, 0, 0])
         x_axis /= np.linalg.norm(x_axis)
         y_axis = np.cross(normal, x_axis)
         basis = np.stack((x_axis, y_axis)).T
         return (points - origin) @ basis

    def get_blocking(self, hstpos, receiver_pos, sun_vec):
         """
         Calculate blocking losses using adaptive ray-tracing method.
         This function calculates the fraction of each heliostat's surface that is blocked
         by other heliostats when reflecting sunlight toward the receiver.
         
         Arguments:
             hstpos (numpy array): Positions of all heliostats in the field (nx3)
             receiver_pos (numpy array or list): Position of the receiver [x, y, z]
             sun_vec (numpy array): Solar vector (3,)
             
         Returns:
             numpy array: Blocking efficiency for each heliostat (values between 0 and 1)
         """
         # Convert receiver_pos to numpy array if it's a list
         if isinstance(receiver_pos, list):
             receiver_pos = np.array(receiver_pos)
         
         num_hst = len(hstpos)
         blocking_efficiency = np.ones(num_hst)  # Initialize with no blocking
         
         # Calculate heliostat normal vectors (same as in get_normals)
         tower_vec = receiver_pos - hstpos
         tower_vec /= np.linalg.norm(tower_vec, axis=1)[:, None]
         
         # Normalize sun vector
         sun_vec_norm = sun_vec / np.linalg.norm(sun_vec)
         
         # Calculate normal vectors for optimal reflection
         normals = sun_vec_norm + tower_vec
         normals /= np.linalg.norm(normals, axis=1)[:, None]

         # Create heliostat data structures
         heliostats = [{
             "id": i,
             "pos": hstpos[i],
             "normal": normals[i],
             "corners": self.get_corners(hstpos[i], normals[i])
         } for i in range(num_hst)]

         # Create spatial index for efficient neighbor searching
         tree = cKDTree(hstpos[:, :2])
         
         # Calculate solar zenith angle for adaptive sampling
         zenith_deg = np.degrees(np.arccos(np.clip(sun_vec_norm[2], -1, 1)))

         for hst in heliostats:
             idx, pos, norm = hst["id"], hst["pos"], hst["normal"]
             
             # Calculate ray direction from heliostat to receiver
             r_vec = receiver_pos - pos
             r_vec /= np.linalg.norm(r_vec)
             
             # Adaptive sampling density based on local conditions
             base = 3
             if len(tree.query_ball_point(pos[:2], 20)) >= 10:  # Dense area
                 base += 1
             if zenith_deg > 60:  # High zenith angle
                 base += 1
             sample_density = min(10, base)
             
             # Create sampling grid on heliostat surface
             xs = np.linspace(-self.heliostat_width/2, self.heliostat_width/2, sample_density)
             ys = np.linspace(-self.heliostat_height/2, self.heliostat_height/2, sample_density)
             grid = np.array([[x, y, 0] for y in ys for x in xs])
             
             # Transform grid to world coordinates
             z = norm
             x = np.cross([0, 0, 1], z)
             if np.linalg.norm(x) < 1e-6:
                 x = np.array([1, 0, 0])
             x /= np.linalg.norm(x)
             y = np.cross(z, x)
             R = np.column_stack((x, y, z))
             world_points = np.dot(grid, R.T) + pos

             # Count blocked sample points
             blocked = 0
             candidates = tree.query_ball_point(pos[:2], 75)  # Search within 75m radius
             
             for pt in world_points:
                 for j in candidates:
                     if j == idx:  # Skip self
                         continue
                         
                     blocker = heliostats[j]
                     
                     # Ray-plane intersection test
                     denom = np.dot(r_vec, blocker["normal"])
                     if abs(denom) < 1e-6:  # Ray parallel to plane
                         continue
                         
                     t = np.dot(blocker["pos"] - pt, blocker["normal"]) / denom
                     if t <= 0:  # Intersection behind the ray origin
                         continue
                         
                     # Calculate intersection point
                     inter = pt + t * r_vec
                     
                     # Check if intersection point is within heliostat boundaries
                     poly = Polygon(self.to_local_2d(blocker["corners"], blocker["pos"], blocker["normal"]))
                     test_pt = self.to_local_2d(np.array([inter]), blocker["pos"], blocker["normal"])[0]
                     
                     if poly.contains(Point(test_pt)):
                         blocked += 1
                         break  # Point is blocked, no need to check other blockers

             # Calculate blocking fraction
             blocking_fraction = blocked / len(world_points)
             blocking_efficiency[idx] = 1.0 - blocking_fraction

         return np.clip(blocking_efficiency, 0, 1)
     '''
    
    # =========================================================================
    # EXECUTION TIMING
    # =========================================================================
    
    execution_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Analysis completed in {execution_time:.2f} seconds")
    print(f"{'='*60}")