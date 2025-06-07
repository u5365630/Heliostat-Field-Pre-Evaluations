#
# This is an example script to simulate a central receiver system (CRS)
# using Solstice via solsticepy
#
import solsticepy
from solsticepy.master import Master
import numpy as np
import os


def solar_noon():
    '''
    C1.1 full field, Pillbox sunshape, solar noon
    '''
    #==================================================
    # INPUT PARAMETERS

    # the sun
    # =========

    DNI = 1000 # W/m2
    sunshape = 'pillbox'
    half_angle_deg = 0.2664
    sun = solsticepy.Sun(dni=DNI, sunshape=sunshape, half_angle_deg=half_angle_deg)

    # S3. sun position
    # e.g. summer solstice, solar noon
    azimuth=270.   # from East to North, deg
    elevation =78. # 0 is horizontal, deg
    # S4. number of rays for the ray-tracing simulation
    num_rays=2000000
    #
    # the field
    # ==========
    # F1.Layout
    layoutfile='./demo_layout.csv'
    hst_field=True # simulate the full heliostat field (if False: give single heliostat information)

    # F2. Heliostat
    hst_w=10. # m
    hst_h=10. # m
    rho_refl=0.95 # mirror reflectivity
    slope_error=2.e-3 # radians
    # F3. Tower
    tower_h=0.01 # tower height
    tower_r=0.01 # tower radius
    #
    # the receiver
    # ============
    # R1. shape
    receiver='flat' # 'flat' or 'stl'
    # R2. Size
    rec_w=8. # width, m
    rec_h=6. # height, m
    # R3. tilt angle
    tilt=0.  # deg
    # R4. position
    loc_x=0. # m
    loc_y=0. # m
    loc_z=62.# m
    # R5. Abosrptivity
    rec_abs=0.9
    rec_mesh=100

    # set the folder name for saving the output files
    # False: an automatically generated name  
    # or
    # string: name of the user defined folder
    # (Note: you must set this folder name if you want to use new_case==False)
    userdefinedfolder='validation-C1-1'

    # NO NEED TO CHANGE THE CONTENT BELOW
    # ===============================================================
    # the ray-tracing scene will be generated 
    # based on the settings above

    if hst_field:
	    # extract the heliostat positions from the loaded CSV file
        layout=np.loadtxt(layoutfile, delimiter=',', skiprows=2)
        hst_pos=layout[:,:3]
        hst_foc=layout[:,3] # F2.5
        hst_aims=layout[:,4:] # F4.
        one_heliostat=False
    else:
        one_heliostat=True

    casefolder=userdefinedfolder

    rec_param=np.r_[rec_w, rec_h, rec_mesh, rec_mesh, loc_x, loc_y, loc_z, tilt]

    master=Master(casedir=casefolder)
    outfile_yaml = master.in_case(folder=casefolder, fn='input.yaml')
    outfile_recv = master.in_case(folder=casefolder, fn='input-rcv.yaml')

    # generate the YAML file from the input parameters specified above
    solsticepy.gen_yaml(sun, hst_pos, hst_foc, hst_aims,hst_w, hst_h
	    , rho_refl, slope_error, receiver, rec_param, rec_abs
	    , outfile_yaml=outfile_yaml, outfile_recv=outfile_recv
	    , hemisphere='North', tower_h=tower_h, tower_r=tower_r,  spectral=False
	    , medium=0, one_heliostat=one_heliostat)

    # run Solstice using the generate inputs, and run all required post-processing
    eta, performance_hst=master.run(azimuth, elevation, num_rays, rho_refl,sun.dni, folder=casefolder, gen_vtk=True, verbose=True)
    print('efficiency', eta)
    # annual solution (see instructions)
    #master.run_annual(nd=5, nh=5, latitude=latitude, num_rays=num_rays, num_hst=len(hst_pos),rho_mirror=rho_refl, dni=DNI)


def morning():
    '''
    C1.2 full field, Pillbox sunshape, morning
    '''
    #==================================================
    # INPUT PARAMETERS

    # the sun
    # =========

    DNI = 1000 # W/m2
    sunshape = 'pillbox'
    half_angle_deg = 0.2664
    sun = solsticepy.Sun(dni=DNI, sunshape=sunshape, half_angle_deg=half_angle_deg)

    # S3. sun position
    # e.g. summer solstice, solar noon
    azimuth=90.-76.   # from East to North, deg
    elevation =90.-68. # 0 is horizontal, deg
    # S4. number of rays for the ray-tracing simulation
    num_rays=2000000
    #
    # the field
    # ==========
    # F1.Layout
    layoutfile='./demo_layout.csv'
    hst_field=True # simulate the full heliostat field (if False: give single heliostat information)

    # F2. Heliostat
    hst_w=10. # m
    hst_h=10. # m
    rho_refl=0.95 # mirror reflectivity
    slope_error=2.e-3 # radians
    # F3. Tower
    tower_h=0.01 # tower height
    tower_r=0.01 # tower radius
    #
    # the receiver
    # ============
    # R1. shape
    receiver='flat' # 'flat' or 'stl'
    # R2. Size
    rec_w=8. # width, m
    rec_h=6. # height, m
    # R3. tilt angle
    tilt=0.  # deg
    # R4. position
    loc_x=0. # m
    loc_y=0. # m
    loc_z=62.# m
    # R5. Abosrptivity
    rec_abs=0.9
    rec_mesh=100

    # set the folder name for saving the output files
    # False: an automatically generated name  
    # or
    # string: name of the user defined folder
    # (Note: you must set this folder name if you want to use new_case==False)
    userdefinedfolder='validation-C1-2'

    # NO NEED TO CHANGE THE CONTENT BELOW
    # ===============================================================
    # the ray-tracing scene will be generated 
    # based on the settings above

    if hst_field:
	    # extract the heliostat positions from the loaded CSV file
        layout=np.loadtxt(layoutfile, delimiter=',', skiprows=2)
        hst_pos=layout[:,:3]
        hst_foc=layout[:,3] # F2.5
        hst_aims=layout[:,4:] # F4.
        one_heliostat=False
    else:
        one_heliostat=True

    casefolder=userdefinedfolder

    rec_param=np.r_[rec_w, rec_h, rec_mesh, rec_mesh, loc_x, loc_y, loc_z, tilt]

    master=Master(casedir=casefolder)
    outfile_yaml = master.in_case(folder=casefolder, fn='input.yaml')
    outfile_recv = master.in_case(folder=casefolder, fn='input-rcv.yaml')

    # generate the YAML file from the input parameters specified above
    solsticepy.gen_yaml(sun, hst_pos, hst_foc, hst_aims,hst_w, hst_h
	    , rho_refl, slope_error, receiver, rec_param, rec_abs
	    , outfile_yaml=outfile_yaml, outfile_recv=outfile_recv
	    , hemisphere='North', tower_h=tower_h, tower_r=tower_r,  spectral=False
	    , medium=0, one_heliostat=one_heliostat)

    # run Solstice using the generate inputs, and run all required post-processing
    eta, performance_hst=master.run(azimuth, elevation, num_rays, rho_refl,sun.dni, folder=casefolder, gen_vtk=True, verbose=True)
    print('efficiency', eta)
    # annual solution (see instructions)
    #master.run_annual(nd=5, nh=5, latitude=latitude, num_rays=num_rays, num_hst=len(hst_pos),rho_mirror=rho_refl, dni=DNI)
if __name__=='__main__':
    #solar_noon()
    morning()
