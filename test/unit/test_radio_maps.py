#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import mitsuba as mi
import numpy as np
import drjit as dr

import sionna.rt as rt
from sionna.rt import load_scene, Transmitter, PlanarArray, ITURadioMaterial,\
    Receiver, PathSolver, RadioMapSolver, BackscatteringPattern
from sionna.rt.utils import dbm_to_watt


####################################################
# Utilities
####################################################

def paths_to_coverage_map(paths):
    """
    Converts paths into the equivalent coverage map values.
    The coverage map is assumed to be a square.
    """
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_real, a_imag = paths.a
    a = a_real.numpy() + 1j*a_imag.numpy()

    # Transmit precoding
    # Assume default precoding
    # [num_rx, num_rx_ant, num_tx, num_paths]
    a /= np.sqrt(a.shape[3])
    a = np.sum(a, axis=3)

    # Sum energy of paths
    a = np.square(np.abs(a))
    # [num_rx, num_tx]
    a = np.sum(a, axis=(1, 3))

    # Swap dims
    # [num_tx, num_rx]
    a = a.T

    # Reshape to coverage map
    n = int(np.sqrt(a.shape[1]))
    shape = [a.shape[0], n, n]
    a = np.reshape(a, shape)

    return a

def validate_cm(los=False,
                specular_reflection=False,
                diffuse_reflection=False,
                refraction=False,
                rm_center_z=1,
                tx_pattern="iso",
                rx_pattern="iso",
                tx_pol="V"):
    """Compares coverage map against exact path calculation for
    different propagation phenomena.
    """

    scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)

    scene.get("reflector").radio_material = ITURadioMaterial("mat-concrete", "concrete", thickness=0.1)
    rm = scene.get("reflector").radio_material
    scene.get("reflector").radio_material.scattering_coefficient = mi.Float(np.sqrt(0.5))
    scene.get("reflector").radio_material.xpd_coefficient = mi.Float(0.3)

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern=tx_pattern,
                                 polarization=tx_pol)

    scene.rx_array =  PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern=rx_pattern,
                                  polarization="VH")

    delta = 0.1
    width = 3
    tx = Transmitter(name="tx",
                     position=mi.Point3f(0, 0, .1),
                     orientation=mi.Point3f(0,0,0))
    scene.add(tx)

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=1,
                   cell_size=mi.Point2f(delta, delta),
                   center=mi.Point3f(0,0,rm_center_z),
                   orientation=mi.Point3f(0.,0.,0.),
                   size=mi.Point2f(width, width),
                   samples_per_tx=int(1e7),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)

    cell_centers = rm.cell_centers.numpy()
    cell_centers = np.reshape(cell_centers, [-1,3])
    for i, pos in enumerate(cell_centers):
        scene.add(Receiver(name=f"rx-{i}",
                           position=mi.Point3f(pos),
                           orientation=mi.Point3f(0., 0., 0.)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src=int(1e4),
                   max_num_paths_per_src=int(1e7),
                   max_depth=1,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)

    rm_theo = paths_to_coverage_map(paths)[0]
    rm_rt =rm.path_gain.numpy()[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(rm_rt - rm_theo) / rm_theo
    nmse_db = 10*np.log10(np.mean(err[np.isfinite(err)]**2))
    return rm, rm_theo, nmse_db


#############################################################
# Tests
#############################################################

def test_random_positions():
    """test that random positions have a valid path loss and min/max
    distance is correctly set."""

    cell_size = mi.Point2f([4., 5.])
    batch_size = 100
    tx_pos = mi.Point3f(-210,73,105) # Top of Frauenkirche

    scene = load_scene(rt.scene.munich)

    tx = Transmitter(name="tx", position=tx_pos)
    scene.add(tx)

    scene.tx_array = PlanarArray(num_rows=4,
                                num_cols=4,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
    scene.rx_array = PlanarArray(num_rows=4,
                                num_cols=4,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")

    # Position of the measurement surface
    radio_map_pos = dr.copy(scene.transmitters["tx"].position)
    radio_map_pos.z = 1.5



    ### Check with centering set to True

    # Generate the radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=radio_map_pos,
                   orientation=(0., 0., 0.),
                   size=(500., 500.),
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   samples_per_tx=int(1e7))
    
    samples_pos, samples_cell_ind = rm.sample_positions(batch_size,
                                                        min_val_db=-110,
                                                        center_pos=True)
    samples_pos = samples_pos.numpy()
    cell_centers = rm.cell_centers.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    samples_cell_ind = samples_cell_ind.numpy()
    samples_cell_ind = np.squeeze(samples_cell_ind, axis=0)
    # Check that the transmitter is always at the center of the cell
    for p, cell_ind in zip(samples_pos, samples_cell_ind):
        cell_center = cell_centers[cell_ind[0], cell_ind[1]]
        d = np.linalg.norm(p - cell_center)
        assert d == 0.

    ### Check with centering set to False

    # Generate the radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=radio_map_pos,
                   orientation=(0., 0., 0.),
                   size=(500., 500.),
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   samples_per_tx=int(1e7))
    samples_pos, samples_cell_ind = rm.sample_positions(batch_size,
                                                        min_val_db=-110,
                                                        center_pos=False)
    samples_pos = samples_pos.numpy()
    cell_centers = rm.cell_centers.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    samples_cell_ind = samples_cell_ind.numpy()
    samples_cell_ind = np.squeeze(samples_cell_ind, axis=0)
    # Check that the transmitter is always in the cell
    for p, cell_ind in zip(samples_pos, samples_cell_ind):
        cell_center = cell_centers[cell_ind[0], cell_ind[1]]
        d = np.abs(p - cell_center)
        assert (d[0] <= cell_size.x*0.5) and (d[1] <= cell_size.y*0.5)\
                and (d[2] == 0.0)

    ### Test min and max distance

    batch_size = 1000
    d_min = 150
    d_max = 300

    # max distance offset due to cell size quantization
    # dist can be off at most by factor 0.5 of diagonal
    d_cell = 0.5*dr.norm(cell_size).numpy()[0]
    low = d_min - d_cell
    high = d_max + d_cell

    samples_pos, _ = rm.sample_positions(batch_size,
                                        min_dist=d_min,
                                        max_dist=d_max,
                                        center_pos=False)
    samples_pos = samples_pos.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    tx_pos = tx.position.numpy().T[0]
    for p in samples_pos:
        d = np.linalg.norm(p - tx_pos)
        assert (d > low) and (d < high)

    ### Test TX associations

    # Remove transmitter
    scene.remove("tx")

    # Add the first transmitter
    tx0 = Transmitter(name='tx0',
                        position=mi.Point3f(150, -100, 20),
                        orientation=mi.Point3f(0., 0., dr.pi*5/6),
                        power_dbm=44)
    scene.add(tx0)

    # Add the second transmitter
    tx1 = Transmitter(name='tx1',
                    position=mi.Point3f(-150, -100, 20),
                    orientation=mi.Point3f(0., 0., dr.pi/60),
                    power_dbm=44)
    scene.add(tx1)

    # Add the third transmitter
    tx2 = Transmitter(name='tx2',
                    position=mi.Point3f(0, 150 * dr.tan(dr.pi/3) - 100, 20),
                    orientation=mi.Point3f(0., 0., -dr.pi/2),
                    power_dbm=44)
    scene.add(tx2)

    # Compute radio map
    rm_solver = RadioMapSolver()
    
    rm = rm_solver(scene,
                   max_depth=5,
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   samples_per_tx=int(1e7))

    metric = 'sinr'
    sm = getattr(rm, 'sinr').numpy()
    cell_to_tx_ideal = np.argmax(sm, axis=0)
    cell_to_tx_ideal[np.max(sm, axis=0)==0] = -1

    _, samples_cell_ind = rm.sample_positions(batch_size,
                                            center_pos=False,
                                            tx_association=True,
                                            metric=metric)
    samples_cell_ind = samples_cell_ind.numpy()

    for tx in range(sm.shape[0]):
        for cell_ind in samples_cell_ind[tx]:
            assert cell_to_tx_ideal[cell_ind[0], cell_ind[1]] == tx

def test_sinr_map():
    """Test SINR map"""

    cm_cell_size = np.array([4., 5.])

    scene = load_scene(rt.scene.munich)

    # Add the first transmitter
    tx0 = Transmitter(name='tx0',
                        position=mi.Point3f(150, -100, 20),
                        orientation=mi.Point3f(0., 0., dr.pi*5/6),
                        power_dbm=44)
    scene.add(tx0)

    # Add the second transmitter
    tx1 = Transmitter(name='tx1',
                    position=mi.Point3f(-150, -100, 20),
                    orientation=mi.Point3f(0., 0., dr.pi/60),
                    power_dbm=44)
    scene.add(tx1)

    # Add the third transmitter
    tx2 = Transmitter(name='tx2',
                    position=mi.Point3f(0, 150 * dr.tan(dr.pi/3) - 100, 20),
                    orientation=mi.Point3f(0., 0., -dr.pi/2),
                    power_dbm=44)
    scene.add(tx2)

    scene.tx_array = PlanarArray(num_rows=4,
                                num_cols=4,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
    scene.rx_array = PlanarArray(num_rows=4,
                                num_cols=4,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")

    rx_pos = dr.copy(scene.transmitters["tx0"].position)
    rx_pos.z = 1.5

    # generate coverage map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=rx_pos,
                   orientation=mi.Point3f(0., 0., 0.),
                   size=mi.Point2f(500., 500.),
                   cell_size=cm_cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   samples_per_tx=int(1e7))

    # retrieve path gain map
    path_gain = rm.path_gain.numpy()

    # retrieve transmit power
    tx_power = [dbm_to_watt(tx.power_dbm).numpy()[0]
                for tx in scene.transmitters.values()]

    # retrieve noise power
    n0 = scene.thermal_noise_power.numpy()[0]

    num_tx = rm.num_tx
    sinr_map_per_tx_ideal = np.zeros(path_gain.shape)

    # compute the SINR via Numpy operations
    for tx in range(num_tx):
        interf_mat = np.zeros(path_gain.shape[1:])
        for tx_interf in range(num_tx):
            # compute interference
            if tx_interf != tx:
                interf_mat += tx_power[tx_interf] * \
                    path_gain[tx_interf, ::]
        # SINR per tx,  assuming that all tiles connect to transmitter tx
        sinr_map_per_tx_ideal[tx, ::] = tx_power[tx] * \
            path_gain[tx, ::] / (interf_mat + n0)

    # Tile to tx association
    tile_to_tx_ideal = np.argmax(sinr_map_per_tx_ideal, axis=0)
    tile_to_tx_ideal[np.max(sinr_map_per_tx_ideal, axis=0) == 0] = -1

    # Get SINR map and RT association from RT
    sinr_map_per_tx = rm.sinr.numpy()
    tile_to_tx = rm.tx_association('sinr').numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        err_sinr_map_per_tx = np.sort(np.abs((sinr_map_per_tx - sinr_map_per_tx_ideal)\
                                             /sinr_map_per_tx_ideal).flatten())
    err_sinr_map_per_tx = err_sinr_map_per_tx[np.isfinite(err_sinr_map_per_tx)]

    err_tile_to_tx = np.max(abs(tile_to_tx_ideal - tile_to_tx))

    # check that 90-th percentile SINR has small error
    assert err_sinr_map_per_tx[int(.99*len(err_sinr_map_per_tx))] < .01
    assert err_tile_to_tx == 0

def test_los():
    """Test that LoS pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(los=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(los=True, tx_pol="cross")
    assert nmse_db < -20

def test_specular_reflection():
    """Test that reflection pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(specular_reflection=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(specular_reflection=True, tx_pol="cross")
    assert nmse_db <  -20

def test_scattering():
    """Test that scattering pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(diffuse_reflection=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(diffuse_reflection=True, tx_pol="V")
    assert nmse_db <  -20

def test_refraction():
    """Test that refraction pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(refraction=True, tx_pol="V", rm_center_z=-1)
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(refraction=True, tx_pol="cross", rm_center_z=-1)
    assert nmse_db <  -20

def test_box_01():
    """Test that field scattered and reflected fields jointly match for
    max_depth=1 in the box scene.
    This test also applies orientation, directive antenna patterns, as well as
    depolariation during scattering.
    """
    los = False
    specular_reflection = True
    diffuse_reflection = True
    refraction=False
    max_depth = 1 # Test only works for max_depth=1
    width=10
    delta = 1

    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = dr.sqrt(0.5)
    scene.objects["box"].radio_material.scattering_pattern =\
        BackscatteringPattern(alpha_r=30, alpha_i=10, lambda_=0.5)

    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    scene.rx_array =  PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="VH")

    scene.add(Transmitter(name="tx",
                  position=mi.Point3f(1.1, 0.8, 2),
                  orientation=mi.Point3f(0,0,0)))
    scene.get("tx").look_at(mi.Point3f(5,5,5))

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=max_depth,
                   cell_size=mi.Point2f(delta, delta),
                   samples_per_tx=int(1e8),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)

    for i, pos in enumerate(np.reshape(rm.cell_centers, [-1,3])):
        scene.add(Receiver(name=f"rx-{i}",
                            position=pos,
                            orientation=mi.Point3f(0, 0, 0)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src = 10000,
                   max_num_paths_per_src=int(1e7),
                   max_depth=max_depth,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)

    a = paths_to_coverage_map(paths)[0]
    nmse_db = 10*np.log10(np.mean( ((rm.path_gain[0]-a)/a)**2 ))
    assert nmse_db < -20

def test_box_02():
    """Test a multiple reflections and LoS in the box scene.
    It includes directive antenna pattern and a complex material"""
    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = 0.2


    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    scene.rx_array =  PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="VH")

    los = True
    specular_reflection = True
    diffuse_reflection = True
    refraction = False
    width=9
    num_cells_x = 20
    delta = width/num_cells_x
    max_depth = 5

    scene.add(Transmitter(name="tx",
                          position=mi.Point3f(-3, -0.3, 4.),
                          orientation=mi.Point3f(0,0,0)))

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=max_depth,
                   cell_size=mi.Point2f(delta, delta),
                   center=mi.Point3f(0.0,0.0,0.5),
                   orientation=mi.Point3f(0.,0.,0),
                   size=mi.Point2f(width, width),
                   samples_per_tx=int(1e7),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)


    for i, pos in enumerate(np.reshape(rm.cell_centers, [-1,3])):
        scene.add(Receiver(name=f"rx-{i}",
                           position=pos,
                           orientation=mi.Point3f(0, 0, 0)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src = int(1e4),
                   max_num_paths_per_src=int(2e7),
                   max_depth=max_depth,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)

    a = paths_to_coverage_map(paths)[0]
    nmse_db = 10*np.log10(np.mean( ((rm.path_gain[0]-a)/a)**2 ))
    assert nmse_db < -20.
