#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the scene rendering feature."""
from __future__ import annotations

from os.path import join
import tempfile

import drjit as dr
import mitsuba as mi
import numpy as np

import sionna.rt as rt
from sionna.rt.radio_materials.itu import itu_material
from sionna.rt.scene import Scene, load_scene
from sionna.rt import PathSolver, RadioMapSolver
from sionna.rt.utils.meshes import transform_mesh


def add_example_radio_devices(scene: Scene):
    # Note: hardcoded for `box_two_screens.xml` as an example.
    scene.add(rt.Transmitter("tr-1", position=[-3.0, 0.0, 1.5]))
    scene.add(rt.Receiver("rc-1", position=[3.0, 0.0, 1.5]))
    scene.add(rt.Receiver("rc-2", position=[1.0, -2.0, 3.5],
                          color=(0.9, 0.9, 0.2), display_radius=0.9))

    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="tr38901",
                                    polarization="VH")
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="tr38901",
                                    polarization="VH")


def get_example_paths(scene: Scene):
    # Ray tracing parameters
    num_samples_per_src = int(1e6)
    max_num_paths = int(1e7)
    max_depth = 3

    solver = PathSolver()
    paths = solver(scene,
                   max_depth=max_depth,
                   max_num_paths_per_src=max_num_paths,
                   samples_per_src=num_samples_per_src)

    return paths


def get_example_radio_map(scene: Scene,
                          measurement_surface_name: str | None = None,
                          scale_factor: float = 1.01,
                          translation: mi.Vector3f | None = None,
                          props: mi.Properties | None = None):
    rm_solver = RadioMapSolver()
    ms = None
    if measurement_surface_name is not None:
        ms = scene.get(measurement_surface_name).clone(props=props)
        # TODO: combine with `clone`
        transform_mesh(ms.mi_mesh, scale=mi.ScalarVector3f(scale_factor),
                       translation=translation)
    return rm_solver(scene, cell_size=(0.1, 0.1), measurement_surface=ms)


def test01_render_with_paths():
    scene = load_scene(rt.scene.box_two_screens)
    # Set the material properties.
    eta_r, sigma = itu_material("metal", 3e9) # ITU material evaluated at 3GHz
    for sh in scene.mi_scene.shapes():
        material = sh.bsdf().radio_material
        material.relative_permittivity = eta_r
        material.conductivity = sigma
        material.scattering_coefficient = 0.01
        material.xpd_coefficient = 0.2

    add_example_radio_devices(scene)
    paths = get_example_paths(scene)
    radio_map = get_example_radio_map(scene)

    # Camera pose to render from
    bbox = scene.mi_scene.bbox()
    to_world = mi.ScalarTransform4f().look_at(
        origin=mi.ScalarVector3f(1.3, 1.0, 1.5) * bbox.max,
        target=mi.ScalarVector3f(1, 1, 0) * bbox.center(),
        up=[0, 0, 1],
    )

    image: mi.Bitmap = scene.render(
        camera=to_world, paths=paths,
        resolution=(256, 256), num_samples=4, fov=70,
        show_devices=True,
        radio_map=radio_map, rm_db_scale=True,
        clip_at=0.9 * scene.mi_scene.bbox().max.z,
        lighting_scale=1.5, return_bitmap=True)

    # fname = join(tempfile.gettempdir(), "scene.png")
    # image.convert(pixel_format=mi.Bitmap.PixelFormat.RGBA, \
    #             component_format=mi.Struct.Type.UInt8, srgb_gamma=True) \
    #     .write(fname)
    # print(f"[+] Rendering saved to: {fname}")

    # It's too brittle to specify the exact result per pixel, so we assert
    # on the proportion of colors instead.
    image_np = np.array(image, copy=False)
    mean_color = np.mean(image_np, axis=(0, 1))
    assert np.all(mean_color > [0.10, 0.15, 0.20, 0.50])


def test02_render_with_envmap():
    scene = load_scene(rt.scene.box_two_screens)

    # Create a temporary envmap of a bright color
    envmap = np.zeros((256, 512, 3)).astype(np.float16)
    envmap[:, :, 1] = 1.0
    envmap_fname = join(tempfile.gettempdir(), "envmap.exr")
    mi.Bitmap(envmap).write(envmap_fname)
    print(f"[i] Created temporary envmap at: {envmap_fname}")

    # Use a Sionna Camera object to specify the rendering viewpoint.
    bbox = scene.mi_scene.bbox()
    cam = rt.Camera(position=bbox.center() + mi.ScalarVector3f(0, 0, 10),
                    look_at=bbox.center())

    image: mi.Bitmap = scene.render(
        camera=cam, resolution=(256, 256), num_samples=4,
        fov=60, envmap=envmap_fname, return_bitmap=True
    )

    image_np = np.array(image, copy=False)
    mean_color = np.mean(image_np, axis=(0, 1))
    # Red and Blue channels should be zero since the light is all green.
    assert np.all(mean_color[[0, 2]] == 0)
    # Green and alpha channels should have high values.
    assert np.all(mean_color[1] >= 0.2)
    assert np.all(mean_color[3] == 1.0)


def test03_render_to_file_with_vertical_cut_plane():
    # TODO: check that color is dominated by radio map values
    # TODO: also check cut plane with a different orientation

    scene = load_scene(rt.scene.box_two_screens)

    bbox = scene.mi_scene.bbox()
    to_world = mi.ScalarTransform4f().look_at(
        origin=mi.ScalarVector3f(1.5, 0.0, 0.5) * bbox.max,
        target=bbox.center(),
        up=[0, 0, 1],
    )

    out_fname = join(tempfile.gettempdir(), "rendering.png")
    image: mi.Bitmap = scene.render_to_file(
        filename=out_fname,
        camera=to_world,
        resolution=(256, 256), num_samples=4, fov=70,
        show_devices=True,
        clip_at=0.9 * bbox.max.x,
        clip_plane_orientation=(-1, 0, 0),
        lighting_scale=1.5)
    print(f"[+] Wrote image to: {out_fname}")
    assert isinstance(image, mi.Bitmap)
    assert image.component_format() == mi.Struct.Type.UInt8

    loaded = mi.Bitmap(out_fname)

    # We should mainly see the purple plane inside of the box.
    mean_color = np.mean(np.array(loaded, copy=False), axis=(0, 1))
    assert np.all((mean_color[[0, 1]] >= 0.33 * 255)
                  & (mean_color[[0, 1]] <= 0.38 * 255))
    assert mean_color[2] >= 0.5 * 255
    assert mean_color[3] >= 0.95 * 255


def test04_render_to_file_with_mesh_radio_map():
    scene = load_scene(rt.scene.simple_street_canyon, merge_shapes=False)
    bbox = scene.mi_scene.bbox()
    to_world = mi.ScalarTransform4f().look_at(
        origin=mi.ScalarVector3f(1.0, 0.0, 1.5) * bbox.max,
        target=bbox.center() - mi.ScalarVector3f(0, 0, 0.7 * bbox.center().z),
        up=[0, 0, 1],
    )
    add_example_radio_devices(scene)
    radio_map = get_example_radio_map(scene, "building_4", scale_factor=1.01)

    configs = {
        "rm": (radio_map, [72.18, 124.39,  90.78, 181.18]),
        "baseline": (None, [151.19, 141.19, 127.90, 179.44]),
    }
    # Compare rendering with & without our mesh-based radio map.
    for config_name, (rm_or_none, expected_mean_color) in configs.items():
        out_fname = join(tempfile.gettempdir(), f"rendering_{config_name}.png")
        out_crop_fname = join(tempfile.gettempdir(), f"rendering_crop_{config_name}.png")

        image: mi.Bitmap = scene.render_to_file(
            filename=out_fname,
            camera=to_world,
            resolution=(256, 256), num_samples=4, fov=70,
            show_devices=True,
            radio_map=rm_or_none,
            # Note: we hardcode the values so that slight differences in simulation
            # results (e.g. CPU vs GPU) do not affect the rendering too much.
            rm_vmin=-105, rm_vmax=-65,
            lighting_scale=1.5
        )
        print(f"[+] Wrote image to: {out_fname}")

        # Check color in the small region that is supposed to be showing the
        # mesh-based radio map.
        h, w = image.height(), image.width()
        image_crop = np.array(image)[int(0.15*h):int(0.6*h), int(0.5*w):int(0.9*w), :]
        mi.Bitmap(image_crop).write(out_crop_fname)
        print(f"[+] Wrote cropped image to: {out_crop_fname}")

        mean_color = np.mean(image_crop, axis=(0, 1))
        assert np.allclose(mean_color, expected_mean_color, atol=1e-2)


def test05_render_mesh_radio_map_orientation():
    """
    By default, area emitters (used to render the color-mapped values onto the
    radio map) are single-sided. Sionna uses a custom emitter plugin to make
    them two-sided. This test checks that it works.
    """

    scene = load_scene(rt.scene.floor_wall, merge_shapes=False)
    bbox = scene.mi_scene.bbox()
    to_world = mi.ScalarTransform4f().look_at(
        origin=mi.ScalarVector3f(0.5, 1.0, 1.5) * bbox.max,
        target=bbox.center() - mi.ScalarVector3f(0, 0, 0.7 * bbox.center().z),
        up=[0, 0, 1],
    )
    add_example_radio_devices(scene)
    radio_map = get_example_radio_map(
        scene,
        "floor",
        scale_factor=0.98,
        translation=mi.ScalarVector3f(0, 0, 0.01 * bbox.extents().z),
    )

    # Compare rendering with & without our mesh-based radio map.
    rendered_images = []
    for flip_normals in (False, True):
        config_name = "flip" if flip_normals else "no_flip"
        out_fname = join(tempfile.gettempdir(), f"rendering_{config_name}.png")

        if flip_normals:
            # Flip face normals by re-ordering the vertices in each face.
            mesh = radio_map.measurement_surface
            assert not mesh.has_vertex_normals() and not mesh.has_face_normals()
            props = mi.traverse(mesh)
            face_indices = dr.unravel(mi.Point3u, props["faces"])
            tmp = type(face_indices.z)(face_indices.z)
            face_indices.z = face_indices.x
            face_indices.x = tmp
            props["faces"] = dr.ravel(face_indices)
            props.update()

        image: mi.Bitmap = scene.render_to_file(
            filename=out_fname,
            camera=to_world,
            resolution=(256, 256), num_samples=4, fov=70,
            # resolution=(720, 720), num_samples=16, fov=70,
            show_devices=True,
            radio_map=radio_map,
            lighting_scale=1.5,
        )
        print(f"[+] Wrote image to: {out_fname}")

        dr.eval(image)
        rendered_images.append(np.array(image))

    # Regardless of the orientation of the measurement surface, the radio map
    # should appear the same on either side.
    assert np.allclose(rendered_images[0], rendered_images[1])
