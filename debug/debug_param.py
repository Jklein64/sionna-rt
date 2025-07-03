import sionna.rt as rt
import mitsuba as mi
import numpy as np
import drjit as dr
import matplotlib.pyplot as plt
from loguru import logger

def main():
    logger.info("Generating terrain mesh...")
    N = 100
    vert_x, vert_y = np.meshgrid(
        np.linspace(0, 10, N), np.linspace(0, 10, N), indexing="xy"
    )
    elevation = np.exp(-((vert_x - 5) ** 2 + (vert_y - 5) ** 2)) / 5
    mesh = rt.utils.geometry.triangulate_elevation(
        mi.TensorXf(elevation), center=[0, 0, 0], size=[1, 1]
    )
    vertices = dr.reshape(mi.Point3f, mesh.vertex_positions_buffer(), (3, -1))

    logger.info("Mesh has vertex texcoords: {}.", mesh.has_vertex_texcoords())
    logger.info(f"Mesh vertex positions buffer shape: {mesh.vertex_positions_buffer().shape}.")
    logger.info(f"Mesh vertex texcoords buffer shape: {mesh.vertex_texcoords_buffer().shape}.")
    logger.info(f"Mesh face index buffer shape: {mesh.faces_buffer().shape}.")
    logger.info("Attempting manual call to eval_parameterization...")
    si = mesh.eval_parameterization(mi.Point2f(0.5, 0.5))
    logger.info(f"Scene interaction at point p = {si.p}.")

    # Create scene
    logger.info("Building scene...")
    props = mi.Properties()
    props["material"] = rt.RadioMaterial(
        name="lunar-highland",
        thickness=11,
        relative_permittivity=2.7,
        conductivity=10 ** (-12.5),
        scattering_coefficient=0.4,
        xpd_coefficient=0.1,
        scattering_pattern="lambertian",
        frequency_update_callback=None,
        color=None,
        props=None,
    )
    # Sionna requires bsdfs to be wrapped in HolderMaterial
    props["material"] = rt.HolderMaterial(props)
    mi_mesh = mi.Mesh(
        name="terrain",
        vertex_count=mesh.vertex_count(),
        face_count=mesh.face_count(),
        props=props,
    )
    mesh_params = mi.traverse(mesh)
    mi_mesh_params = mi.traverse(mi_mesh)
    mi_mesh_params["vertex_positions"] = dr.ravel(vertices)
    mi_mesh_params["faces"] = mesh_params["faces"]
    mi_mesh_params.update()
    scene = rt.Scene(mi.load_dict({"type": "scene", "terrain": mi_mesh}))

    scene.tx_array = default_array()
    scene.rx_array = default_array(polarization="VH")
    tx = rt.Transmitter(
        name="tx",
        #  position=mi.Point3f(0, 0, 0.1),
        position=mi.Point3f(0.2, 0.2, 0.1),
        orientation=mi.Point3f(0, 0, 0),
        power_dbm=100,
    )
    tx.display_radius = 0.01
    scene.add(tx)
    logger.info("Finished building scene.")

    sim_params = dict(
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=True,
    )

    # Compute the radio map
    logger.info("Starting radio map solver...")
    rm_solver = rt.RadioMapSolver()
    rm = rm_solver(
        scene=scene,
        center=[0, 0, 0],
        size=[1, 1],
        cell_size=[0.05, 0.05],
        # evaluate 5cm above the surface
        digital_elevation_model=mi.TensorXf(elevation) + 0.05,
        samples_per_tx=int(1e4),
        **sim_params,
    )

    # Check for NaN or Inf
    assert not (dr.any(dr.isinf(rm.path_gain)) or dr.any(dr.isnan(rm.path_gain)))
    assert rm.measurement_surface.has_vertex_texcoords()
    logger.info("Finished radio map solver.")

    meas_surface = rm.measurement_surface
    dr.eval(meas_surface)
    logger.info("Measurement surface has vertex texcoords: {}.", meas_surface.has_vertex_texcoords())
    logger.info(f"Measurement surface vertex positions buffer shape: {meas_surface.vertex_positions_buffer().shape}")
    logger.info(f"Measurement surface vertex texcoords buffer shape: {meas_surface.vertex_texcoords_buffer().shape}")
    logger.info(f"Measurement surface face index buffer shape: {meas_surface.faces_buffer().shape}")
    logger.info("Writing measurement surface ply to /tmp/meas_surface.ply...")
    meas_surface.write_ply("/tmp/meas_surface.ply")
    params = mi.traverse(meas_surface)
    params.update()
    # logger.info("Attempting manual call to eval_parameterization...")
    # u = [0.1, 0.2, 0.5]
    # v = [0.07, 0.3, 0.9]
    # u, v = dr.meshgrid(
    #     (dr.arange(mi.UInt, size=3) + 0.5) / 3,
    #     (dr.arange(mi.UInt, size=3) + 0.5) / 3
    # )
    # cells_per_dim_x = rm._cells_per_dim.x[0]
    # cells_per_dim_y = rm._cells_per_dim.y[0]
    # u, v = dr.meshgrid(
    #     (dr.arange(mi.UInt, size=cells_per_dim_x) + 0.5) / cells_per_dim_x,
    #     (dr.arange(mi.UInt, size=cells_per_dim_y) + 0.5) / cells_per_dim_y
    # )
    # point = mi.Point2f(0.5, 0.5)
    # si = meas_surface.eval_parameterization(point)
    # logger.info(f"Result for {point} is {si.p}.")

    logger.info("Attempting to call cell_centers...")
    cell_centers = rm.cell_centers
    logger.info("Cell centers has shape {}", dr.shape(cell_centers))


def default_array(
    num_rows: int = 1,
    num_cols: int = 1,
    vertical_spacing: float = 0.5,
    horizontal_spacing: float = 0.5,
    pattern: str = "iso",
    polarization: str = "V",
):
    return rt.PlanarArray(
        num_rows=num_rows,
        num_cols=num_cols,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        pattern=pattern,
        polarization=polarization,
    )


if __name__ == "__main__":
    main()
