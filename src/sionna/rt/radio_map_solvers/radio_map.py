#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Radio map object"""

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import from_levels_and_colors
import warnings
from typing import Tuple, List

from sionna.rt.utils import watt_to_dbm, log10
from sionna.rt.scene import Scene
from sionna.rt.constants import DEFAULT_TRANSMITTER_COLOR,\
    DEFAULT_RECEIVER_COLOR


class RadioMap:
    r"""
    Radio Map

    A radio map is generated for the loaded scene for all transmitters using
    a :doc:`radio map solver <radio_map_solvers>`.
    Please refer to the documentation of this module for further details.

    :param scene: Scene for which the radio map is computed
    :param meas_surf: Mitsuba mesh to use as the measurement surface
    :param cell_size: Size of a cell of the radio map [m]
    :param area_samples: Total number of samples to use for cell area estimates
    :param area_samples_per_round: Number of simultaneously evaluated samples
    """

    def __init__(self,
                 scene : Scene,
                 meas_surf : mi.Mesh,
                 cell_size : mi.Point2f,
                 proj_normal : mi.Vector3f | None = None,
                 area_samples = 1_000_000,
                 area_samples_per_round = 10_000_000):

        # create parameterization of measurement surface
        params = mi.traverse(meas_surf)
        faces = dr.reshape(mi.Vector3u, params["faces"], (3, -1))
        n = meas_surf.face_normal(dr.arange(mi.UInt, dr.width(faces)))

        n0 = meas_surf.face_normal(0)
        is_flat = dr.all(dr.norm(n - n0) < dr.epsilon(mi.Float))[0]
        if proj_normal is None:
            # infer the projection normal if meas_surf is flat
            proj_normal = n0 if is_flat else mi.Vector3f(0, 0, 1)
        self._proj_frame = mi.Frame3f(proj_normal)

        # The parameterization works by associating each vertex with a texcoord
        # that is just a scaled and shifted copy of the vertex's projection
        # onto a plane with the given normal vector. The parameterization is not
        # bijective when the mesh "folds over" itself when viewed along the
        # given normal vector. Assuming the input mesh is connected and its
        # geometric normals do not instantaneously change direction, the mesh
        # "folds over" itself when any geometric normal points in the opposite
        # direction of the given projection plane normal vector.
        if not dr.all(dr.dot(n, proj_normal) > 0):
            warnings.warn(
                "The mesh parameterization is not bijective, which may "
                "significantly impact the accuracy of cell surface area "
                "computation. Try using a different projection normal.",
                category=RuntimeWarning,
            )

        vert_world = dr.reshape(mi.Point3f, params["vertex_positions"], (3, -1))
        vert_local = self._proj_frame.to_local(vert_world)
        bbox = mi.BoundingBox2f(min=dr.min(vert_local.xy, axis=1),
                                max=dr.max(vert_local.xy, axis=1))
        texcoords = (vert_local.xy - bbox.min) / bbox.extents()

        meas_surf = mi.Mesh(name="measurement_surface",
                            vertex_count=dr.width(vert_world),
                            face_count=dr.width(faces),
                            has_vertex_texcoords=True)
        params = mi.traverse(meas_surf)
        params["vertex_positions"] = dr.ravel(vert_world)
        params["vertex_texcoords"] = dr.ravel(texcoords)
        params["faces"] = dr.ravel(faces)
        params.update()

        meas_surf.recompute_bbox()
        self._meas_surf = meas_surf
        self._meas_surf_scene = mi.load_dict({
            "type": "scene",
            "measurement_surface": meas_surf
        })

        # Number of cells
        self._cell_size = mi.Point2f(cell_size)
        num_cells = mi.Point2u(dr.ceil(bbox.extents() / self._cell_size))
        self._num_cells = num_cells

        # Precompute cell areas
        cell_area_shape = (num_cells.y[0], num_cells.x[0])
        # bbox is in local coordinates of proj_frame, but that only applies
        # a rotation, so with unit proj_normal, scale is still preserved
        cell_area_flat = (self._cell_size.x * self._cell_size.y)[0]
        self._cell_areas = dr.full(mi.TensorXf, cell_area_flat, cell_area_shape)
        if not is_flat:
            self._cell_areas *= self._compute_cell_area_scaling_factors(
                mesh=meas_surf,
                rows=num_cells.y[0],
                cols=num_cells.x[0],
                proj_normal=proj_normal,
                samples=area_samples,
                samples_per_round=area_samples_per_round,
            )

        self._size = mi.Point2f(bbox.extents())

        self._center = meas_surf.bbox().center()
        self._thermal_noise_power = scene.thermal_noise_power
        self._wavelength = scene.wavelength

        # Positions of the transmitters
        tx_positions_x = mi.Float([tx.position.x[0]
                                for tx in scene.transmitters.values()])
        tx_positions_y = mi.Float([tx.position.y[0]
                                for tx in scene.transmitters.values()])
        tx_positions_z = mi.Float([tx.position.z[0]
                                for tx in scene.transmitters.values()])
        self._tx_positions = mi.Point3f(tx_positions_x,
                                        tx_positions_y,
                                        tx_positions_z)

        # Powers of the transmitters
        self._tx_powers = mi.Float([tx.power[0]
                                    for tx in scene.transmitters.values()])

        # Positions of the receivers
        rx_positions_x = mi.Float([rx.position.x[0]
                                for rx in scene.receivers.values()])
        rx_positions_y = mi.Float([rx.position.y[0]
                                for rx in scene.receivers.values()])
        rx_positions_z = mi.Float([rx.position.z[0]
                                for rx in scene.receivers.values()])
        self._rx_positions = mi.Point3f(rx_positions_x,
                                        rx_positions_y,
                                        rx_positions_z)

        # Initialize the pathgain map to zero
        self._pathgain_map = dr.zeros(
            mi.TensorXf, [self.num_tx, num_cells.y[0], num_cells.x[0]]
        )

        # Sampler used to randomly sample user positions using
        # sample_positions()
        self._sampler = mi.load_dict({'type': 'independent'})

    @property
    def measurement_surface(self):
        r"""Mitsuba mesh corresponding to the radio map measurement surface

        :type: :py:class:`mi.Mesh`
        """
        return self._meas_surf

    @property
    def measurement_surface_scene(self):
        r"""Mitsuba scene containing just the radio map measurement surface

        :type: :py:class:`mi.Scene`
        """
        return self._meas_surf_scene

    @property
    def cell_centers(self):
        r"""Positions of the centers of the cells in the global coordinate
        system

        :type: :py:class:`mi.TensorXf [num_cells_y, num_cells_x, 3]`
        """
        num_cells_x = self._num_cells.x[0]
        num_cells_y = self._num_cells.y[0]
        center_uv = mi.Point2f(*dr.meshgrid(
            (dr.arange(mi.Float, num_cells_x) + 0.5) / num_cells_x,
            (dr.arange(mi.Float, num_cells_y) + 0.5) / num_cells_y))
        # (num_cells_x * num_cells_y, 3)
        center_xyz = self._meas_surf.eval_parameterization(center_uv).p
        # (num_cells_y, num_cells_x, 3)
        return dr.reshape(dtype=mi.TensorXf,
                          value=dr.ravel(center_xyz),
                          shape=(num_cells_y, num_cells_x, 3))

    @property
    def cell_size(self):
        r"""Size of a cell of the radio map [m]

        :type: :py:class:`mi.Point2f`
        """
        return self._cell_size

    @property
    def center(self):
        r"""Center of the radio map in the global coordinate system

        :type: :py:class:`mi.Point3f`
        """
        return self._center

    @property
    def projection_frame(self):
        r"""Coordinate frame whose :math:`z` component is the normal vector of 
        the projection plane.

        :type: :py:class:`mi.Frame3f`
        """
        return self._proj_frame

    @property
    def num_cells(self):
        r"""Number of cells along the local X and Y axes

        :type: :py:class:`mi.Point2u`
        """
        return self._num_cells

    @property
    def num_tx(self):
        r"""Number of transmitters

        :type: :py:class:`int`
        """
        return dr.width(self._tx_positions)

    @property
    def num_rx(self):
        r"""Number of receivers

        :type: :py:class:`int`
        """
        return dr.width(self._rx_positions)

    @property
    def size(self):
        r"""Size of the radio map [m]

        :type: :py:class:`mi.Point2f`
        """
        return self._size

    @property
    def tx_cell_indices(self):
        r"""Cell index position of each transmitter in the format
        `(column, row)`

        :type: :py:class:`mi.Point2u`
        """
        return self._global_to_cell_ind(self._tx_positions)

    @property
    def rx_cell_indices(self):
        r"""Computes and returns the cell index positions corresponding to
        receivers in the format `(column, row)`

        :type: :py:class:`mi.Point2u`
        """
        return self._global_to_cell_ind(self._rx_positions)

    @property
    def path_gain(self):
        r"""Path gains across the radio map from all transmitters

        :type: :py:class:`mi.TensorXf [num_tx, num_cells_y, num_cells_x]`
        """
        return self._pathgain_map

    @property
    def rss(self):
        r"""Received signal strength (RSS) across the radio map from all
        transmitters

        :type: :py:class:`mi.TensorXf [num_tx, num_cells_y, num_cells_x]`
        """
        tx_powers = dr.reshape(mi.TensorXf, self._tx_powers,
                               [self.num_tx, 1, 1])
        rss_map = self.path_gain*tx_powers
        return rss_map

    @property
    def sinr(self):
        r"""SINR across the radio map from all transmitters

        :type: :py:class:`mi.TensorXf [num_tx, num_cells_y, num_cells_x]`
        """

        num_cells = self.num_cells
        rss = self.rss

        # Total received power from all transmitters
        # [num_cells_y, num_cells_x]
        total_pow = dr.zeros(mi.TensorXf, [1, num_cells.y[0], num_cells.x[0]])
        total_pow = dr.sum(rss, axis=0)
        # [1, num_cells_y, num_cells_x]
        total_pow = dr.reshape(mi.TensorXf, total_pow.array,
                            [1, num_cells.y[0], num_cells.x[0]])

        # Interference for each transmitter
        # Numerical issue can cause this value to be slightly negative
        interference = total_pow - rss

        # Thermal noise
        noise = self._thermal_noise_power

        # SINR
        sinr_map = rss / (interference + noise)
        return sinr_map

    def tx_association(self, metric : str = "path_gain") -> mi.TensorXi:
        r"""Computes cell-to-transmitter association

        Each cell is associated with the transmitter providing the highest
        metric, such as path gain, received signal strength (RSS), or
        SINR.

        :param metric: Metric to be used
        :type metric: "path_gain" | "rss" | "sinr"

        :return: Cell-to-transmitter association
        """
        num_cells = self.num_cells
        cells_count = num_cells.x[0]*num_cells.y[0]

        # Get tensor for desired metric
        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")
        radio_map = getattr(self, metric)

        # Equivalent to argmax
        max_val = dr.tile(dr.max(radio_map, axis=0).array, self.num_tx)
        active = max_val > 0.
        radio_map_flat = radio_map.array
        i = dr.compress((max_val == radio_map_flat) & active)
        n_tx = mi.Int(i // cells_count)
        cell_ind_flat = i % cells_count

        # Fill the tx association map
        # No transmitter assignment for the cells with no coverage
        tx_association = dr.full(mi.TensorXi, -1,
                                 [num_cells.y[0], num_cells.x[0]])
        dr.scatter(tx_association.array, n_tx, cell_ind_flat)

        return tx_association

    def add(
        self,
        e_fields : mi.Vector4f,
        solid_angle : mi.Float,
        array_w : List[mi.Float],
        si_mp : mi.SurfaceInteraction3f,
        k_world : mi.Vector3f,
        tx_indices : mi.UInt,
        hit : mi.Bool
        ) -> None:
        r"""
        Adds the contribution of the rays that hit the measurement surface to
        the radio maps

        The radio maps are updated in place.

        :param e_fields: Electric fields as real-valued vectors of dimension 4
        :param solid_angle: Ray tubes solid angles [sr]
        :param array_w: Weighting used to model the effect of the transmitter
            array
        :param si_mp: Informations about the interaction with the measurement
            surface
        :param k_world: Directions of propagation of the rays
        :param tx_indices: Indices of the transmitters from which the rays
            originate
        :param hit: Flags indicating if the rays hit the measurement surface
        """

        # Indices of the hit cells
        cell_ind = self._local_to_cell_ind(si_mp.uv)
        # Indices of the item in the tensor storing the radio maps
        num_cells = self._num_cells.x[0]*self._num_cells.y[0]
        tensor_ind = tx_indices*num_cells + cell_ind

        # Contribution to the path loss map
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw@e_field
        a = dr.squared_norm(a)

        # Ray weight
        cos_theta = dr.abs(dr.dot(k_world, si_mp.n))
        w = solid_angle*dr.rcp(cos_theta)

        a *= w

        # Update the path loss map
        dr.scatter_reduce(dr.ReduceOp.Add, self._pathgain_map.array, value=a,
                          index=tensor_ind, active=hit)

    def finalize(self) -> None:
        r"""Finalizes the computation of the radio map"""

        # Scale the pathloss map
        wavelength = self._wavelength
        scaling = dr.square(wavelength*dr.rcp(4.*dr.pi)) / self._cell_areas
        self._pathgain_map *= scaling

    def show(
        self,
        metric : str = "path_gain",
        tx : int | None = None,
        vmin : float | None = None,
        vmax : float | None = None,
        show_tx : bool = True,
        show_rx : bool = False
        ) -> plt.Figure:
        r"""Visualizes a radio map

        The position of the transmitters is indicated by "+" markers.
        The positions of the receivers are indicated by "x" markers.

        :param metric: Metric to show
        :type metric: "path_gain" | "rss" | "sinr"

        :param tx: Index of the transmitter for which to show the radio
            map. If `None`, the maximum value over all transmitters for each
            cell is shown.

        :param vmin: Defines the minimum value [dB] for the colormap covers.
            If set to `None`, then the minimum value across all cells is used.

        :param vmax: Defines the maximum value [dB] for the colormap covers.
            If set to `None`, then the maximum value across all cells is used.

        :param show_tx: If set to `True`, then the position of the transmitters
            are shown.

        :param show_rx: If set to `True`, then the position of the receivers are
            shown.

        :return: Figure showing the radio map
        """

        tx_cell_indices = self.tx_cell_indices
        rx_cell_indices = self.rx_cell_indices
        tensor = self.transmitter_radio_map(metric, tx)

        # Convert to dB-scale
        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                tensor = 10.*log10(tensor)
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strengmth to dBm
                tensor = watt_to_dbm(tensor)

        # Set label
        if metric == "path_gain":
            colorbar_label = "Path gain [dB]"
            title = "Path gain"
        elif metric == "rss":
            colorbar_label = "Received signal strength (RSS) [dBm]"
            title = 'RSS'
        else:
            colorbar_label = "Signal-to-interference-plus-noise ratio (SINR)"\
                             " [dB]"
            title = 'SINR'

        # Visualization the radio map
        fig_cm = plt.figure()
        plt.imshow(tensor.numpy(), origin='lower', vmin=vmin, vmax=vmax)

        # Set label
        if (tx is None) & (self.num_tx > 1):
            title = 'Highest ' + title + ' across all TXs'
        elif tx is not None:
            title = title + f" for TX '{tx}'"
        plt.colorbar(label=colorbar_label)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title(title)

        # Show transmitter, receiver
        if show_tx:
            if tx is not None:
                fig_cm.axes[0].scatter(tx_cell_indices.x[tx],
                                       tx_cell_indices.y[tx],
                                       marker='P',
                                       color=DEFAULT_TRANSMITTER_COLOR)
            else:
                for tx_ in range(self.num_tx):
                    fig_cm.axes[0].scatter(tx_cell_indices.x[tx_],
                                           tx_cell_indices.y[tx_],
                                           marker='P',
                                           color=DEFAULT_TRANSMITTER_COLOR)

        if show_rx:
            for rx in range(self.num_rx):
                fig_cm.axes[0].scatter(rx_cell_indices.x[rx],
                                       rx_cell_indices.y[rx],
                                       marker='x',
                                       color=DEFAULT_RECEIVER_COLOR)

        return fig_cm

    def show_association(
        self,
        metric : str = "path_gain",
        show_tx : bool = True,
        show_rx : bool = False
        ) -> plt.Figure:
        r"""Visualizes cell-to-tx association for a given metric

        The positions of the transmitters and receivers are indicated
        by "+" and "x" markers, respectively.

        :param metric: Metric to show
        :type metric: "path_gain" | "rss" | "sinr"

        :param show_tx: If set to `True`, then the position of the transmitters
            are shown.

        :param show_rx: If set to `True`, then the position of the receivers are
            shown.

        :return: Figure showing the cell-to-transmitter association
        """

        tx_cell_indices = self.tx_cell_indices
        rx_cell_indices = self.rx_cell_indices

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        # Create the colormap and normalization
        colors = mpl.colormaps['Dark2'].colors[:self.num_tx]
        cmap, norm = from_levels_and_colors(
            list(range(self.num_tx+1)), colors)
        fig_tx = plt.figure()
        plt.imshow(self.tx_association(metric).numpy(),
                    origin='lower', cmap=cmap, norm=norm)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title('Cell-to-TX association')
        cbar = plt.colorbar(label="TX")
        cbar.ax.get_yaxis().set_ticks([])
        for tx in range(self.num_tx):
            cbar.ax.text(.5, tx + .5, str(tx), ha='center', va='center')

        # Show transmitter, receiver
        if show_tx:
            for tx in range(self.num_tx):
                fig_tx.axes[0].scatter(tx_cell_indices.x[tx],
                                       tx_cell_indices.y[tx],
                                       marker='P',
                                       color=DEFAULT_TRANSMITTER_COLOR)

        if show_rx:
            for rx in range(self.num_rx):
                fig_tx.axes[0].scatter(rx_cell_indices.x[rx],
                                       rx_cell_indices.y[rx],
                                       marker='x',
                                       color=DEFAULT_RECEIVER_COLOR)

        return fig_tx

    def sample_positions(
        self,
        num_pos : int,
        metric : str = "path_gain",
        min_val_db : float | None = None,
        max_val_db : float | None = None,
        min_dist : float | None = None,
        max_dist : float | None = None,
        tx_association : bool = True,
        center_pos : bool = False,
        seed : int = 1
        ) -> Tuple[mi.TensorXf, mi.TensorXu]:
        # pylint: disable=line-too-long
        r"""Samples random user positions in a scene based on a radio map

        For a given radio map, ``num_pos`` random positions are sampled
        around each transmitter, such that the selected metric, e.g., SINR, is
        larger than ``min_val_db`` and/or smaller than ``max_val_db``.
        Similarly, ``min_dist`` and ``max_dist`` define the minimum and maximum
        distance of the random positions to the transmitter under consideration.
        By activating the flag ``tx_association``, only positions are sampled
        for which the selected metric is the highest across all transmitters.
        This is useful if one wants to ensure, e.g., that the sampled positions
        for each transmitter provide the highest SINR or RSS.

        Note that due to the quantization of the radio map into cells it is
        not guaranteed that all above parameters are exactly fulfilled for a
        returned position. This stems from the fact that every
        individual cell of the radio map describes the expected *average*
        behavior of the surface within this cell. For instance, it may happen
        that half of the selected cell is shadowed and, thus, no path to the
        transmitter exists but the average path gain is still larger than the
        given threshold. Please enable the flag ``center_pos`` to sample only
        positions from the cell centers.

        .. code-block:: Python

            import numpy as np
            import sionna
            from sionna.rt import load_scene, PlanarArray, Transmitter,\
                                  RadioMapSolver, Receiver

            scene = load_scene(sionna.rt.scene.munich)

            # Configure antenna array for all transmitters
            scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.7,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
            # Add a transmitters
            tx = Transmitter(name="tx",
                        position=[-195,-240,30],
                        orientation=[0,0,0])
            scene.add(tx)

            solver = RadioMapSolver()
            rm = solver(scene, cell_size=(1., 1.), samples_per_tx=100000000)

            positions,_ = rm.sample_positions(num_pos=200, min_val_db=-100.,
                                              min_dist=50., max_dist=80.)
            positions = positions.numpy()
            positions = np.squeeze(positions, axis=0)

            for i,p in enumerate(positions):
                rx = Receiver(name=f"rx-{i}",
                            position=p,
                            orientation=[0,0,0])
                scene.add(rx)

            scene.preview(clip_at=10.);

        .. figure:: ../figures/rm_user_sampling.png
            :align: center

        The above example shows an example for random positions between 50m and
        80m from the transmitter and a minimum path gain of -100 dB.
        Keep in mind that the transmitter can have a different height than the
        radio map which also contributes to this distance.
        For example if the transmitter is located 20m above the surface of the
        radio map and a ``min_dist`` of 20m is selected, also positions
        directly below the transmitter are sampled.

        :param num_pos: Number of returned random positions for each transmitter

        :param metric: Metric to be considered for sampling positions
        :type metric: "path_gain" | "rss" | "sinr"

        :param min_val_db: Minimum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Positions are only sampled from cells where the selected metric is
            larger than or equal to this value. Ignored if `None`.

        :param max_val_db: Maximum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Positions are only sampled from cells where the selected metric is
            smaller than or equal to this value.
            Ignored if `None`.

        :param min_dist:  Minimum distance [m] from transmitter for all random
            positions. Ignored if `None`.

        :param max_dist: Maximum distance [m] from transmitter for all random
            positions. Ignored if `None`.

        :param tx_association: If `True`, only positions associated with a
            transmitter are chosen, i.e., positions where the chosen metric is
            the highest among all all transmitters. Else, a user located in a
            sampled position for a specific transmitter may perceive a higher
            metric from another TX.

        :param center_pos: If `True`, all returned positions are sampled from
            the cell center (i.e., the grid of the radio map). Otherwise, the
            positions are randomly drawn from the surface of the cell.

        :return: Random positions :math:`(x,y,z)` [m]
            (shape : :py:class:`[num_tx, num_pos, 3]`) that are in cells
            fulfilling the configured constraints

        :return: Cell indices (shape :py:class:`[num_tx, num_pos, 2]`)
            corresponding to the random positions in the format `(column, row)`
        """

        num_tx = self.num_tx
        num_cells = self.num_cells

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        if not isinstance(num_pos, int):
            raise ValueError("num_pos must be int.")

        if min_val_db is None:
            min_val_db = float("-inf")
        min_val_db = float(min_val_db)

        if max_val_db is None:
            max_val_db = float("inf")
        max_val_db = float(max_val_db)

        if min_val_db > max_val_db:
            raise ValueError("min_val_d cannot be larger than max_val_db.")

        if min_dist is None:
            min_dist = 0.
        min_dist = float(min_dist)

        if max_dist is None:
            max_dist = float("inf")
        max_dist = float(max_dist)

        if min_dist > max_dist:
            raise ValueError("min_dist cannot be larger than max_dist.")

        # Select metric to be used
        cm = getattr(self, metric)

        # Convert to dB-scale
        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                cm = 10.*log10(cm)
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strengmth to dBm
                cm = watt_to_dbm(cm)

        # Transmitters positions
        tx_pos = self._tx_positions
        tx_pos = dr.ravel([tx_pos.x, tx_pos.y, tx_pos.z])
        # [num_tx, num_cells_y. num_cells_x, 3]
        tx_pos = dr.reshape(mi.TensorXf, tx_pos, [num_tx, 1, 1, 3])

        # Compute distance from each tx to all cells
        # [num_cells_y, num_cells_x, 3]
        cell_centers = self.cell_centers
        # [1, num_cells_y, num_cells_x, 3]
        cell_centers_ = dr.reshape(mi.TensorXf, cell_centers.array,
                                  [1, num_cells.y[0], num_cells.x[0], 3])
        # [num_tx, num_cells_y, num_cells_x]
        cell_distance_from_tx = dr.sqrt(dr.sum(dr.square(cell_centers_-tx_pos),
                                               axis=3))

        # [num_tx, num_cells_y. num_cells_x]
        distance_mask = ( (cell_distance_from_tx >= min_dist) &
                          (cell_distance_from_tx <= max_dist) )

        # Get cells for which metric criterion is valid
        # [num_tx, num_cells_y. num_cells_x]
        cm_mask = (cm >= min_val_db) & (cm <= max_val_db)

        # Get cells for which the tx association is valid
        # [num_tx, num_cells_y. num_cells_x]
        tx_ids = dr.arange(mi.UInt, num_tx)
        tx_ids = dr.reshape(mi.TensorXu, tx_ids,
                            [num_tx, 1, 1])
        tx_a = self.tx_association(metric)
        tx_a = dr.reshape(mi.TensorXu, tx_a,
                          [1, num_cells.y[0], num_cells.x[0]])
        association_mask = tx_ids == tx_a

        # Compute combined mask
        # [num_tx, num_cells_y. num_cells_x]
        active_cells = distance_mask & cm_mask
        if tx_association:
            active_cells = active_cells & association_mask

        # Loop over transmitters and sample for each transmitters active cells
        self._sampler.seed(seed, num_pos)
        # [num_cells, 3]
        cell_centers_flat = dr.reshape(mi.TensorXf, cell_centers, [-1, 3])
        # Sampled positions
        # [num_tx, num_pos, 3]
        sampled_pos = dr.zeros(mi.TensorXf, [num_tx, num_pos, 3])
        sampled_cells = dr.zeros(mi.TensorXu, [num_tx, num_pos, 2])
        scatter_ind = dr.arange(mi.UInt, num_pos)
        # Scaled directions in which to add offsets if required
        # [3]
        y_dir = 0.5*(cell_centers[1,0] - cell_centers[0,0])
        x_dir = 0.5*(cell_centers[0,1] - cell_centers[0,0])
        for n in range(num_tx):
            active_cells_tx = active_cells[n].array
            # Indices of the active cells for this transmitter
            active_cells_ind = dr.compress(active_cells_tx)
            active_cells_count = dr.width(active_cells_ind)
            if active_cells_count == 0:
                continue
            # Sample cells ids
            # Float in (0,1)
            cell_ids = self._sampler.next_1d()
            # Int
            cell_ids = dr.floor(cell_ids*active_cells_count)
            cell_ids = mi.UInt(cell_ids)
            cell_ids = dr.gather(mi.UInt, active_cells_ind, cell_ids)
            cell_ids_y = cell_ids // self.num_cells.x[0]
            cell_ids_x = cell_ids % self.num_cells.x[0]
            # Sampled cells center positions
            sampled_pos_x = dr.gather(mi.Float, cell_centers_flat.array,
                                      3*cell_ids)
            sampled_pos_y = dr.gather(mi.Float, cell_centers_flat.array,
                                      3*cell_ids + 1)
            sampled_pos_z = dr.gather(mi.Float, cell_centers_flat.array,
                                      3*cell_ids + 2)
            # Add random offset within cell-size, if positions should not be
            # centered
            if not center_pos:
                # Random offsets
                # Point2f, batch size: num_pos
                offsets = self._sampler.next_2d()*2. - 1.
                # mi.Float, [num_pos]
                sampled_pos_x += offsets.x*x_dir[0].array\
                                                    + offsets.y*y_dir[0].array
                sampled_pos_y += offsets.x*x_dir[1].array\
                                                    + offsets.y*y_dir[1].array
                sampled_pos_z += offsets.x*x_dir[2].array \
                                                    + offsets.y*y_dir[2].array
            # Store sampled positions
            dr.scatter(sampled_pos.array, sampled_pos_x,
                       scatter_ind*3 + n*num_pos*3)
            dr.scatter(sampled_pos.array, sampled_pos_y,
                       scatter_ind*3 + n*num_pos*3 + 1)
            dr.scatter(sampled_pos.array, sampled_pos_z,
                       scatter_ind*3 + n*num_pos*3 + 2)
            #
            dr.scatter(sampled_cells.array, cell_ids_y,
                       scatter_ind*2 + n*num_pos*2)
            dr.scatter(sampled_cells.array, cell_ids_x,
                       scatter_ind*2 + n*num_pos*2 + 1)

        return sampled_pos, sampled_cells

    def cdf(
        self,
        metric : str = "path_gain",
        tx : int | None = None,
        bins : int = 200
        ) -> Tuple[plt.Figure, mi.TensorXf, mi.Float]:
        r"""Computes and visualizes the CDF of a metric of the radio map

        :param metric: Metric to be shown
        :type metric: "path_gain" | "rss" | "sinr"

        :param tx: Index or name of the transmitter for which to show the radio
            map. If `None`, the maximum value over all transmitters for each
            cell is shown.

        :param bins: Number of bins used to compute the CDF

        :return: Figure showing the CDF

        :return: Data points for the chosen metric

        :return: Cummulative probabilities for the data points
        """

        tensor = self.transmitter_radio_map(metric, tx)
        # Flatten tensor
        tensor = dr.ravel(tensor)

        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                tensor = 10.*log10(tensor)
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strengmth to dBm
                tensor = watt_to_dbm(tensor)

        # Compute the CDF

        # Cells with no coverage are excluded
        active = tensor != float("-inf")
        num_active = dr.count(active)
        # Compute the range
        max_val = dr.max(tensor)
        if max_val == float("inf"):
            raise ValueError("Max value is infinity")
        tensor_ = dr.select(active, tensor, float("inf"))
        min_val = dr.min(tensor_)
        range_val = max_val - min_val
        # Compute the cdf
        ind = mi.UInt(dr.floor((tensor - min_val)*bins/range_val))
        cdf = dr.zeros(mi.UInt, bins)
        dr.scatter_inc(cdf, ind, active)
        cdf = mi.Float(dr.cumsum(cdf))
        cdf /= num_active
        # Values
        x = dr.arange(mi.Float, 1, bins+1)/bins*range_val + min_val

        # Plot the CDF

        fig, _ = plt.subplots()
        plt.plot(x.numpy(), cdf.numpy())
        plt.grid(True, which="both")
        plt.ylabel("Cummulative probability")

        # Set x-label and title
        if metric=="path_gain":
            xlabel = "Path gain [dB]"
            title = "Path gain"
        elif metric=="rss":
            xlabel = "Received signal strength (RSS) [dBm]"
            title = "RSS"
        else:
            xlabel = "Signal-to-interference-plus-noise ratio (SINR) [dB]"
            title = "SINR"
        if (tx is None) & (self.num_tx > 1):
            title = 'Highest ' + title + ' across all TXs'
        elif tx is not None:
            title = title + f' for TX {tx}'

        plt.xlabel(xlabel)
        plt.title(title)

        return fig, x, cdf

    ###############################################
    # Internal methods
    ###############################################

    def _local_to_cell_ind(self, p_local : mi.Point2f) -> mi.Int:
        r"""
        Computes the indices of the hitted cells of the map from the local
        :math:`(x,y)` coordinates

        :param p_local: Coordinates of the intersected points in the
            measurement surface local frame

        :return: Cell indices in the flattened measurement surface
        """
        # Cell indices in the 2D measurement surface
        cell_ind = mi.Point2u(dr.floor(p_local * self._num_cells))

        # Cell indices for the flattened measurement surface
        cell_ind = cell_ind[1] * self._num_cells[0] + cell_ind[0]

        return cell_ind

    def _global_to_cell_ind(self, p_global : mi.Point3f) -> mi.Point2u:
        r"""
        Computes the indices of the hitted cells of the map from the global
        :math:`(x,y,z)` coordinates

        :param p_global: Coordinates of the a point on the measurement surface
            in the global frame

        :return: `(x, y)` indices of the cell which contains `p_global`
        """

        if dr.width(p_global) == 0:
            return mi.Point2u()

        proj_normal = self._proj_frame.n
        scene = self._meas_surf_scene
        # project to measurement surface along proj_normal
        ray = mi.Ray3f(p_global, proj_normal)
        si = scene.ray_intersect(ray)
        # recompute with flipped directions if the ray missed
        if dr.width(missed := dr.isinf(si.t)) > 0:
            ray.d[missed] = -proj_normal
            si = scene.ray_intersect(ray)

        return mi.Point2u(dr.floor(si.uv * self._num_cells))

    def _compute_cell_area_scaling_factors(
            self,
            mesh : mi.Mesh,
            rows : int,
            cols : int,
            proj_normal : mi.Vector3f,
            samples : int,
            samples_per_round: int):
        spp_total = 0
        samples_remaining = samples
        f_sum = dr.zeros(mi.Float, rows * cols)
        # Divide into blocks. samples_per_block should be chosen based on the
        # System's memory. Too large and things slow down significantly
        while samples_remaining > 0:
            if samples_remaining // samples_per_round == 0:
                spp = max(1, int(samples_remaining / (rows * cols)))
                samples_remaining = 0
            else:
                spp = max(1, int(samples_per_round / (rows * cols)))
                samples_remaining -= samples_per_round

            spp_total += spp

            # Generate spp samples per grid cell
            center_u, center_v = dr.meshgrid(
                (dr.arange(mi.Float, cols) + 0.5) / cols,
                (dr.arange(mi.Float, rows) + 0.5) / rows,
            )
            center_uv = mi.Point2f(
                dr.repeat(center_u, count=spp),
                dr.repeat(center_v, count=spp),
            )
            rng = dr.auto.ad.PCG32(size=2 * spp * rows * cols)
            jitter = dr.reshape(mi.Vector2f, rng.next_float32(), shape=(2, -1))
            jitter = (jitter - 0.5) / mi.Vector2f(cols, rows)
            sample_uv = center_uv + jitter

            # Query the mesh parameterization. si.t is inf when it misses
            si = mesh.eval_parameterization(sample_uv)
            sample_n = si.n

            # Calculate and average scaling factors
            f = dr.rcp(dr.abs_dot(sample_n, proj_normal))
            all_idx = dr.arange(mi.UInt, spp * rows * cols)
            # Slightly faster than using dr.compress() to make an index array
            dr.scatter(f, value=0, index=all_idx, active=dr.isinf(si.t))
            f_sum += dr.block_sum(value=f, block_size=spp)

        cell_areas = dr.reshape(dtype=mi.TensorXf,
                                value=f_sum / spp_total,
                                shape=(rows, cols))
        return cell_areas

    def transmitter_radio_map(
        self,
        metric : str = "path_gain",
        tx : int | None = None
        ) -> mi.TensorXf:
        r"""Returns the radio map values corresponding to transmitter ``tx``
        and a specific ``metric``

        If ``tx`` is `None`, then returns for each cell the maximum value
        accross the transmitters.

        :param metric: Metric for which to return the radio map
        :type metric: "path_gain" | "rss" | "sinr"
        """

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")
        tensor = getattr(self, metric)

        if isinstance(tx, int):
            if tx >= self.num_tx:
                raise ValueError("Invalid transmitter index")
        elif tx is not None:
            raise ValueError("Invalid type for `tx`: Must be am int, or None")

        # Select metric for a specific transmitter or compute max
        if tx is not None:
            tensor = tensor[tx]
        else:
            tensor = dr.max(tensor, axis=0)

        return tensor
