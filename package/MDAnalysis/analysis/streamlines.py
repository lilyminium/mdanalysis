# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

import typing
import numpy as np

from .base import AnalysisBase

class Streamlines(AnalysisBase):
    def __init__(
        self,
        universe,
        select: str = "all",
        grid_spacing: float = 20,
        x_min: typing.Optional[float] = None,
        x_max: typing.Optional[float] = None,
        y_min: typing.Optional[float] = None,
        y_max: typing.Optional[float] = None,
        z_min: typing.Optional[float] = None,
        z_max: typing.Optional[float] = None,
        in_3D: bool = False,
        maximum_delta_magnitude: float = None,
        **kwargs
    ):
        super().__init__(universe.universe.trajectory, **kwargs)
        self.atomgroup = universe.atoms.select_atoms(select)
        self.grid_spacing = grid_spacing
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.z_max = z_max
        self.z_min = z_min
        self.in_3D = in_3D
        self.maximum_delta_magnitude = maximum_delta_magnitude
    

    def _set_grid_bounds(self):
        if self.x_min is None:
            self.x_min = self.atomgroup.positions.T[0].min()
        if self.x_max is None:
            self.x_max = self.atomgroup.positions.T[0].max()
        if self.y_min is None:
            self.y_min = self.atomgroup.positions.T[1].min()
        if self.y_max is None:
            self.y_max = self.atomgroup.positions.T[1].max()
        if self.in_3D:
            if self.z_min is None:
                self.z_min = self.atomgroup.positions.T[2].min()
            if self.z_max is None:
                self.z_max = self.atomgroup.positions.T[2].max()

    def _prepare(self):
        self._set_grid_bounds()

        self.results.x_edges = np.arange(
            self.x_min, self.x_max, self.grid_spacing
        )
        self.results.n_x = len(self.results.x_edges) - 1
        self.results.y_edges = np.arange(
            self.y_min, self.y_max, self.grid_spacing
        )
        self.results.n_y = len(self.results.y_edges) - 1

        if self.in_3D:
            self.results.z_edges = np.arange(
                self.z_min, self.z_max, self.grid_spacing
            )
            self.results.n_z = len(self.results.z_edges) - 1
        else:
            self.results.z_edges = []
            self.results.n_z = 1

        n_xyz = self.results.n_x * self.results.n_y * self.results.n_z
        centroid_shape = (self.n_frames, n_xyz, 3)

        self._framewise_atom_grid_index = np.zeros(
            (self.n_frames, self.atomgroup.n_atoms), dtype=int
        )
        self._framewise_grid_centroids = np.full(centroid_shape, np.nan)
        self._next_grid_centroids = np.full(centroid_shape, np.nan)

    def _single_frame(self):
        x, y, z = self.atomgroup.positions.T

        # ignore anything under the lower bound or above upper bound
        # we do this by setting anything too low or too high to -1
        x_indices = np.digitize(x, self.results.x_edges)
        x_indices[x_indices == self.results.n_x + 1] = 0
        x_indices -= 1
        y_indices = np.digitize(y, self.results.y_edges)
        y_indices[y_indices == self.results.n_y + 1] = 0
        y_indices -= 1

        if self.in_3D:
            z_indices = np.digitize(z, self.results.z_edges)
            z_indices[z_indices == self.results.n_z + 1] = 0
            z_indices -= 1
        else:
            z_indices = np.zeros_like(z, dtype=int)
        
        # flatten 3d indices to 1d
        grid_indices = (
            (x_indices * self.results.n_y * self.results.n_z)
            + (y_indices * self.results.n_z) + z_indices
        )

        # grid_indices = (x_indices * self.results.n_x) + y_indices
        mask = (x_indices == -1) | (y_indices == -1) | (z_indices == -1)
        grid_indices[mask] = -1

        self._framewise_atom_grid_index[self._frame_index] = grid_indices

        # dx is centroid computed *this* frame using indices from last frame,
        # minus centroid computed last frame using indices from last frame

        unique_indices = np.unique(grid_indices)
        for grid_idx in unique_indices:
            if grid_idx == -1:
                continue

            # first compute centroids this frame using indices this frame
            indices_this_frame = np.where(grid_indices == grid_idx)[0]
            positions_this_frame = self.atomgroup.positions[indices_this_frame]
            centroid_this_frame = np.mean(positions_this_frame, axis=0)
            key = (self._frame_index, grid_idx)
            self._framewise_grid_centroids[key] = centroid_this_frame

            # now compute centroids this frame using indices from last frame
            grid_last_frame = self._framewise_atom_grid_index[self._frame_index - 1]
            indices_last_frame = np.where(grid_last_frame == grid_idx)[0]
            positions_last_frame = self.atomgroup.positions[indices_last_frame]
            centroid_last_frame = np.mean(positions_last_frame, axis=0)
            self._next_grid_centroids[key] = centroid_last_frame

    def _conclude(self):
        print(self._next_grid_centroids.shape)
        print(self._framewise_grid_centroids.shape)
        displacements = (
            self._next_grid_centroids[1:] - self._framewise_grid_centroids[:-1]
        )
        if self.maximum_delta_magnitude is not None:
            max_delta = np.abs(self.maximum_delta_magnitude)
            displacements[displacements > max_delta] = np.nan
            displacements[displacements < -max_delta] = np.nan
        displacements[np.isnan(displacements)] = 0


        shape = (self.n_frames - 1, self.results.n_x, self.results.n_y, self.results.n_z, 3)
        print(displacements)
        displacements = displacements.reshape(shape)
        self.results.dx_array = displacements[..., 0]
        self.results.dy_array = displacements[..., 1]
        self.results.dz_array = displacements[..., 2]
        # self.results.dx_array = displacements[..., 0].reshape(shape)
        # self.results.dy_array = displacements[..., 1].reshape(shape)
        
        disp_sq = (
            self.results.dx_array ** 2 + self.results.dy_array ** 2
        )
        if self.in_3D:
            disp_sq += self.results.dz_array ** 2

        self.results.displacements = np.sqrt(disp_sq)
        self.results.displacement_mean = np.mean(self.results.displacements)
        self.results.displacement_std = np.std(self.results.displacements)




