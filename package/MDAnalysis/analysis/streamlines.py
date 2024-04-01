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
import multiprocessing
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
        self.maximum_delta_magnitude = maximum_delta_magnitude
    

    def _set_grid_bounds(self):
        get_min_axes = []
        get_max_axes = []
        if self.x_min is None:
            get_min_axes.append(0)
        if self.x_max is None:
            get_max_axes.append(0)
        if self.y_min is None:
            get_min_axes.append(1)
        if self.y_max is None:
            get_max_axes.append(1)

        all_mins = []
        all_maxs = []
        for _ in self._sliced_trajectory:
            all_mins.append(
                self.atomgroup.positions.T[get_min_axes].min(axis=1)
            )
            all_maxs.append(
                self.atomgroup.positions.T[get_max_axes].max(axis=1)
            )
        mins = np.min(all_mins, axis=0)
        maxs = np.max(all_maxs, axis=0)
        if self.x_min is None:
            self.x_min = mins[0]
        if self.x_max is None:
            self.x_max = maxs[0]
        if self.y_min is None:
            self.y_min = mins[-1]
        if self.y_max is None:
            self.y_max = maxs[-1]

    def _prepare(self):
        print("n_frames", self.n_frames, self.start, self.stop)
        bounds = [self.x_min, self.x_max, self.y_min, self.y_max]
        if any(b is None for b in bounds):
            self._set_grid_bounds()

        self.results.x_edges = np.arange(
            self.x_min, self.x_max, self.grid_spacing
        )
        self.results.n_x = len(self.results.x_edges) - 1
        self.results.y_edges = np.arange(
            self.y_min, self.y_max, self.grid_spacing
        )
        self.results.n_y = len(self.results.y_edges) - 1
        self._framewise_atom_grid_index = np.zeros(
            (self.n_frames, self.atomgroup.n_atoms), dtype=int
        )
        self._framewise_grid_centroids = np.full(
            (self.n_frames, self.results.n_x * self.results.n_y, 2),
            np.nan
        )
        # print(np.mgrid[self.x_min:self.x_max:self.grid_spacing, self.y_min:self.y_max:self.grid_spacing])
        print(self.x_min, self.x_max, self.results.x_edges)
        print(self.y_min, self.y_max, self.results.y_edges)


    def _single_frame(self):
        x = self.atomgroup.positions.T[0]
        y = self.atomgroup.positions.T[1]

        # ignore anything under the lower bound or above upper bound
        # we do this by setting anything too low or too high to -1
        x_indices = np.digitize(x, self.results.x_edges)
        x_indices[x_indices == self.results.n_x + 1] = 0
        x_indices -= 1
        y_indices = np.digitize(y, self.results.y_edges)
        y_indices[y_indices == self.results.n_y + 1] = 0
        y_indices -= 1
        grid_indices = (x_indices * self.results.n_x) + y_indices
        mask = (x_indices == -1) | (y_indices == -1)
        grid_indices[mask] = -1

        self._framewise_atom_grid_index[self._frame_index] = grid_indices

        unique_indices, inverse_indices = np.unique(
            grid_indices, return_inverse=True
        )
        print(unique_indices)
        for grid_idx in unique_indices:
            if grid_idx == -1:
                continue
            print(grid_idx, self._framewise_grid_centroids.shape)
            inverse_idx = np.where(inverse_indices == grid_idx)[0]
            atoms_in_square = self.atomgroup[inverse_idx]
            centroid = np.mean(atoms_in_square.positions, axis=0)
            self._framewise_grid_centroids[
                self._frame_index, grid_idx
            ] = centroid[:2]
        # for grid_idx, inverse_idx in zip(unique_indices, inverse_indices):
        #     atoms_in_square = self.atomgroup[inverse_idx]
        #     self._framewise_grid_centroids[
        #         self._frame_index, grid_idx
        #     ] = np.mean(atoms_in_square.positions, axis=0)


    def _conclude(self):
        centroids = self._framewise_grid_centroids
        print(centroids.shape)
        displacements = centroids[1:] - centroids[:-1]
        if self.maximum_delta_magnitude is not None:
            max_delta = np.abs(self.maximum_delta_magnitude)
            displacements[displacements > max_delta] = np.nan
            displacements[displacements < -max_delta] = np.nan
        displacements[np.isnan(displacements)] = 0
        self.results.dx_array = displacements[..., 0]
        self.results.dy_array = displacements[..., 1]
        self.results.displacements = np.sqrt(
            self.results.dx_array ** 2 + self.results.dy_array ** 2
        )
        self.results.displacement_mean = np.mean(self.results.displacements)
        self.results.displacement_std = np.std(self.results.displacements)

