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

import numpy as np

from .base import LeafletAnalysis
from .. import distances
from ...lib.c_distances import unwrap_around


def lipid_area(headgroup_coordinate,
               neighbor_coordinates,
               other_coordinates=None,
               box=None, plot=False):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    
    # preprocess coordinates
    headgroup_coordinate = np.asarray(headgroup_coordinate)
    if len(headgroup_coordinate.shape) > 1:
        if box is not None:
            headgroup_coordinates = unwrap_around(headgroup_coordinate.copy(),
                                                headgroup_coordinate[0],
                                                box)
        headgroup_coordinate = headgroup_coordinates.mean(axis=0)
    if box is not None:
        neighbor_coordinates = unwrap_around(neighbor_coordinates.copy(),
                                             headgroup_coordinate,
                                             box)
        if other_coordinates is not None:
            other_coordinates = np.asarray(other_coordinates).copy()
            other_coordinates = unwrap_around(other_coordinates,
                                              headgroup_coordinate,
                                              box)
    points = np.r_[[headgroup_coordinate], neighbor_coordinates]
    points -= headgroup_coordinate
    center = points.mean(axis=0)
    points -= center

    Mt_M = np.matmul(points.T, points)
    u, s, vh = np.linalg.linalg.svd(Mt_M)
    # project onto plane
    if other_coordinates is not None:
        points = np.r_[points, other_coordinates-center]
    xy = np.matmul(points, vh[:2].T)
    # voronoi
    vor = Voronoi(xy)
    headgroup_cell_int = vor.point_region[0]
    headgroup_cell = vor.regions[headgroup_cell_int]
    if plot:
        import matplotlib.pyplot as plt
        fig = voronoi_plot_2d(vor, show_vertices=False, line_alpha=0.6)
        plt.plot([vor.points[0][0]], [vor.points[0][1]], 'r+', markersize=12)
        plt.show()

    if not all(vertex != -1 for vertex in headgroup_cell):
        print(len(neighbor_coordinates))
        import matplotlib.pyplot as plt
        fig = voronoi_plot_2d(vor, show_vertices=False, line_alpha=0.6)
        plt.plot([vor.points[0][0]], [vor.points[0][1]], 'r+', markersize=12)
        plt.show()
        raise ValueError("headgroup not bounded by Voronoi cell points: "
                         f"{headgroup_cell}. "
                         "Try including more neighbor points")
    # x and y should be ordered clockwise
    x, y = np.array([vor.vertices[x] for x in headgroup_cell]).T
    area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    area += (x[-1] * y[0] - y[-1] * x[0])
    lipid_area = 0.5 * np.abs(area)

    
    # if lipid_area < 5 or lipid_area > 100:
    #     print(lipid_area)
        # fig = voronoi_plot_2d(vor, show_vertices=False, line_alpha=0.6)
        # plt.plot([vor.points[0][0]], [vor.points[0][1]], 'r+', markersize=12)
        # plt.show()
    return lipid_area


class AreaPerLipid(LeafletAnalysis):

    def __init__(self, universe, *args, cutoff=50, cutoff_other=None, select_other=None,
                 **kwargs):
        super().__init__(universe, *args, **kwargs)
        if select_other is None:
            self.other = (self.universe.residues - self.residues).atoms
        else:
            self.other = universe.select_atoms(select_other) - self.residues.atoms
        self.cutoff = cutoff
        if cutoff_other is None:
            cutoff_other = cutoff
        self.cutoff_other = cutoff_other
        self.unique_ids = np.unique(self.ids)
        self.resindices = self.residues.resindices
        self.rix2ix = {x.resindex: i for i, x in enumerate(self.residues)}
        self.n_per_res = np.array([len(x) for x in self.headgroups])

    def _prepare(self):
        super()._prepare()
        self.areas = np.zeros((self.n_frames, self.n_residues))
        self.areas_by_attr = []
        for i in range(self.n_leaflets):
            dct = {}
            for each in self.unique_ids:
                dct[each] = []
            self.areas_by_attr.append(dct)
    
    def _single_frame(self):
        other = self.other.positions
        box = self.universe.dimensions
        rix2lfi = {}
        components = []
        leaflets = []

        for i, x in enumerate(self.leafletfinder.leaflets):
            ix = []
            atoms = []
            for y in x.residues.resindices:
                rix2lfi[y] = i
                if y in self.resindices:
                    ix.append(self.rix2ix[y])
                    atoms.extend(self.headgroups[self.rix2ix[y]])
            components.append(np.array(ix))
            leaflets.append(sum(atoms))

        hg_coords = [unwrap_around(x.positions.copy(), x.positions.copy()[0], box)
                     for x in self.headgroups]
        hg_mean = np.array([x.mean(axis=0) for x in hg_coords])

        all_wrapped = [hg_mean[x] for x in components]

        
        for i, rix in enumerate(self.resindices):
            hg_xyz = hg_mean[i]
            try:
                lf_i = rix2lfi[rix]
            except KeyError:
                self.areas[self._frame_index][i] = np.nan
                continue
            potential_xyz = all_wrapped[lf_i]
            # hg_xyz = self.headgroups[i].positions
            # potential_xyz = leaflets[lf_i].positions

            pairs, dist = distances.capped_distance(hg_xyz,
                                                    potential_xyz,
                                                    self.cutoff,
                                                    box=self.selection.dimensions,
                                                    return_distances=True)

            if not len(pairs):
                continue            
            pairs = pairs[dist>0]
            js = np.unique(pairs[:, 1])
            neighbor_xyz = potential_xyz[js]

            # get protein / etc ones
            pairs2 = distances.capped_distance(hg_xyz, other, self.cutoff_other,
                                               box=self.selection.dimensions,
                                               return_distances=False)
            if len(pairs2):
                other_xyz = other[np.unique(pairs2[:, 1])]
            else:
                other_xyz = None
            res = self.residues[i]
            try:
                area = lipid_area(hg_xyz, neighbor_xyz,
                                other_coordinates=other_xyz,
                                box=self.selection.dimensions)
            except:
                print(i)
                raise ValueError()
            self.areas[self._frame_index][i] = area
            self.areas_by_attr[lf_i][self.ids[i]].append(area)

    # def _conclude(self):
    #     super()._conclude()
    #     self.areas_by_attr = {}
    #     self.mean_area_by_attr = {}
    #     self.std_area_by_attr = {}
    #     for id_ in self.ids:
    #         values = np.concatenate(self.areas[:, self.ids == id_])
    #         self.areas_by_attr[id_] = values
    #         self.mean_area_by_attr[id_] = values.mean()
    #         self.std_area_by_attr[id_] = values.std()