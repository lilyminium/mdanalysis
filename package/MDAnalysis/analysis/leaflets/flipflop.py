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
from ..distances import capped_distance, distance_array, calc_bonds
from ...lib.c_distances import unwrap_around, mean_unwrap_around
from ...lib.mdamath import norm
from .utils import get_centers_by_residue

class LipidFlipFlop(LeafletAnalysis):
    """Quantify lipid flip-flops between leaflets.

    Parameters
    ----------

    universe : Universe
        :class:`~MDAnalysis.core.universe.Universe` object.
    select : AtomGroup or str
        A AtomGroup instance or a
        :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        universe.atoms.PO4 or "name PO4" or "name P*"



    """

    def __init__(self, universe, *args, select="resname CHOL",
                 cutoff=50, buffer_zone=8, leaflet_width=8, **kwargs):
        super().__init__(universe, *args, select=select, **kwargs)
        self.cutoff = cutoff
        self.leaflet_width = leaflet_width
        self.buffer_zone = buffer_zone
        other_i = [i for i, x in enumerate(self.leafletfinder.residues)
                   if x not in self.residues]
        self.other_i = set(other_i)
    
    def _prepare(self):
        self.residue_leaflet = np.ones((self.n_frames, self.n_residues), dtype=int)
        self.residue_leaflet *= -1

    def _single_frame(self):
        box = self.universe.dimensions
        coords = self.leafletfinder.coordinates.copy()
        coords[:, :2] = 0
        upper_comp, lower_comp = self.leafletfinder.components[:2]
        # upper_z = coords[upper_comp]
        # lower_z = coords[lower_comp]
        # thickness = upper_z.mean() - lower_z.mean()
        upper_comp = set(list(upper_comp))
        lower_comp = set(list(lower_comp))

        upper_i = 0
        lower_i = 1
        inter_i = -1

        row = self.residue_leaflet[self._frame_index]

        res_positions = get_centers_by_residue(self.selection)

        pairs = capped_distance(res_positions,
                                self.leafletfinder.coordinates,
                                box=box,
                                max_cutoff=self.cutoff,
                                return_distances=False)
        all_pairs = np.sort(pairs, axis=0)
        splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
        plist = np.split(pairs, splix)

        res_positions[:, :2] = 0

        for pairs_ in plist:
            rix = set(pairs_[:, 1])
            i = pairs_[0, 0]
            # rix = set(all_pairs[all_pairs[:, 0] == i][:, 1])

        # for i, r in enumerate(self.headgroups):
        #     res = r.positions.copy()

        #     pairs = capped_distance(self.leafletfinder.coordinates,
        #                             res, box=box,
        #                             max_cutoff=self.cutoff,
        #                             return_distances=False)

            # rix = set(list(pairs[:, 0]))
            upper_idx = sorted(rix & upper_comp)
            lower_idx = sorted(rix & lower_comp)
            upper = coords[upper_idx]
            lower = coords[lower_idx]
            res = res_positions[i]

            # upper = dists[np.in1d(rix, upper_comp)]
            # lower = dists[np.in1d(rix, lower_comp)]

            if not len(lower) and len(upper):
                row[i] = upper_i
                continue

            if not len(upper) and len(lower):
                row[i] = lower_i
                continue

            upper = mean_unwrap_around(upper, upper[0], box)
            lower = mean_unwrap_around(lower, lower[0], box)

            # thickness = distance_array(upper, lower, box=box).mean()
            # thickness = calc_bonds(upper, lower, box=box)
            # dist_threshold = max(thickness/2 - self.buffer_zone, 1)
            
            
            # print("thickness", dist_threshold)

            # upper_dist = distance_array(upper, res, box=box).mean()
            # lower_dist = distance_array(lower, res, box=box).mean()

            upper_dist = calc_bonds(upper, res)
            lower_dist = calc_bonds(lower, res)

            
            if upper_dist <= self.leaflet_width:
                row[i] = upper_i
            elif lower_dist <= self.leaflet_width:
                row[i] = lower_i
            else:
                row[i] = inter_i
            

    def _conclude(self):
        self.flips = np.zeros(self.n_residues)
        self.flops = np.zeros(self.n_residues)
        self.flip_sections = np.zeros(self.n_residues)
        self.flop_sections = np.zeros(self.n_residues)

        if not self.n_frames:
            return

        for i in range(self.n_residues):
            trans = self.residue_leaflet[:, i]
            trans = trans[trans != -1]
            diff = trans[1:] - trans[:-1]

            self.flips[i] = np.sum(diff > 0)  # 0: upper, 1: lower
            self.flops[i] = np.sum(diff < 0)
        
        self.translocations = self.flips + self.flops

        self.flips_by_attr = {}
        self.flops_by_attr = {}
        self.translocations_by_attr = {}

        for each in np.unique(self.ids):
            mask = self.ids == each
            self.flips_by_attr[each] = int(sum(self.flips[mask]))
            self.flops_by_attr[each] = int(sum(self.flops[mask]))
            self.translocations_by_attr[each] = int(sum(self.translocations[mask]))
