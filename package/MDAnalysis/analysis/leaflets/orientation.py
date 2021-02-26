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

from ..base import AnalysisBase
from .base import LeafletAnalysis
from .utils import get_centers_by_residue, get_orientations

class Orientation(AnalysisBase):

    def __init__(self, universe, select="name ROH"):
        super().__init__(universe.universe.trajectory)
        self.universe = universe.universe
        self.selection = universe.select_atoms(select)
        self.headgroups = self.selection.split("residue")
        self.residues = self.selection.residues
        self.n_residues = len(self.residues)

    def _prepare(self):
        self.orientations = np.zeros((self.n_frames, self.n_residues))
    
    def _single_frame(self):
        box = self.universe.dimensions
        coordinates = get_centers_by_residue(self.selection,
                                             box=box)
        orientations = get_orientations(self.residues,
                                        self.selection,
                                        box=box,
                                        headgroup_centers=coordinates,
                                        normalize=True)
        
        self.orientations[self._frame_index] = np.arccos(orientations[:, 2])

    def _conclude(self):
        self.degrees = np.rad2deg(self.orientations)
    
    
class LipidOrientation(LeafletAnalysis):

    def __init__(self, universe, *args, full_sel=None, **kwargs):
        super().__init__(universe, *args, **kwargs)
        self.full_sel = self.universe.select_atoms(full_sel)
        print(self.full_sel.atoms.names)
        self.norm = np.linalg.norm([0, 0, 1])

    def _prepare(self):
        shape = (self.n_frames, self.n_leaflets, self.n_residues)
        self.orientations = np.zeros(shape)
        self.orientations[:] = np.nan

    def _single_frame(self):
        box = self.universe.dimensions
        frame = self.orientations[self._frame_index]
        for leaflet_row, res_indices in zip(frame, self._relevant_rix):
            coords = self.get_leaflet_coordinates(res_indices)
            orientations = get_orientations(self.residues,
                                            self.selection,
                                            box=box,
                                            headgroup_centers=coords,
                                            normalize=True)
            
            angles = [np.dot(x, [0, 0, 1])/(np.linalg.norm(x)*self.norm) for x in orientations]
            leaflet_row[res_indices] = np.arccos(angles)
    
    def _conclude(self):
        self.degrees = np.rad2deg(self.orientations)