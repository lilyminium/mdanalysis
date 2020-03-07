# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2020 The MDAnalysis Development Team and contributors
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

"""Secondary structure analysis --- :mod:`MDAnalysis.analysis.secondary_structure`
==================================================================================

:Authors: Lily Wang
:Year: 2020
:Copyright: GNU Public License v3

.. versionadded:: 1.0.0

This module contains classes for computing secondary structure. MDAnalysis provides
a wrapper for the external DSSP_ and STRIDE_ programs to be run on an MD trajectory.

Both DSSP_ and STRIDE_ need to be installed separately.

.. _DSSP: https://swift.cmbi.umcn.nl/gv/dssp/
.. _STRIDE: http://webclu.bio.wzw.tum.de/stride/


Classes and functions
---------------------

.. autoclass:: DSSP
.. autoclass:: STRIDE

"""

import errno
import subprocess
import os
import numpy as np
import pandas as pd

from ..core.topologyattrs import ResidueAttr
from ..lib import util
from .base import AnalysisBase

class SecondaryStructure(ResidueAttr):
    """Single letter code for secondary structure"""
    attrname = 'secondary_structures'
    singular = 'secondary_structure'
    dtype = '<U1'

    @staticmethod
    def _gen_initial_values(na, nr, ns):
        return np.empty(nr, dtype=dtype)

class SecondaryStructureBase(AnalysisBase):
    """Base class for implementing secondary structure analysis.

    Subclasses should implement ``_compute_dssp``. 

    Parameters
    ----------
    universe: Universe or AtomGroup
        The Universe or AtomGroup to apply the analysis to. As secondary 
        structure is a residue property, the analysis is applied to every 
        residue in your chosen atoms.
    select: string, optional
        The selection string for selecting atoms from ``universe``. The 
        analysis is applied to the residues in this subset of atoms.
    add_topology_attr: bool, optional
        Whether to add the most common secondary structure as a topology 
        attribute ``secondary_structure`` to your residues. 
    verbose: bool, optional
        Turn on more logging and debugging.


    Attributes
    ----------
    residues: :class:`~MDAnalysis.core.groups.ResidueGroup`
        The residues to which the analysis is applied.
    ss_codes: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Single-letter secondary structure codes
    ss_names: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Secondary structure names
    ss_simple: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Simplified secondary structure names
    phi: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Phi angles
    psi: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Psi angles
    sasa: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Solvent-accessible surface area
    ss_counts: dict of {code: :class:`numpy.ndarray` of shape (n_frames,)}
        Dictionary that counts number of residues with each secondary
        structure code, for each frame
    simple_counts: dict of {name: :class:`numpy.ndarray` of shape (n_frames,)}
        Dictionary that counts number of residues with each simplified 
        secondary structure, for each frame
    ss_mode: :class:`numpy.ndarray` of shape (n_residues,)
        The most common secondary structure for each residue
    ss_codes_to_names: dict of {code: name}
        Dictionary converting each single-letter code to the full name of
        the secondary structure
    ss_codes_to_simple: dict of {code: name}
        Dictionary converting each single-letter code to simplified 
        secondary structures

    """

    ss_codes_to_names = {
        'G': 'Helix (3-10)',
        'H': 'Helix (alpha)'
        'I': 'Helix (pi)',
        'E': 'Strand',
        'B': 'Bridge',
        'T': 'Turn',
        'S': 'Bend',
        'C': 'Coil',
    }

    ss_codes_to_simple = {
        'G': 'Helix',
        'H': 'Helix',
        'I': 'Helix',
        'E': 'Strand',
        'B': 'Strand',
        'T': 'Coil',
        'S': 'Coil',
        'C': 'Coil'
    }

    def __init__(self, executable, universe, select='protein', verbose=False,
                 add_topology_attr=False):
        super(SecondaryStructure, self).__init__(universe.universe.trajectory,
                                                 verbose=verbose)
        self._universe = universe.universe
        self.residues = universe.select_atoms(select).residues
        self._add_topology_attr = add_topology_attr
    
    def _prepare(self):
        nf = self.n_frames
        nr = len(self.residues)
        self.ss_codes = np.empty((nf, nr), dtype='<U1')
        self.ss_counts = dict.fromkeys(self.ss_codes_to_simple)
        for k in self.ss_counts:
            self.ss_counts[k] = np.zeros(nf)
        self.phi = np.zeros((nf, nr), dtype=float)
        self.psi = np.zeros((nf, nr), dtype=float)
        self.sasa = np.zeros((nf, nr), dtype=float)

    def _compute_dssp(self):
        raise NotImplementedError

    def _single_frame(self):
        self._compute_dssp()
        row = self.ss_codes[self._frame_index]
        codes, counts = np.unique(row, return_counts=True)
        for c, n in zip(codes, counts):
            self.ss_counts[c][self._frame_index] = n
    
    def _conclude(self):
        # convert to full names and simple codes
        codes, idx = np.unique(self.ss_codes, return_inverse=True)
        names = np.array([self.ss_codes_to_names[c] for c in codes])
        self.ss_names = names[idx].reshape(self.ss_codes.shape)
        simp = np.array([self.ss_codes_to_simple[c] for c in codes])
        self.ss_simple = simp[idx].reshape(self.ss_codes.shape)

        # count number of simple structures
        self.simple_counts = {'Helix': np.zeros(self.n_frames),
                              'Strand': np.zeros(self.n_frames),
                              'Coil': np.zeros(self.n_frames),
                              }
        for code, counts in self.ss_counts.items():
            simple = self.ss_codes_to_simple[code]
            self.simple_counts[simple] += counts
        
        # get most common secondary structure
        self.ss_mode = np.empty(len(self.residues), dtype='<U1')
        for i, col in enumerate(self.ss_codes.T):
            code, counts = np.unique(col, return_counts=True)
            self.ss_mode[i] = code[np.argmax(counts)]

        # possibly add this as attribute
        if self._add_topology_attr:
            attr = SecondaryStructure.attrname
            self._universe.add_TopologyAttr(attr)
            setattr(self.residues, attr, self.ss_mode)
            

    def plot_content(self, ax=None, simple=False):
        """Plot the counts of secondary structures over frames.

        Parameters
        ----------
        ax: :class: `matplotlib.axes.Axes`, optional
            If no `ax` is supplied or set to ``None`` then the plot will
            be created on new axes.
        simple: bool, optional
            If ``True``, plots the counts of the simple secondary 
            structures. If ``False``, plots the counts of each distinct 
            category.

        Returns
        -------
        ax: :class: `matplotlib.axes.Axes`

        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if not simple:
            data = self.ss_counts
        else:
            data = self.simple_counts
        
        df = pd.DataFrame(data)
        ax = df.plot.bar(stacked=True)
        return ax

class SecondaryStructureWrapper(SecondaryStructureBase):
    """Base class for secondary structure analysis that wraps an 
    external program.
    """
    cmd = ''

    @property
    def exe_name(self):
        raise NotImplementedError

    def __init__(self, executable, universe, select='protein', verbose=False,
                 add_topology_attr=False):

        exe = util.which(executable)
        if exe is None:
            msg = ('{name} executable not found at {executable}. '
                   '{exe_name} must be on the PATH, or the path must '
                   'be provided with the keyword argument executable')
            raise OSError(errno.ENOENT, msg.format(name=type(self).__name__,
                                                   executable=executable,
                                                   exe_name=self.exe_name))
        self.exe = exe
        super(SecondaryStructureWrapper, self).__init__(universe,
                                                        select=select,
                                                        verbose=verbose,
                                                        add_topology_attr=add_topology_attr)
    
    def _execute(self):
        """Run the wrapped program."""
        fd, pdbfile = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)
        try:
            self.residues.atoms.write(pdbfile)
            cmd_args = [self.exe] + self.cmd.format(pdb=pdbfile).split()
            proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
        finally:
            try:
                os.unlink(pdbfile)
            except OSError:
                pass
        return output
    
    def _compute_dssp(self):
        output = self._execute()
        self._process_output(output)
    

class DSSP(SecondaryStructureWrapper):
    """Runs :program:`mkdssp` on a trajectory.

    :program:`mkdssp` implements the DSSP algorithm to determine secondary
    structure.

    This class creates temporary PDB files for each frame and runs ``mkdssp`` on them.

    Parameters
    ----------
    universe: Universe or AtomGroup
        The Universe or AtomGroup to apply the analysis to. As secondary 
        structure is a residue property, the analysis is applied to every 
        residue in your chosen atoms.
    select: string, optional
        The selection string for selecting atoms from ``universe``. The 
        analysis is applied to the residues in this subset of atoms.
    executable: str, optional
        Path to the ``mkdssp`` executable.
    add_topology_attr: bool, optional
        Whether to add the most common secondary structure as a topology 
        attribute ``secondary_structure`` to your residues. 
    verbose: bool, optional
        Turn on more logging and debugging.

    Attributes
    ----------
    residues: :class:`~MDAnalysis.core.groups.ResidueGroup`
        The residues to which the analysis is applied.
    ss_codes: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Single-letter secondary structure codes
    ss_names: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Secondary structure names
    ss_simple: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Simplified secondary structure names
    phi: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Phi angles
    psi: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Psi angles
    sasa: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Solvent-accessible surface area
    ss_counts: dict of {code: :class:`numpy.ndarray` of shape (n_frames,)}
        Dictionary that counts number of residues with each secondary
        structure code, for each frame
    simple_counts: dict of {name: :class:`numpy.ndarray` of shape (n_frames,)}
        Dictionary that counts number of residues with each simplified 
        secondary structure, for each frame
    ss_mode: :class:`numpy.ndarray` of shape (n_residues,)
        The most common secondary structure for each residue
    ss_codes_to_names: dict of {code: name}
        Dictionary converting each single-letter code to the full name of
        the secondary structure
    ss_codes_to_simple: dict of {code: name}
        Dictionary converting each single-letter code to simplified 
        secondary structures
    """

    exe_name = 'mkdssp'
    cmd = '-i {pdb} -o stdout'

    columns = {
        'ss': 17,
        'sasa': (35, 39),
        'phi': (105, 110),
        'psi': (111, 116),
    }

    def __init__(self, universe, select='protein', executable='mkdssp',
                 verbose=False, add_topology_attr=False):
        super(DSSP, self).__init__(executable, universe, 
                                   select=select, verbose=verbose,
                                   add_topology_attr=add_topology_attr)
        

    def _process_output(self, output):
        frame = self._frame_index
        per_res = output.split('  #  RESIDUE AA STRUCTURE BP1 BP2  ACC')[1]
        lines = per_res.split()[1:]
        for i, line in enumerate(lines):
            ss = line[self.columns['ss']]
            if not ss:
                ss = 'C'
            self.ss_codes[frame][i] = ss
            for kw in ('phi', 'psi', 'sasa'):
                _i, _j = self.columns[kw]
                getattr(self, kw)[frame][i] = line[_i:_j]
        


class STRIDE(SecondaryStructureWrapper):
    """Runs :program:`stride` on a trajectory.

    :program:`stride` implements the STRIDE algorithm to determine secondary
    structure.

    This class creates temporary PDB files for each frame and runs ``stride`` on them.

    Parameters
    ----------
    universe: Universe or AtomGroup
        The Universe or AtomGroup to apply the analysis to. As secondary 
        structure is a residue property, the analysis is applied to every 
        residue in your chosen atoms.
    select: string, optional
        The selection string for selecting atoms from ``universe``. The 
        analysis is applied to the residues in this subset of atoms.
    executable: str, optional
        Path to the ``stride`` executable.
    add_topology_attr: bool, optional
        Whether to add the most common secondary structure as a topology 
        attribute ``secondary_structure`` to your residues. 
    verbose: bool, optional
        Turn on more logging and debugging.
        
    Attributes
    ----------
    residues: :class:`~MDAnalysis.core.groups.ResidueGroup`
        The residues to which the analysis is applied.
    ss_codes: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Single-letter secondary structure codes
    ss_names: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Secondary structure names
    ss_simple: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Simplified secondary structure names
    phi: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Phi angles
    psi: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Psi angles
    sasa: :class:`numpy.ndarray` of shape (n_frames, n_residues)
        Solvent-accessible surface area
    ss_counts: dict of {code: :class:`numpy.ndarray` of shape (n_frames,)}
        Dictionary that counts number of residues with each secondary
        structure code, for each frame
    simple_counts: dict of {name: :class:`numpy.ndarray` of shape (n_frames,)}
        Dictionary that counts number of residues with each simplified 
        secondary structure, for each frame
    ss_mode: :class:`numpy.ndarray` of shape (n_residues,)
        The most common secondary structure for each residue
    ss_codes_to_names: dict of {code: name}
        Dictionary converting each single-letter code to the full name of
        the secondary structure
    ss_codes_to_simple: dict of {code: name}
        Dictionary converting each single-letter code to simplified 
        secondary structures
    """

    exe_name = 'stride'
    cmd = '{pdb}'

    def __init__(self, universe, select='protein', executable='stride',
                 verbose=False, add_topology_attr=False):
        super(STRIDE, self).__init__(executable, universe, 
                                     select=select, verbose=verbose,
                                     add_topology_attr=add_topology_attr)

    def _process_output(self, output):
        lines = output.split('\nASG ')[1:]
        lines[-1] = lines[-1].split('\n')[0]
        frame = self._frame_index
        for i, line in enumerate(lines):
            resname, chain, resid, resnum, ss, ssname, phi, psi, area = line.split()
            self.ss_codes[frame][i] = ss
            self.phi[frame][ss] = phi
            self.psi[frame][ss] = psi
            self.sasa[frame][ss] = area


    
