# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
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

from .base import GuesserBase
import numpy as np
import warnings

import re

from ..lib import distances
from . import tables


class DefaultGuesser(GuesserBase):
    context = 'default'

    def __init__(self, atoms, **kwargs):
        super().__init__(atoms, **kwargs)
        self._guess = {'masses': self.guess_masses,
                       'types': self.guess_types,
                       'angles': self.guess_angles,
                       'dihedrals': self.guess_dihedrals,
                       'bonds': self.guess_bonds}
        self._rank = {'masses': 1,
                      'types': 0,
                      'bonds': 0,
                      'angles': 1,
                      'dihedrals': 2
                      }

    def guess_masses(self):
        """Guess the mass of many atoms based upon their type

        Returns
        -------
        atom_masses : np.ndarray dtype float64
        """
        if hasattr(self._atoms, 'elements'):
            atom_types = self._atoms.elements
        else:
            atom_types = self._atoms.types

        self.validate_atom_types(atom_types)
        masses = np.array([self.get_atom_mass(atom_t)
                           for atom_t in atom_types], dtype=np.float64)
        return masses

    def validate_atom_types(self, atom_types):
        """Validates the atom types based on whether they are available
        in our tables

        Parameters
        ----------
        atom_types
          Type of each atom

        Returns
        -------
        None

        .. versionchanged:: 0.20.0
           Try uppercase atom type name as well
        """
        for atom_type in np.unique(atom_types):
            try:
                tables.masses[atom_type]
            except KeyError:
                try:
                    tables.masses[atom_type.upper()]
                except KeyError:
                    warnings.warn
                    ("Failed to guess the mass for the following atoms: {}"
                     .format(atom_type))

    def get_atom_mass(self, element):
        """Return the atomic mass in u for *element*.

        Masses are looked up in :data:`MDAnalysis.topology.tables.masses`.

        .. Warning:: Unknown masses are set to 0.0

        .. versionchanged:: 0.20.0
           Try uppercase atom type name as well
        """
        try:
            return tables.masses[element]
        except KeyError:
            try:
                return tables.masses[element.upper()]
            except KeyError:
                return 0.0

    def guess_atom_mass(self, atomname):
        """Guess a mass based on the atom name.

        :func:`guess_atom_element` is used to determine the kind of atom.

        .. warning:: Anything not recognized is simply set to 0;
        if you rely on the masses you might want to double check.
        """
        return self.get_atom_mass(self.guess_atom_element(atomname))

    def guess_types(self):
        """Guess the atom type of many atoms based on atom name

        Parameters
        ----------
        atom_names
          Name of each atom

        Returns
        -------
        atom_types : np.ndarray dtype object
        """
        names = self._atoms.names
        return np.array([self.guess_atom_element(n) for n in names], dtype=object)

    NUMBERS = re.compile(r'[0-9]')  # match numbers
    SYMBOLS = re.compile(r'[*+-]')  # match *, +, -

    def guess_atom_element(self, atomname):
        """Guess the element of the atom from the name.

        Looks in dict to see if element is found, otherwise it uses the first
        character in the atomname. The table comes from CHARMM and AMBER atom
        types, where the first character is not sufficient to determine the
        atom type. Some GROMOS ions have also been added.

        .. Warning: The translation table is incomplete.
           This will probably result in some mistakes,
           but it still better than nothing!

        See Also
        --------
        :func:`guess_atom_type`
        :mod:`MDAnalysis.topology.tables`
        """
        if atomname == '':
            return ''
        try:
            return tables.atomelements[atomname.upper()]
        except KeyError:
            # strip symbols and numbers
            no_symbols = re.sub(self.SYMBOLS, '', atomname)
            name = re.sub(self.NUMBERS, '', no_symbols).upper()

            # just in case
            if name in tables.atomelements:
                return tables.atomelements[name]

            while name:
                if name in tables.elements:
                    return name
                if name[:-1] in tables.elements:
                    return name[:-1]
                if name[1:] in tables.elements:
                    return name[1:]
                if len(name) <= 2:
                    return name[0]
                name = name[:-1]  # probably element is on left not right

            # if it's numbers
            return no_symbols

    def guess_bonds(self):
        atoms = self._atoms
        coords = self._atoms.positions
        box = self._kwargs['box']
        
        
        r"""Guess if bonds exist between two atoms based on their distance.

        Bond between two atoms is created, if the two atoms are within

        .. math::

              d < f \cdot (R_1 + R_2)

        of each other, where :math:`R_1` and :math:`R_2` are the VdW radii
        of the atoms and :math:`f` is an ad-hoc *fudge_factor*. This is
        the `same algorithm that VMD uses`_.

        Parameters
        ----------
        atoms : AtomGroup
             atoms for which bonds should be guessed
        coords : array
             coordinates of the atoms (i.e., `AtomGroup.positions)`)
        fudge_factor : float, optional
            The factor by which atoms must overlap eachother to be considered a
            bond.  Larger values will increase the number of bonds found. [0.55]
        vdwradii : dict, optional
            To supply custom vdwradii for atoms in the algorithm. Must be a dict
            of format {type:radii}. The default table of van der Waals radii is
            hard-coded as :data:`MDAnalysis.topology.tables.vdwradii`.  Any user
            defined vdwradii passed as an argument will supercede the table
            values. [``None``]
        lower_bound : float, optional
            The minimum bond length. All bonds found shorter than this length will
            be ignored. This is useful for parsing PDB with altloc records where
            atoms with altloc A and B maybe very close together and there should be
            no chemical bond between them. [0.1]
        box : array_like, optional
            Bonds are found using a distance search, if unit cell information is
            given, periodic boundary conditions will be considered in the distance
            search. [``None``]

        Returns
        -------
        list
            List of tuples suitable for use in Universe topology building.

        Warnings
        --------
        No check is done after the bonds are guessed to see if Lewis
        structure is correct. This is wrong and will burn somebody.

        Raises
        ------
        :exc:`ValueError` if inputs are malformed or `vdwradii` data is missing.


        .. _`same algorithm that VMD uses`:
           http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.1/ug/node26.html

        .. versionadded:: 0.7.7
        .. versionchanged:: 0.9.0
           Updated method internally to use more :mod:`numpy`, should work
           faster.  Should also use less memory, previously scaled as
           :math:`O(n^2)`.  *vdwradii* argument now augments table list
           rather than replacing entirely.
        """
        # why not just use atom.positions?
        if len(atoms) != len(coords):
            raise ValueError("'atoms' and 'coord' must be the same length")

        fudge_factor = self._kwargs.get('fudge_factor', 0.55)

        vdwradii = tables.vdwradii.copy()  # so I don't permanently change it
        user_vdwradii = self._kwargs.get('vdwradii', None)
        if user_vdwradii:  # this should make algo use their values over defaults
            vdwradii.update(user_vdwradii)

        # Try using types, then elements
        atomtypes = atoms.types

        # check that all types have a defined vdw
        if not all(val in vdwradii for val in set(atomtypes)):
            raise ValueError(("vdw radii for types: " +
                              ", ".join([t for t in set(atomtypes) if
                                         not t in vdwradii]) +
                              ". These can be defined manually using the" +
                              " keyword 'vdwradii'"))

        lower_bound = self._kwargs.get('lower_bound', 0.1)

        if box is not None:
            box = np.asarray(box)

        # to speed up checking, calculate what the largest possible bond
        # atom that would warrant attention.
        # then use this to quickly mask distance results later
        max_vdw = max([vdwradii[t] for t in atomtypes])

        bonds = []

        pairs, dist = distances.self_capped_distance(coords,
                                                     max_cutoff=2.0*max_vdw,
                                                     min_cutoff=lower_bound,
                                                     box=box)
        for idx, (i, j) in enumerate(pairs):
            d = (vdwradii[atomtypes[i]] + vdwradii[atomtypes[j]])*fudge_factor
            if (dist[idx] < d):
                bonds.append((atoms[i].index, atoms[j].index))
        return tuple(bonds)

    def guess_angles(self):
        """Given a list of Bonds, find all angles that exist between atoms.

        Works by assuming that if atoms 1 & 2 are bonded, and 2 & 3 are bonded,
        then (1,2,3) must be an angle.

        Returns
        -------
        list of tuples
            List of tuples defining the angles.
            Suitable for use in u._topology


        See Also
        --------
        :meth:`guess_bonds`


        .. versionadded 0.9.0
        """
        bonds = self._atoms.bonds
        angles_found = set()

        for b in bonds:
            for atom in b:
                other_a = b.partner(atom)  # who's my friend currently in Bond
                for other_b in atom.bonds:
                    if other_b != b:  # if not the same bond I start as
                        third_a = other_b.partner(atom)
                        desc = tuple([other_a.index, atom.index, third_a.index])
                        if desc[0] > desc[-1]:  # first index always less than last
                            desc = desc[::-1]
                        angles_found.add(desc)

        return tuple(angles_found)

    def guess_dihedrals(self):
        """Given a list of Angles, find all dihedrals that exist between atoms.

        Works by assuming that if (1,2,3) is an angle, and 3 & 4 are bonded,
        then (1,2,3,4) must be a dihedral.

        Returns
        -------
        list of tuples
            List of tuples defining the dihedrals.
            Suitable for use in u._topology

        .. versionadded 0.9.0
        """
        angles = self.atoms.angles
        dihedrals_found = set()

        for b in angles:
            a_tup = tuple([a.index for a in b])  # angle as tuple of numbers
            # if searching with b[0], want tuple of (b[2], b[1], b[0], +new)
            # search the first and last atom of each angle
            for atom, prefix in zip([b.atoms[0], b.atoms[-1]],
                                    [a_tup[::-1], a_tup]):
                for other_b in atom.bonds:
                    if not other_b.partner(atom) in b:
                        third_a = other_b.partner(atom)
                        desc = prefix + (third_a.index,)
                        if desc[0] > desc[-1]:
                            desc = desc[::-1]
                        dihedrals_found.add(desc)

        return tuple(dihedrals_found)


    def guess_improper_dihedrals(angles):
        """Given a list of Angles, find all improper dihedrals that exist between
        atoms.

        Works by assuming that if (1,2,3) is an angle, and 2 & 4 are bonded,
        then (2, 1, 3, 4) must be an improper dihedral.
        ie the improper dihedral is the angle between the planes formed by
        (1, 2, 3) and (1, 3, 4)

        Returns
        -------
            List of tuples defining the improper dihedrals.
            Suitable for use in u._topology

        .. versionadded 0.9.0
        """
        dihedrals_found = set()

        for b in angles:
            atom = b[1]  # select middle atom in angle
            # start of improper tuple
            a_tup = tuple([b[a].index for a in [1, 2, 0]])
            # if searching with b[1], want tuple of (b[1], b[2], b[0], +new)
            # search the first and last atom of each angle
            for other_b in atom.bonds:
                other_atom = other_b.partner(atom)
                # if this atom isn't in the angle I started with
                if not other_atom in b:
                    desc = a_tup + (other_atom.index,)
                    if desc[0] > desc[-1]:
                        desc = desc[::-1]
                    dihedrals_found.add(desc)

        return tuple(dihedrals_found)


