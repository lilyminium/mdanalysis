# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
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
import MDAnalysis as mda
import pytest
import numpy as np
import networkx as nx
import textwrap

from numpy.testing import assert_equal, assert_almost_equal

from MDAnalysis.analysis.leaflets import LeafletFinder
from MDAnalysis.analysis.leaflets.leafletfinder import optimize_cutoff
from MDAnalysisTests.datafiles import (Martini_membrane_gro,
                                       DPPC_vesicle_only,
                                       )

@pytest.fixture()
def single_planar():
    return mda.Universe(Martini_membrane_gro)


@pytest.fixture()
def single_planar_heads(single_planar):
    return single_planar.select_atoms("name PO4")


@pytest.fixture()
def lfinder(single_planar):
    return LeafletFinder(single_planar, select="name PO4",
                         method="graph", cutoff=15.0, pbc=True)


def test_optimize_cutoff(single_planar_heads):
    cutoff, N = optimize_cutoff(single_planar_heads, pbc=True)
    assert N == 2
    assert_almost_equal(cutoff, 10.5, decimal=4)


def test_graph_pbc_on_off(single_planar_heads):
    lfls_pbc_on = LeafletFinder(single_planar_heads, pbc=True)
    lfls_pbc_off = LeafletFinder(single_planar_heads, pbc=False)
    pbc_on_size = lfls_pbc_on.clusters.predictor.size()
    pbc_off_size = lfls_pbc_off.clusters.predictor.size() 
    assert pbc_on_size > pbc_off_size


def test_pbc_on_off_difference(single_planar_heads):
    lfls_pbc_on = LeafletFinder(single_planar_heads, cutoff=7, pbc=True)
    lfls_pbc_off = LeafletFinder(single_planar_heads, cutoff=7, pbc=False)
    pbc_on_graph = lfls_pbc_on.clusters.predictor
    pbc_off_graph = lfls_pbc_off.clusters.predictor
    diff_graph = nx.difference(pbc_on_graph, pbc_off_graph)
    assert_equal(set(diff_graph.edges), {(69, 153), (73, 79),
                                        (206, 317), (313, 319)})


@pytest.mark.parametrize("sparse", [True, False, None])
def test_sparse_on_off_none(single_planar_heads, sparse):
    lfls_ag = LeafletFinder(single_planar_heads, cutoff=15.0, pbc=True,
                            sparse=sparse)
    graph = lfls_ag.clusters.predictor
    assert_almost_equal(len(graph.edges), 1903, decimal=4)


def test_cutoff_update(lfinder):
    lfinder.update(cutoff=1.0)
    assert_almost_equal(lfinder.cutoff, 1.0, decimal=4)
    assert_almost_equal(len(lfinder.clusters), 360, decimal=4)


def test_cutoff_update_default(lfinder):
    lfinder.update()
    assert_almost_equal(lfinder.cutoff, 15.0, decimal=4)
    assert_almost_equal(len(lfinder.clusters), 2, decimal=4)


def test_write_selection(lfinder, tmpdir):
    with tmpdir.as_cwd():
        filename = lfinder.write_selection('leaflet.vmd')
        text = """\
        # Leaflets found by LeafletFinder(select='name PO4', cutoff=15.0 Å, pbc=True)

        # MDAnalysis VMD selection
        atomselect macro leaflet_1 {index 1 13 25 37 49 61 73 85 \\
        97 109 121 133 145 157 169 181 \\
        193 205 217 229 241 253 265 277 \\
        289 301 313 325 337 349 361 373 \\
        385 397 409 421 433 445 457 469 \\
        481 493 505 517 529 541 553 565 \\
        577 589 601 613 625 637 649 661 \\
        673 685 697 709 721 733 745 757 \\
        769 781 793 805 817 829 841 853 \\
        865 877 889 901 913 925 937 949 \\
        961 973 985 997 1009 1021 1033 1045 \\
        1057 1069 1081 1093 1105 1117 1129 1141 \\
        1153 1165 1177 1189 1201 1213 1225 1237 \\
        1249 1261 1273 1285 1297 1309 1321 1333 \\
        1345 1357 1369 1381 1393 1405 1417 1429 \\
        1441 1453 1465 1477 1489 1501 1513 1525 \\
        1537 1549 1561 1573 1585 1597 1609 1621 \\
        1633 1645 1657 1669 1681 1693 1705 1717 \\
        1729 1741 1753 1765 1777 1789 1801 1813 \\
        1825 1837 1849 1861 1873 1885 1897 1909 \\
        1921 1933 1945 1957 1969 1981 1993 2005 \\
        2017 2029 2041 2053 2065 2077 2089 2101 \\
        2113 2125 2137 2149 }
        # MDAnalysis VMD selection
        atomselect macro leaflet_2 {index 2521 2533 2545 2557 2569 2581 2593 2605 \\
        2617 2629 2641 2653 2665 2677 2689 2701 \\
        2713 2725 2737 2749 2761 2773 2785 2797 \\
        2809 2821 2833 2845 2857 2869 2881 2893 \\
        2905 2917 2929 2941 2953 2965 2977 2989 \\
        3001 3013 3025 3037 3049 3061 3073 3085 \\
        3097 3109 3121 3133 3145 3157 3169 3181 \\
        3193 3205 3217 3229 3241 3253 3265 3277 \\
        3289 3301 3313 3325 3337 3349 3361 3373 \\
        3385 3397 3409 3421 3433 3445 3457 3469 \\
        3481 3493 3505 3517 3529 3541 3553 3565 \\
        3577 3589 3601 3613 3625 3637 3649 3661 \\
        3673 3685 3697 3709 3721 3733 3745 3757 \\
        3769 3781 3793 3805 3817 3829 3841 3853 \\
        3865 3877 3889 3901 3913 3925 3937 3949 \\
        3961 3973 3985 3997 4009 4021 4033 4045 \\
        4057 4069 4081 4093 4105 4117 4129 4141 \\
        4153 4165 4177 4189 4201 4213 4225 4237 \\
        4249 4261 4273 4285 4297 4309 4321 4333 \\
        4345 4357 4369 4381 4393 4405 4417 4429 \\
        4441 4453 4465 4477 4489 4501 4513 4525 \\
        4537 4549 4561 4573 4585 4597 4609 4621 \\
        4633 4645 4657 4669 }\n"""

        with open("leaflet.vmd") as f:
            assert f.read() == textwrap.dedent(text)

class BaseTestPlanar(object):
    select = "name PO4"

    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)

    def test_leafletfinder(self, universe, method, kwargs):
        lf = LeafletFinder(universe, select=self.select, pbc=True,
                           method=method, **kwargs)

        for found, given in zip(lf.leaflets, self.leaflet_resix):
            assert_equal(found.resindices, given,
                         err_msg="Found wrong leaflet lipids")
        

@pytest.mark.parametrize("method,kwargs", [
    ("graph", {"cutoff": 20}),
])
class TestSinglePlanar(BaseTestPlanar):
    file = Martini_membrane_gro
    leaflet_resix = [np.arange(180), np.arange(225, 405)]

    @pytest.fixture()
    def lf(self, universe, method, kwargs):
        return LeafletFinder(universe, select=self.select,
                             pbc=True, method=method, **kwargs)

    def test_atomgroup(self, universe, method, kwargs):
        ag = universe.select_atoms(self.select)
        lf = LeafletFinder(ag, pbc=True, method=method, **kwargs)
        for found, given in zip(lf.leaflets, self.leaflet_resix):
            assert_equal(found.resindices, given,
                         err_msg="Found wrong leaflet lipids")


class BaseTestVesicle:
    file = DPPC_vesicle_only
    select = "name PO4"
    n_leaflets = 2

    full_20 = ([0,   43,   76,  112,  141,  172,  204,
                234,  270,  301,  342,  377,  409,  441,
                474,  513,  544,  579,  621,  647,  677,
                715,  747,  771,  811,  847,  882,  914,
                951,  982, 1016, 1046, 1084, 1116, 1150,
                1181, 1210, 1246, 1278, 1312, 1351, 1375,
                1401, 1440, 1476, 1505, 1549, 1582, 1618,
                1648, 1680, 1713, 1740, 1780, 1810, 1841,
                1864, 1899, 1936, 1974, 1999, 2033, 2066,
                2095, 2127, 2181, 2207, 2243, 2278, 2311,
                2336, 2368, 2400, 2427, 2456, 2482, 2515,
                2547, 2575, 2608, 2636, 2665, 2693, 2720,
                2748, 2792, 2822, 2860, 2891, 2936, 2960,
                2992, 3017],
               [ 3,   36,   89,  139,  198,  249,  298,
                340,  388,  435,  491,  528,  583,  620,
                681,  730,  794,  831,  877,  932,  979,
                1032, 1073, 1132, 1180, 1238, 1286, 1328,
                1396, 1441, 1490, 1528, 1577, 1625, 1688,
                1742, 1782, 1839, 1910, 1945, 2005, 2057,
                2111, 2153, 2180, 2236, 2286, 2342, 2401,
                2470, 2528, 2584, 2649, 2722, 2773, 2818,
                2861, 2905, 2961])
    
    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)


@pytest.mark.parametrize("method,kwargs", [
    ("graph", {"cutoff": 25}),
])
class TestVesicleFull(BaseTestVesicle):
    def test_full(self, universe, method, kwargs):
        lf = LeafletFinder(universe.atoms, select=self.select,
                           n_leaflets=self.n_leaflets, pbc=True,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflets, self.full_20):
            assert_equal(found.residues.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")
