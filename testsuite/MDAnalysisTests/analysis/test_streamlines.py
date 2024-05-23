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
import numpy as np
from numpy.testing import assert_allclose
import MDAnalysis
from MDAnalysis.visualization import (streamlines,
                                      streamlines_3D)
from MDAnalysis.coordinates.XTC import XTCWriter
from MDAnalysisTests.datafiles import Martini_membrane_gro
import pytest
from pytest import approx
from MDAnalysis.analysis.streamlines import Streamlines
import matplotlib.pyplot as plt
import os

@pytest.fixture(scope="session")
def membrane_xtc(tmpdir_factory):
    univ = MDAnalysis.Universe(Martini_membrane_gro)
    x_delta, y_delta, z_delta  = 0.5, 0.3, 0.2
    tmp_xtc = tmpdir_factory.mktemp('streamlines').join('dummy.xtc')

    with XTCWriter(str(tmp_xtc), n_atoms=univ.atoms.n_atoms) as xtc_writer:
        for i in range(5):
           univ.atoms.translate([x_delta, y_delta, z_delta])
           xtc_writer.write(univ.atoms)
           x_delta += 0.1
           y_delta += 0.08
           z_delta += 0.02
    return str(tmp_xtc)

@pytest.fixture(scope="session")
def univ(membrane_xtc):
    u = MDAnalysis.Universe(Martini_membrane_gro, membrane_xtc)
    return u


def test_streamplot_2D(univ):
    # regression test ported from previous visualization form
    xmin, xmax, ymin, ymax = (3.5300002, 117.520004, 2.33, 116.28)
    analysis = Streamlines(
        univ,
        select="name PO4",
        grid_spacing=20,
        x_min=xmin,
        x_max=xmax,
        y_min=ymin,
        y_max=ymax,
        maximum_delta_magnitude=2.0,
    )
    analysis.run(start=2, stop=4)

    # test grid bounds
    assert_allclose(
        analysis.results.x_edges,
        [3.5300002, 23.5300002, 43.5300002, 63.5300002, 83.5300002, 103.5300002],
    )
    assert_allclose(
        analysis.results.y_edges,
        [2.32999992, 22.32999992, 42.32999992, 62.32999992, 82.32999992, 102.32999992],
    )
    assert analysis._framewise_grid_centroids.shape == (2, 25, 2)

    assert_allclose(
        analysis.results.dx_array[0],
        np.array([[0.79999924, 0.79999924, 0.80000687, 0.79999542, 0.79998779],
                  [0.80000019, 0.79999542, 0.79999924, 0.79999542, 0.80001068],
                  [0.8000021, 0.79999924, 0.80001068, 0.80000305, 0.79999542],
                  [0.80000019, 0.79999542, 0.80001068, 0.80000305, 0.80000305],
                  [0.79999828, 0.80000305, 0.80000305, 0.80000305, 0.79999542]]),
        atol=1e-4
    )
    assert_allclose(
        analysis.results.dy_array[0],
        np.array([[0.53999901, 0.53999996, 0.53999996, 0.53999996, 0.54000092],
                    [0.5399971, 0.54000092, 0.54000092, 0.54000092, 0.5399971 ],
                    [0.54000473, 0.54000473, 0.54000092, 0.5399971, 0.54000473],
                    [0.54000092, 0.53999329, 0.53999329, 0.53999329, 0.54000092],
                    [0.54000092, 0.53999329, 0.53999329, 0.54000092, 0.53999329]]),
        atol=1e-4,
    )
    assert analysis.results.displacement_mean == pytest.approx(0.965194167)
    assert analysis.results.displacement_std == pytest.approx(4.444808820e-06)

# def test_streamplot_2D_zero_return(membrane_xtc, univ, tmpdir):
#     # simple roundtrip test to ensure that
#     # zeroed arrays are returned by the 2D streamplot
#     # code when called with an empty selection
#     u1, v1, avg, std = streamlines.generate_streamlines(topology_file_path=Martini_membrane_gro,
#                                                         trajectory_file_path=membrane_xtc,
#                                                         grid_spacing=20,
#                                                         MDA_selection='name POX',
#                                                         start_frame=1,
#                                                         end_frame=2,
#                                                         xmin=univ.atoms.positions[...,0].min(),
#                                                         xmax=univ.atoms.positions[...,0].max(),
#                                                         ymin=univ.atoms.positions[...,1].min(),
#                                                         ymax=univ.atoms.positions[...,1].max(),
#                                                         maximum_delta_magnitude=2.0,
#                                                         num_cores=1)
#     assert_allclose(u1, np.zeros((5, 5)))
#     assert_allclose(v1, np.zeros((5, 5)))
#     assert avg == approx(0.0)
#     assert std == approx(0.0)


# def test_streamplot_3D(membrane_xtc, univ, tmpdir):
#     # because mayavi is too heavy of a dependency
#     # for a roundtrip plotting test, simply
#     # aim to check for sensible values
#     # returned by generate_streamlines_3d
#     dx, dy, dz = streamlines_3D.generate_streamlines_3d(topology_file_path=Martini_membrane_gro,
#                                                         trajectory_file_path=membrane_xtc,
#                                                         grid_spacing=20,
#                                                         MDA_selection='name PO4',
#                                                         start_frame=1,
#                                                         end_frame=2,
#                                                         xmin=univ.atoms.positions[...,0].min(),
#                                                         xmax=univ.atoms.positions[...,0].max(),
#                                                         ymin=univ.atoms.positions[...,1].min(),
#                                                         ymax=univ.atoms.positions[...,1].max(),
#                                                         zmin=univ.atoms.positions[...,2].min(),
#                                                         zmax=univ.atoms.positions[...,2].max(),
#                                                         maximum_delta_magnitude=2.0,
#                                                         num_cores=1)
#     assert dx.shape == (5, 5, 2)
#     assert dy.shape == (5, 5, 2)
#     assert dz.shape == (5, 5, 2)
#     assert dx[4, 4, 0] == approx(0.700004, abs=1e-5)
#     assert dy[0, 0, 0] == approx(0.460000, abs=1e-5)
#     assert dz[2, 2, 0] == approx(0.240005, abs=1e-5)
