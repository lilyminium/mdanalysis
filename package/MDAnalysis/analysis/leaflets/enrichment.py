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


import scipy
import numpy as np
import pandas as pd

from .. import distances
from .base import LeafletAnalysis

class LipidEnrichment(LeafletAnalysis):
    def __init__(self, universe, *args, select_protein="protein",
                 cutoff=6, compute_headgroup_only=True,
                 distribution="binomial", compute_p_value=True,
                 buffer=0, beta=4, **kwargs):
        super(LipidEnrichment, self).__init__(universe, *args, **kwargs)

        self.distribution = distribution.lower()
        if self.distribution == "binomial":
            self._fit_distribution = self._fit_binomial
        elif self.distribution == "gaussian":
            self._fit_distribution = self._fit_gaussian
        else:
            raise ValueError("`distribution` should be either "
                             "'binomial' or 'gaussian'")

        self.compute_p_value = compute_p_value

        if self.compute_p_value:
            if buffer:
                self._compute_p = self._compute_p_hypergeom_gamma
            elif self.distribution == "gaussian":
                self._compute_p = self._compute_p_gaussian
            else:
                self._compute_p = self._compute_p_hypergeom

        self.protein = universe.select_atoms(select_protein)

        if compute_headgroup_only:
            self._compute_atoms = self.selection
        else:
            self._compute_atoms = self.residues.atoms
        self._resindices = self._compute_atoms.resindices
        self.cutoff = cutoff
        self.buffer = buffer
        self.beta = beta
        self.leaflets = []
        self.leaflets_summary = []


    def _prepare(self):
        # in case of change + re-run
        self.mid_buffer = self.buffer / 2.0
        self.max_cutoff = self.cutoff + self.buffer
        self._buffer_sigma = self.buffer / self.beta
        if self._buffer_sigma:
            self._buffer_coeff = 1 / (self._buffer_sigma * np.sqrt(2 * np.pi))
        self.ids = np.unique(getattr(self.residues, self.group_by_attr))

        # results
        self.near_counts = np.zeros((self.n_leaflets, len(self.ids),
                                     self.n_frames))
        self.residue_counts = np.zeros((self.n_leaflets, len(self.ids),
                                        self.n_frames))
        self.total_counts = np.zeros((self.n_leaflets, self.n_frames))
        self.leaflet_residues = np.zeros((self.n_frames, self.n_leaflets),
                                         dtype=object)

    def _update_leaflets(self):
        self.leafletfinder.run()
        self._current_leaflets = [l.residues for l in self.leafletfinder.leaflets[:self.n_leaflets]]
        self._current_ids = [getattr(r, self.group_by_attr) for r in self._current_leaflets]

    def _single_frame(self):
        # initial scoop for nearby groups
        coords_ = self._compute_atoms.positions
        pairs = distances.capped_distance(self.protein.positions,
                                          coords_,
                                          self.cutoff, box=self.protein.dimensions,
                                          return_distances=False)
        if pairs.size > 0:
            indices = np.unique(pairs[:, 1])
        else:
            indices = []

        # now look for groups in the buffer
        if len(indices) and self.buffer:
            pairs2, dist = distances.capped_distance(self.protein.positions,
                                                     coords_, self.max_cutoff,
                                                     min_cutoff=self.cutoff,
                                                     box=self.protein.dimensions,
                                                     return_distances=True)
            
            # don't count things in inner cutoff
            mask = [x not in indices for x in pairs2[:, 1]]
            pairs2 = pairs2[mask]
            dist = dist[mask]

            if pairs2.size > 0:
                _ix = np.argsort(pairs2[:, 1])  
                indices2 = pairs2[_ix][:, 1]
                dist = dist[_ix] - self.cutoff

                init_resix2 = self._resindices[indices2]
                # sort through for minimum distance
                ids2, splix = np.unique(init_resix2, return_index=True)
                resix2 = init_resix2[splix]
                split_dist = np.split(dist, splix[1:])
                min_dist = np.array([x.min() for x in split_dist])

                # logistic function
                for i, leaf in enumerate(self._current_leaflets):
                    ids = self._current_ids[i]
                    match, rix, lix = np.intersect1d(resix2, leaf.residues.resindices,
                                                     assume_unique=True,
                                                     return_indices=True)
                    subdist = min_dist[rix]
                    subids = ids[lix]
                    for j, x in enumerate(self.ids):
                        mask = (subids == x)
                        xdist = subdist[mask]
                        exp = -0.5 * ((xdist/self._buffer_sigma) ** 2)
                        n = self._buffer_coeff * np.exp(exp)
                        self.near_counts[i, j, self._frame_index] += n.sum()

        soft = self.near_counts[:, :, self._frame_index].sum()

        init_resix = self._resindices[indices]
        resix = np.unique(init_resix)
        for i, leaf in enumerate(self._current_leaflets):
            ids = self._current_ids[i]
            _, ix1, ix2 = np.intersect1d(resix, leaf.residues.resindices,
                                         assume_unique=True,
                                         return_indices=True)
            self.total_counts[i, self._frame_index] = len(ix1)
            subids = ids[ix2]
            for j, x in enumerate(self.ids):
                self.residue_counts[i, j, self._frame_index] += sum(ids == x)
                self.near_counts[i, j, self._frame_index] += sum(subids == x)

        both = self.near_counts[:, :, self._frame_index].sum()

    def _conclude(self):
        self.leaflets = []
        self.leaflets_summary = []

        for i in range(self.n_leaflets):
            timeseries = {}
            summary = {}
            res_counts = self.residue_counts[i]
            near_counts = self.near_counts[i]

            near_all = near_counts.sum(axis=0)
            total_all = res_counts.sum(axis=0)
            n_near_tot = near_all.sum()
            n_all_tot = total_all.sum()
            d, s = self._collate(near_all, near_all, total_all,
                                 total_all, n_near_tot, n_all_tot)
            timeseries['all'] = d
            summary['all'] = s
            for j, resname in enumerate(self.ids):
                near_species = near_counts[j]
                total_species = res_counts[j]
                d, s = self._collate(near_species, near_all, total_species,
                                     total_all, n_near_tot, n_all_tot)
                timeseries[resname] = d
                summary[resname] = s
            self.leaflets.append(timeseries)
            self.leaflets_summary.append(summary)


    def _fit_gaussian(self, data, *args, **kwargs):
        """Treat each frame as an independent observation in a gaussian
        distribution.

        Appears to be original method of [Corradi2018]_.

        .. note::

            The enrichment p-value is calculated from a two-tailed
            sample T-test, following [Corradi2018]_.

        """
        near = data['Near protein']
        frac = data['Fraction near protein']
        dei = data['Enrichment']
        summary = {
            'Mean near protein': near.mean(),
            'SD near protein': near.std(),
            'Mean fraction near protein': frac.mean(),
            'SD fraction near protein': frac.std(),
            'Mean enrichment': dei.mean(),
            'SD enrichment': dei.std()
        }
        if self.compute_p_value:
            # sample T-test, 2-tailed
            t, p = scipy.stats.ttest_1samp(dei, 1)
            summary['Enrichment p-value'] = p

        return summary

    def _fit_binomial(self, data: dict, n_near_species: np.ndarray,
                      n_near: np.ndarray, n_species: np.ndarray,
                      n_all: np.ndarray, n_near_tot: int, n_all_tot: int):
        """
        This function computes the following approximate probability
        distributions and derives statistics accordingly.

        * The number of lipids near the protein is represented as a
        normal distribution.
        * The fraction of lipids near the protein follows a
        hypergeometric distribution.
        * The enrichment is represented as the log-normal distribution
        derived from the ratio of two binomial convolutions of the
        frame-by-frame binomial distributions.

        All these approximations assume that each frame or observation is
        independent. The binomial approximation assumes that:

        * the number of the lipid species near the protein is
        small compared to the total number of that lipid species
        * the total number of all lipids is large
        * the fraction (n_species / n_all) is not close to 0 or 1.

        .. note::

            The enrichment p-value is calculated from the log-normal
            distribution of the null hypothesis: that the average
            enrichment is representative of the ratio of
            n_species : n_all

        """

        summary = {"Total # lipids, all": n_all_tot,
                   "Total # lipids, shell": n_near_tot}
        p_time = data['Fraction near protein']
        summary['Total # species, shell'] = N = n_near_species.sum()
        summary['Total # species, all'] = N_sp = n_species.sum()
        if n_near_tot:  # catch zeros
            p_shell = N / n_near_tot
        else:
            p_shell = 0
        if n_all_tot:
            p_null = N_sp / n_all_tot
        else:
            p_null = 0

        # n events: assume normal
        summary['Mean # species, shell'] = n_near_species.mean()
        summary['SD # species, shell'] = sd = n_near_species.std()

        # actually hypergeometric, but binomials are easier
        # X ~ B(n_near_tot, p_shell)
        summary['Mean fraction of species, shell'] = p_shell
        summary['SD fraction of species, shell'] = sd_frac = sd / n_near.mean()


        if p_null == 0:
            summary['Mean enrichment'] = 1
            summary['SD enrichment'] = 0
            # summary['Analytical SD enrichment'] = 0
            # summary['Sample SD enrichment'] = 0
        
        else:
            summary['Mean enrichment'] = p_shell / p_null
            summary['SD enrichment'] = sd_frac / p_null
            # summary['Analytical SD enrichment'] = dist_sd / p_null
            # summary['Sample SD enrichment'] = samp_sd / p_null


        if self.compute_p_value:
            p = scipy.stats.binom_test(N, n_near_tot, p_null)
            summary['Enrichment p-value'] = p

        return summary

    def _collate(self, n_near_species: np.ndarray, n_near: np.ndarray,
                 n_species: np.ndarray, n_all: np.ndarray,
                 n_near_tot: int, n_all_tot: int):
        data = {}
        data['Near protein'] = n_near_species
        frac = np.nan_to_num(n_near_species / n_near, nan=0.0)
        data['Fraction near protein'] = frac
        data['Total number'] = n_species
        n_total = np.nan_to_num(n_species / n_all, nan=0.0)
        # data['Fraction total'] = n_total
        n_total[n_total == 0] = np.nan
        dei = np.nan_to_num(frac / n_total, nan=0.0)
        data['Enrichment'] = dei
        # if self.compute_p_value:
        #     pval = np.zeros(len(frac))
        #     for i, args in enumerate(zip(dei, n_near_species, n_all,
        #                                  n_species, n_near_species)):
        #         pval[i] = self._compute_p(*args)
        #     data['Enrichment p-value'] = pval

        summary = self._fit_distribution(data, n_near_species, n_near,
                                         n_species, n_all, n_near_tot,
                                         n_all_tot)
        return data, summary

    def summary_as_dataframe(self):
        """Convert the results summary into a pandas DataFrame.

        This requires pandas to be installed.
        """

        if not self.leaflets_summary:
            raise ValueError('Call run() first to get results')
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('pandas is required to use this function '
                              'but is not installed. Please install with '
                              '`conda install pandas` or '
                              '`pip install pandas`.') from None

        dfs = [pd.DataFrame.from_dict(d, orient='index')
               for d in self.leaflets_summary]
        for i, df in enumerate(dfs, 1):
            df['Leaflet'] = i
        df = pd.concat(dfs)
        return df

    def _compute_p_ttest(self, dei, *args):
        # sample T-test, 2-tailed
        t, p = scipy.stats.ttest_1samp(dei, 1)
        return p

    def _compute_p_hypergeom(self, dei, k, N, K, n):
        kn = k/n
        KN = K/N
        if kn <= KN:
            return scipy.stats.hypergeom.cdf(k, N, K, n)
        return scipy.stats.hypergeom.sf(k, N, K, n)

    def _compute_p_hypergeom_gamma(self, dei, k, N, K, n):
        K_k = binomial_gamma(K, k)
        N_k = binomial_gamma(N - K, n - k)
        N_n = binomial_gamma(N, n)
        return K_k * N_k / N_n

    def _compute_p_gaussian(self, dei, k, N, K, n):
        kn = k/n
        KN = K/N
        sigma = KN / 2.5
        if kn <= KN:
            return scipy.stats.norm.cdf(kn, KN, sigma)
        return scipy.stats.norm.sf(kn, KN, sigma)