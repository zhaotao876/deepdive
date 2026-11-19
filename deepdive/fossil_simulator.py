import sys
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
from collections.abc import Iterable
from .feature_extraction import *
from .feature_extraction import _accumulate_reduce
from .utilities import get_rnd_gen
SMALL_NUMBER = 1e-10

# [PERF 2026-11-19] Number of species processed at a time when a
# (n_species x n_areas x n_time_bins) array is built in pieces instead of all at
# once. For n_species = 350k, n_areas = 10, n_bins = 40 a single such array is
# 1.07 GiB, so this bounds the temporary buffers to roughly 80 MiB each.
# The chunk size does not affect the results: all the reductions used here are
# performed with an exact, order preserving accumulation (see
# _accumulate_reduce below), so the output is bitwise identical for any chunk
# size. It is only a memory / cache trade-off.
DEFAULT_CHUNK_SIZE = 25000


# [PERF 2026-11-19] _accumulate_reduce() is defined in feature_extraction.py
# (next to the trajectory functions that use it) and imported explicitly here,
# because ``from .feature_extraction import *`` does not export names starting
# with an underscore.


class fossil_simulator():
    def __init__(self,
                 n_areas=30,
                 n_bins=10,  # number of time bins using a numpy
                 eta=1.5,  # cannot be smaller than 1. At 1, there is no stochasticity.
                 # e.g. 2 = 2-fold variation across areas
                 p_gap=0.1,  # probability of 0 preservation in a time bin - can be an iterable
                 dispersal_rate=None,  # if none, use disp rate mean and variance
                 max_dist=1,
                 disp_rate_mean=1,  # todo: adjust these values
                 disp_rate_variance=0.5,
                 area_mean=1,
                 area_variance=1,
                 size_concentration_parameter=1,  # positive. With larger values area sizes tend to be more similar
                 # can be an iterable
                 link_area_size_carrying_capacity=10,  # positive, larger numbers = stronger link between area size
                 # and carrying capacity can be an iterable
                 p_origination_a_slope_mean=1,  # slope of the probabilitiy of origination areas mean
                 p_origination_a_slope_sd=0.01,
                 sp_mean=1,  # can be iterable
                 sp_variance=0.5,
                 sp_bimodal=False,  # if true draw from min(sp_mean) and from max(sp_mean)
                 sp_bimodal_p=None,  # default: 50-50 otherwise e.g. [0.9, 0.1] 0.9 prob of low rate in the bimodal
                 singletons_frequency=None,  # if set to e.g. 0.3 it will inflate frequency of singleton species
                 time_bins=None,
                 slope=0,  # change in log-sampling rate through time (log-linear) (can be iterable)
                 intercept=0.05,  # initial sampling rate (can be iterable)
                 sd_through_time=0.5,  # st dev in log-sampling rate through time (can be iterable)
                 maximum_localities_per_bin=10, # max locality rate per bin per area
                 sd_through_time_skyline=2,     # locs ~ Poi(min(rate, maximum_localities_per_bin))
                 mean_n_epochs_skyline=4,
                 mean_rate_skyline=None,
                 fraction_skyline_sampling=0.5,
                 bin_sampling=0.,
                 bin_std_rates=None,
                 bin_mean_rates=None,
                 locality_rate_multiplier=None,  # array of shape = (n_areas x n_time_bins)
                 carrying_capacity_multiplier=None,  # array of shape = (n_species x n_areas). Must be between 0 and 1
                 sampling_rate_multiplier=None, # array of shape = (n_areas x n_time_bins)
                 target_n_occs=None,  # Integer
                 target_n_occs_range=10, # [target_n_occs / target_n_occs_range, target_n_occs * target_n_occs_range]
                 species_per_locality_multiplier=None,
                 seed=None,
                 verbose=False):

        self.n_areas = n_areas
        self.n_bins = n_bins
        self.eta = eta  # stochasticity among areas
        self.p_gap = p_gap
        self.dispersal_rate = dispersal_rate
        self.max_dist = max_dist
        self.disp_rate_mean = disp_rate_mean
        self.disp_rate_variance = disp_rate_variance
        self.area_scale = area_variance / area_mean
        self.area_shape = area_mean / self.area_scale
        self.size_concentration_parameter = size_concentration_parameter
        self.link_area_size_carrying_capacity = link_area_size_carrying_capacity
        self.p_origination_a_slope_mean = p_origination_a_slope_mean
        self.p_origination_a_slope_sd = p_origination_a_slope_sd
        self.sp_mean = sp_mean
        self.sp_variance = sp_variance
        self.sp_bimodal = sp_bimodal
        self.sp_bimodal_p = sp_bimodal_p
        self.singletons_frequency = singletons_frequency
        self.sd_through_time = sd_through_time
        self.time_bins = time_bins
        self.slope = slope
        self.intercept = intercept
        self.max_localities = maximum_localities_per_bin
        self.sd_through_time_skyline = sd_through_time_skyline
        self.mean_rate_skyline = mean_rate_skyline
        self.mean_n_epochs_skyline = mean_n_epochs_skyline
        self.fraction_skyline_sampling = fraction_skyline_sampling
        self.bin_sampling = bin_sampling
        self.bin_mean_rates = np.tile(bin_mean_rates, (n_areas, 1)) # shape 'at'
        self.bin_std_rates = np.tile(bin_std_rates, (n_areas, 1))
        self.locality_rate_multiplier = locality_rate_multiplier
        self.carrying_capacity_multiplier = carrying_capacity_multiplier
        self.sampling_rate_multiplier = sampling_rate_multiplier
        self.target_n_occs = target_n_occs
        self.target_n_occs_range = target_n_occs_range
        self.species_per_locality_multiplier = species_per_locality_multiplier
        self._additional_info = None
        self._rs = get_rnd_gen(seed)
        self.verbose = verbose

    def reset_seed(self, seed):
        self._rs = get_rnd_gen(seed)


    # Generate presence/absence geographic range array.
    # Define a 2D array where 0 means absence and 1 means presence of a species in a set number of defined geographic
    # areas. Could add evolution of the geographic range and allow some species to go extinct in certain areas etc
    # or change the additional ranges to account for a distance between areas (constant range, but accounts for some
    # areas closer than others)
    def set_locality_rate_multiplier(self, locality_rate_multiplier):
        self.locality_rate_multiplier = locality_rate_multiplier

    def set_carrying_capacity_multiplier(self,
                                         m_species_origin,  # shape = (s, a)
                                         m_sp_area_time  # shape = (s, a, t)
                                         ):
        self.carrying_capacity_multiplier = [m_species_origin, m_sp_area_time]

    def set_sampling_rate_multiplier(self, sampling_rate_multiplier):
        self.sampling_rate_multiplier = sampling_rate_multiplier

    def reset(self):
        self.locality_rate_multiplier = None
        self.carrying_capacity_multiplier = None
        self.sampling_rate_multiplier = None

    def bd_sim_setup(self,
                     sp_x,
                     min_age=None,
                     max_age=None):
        self.sp_x = sp_x
        if min_age is not None:
            self.min_age = min_age
        else:
            self.min_age = np.min(sp_x)
        if max_age is not None:
            self.max_age = max_age
        else:
            self.max_age = np.max(sp_x)

        self.n_species = len(sp_x)
        if self.time_bins is None:
            self.time_bins = np.linspace(self.min_age, self.max_age, self.n_bins + 1)
            # draws equally spaced numbers between 0 and 100

        self.time_bins_duration = np.diff(self.time_bins)
        mid_time_bins = []
        for i in range(0, self.n_bins):
            mid_time_i = self.time_bins[i] + 0.5 * self.time_bins_duration[i]
            mid_time_bins.append(mid_time_i)
        self.mid_time_bins = np.array(mid_time_bins)

        # print("self.mid_time_bins ", self.mid_time_bins )

    def simulate_geo_ranges(self,
                            prob_origination_area=None,  # if None : random
                            distance_matrix=None,
                            area_size=None
                            ):
        if prob_origination_area is None:
            if isinstance(self.link_area_size_carrying_capacity, Iterable):
                # if it is an iterable draw at random from provided min/max
                kappa = self._rs.uniform(np.min(self.link_area_size_carrying_capacity),
                                         np.max(self.link_area_size_carrying_capacity))
                # phi = kappa + slope*self.n_bins  # what is used for slope?
                # K = np.exp(phi)/sum(np.exp(phi))
            else:
                kappa = self.link_area_size_carrying_capacity
                # phi = kappa + slope * self.n_bins  # what is used for slope?
                # K = np.exp(phi) / sum(np.exp(phi))
            if area_size is None:
                prob_origination_area = self._rs.dirichlet(np.ones(self.n_areas) * kappa)
            else:
                prob_origination_area = self._rs.dirichlet(area_size * kappa)

            # turn prob_origination_area into [s x a] array
            # print(prob_origination_area)
            exp_pr = np.log(prob_origination_area + SMALL_NUMBER)  # intercept at time 0
            slope_pr = self._rs.normal(self.p_origination_a_slope_mean, self.p_origination_a_slope_sd, self.n_areas)
            time_of_origination = self.sp_x[:, 0]
            # print("prob_origination_area", prob_origination_area, self.carrying_capacity_multiplier,
            #       np.einsum('s,a -> sa', time_of_origination, slope_pr))
            exp_pr_at_origination = np.einsum('s,a -> sa', time_of_origination, slope_pr) + exp_pr
            if self.carrying_capacity_multiplier is not None:
                exp_pr_at_origination *= np.log(self.carrying_capacity_multiplier[0] + SMALL_NUMBER)
            # print("\n\n", exp_pr_at_origination, "\n\n")
            if np.nanmax(exp_pr_at_origination) > 100:
                exp_pr_at_origination /= np.nanmax(exp_pr_at_origination)

            pr_at_origination = np.einsum('sa, s -> sa', np.exp(exp_pr_at_origination),
                                          1 / (np.sum(np.exp(exp_pr_at_origination), 1)))
            if np.all(np.isfinite(pr_at_origination)):
                pass
            else:
                # prevent overflows
                # print(pr_at_origination)
                # quit()
                p_tmp = self.carrying_capacity_multiplier[0] + SMALL_NUMBER
                pr_at_origination = np.einsum('sa, s -> sa', p_tmp,
                                              1 / (np.sum(p_tmp, 1)))
                if self.verbose:
                    print(pr_at_origination, "overflow prevented!")
        else:
            sys.exit('fixed prob_origination_area not implemented')
        if distance_matrix is None:
            dist_matrix = self._rs.uniform(0, self.max_dist, (self.n_areas, self.n_areas))  # e.g. in Km
            X = 0 + dist_matrix
            X = np.triu(X, 1)  # set lower and diagonal to 0
            distance_matrix = X + X.T  # add transposed matrix to set lower tr = upper tr
        if self.dispersal_rate == "None":
            self.dispersal_rate = None
        if self.dispersal_rate is None:
            if isinstance(self.disp_rate_mean, Iterable):  # if iterable
                disp_rate_mean = self._rs.uniform(np.min(self.disp_rate_mean),
                                                  np.max(self.disp_rate_mean))
            else:
                disp_rate_mean = self.disp_rate_mean
            disp_rate_scale = self.disp_rate_variance / disp_rate_mean
            disp_rate_shape = disp_rate_mean / disp_rate_scale

            dispersal_rate = self._rs.gamma(disp_rate_shape, scale=disp_rate_scale, size=self.n_species)
        else:
            dispersal_rate = self.dispersal_rate * np.ones(self.n_species)
            disp_rate_mean = dispersal_rate
            self.disp_rate_variance = 0
        # print(np.min(dispersal_rate), np.max(dispersal_rate),
        #       disp_rate_mean, np.mean(dispersal_rate),
        #       disp_rate_shape, disp_rate_scale)
        dispersal_rate += SMALL_NUMBER  # avoid divide-by-zero

        # [PERF 2026-11-19] The dispersal probability tensor is no longer built.
        #
        # The previous implementation was:
        #     m = np.einsum('ab,s->sab', distance_matrix, 1 / dispersal_rate)
        #     dispersal_pr = np.exp(-m)
        # which materialises two (n_species x n_areas x n_areas) float64
        # arrays. For n_species = 350k and n_areas = 10 that is 2 x 267 MiB =
        # 534 MiB, and it was the single largest memory consumer of the
        # simulation.
        # (With n_areas = 30 the same tensors would be 2 x 2.35 GiB = 4.7 GiB.)
        #
        # The tensor was never used as a whole: the loop below only ever reads
        # one (n_areas, n_areas) slice per species, and in fact only one *row*
        # of that slice:
        #     row = dispersal_pr[sp_i]              # (a, b)
        #     np.fill_diagonal(row, 0)               # row[a, a] = 0
        #     p_vec = row[sp_init_range]             # row (sp_init_range, :)
        # so that p_vec[b] = exp(-distance_matrix[sp_init_range, b] / rate)
        # for every b, with p_vec[sp_init_range] = 0 because of the zeroed
        # diagonal.
        #
        # Computing that single row directly per species gives bitwise
        # identical results (exp is elementwise, and the multiplication
        # distance_matrix[sp_init_range, b] * (1 / rate) is the same operation
        # that einsum performs) while using a few hundred bytes instead of
        # gigabytes.
        inv_dispersal_rate = 1 / dispersal_rate
        # init geo table
        geo_table = np.zeros((self.n_species, self.n_areas))
        # init first sp area
        # sp_range = self._rs.choice(np.arange(self.n_areas), p=prob_origination_area, size=self.n_species)
        # geo_table[np.arange(self.n_species), sp_range] = 1

        # add areas based on dispersal and distance from init range
        area_indices = np.arange(self.n_areas)
        for sp_i in range(self.n_species):
            # init first sp area
            sp_init_range = self._rs.choice(area_indices, p=pr_at_origination[sp_i])
            geo_table[sp_i, sp_init_range] = 1
            # the dispersal probability of reaching each area from the area of
            # origin, i.e. one row of exp(-distance_matrix / dispersal_rate).
            dispersal_pr_sp_i = np.exp(-(distance_matrix[sp_init_range] * inv_dispersal_rate[sp_i]))
            # the diagonal (the area of origin itself) is set to zero
            dispersal_pr_sp_i[sp_init_range] = 0.0
            add_areas = self._rs.binomial(1, dispersal_pr_sp_i)
            geo_table[sp_i] += add_areas

        if self.carrying_capacity_multiplier is not None:
            # remove species from areas with 0-carrying capacity
            # print("geo_table", geo_table.shape, self.carrying_capacity_multiplier[1].shape)
            geo_table *= self.carrying_capacity_multiplier[0]

        return geo_table, slope_pr, pr_at_origination, kappa, disp_rate_mean

    # check the size simulation works (higher div in larger areas).
    def generate_area_size_table(self):
        if isinstance(self.size_concentration_parameter, Iterable):  # if eta is an iterable
            alpha = self._rs.uniform(np.min(self.size_concentration_parameter),
                                     np.max(self.size_concentration_parameter))
        else:
            alpha = self.size_concentration_parameter
        alphas = np.ones(self.n_areas) * alpha
        a_sizes = self._rs.dirichlet(alphas) * self.n_areas
        # scales to mean = 1 so area_rate retains the same meaning
        return a_sizes, alphas

    def preservation_rate_through_time_table(self):
        if isinstance(self.slope, Iterable):  # if iterable
            slope = self._rs.uniform(np.min(self.slope),
                                     np.max(self.slope),
                                     self.n_areas)
        else:
            slope = np.ones(self.n_areas) * self.slope

        if isinstance(self.intercept, Iterable):  # if iterable
            intercept = self._rs.uniform(np.min(self.intercept),
                                         np.max(self.intercept))
        else:
            intercept = self.intercept

        if isinstance(self.sd_through_time, Iterable):  # if iterable
            sd_through_time = self._rs.uniform(np.min(self.sd_through_time),
                                               np.max(self.sd_through_time))
        else:
            sd_through_time = self.sd_through_time

        ### SKYLINE MODEL
        if self._rs.random() > self.fraction_skyline_sampling:
            mu = np.einsum('a, t -> at', slope, self.mid_time_bins) + np.log(intercept)
            # print(mu)
            loc_rates = self._rs.normal(loc=mu, scale=sd_through_time)
            if self.verbose:
                print("continuous rates", np.exp(loc_rates)[0])
        else:

            if self._rs.random() < self.bin_sampling:
                loc_rates = self._rs.normal(loc=self.bin_mean_rates,
                                            scale=self.bin_std_rates)
                if self.verbose:
                    print("bin_sampling", np.exp(loc_rates)[0])
            else:
                if self.mean_rate_skyline is None:
                    mu = np.mean(np.einsum('a, t -> at', slope, self.mid_time_bins) + np.log(intercept))
                elif isinstance(self.mean_rate_skyline, Iterable):
                    mu = self._rs.uniform(np.min(self.mean_rate_skyline),
                                          np.max(self.mean_rate_skyline))
                else:
                    mu = self.mean_rate_skyline

                n_preservation_bins = np.maximum(1, self._rs.poisson(self.mean_n_epochs_skyline * self.n_areas))
                time_id = np.sort(self._rs.integers(0, n_preservation_bins, self.n_areas * self.n_bins))
                time_id_area = time_id.reshape((self.n_areas, self.n_bins))
                loc_rates_b = self._rs.normal(loc=mu,
                                              scale=self.sd_through_time_skyline,
                                              size=(n_preservation_bins))
                loc_rates = loc_rates_b[time_id_area]
                if self.verbose:
                    print("skyline", np.exp(loc_rates)[0])
        return np.exp(loc_rates), slope, intercept, sd_through_time

    # consider that each area has its own preservation and sampling rate, generate preservation rate per area function
    def generate_preservation_rate_per_area_table(self):
        return self._rs.gamma(shape=self.area_shape, scale=self.area_scale, size=self.n_areas)

    # 2D area_time table - stochasticity through time in an area. Not all areas will follow the same trajectory.
    # draw a matrix of 0's and 1's from a binomial distribution
    def multiplier_vector(self, q, d=1.05):
        S = q.shape
        u = self._rs.random(S)
        l = 2 * np.log(d)
        m = np.exp(l * (u - .5))
        new_q = q * m
        U = np.sum(np.log(m))
        return new_q, U

    def a_variance(self):
        if isinstance(self.eta, Iterable):  # if eta is an iterable
            eta_r = self._rs.uniform(np.min(self.eta), np.max(self.eta))
        else:
            eta_r = self.eta
        if isinstance(self.p_gap, Iterable):  # if eta is an iterable
            p_gap = self._rs.uniform(np.min(self.p_gap), np.max(self.p_gap))
        else:
            p_gap = self.p_gap

        # # binomial(n=1) -> Bernoulli
        # a_var_0_1 = self._rs.binomial(n=1, p=1 - p_gap, size=(self.n_areas, self.n_bins))
        a_var_0_1 = np.ones((self.n_areas, self.n_bins))
        a_var_noise, _ = self.multiplier_vector(a_var_0_1, eta_r)
        a_var = a_var_0_1 * a_var_noise
        return a_var, eta_r, p_gap

    # Generate species specific preservation rate. Can adjust the default settings.
    def generate_sp_preservation_rate_table(self):  # arguments with a default should go to the end
        if isinstance(self.sp_mean, Iterable):
            if self.sp_bimodal:
                sp_mean = self._rs.choice(self.sp_mean, self.n_species, p=self.sp_bimodal_p)
            else:
                sp_mean = self._rs.uniform(np.min(self.sp_mean), np.max(self.sp_mean))
        else:
            sp_mean = self.sp_mean
        sp_scale = self.sp_variance / sp_mean
        sp_shape = sp_mean / sp_scale
        return self._rs.gamma(shape=sp_shape, scale=sp_scale, size=self.n_species), sp_mean

    # ---- NEW CODE with localities
    # draw a no of localities, and within each draw species occurrences. The rate at which you have a fossil site in an
    # area draw as a function of the area rate in time. Poisson distribution with a rate = area rate*area_size x time
    # rate x avar.
    def draw_localities(self, area_rate, area_size, time_rate, a_var, p_gap=None):
        locality_rate = np.einsum('a, at, at -> at', area_rate * area_size, time_rate, a_var)
        l = locality_rate * self.time_bins_duration
        l[l > self.max_localities] = self.max_localities  # truncate the number of localities that can be drawn per bin
        # locality_rate[locality_rate > self.max_localities] = self.max_localities

        if self.locality_rate_multiplier is not None:
            l = np.einsum('at, a -> at', l, self.locality_rate_multiplier)
        # print(self.time_bins_duration)

        rnd_l = self._rs.poisson(l)

        # apply p_gap
        if p_gap is not None:
            # binomial(n=1) -> Bernoulli
            current_p_gap = np.sum(rnd_l == 0) / rnd_l.size
            # print("\ncurrent_p_gap", current_p_gap, p_gap)
            if current_p_gap < p_gap:
                p_gap = p_gap - current_p_gap
                a_var_0_1 = self._rs.binomial(n=1, p=1 - p_gap, size=(self.n_areas, self.n_bins))
                rnd_l = rnd_l * a_var_0_1

            # print("\nnew_p_gap", np.sum(rnd_l == 0) / rnd_l.size)

        return rnd_l, l

    def add_singletons(self, p_3d_no_fossil):
        if isinstance(self.singletons_frequency, Iterable):
            f_sngl = self._rs.uniform(self.singletons_frequency[0], self.singletons_frequency[1])
        else:
            f_sngl = self.singletons_frequency

        p_3d_no_fossil = np.round(p_3d_no_fossil)
        foss_per_species = np.einsum('sat-> s', p_3d_no_fossil)
        p_sngl = 1 / (foss_per_species + SMALL_NUMBER)
        n_sngl = int(
            len(foss_per_species[foss_per_species > 0]) * f_sngl - len(foss_per_species[foss_per_species == 1]))
        p_sngl = p_sngl[foss_per_species > 1]
        p_sngl = p_sngl / np.sum(p_sngl)

        if n_sngl > 0:
            sngl = self._rs.choice(np.where(foss_per_species > 1)[0],
                                   n_sngl,
                                   p=p_sngl,
                                   replace=False)
            for i in sngl:
                sp_i = p_3d_no_fossil[i, :, :]
                sp_i = sp_i + self._rs.uniform(0, 1, sp_i.shape)
                # print(i, np.sum(sp_i), np.sum(p_3d_no_fossil[i+1, :, :]))
                sp_i_sngl = np.zeros(sp_i.shape)
                sp_i_sngl[sp_i == np.max(sp_i)] = 1
                p_3d_no_fossil[i, :, :] = sp_i_sngl
                # print(sp_i, p_3d_no_fossil[i, :, :])

            # x = np.einsum('sat-> s', p_3d_no_fossil)
            # print("SINGL", len(x[x == 1]) / len(x[x > 0]))
            # print("n_sngl",n_sngl,
            #       np.einsum('sat-> s', p_3d_no_fossil),
            #       )

        return p_3d_no_fossil

    # Draw fossils from localities
    #
    # [PERF 2026-11-19] This method was rewritten to build the
    # (n_species x n_areas x n_time_bins) arrays in chunks along the species
    # axis instead of materialising them in full.
    #
    # Before the change the method needed up to five full size
    # (n_species x n_areas x n_time_bins) arrays alive at the same time
    # (p_3d, p_3d_no_fossil, the mask, n_localities * mask and
    # fossils_per_area), i.e. about 5 x 1.07 GiB = 5.3 GiB for
    # n_species = 350k, n_areas = 10, n_bins = 40. The chunked version keeps
    # only small buffers alive and reduces the peak to well under one full
    # size array.
    #
    # The results are bitwise identical to the previous implementation. Two
    # details are required for that and must not be changed:
    #   1. The random draws happen in the same order. numpy's binomial()
    #      consumes random numbers in C order, element by element, so
    #      splitting an array along the first axis does not change the
    #      sequence of draws. However, the *original* code draws the whole
    #      mask first and only then the whole fossil count. Drawing mask and
    #      fossils interleaved per chunk would consume the random stream in a
    #      different order and hence produce different results, so the mask is
    #      drawn in a separate pass over the species axis.
    #   2. The reduction over the species axis must preserve the summation
    #      order of einsum (which sums sequentially over species). See
    #      _accumulate_reduce() for the details.
    def draw_fossils(self, species_specific_rate,
                     geo_table,
                     species_presence_absence_data,
                     number_of_localities):
        # sp specific probability is multiplied by its presence/absence in space/time
        n_species = geo_table.shape[0]
        n_areas = geo_table.shape[1]
        n_bins = species_presence_absence_data.shape[1]
        chunk = min(DEFAULT_CHUNK_SIZE, n_species)

        # carrying_capacity_multiplier[1] is a (s, a, t) multiplier applied to
        # the space/time fraction. It is applied inside the chunk loop so that
        # the full size array never has to exist.
        ccm = self.carrying_capacity_multiplier[1] if self.carrying_capacity_multiplier is not None else None

        def frac_chunks():
            """Yield the (chunk, a, t) space/time fraction, chunk by chunk."""
            for lo in range(0, n_species, chunk):
                hi = min(lo + chunk, n_species)
                f = np.einsum('sa, st -> sat', geo_table[lo:hi], species_presence_absence_data[lo:hi])
                if ccm is not None:
                    f *= ccm[lo:hi]
                yield f

        def p_3d_chunks():
            """Yield the (chunk, a, t) sampling probability, chunk by chunk."""
            lo = 0
            for f in frac_chunks():
                hi = lo + f.shape[0]
                p = np.einsum('s, sat -> sat', (1 - np.exp(-species_specific_rate[lo:hi])), f)
                if self.sampling_rate_multiplier is not None:
                    p = np.einsum('sat, at -> sat', p, self.sampling_rate_multiplier)
                lo = hi
                yield p

        # ---
        # product of probabilities (in log space to use einsum)
        # The log of (1 - p) is summed over the species axis; the summation is
        # done with an order preserving accumulation so that the result is
        # bitwise identical to the previous, non chunked implementation.
        def log_no_fossil_chunks():
            for p in p_3d_chunks():
                # np.nan_to_num() is a no-op here: p is a probability, so
                # (1 - p) is in [0, 1] and contains neither NaN nor Inf.
                pnf = 1 - p
                pnf[pnf == 0] += SMALL_NUMBER
                yield np.log(pnf)

        p_no_fossil_in_locality = np.exp(_accumulate_reduce(log_no_fossil_chunks(), (n_areas, n_bins)))
        expected_n_localities_with_fossils = self._rs.binomial(number_of_localities, 1 - p_no_fossil_in_locality)
        # print("expected_n_localities_with_fossils",
        #       np.einsum("at -> t", expected_n_localities_with_fossils))
        # expected_n_localities_with_fossils.shape = (at)
        # ---
        # print("number_of_localities", number_of_localities.shape, number_of_localities.dtype)
        # even with max sampling rate localities will only record ~10% of the species
        if self.species_per_locality_multiplier is None:
            mul = 1
        elif isinstance(self.species_per_locality_multiplier, Iterable):
            mul = self._rs.uniform(self.species_per_locality_multiplier[0],
                                   self.species_per_locality_multiplier[len(self.species_per_locality_multiplier) - 1])
        else:
            mul = self.species_per_locality_multiplier

        # Pass 1: draw the mask for every species. This must be a separate pass
        # over the species axis so that the random numbers are consumed in the
        # same order as in the original implementation.
        mask = np.empty((n_species, n_areas, n_bins), dtype=np.int8)
        for lo in range(0, n_species, chunk):
            hi = min(lo + chunk, n_species)
            mask[lo:hi] = self._rs.binomial(1, mul, (hi - lo, n_areas, n_bins))

        # Pass 2: draw the fossil counts, using the mask drawn above.
        fossils_per_area = np.empty((n_species, n_areas, n_bins), dtype=np.int64)
        lo = 0
        for p in p_3d_chunks():
            hi = lo + p.shape[0]
            fossils_per_area[lo:hi] = self._rs.binomial(number_of_localities[None, :, :] * mask[lo:hi], p)
            lo = hi
        # print("fossils_per_area", fossils_per_area.shape)
        # print(np.sum(fossils_per_area))
        if self.singletons_frequency is not None:
            fossils_per_area = self.add_singletons(fossils_per_area)
        # print(np.sum(fossils_per_area))

        if self.target_n_occs is not None:
            n_fossils = np.max([1, np.sum(fossils_per_area)])
            if n_fossils > self.target_n_occs * self.target_n_occs_range:
                m = self.target_n_occs * self.target_n_occs_range
            elif n_fossils < self.target_n_occs / self.target_n_occs_range:
                m = self.target_n_occs / self.target_n_occs_range
            else:
                m = n_fossils

            fossils_per_area = np.round(fossils_per_area / n_fossils * m).astype(int)
            # print("fossils_per_area", fossils_per_area.shape)
            max_n_loc = np.einsum('sat -> at', fossils_per_area)
            expected_n_localities_with_fossils[
                expected_n_localities_with_fossils > max_n_loc] = max_n_loc[
                expected_n_localities_with_fossils > max_n_loc] + 0

        # print("fossils_per_area",
        #           np.einsum('sat -> t', fossils_per_area),
        #       np.einsum('sat -> t', p_3d))

        # fossils_per_area is the number of localities per area in which a species is sampled
        return fossils_per_area, expected_n_localities_with_fossils

    # ----

    # species_presence_absence_in_time_bin estimates the fraction of each time bin a species is present for
    def species_presence_absence_in_time_bin(self):
        # This is a fraction of how much of the time bin the species exists in.
        # [PERF 2026-11-13] The original implementation was a nested
        #   Python loop over (n_bins x n_species):
        #   with n_species = 350k and n_bins = 20 that is 7 million iterations
        #   (each performing an np.min/np.max call), estimated at 30-60 s per
        #   simulation.
        #   It is replaced by a single vectorized interval-overlap computation
        #   per bin; the result is identical element by element (this method
        #   draws no random numbers).
        #   To keep the peak memory bounded (the temporary float64 arrays are
        #   n_species x n_bins), the species are processed in chunks, which
        #   keeps each temporary array at roughly 8 MB.
        species_presence_absence_data = np.zeros((self.n_species, self.n_bins))

        fa_all = self.sp_x[:, 0]   # FAD (origination time)
        la_all = self.sp_x[:, 1]   # LAD (extinction time)

        # The interval of bin b is [time_bins[b], time_bins[b+1]) (older end first)
        starts = self.time_bins[1:]    # (n_bins,)  older end = start of the bin
        ends   = self.time_bins[:-1]   # (n_bins,)  younger end = end of the bin
        delta_t = starts - ends
        delta_t = np.where(delta_t == 0, SMALL_NUMBER, delta_t)

        chunk = 50000
        for lo in range(0, self.n_species, chunk):
            hi = min(lo + chunk, self.n_species)
            fa = fa_all[lo:hi][:, None]   # (c, 1)
            la = la_all[lo:hi][:, None]   # (c, 1)

            # Presence condition: fa >= end and la <= start
            present = (fa >= ends[None, :]) & (la <= starts[None, :])

            # Overlap duration = min(start, fa) - max(end, la)
            dur = np.minimum(starts[None, :], fa) - np.maximum(ends[None, :], la)
            dur = np.where(dur == 0, dur + SMALL_NUMBER, dur)

            rel = dur / delta_t[None, :]
            species_presence_absence_data[lo:hi] = np.where(present, rel, 0.0)
        return species_presence_absence_data

    # match geographic ranges to time in a 3D array with no. sp, time bins and areas
    def einsum_species_space_time_fraction(self, geo_table, species_presence_absence_data):
        tmp = np.einsum('sa, st -> sat', geo_table, species_presence_absence_data)

        return tmp

    # einsum_species_space_time_table/n_species==einsum_species_space_time_table_fraction

    # [REMOVED 2026-11-19] The following two methods were left in place as dead code.
    #
    # def einsum_species_space_time(self, geo_table, species_presence_absence_data):
    #     """Einsum species_space_time. How many my a species lived in each area in each time bin."""
    #     tmp = np.einsum('sa, st, t -> sat', geo_table, species_presence_absence_data, self.time_bins_duration)
    #     return tmp
    #
    # def einsum_fossil_record(self, e_species_space_time_table, preservation_sampling_rate):
    #     """Multiply the 3D matrix einsum_species_space_time_table with the 3D matrix preservation_sampling_rates."""
    #     return np.einsum('sat, sat -> sat', e_species_space_time_table, preservation_sampling_rate)
    #
    # Why they were removed:
    #   - einsum_species_space_time() built the array "e_species_space_time_table"
    #     (species x areas x time bins = the time a species lived in an area within a
    #     time bin). It differs from einsum_species_space_time_fraction() only by the
    #     factor time_bins_duration, so it was a duplicate computation of the same
    #     quantity in a different unit (absolute duration vs. fraction of the bin).
    #   - einsum_fossil_record() was its only consumer, and it was never called
    #     anywhere in the codebase: the Poisson sampling path that used it was
    #     already commented out in run_simulation(). The actual fossil sampling is
    #     done by draw_fossils(), which uses e_species_space_time_table_fraction.
    #   - The array was never used by any downstream feature extraction or by the
    #     "keys" list saved in deepdiver_utilities.py, so removing it does not
    #     change any simulation output.
    #   - Memory benefit: the array is n_species x n_areas x n_bins float64 values
    #     (~1.07 GiB for n_species=350k, n_areas=10, n_bins=40) and it was held in
    #     memory simultaneously with the "_fraction" version, doubling the peak
    #     footprint of the S x A x T arrays for no benefit.
    #   - Note: the old e_species_space_time_table was also never multiplied by
    #     carrying_capacity_multiplier[1], so it did not even reflect the area
    #     constraints applied to the "_fraction" version actually used for sampling.
    #     The methods are kept commented out above for reference.

    # new 3D array, each cell with the sampling rate for a given species for an area and a time

    def add_info(self, sim, info):
        # use to add info about later manipulation of the object (e.g. extract_features_age_uncertainty)
        if sim['additional_info'] is None:
            sim['additional_info'] = info
        else:
            sim['additional_info'] = [sim['additional_info'], info]
        return sim

    def run_simulation(self,
                       sp_x,
                       min_age=None,
                       max_age=None,
                       return_sqs_data=False,
                       area_specific_rate=None,
                       time_specific_rate=None):

        self.bd_sim_setup(sp_x, min_age, max_age)

        if time_specific_rate is None:
            time_specific_rate, slope, intercept, sd_through_time = self.preservation_rate_through_time_table()
            # print("time_specific_rate", time_specific_rate.shape)
        else:
            slope, intercept, sd_through_time = np.nan, np.nan, np.nan
        # create 3D array of the preservation and sampling rates
        species_specific_rate, sp_mean_rate = self.generate_sp_preservation_rate_table()
        if area_specific_rate is None:
            area_specific_rate = self.generate_preservation_rate_per_area_table()

        area_size, alphas = self.generate_area_size_table()  # init area sizes based on `size_concentration_parameter`
        geo_table, slope_pr, pr_at_origination, kappa, disp_rate_mean = self.simulate_geo_ranges(area_size=area_size)
        # init geo ranges based on prob_origination area, which are themselves initialized based on `area_size` and
        # on `link_area_size_carrying_capacity`

        species_presence_absence_data = self.species_presence_absence_in_time_bin()

        # [PERF 2026-11-19] The (n_species x n_areas x n_time_bins) array
        # "e_species_space_time_table_fraction" is no longer built here. It was
        # the largest array in the whole simulation (1.07 GiB for n_species =
        # 350k, n_areas = 10, n_bins = 40) and was only used for two things:
        #   * the global / local "true" diversity trajectories, and
        #   * the sampling probabilities used by draw_fossils().
        # Both of them are now computed chunk by chunk from geo_table and
        # species_presence_absence_data, so the full size array never exists.
        # The results are bitwise identical to before, because the chunked
        # reductions preserve the summation order of the original einsum calls
        # (see _accumulate_reduce()).
        ccm = self.carrying_capacity_multiplier[1] if self.carrying_capacity_multiplier is not None else None

        def frac_chunks():
            """Yield the (chunk, a, t) space/time fraction, chunk by chunk."""
            chunk = min(DEFAULT_CHUNK_SIZE, self.n_species)
            for lo in range(0, self.n_species, chunk):
                hi = min(lo + chunk, self.n_species)
                f = np.einsum('sa, st -> sat', geo_table[lo:hi], species_presence_absence_data[lo:hi])
                if ccm is not None:
                    f *= ccm[lo:hi]
                yield f

        # [REMOVED 2026-11-19] e_species_space_time_table is no longer computed.
        # It was the S x A x T (species x areas x time bins) array giving the time each
        # species lived in each area within each time bin. It was redundant with
        # e_species_space_time_table_fraction (which differs only by the
        # time_bins_duration factor and is the one actually used for fossil sampling),
        # it had no downstream consumer anywhere in the codebase, and it cost about
        # 1.07 GiB of extra peak memory for n_species=350k, n_areas=10, n_bins=40.
        # No simulation output changes: the computation is deterministic and draws no
        # random numbers, so the random number stream and all results are unchanged.
        # e_species_space_time_table = self.einsum_species_space_time(geo_table,
        #                                                             species_presence_absence_data)

        a_var, r_eta, r_p_gap = self.a_variance()  # no. areas x no.bins


        # global_true_trajectory: for every time bin, the number of species
        # that are present in at least one area.
        # _accumulate_reduce() already performs the final reduction over the
        # species axis, in the same order as the original einsum('st -> t').
        def global_true_chunks():
            for f in frac_chunks():
                # temp = sum over areas, then clipped to 1 per species / time bin
                temp = np.einsum('sat -> st', f)
                x = temp + 0
                x[temp > 1] = 1
                yield x

        global_true_trajectory = _accumulate_reduce(global_true_chunks(), (self.n_bins,))

        # local_true_trajectory: for every area and time bin, the number of
        # species present in that area.
        def local_true_chunks():
            for f in frac_chunks():
                x = f + 0
                x[x > 1] = 1
                yield x.astype(float)

        local_true_trajectory = _accumulate_reduce(local_true_chunks(), (self.n_areas, self.n_bins))

        # ---- sampled localities
        # print("\narea_specific_rate", area_specific_rate.shape,
        #       "\narea_size", area_size.shape,
        #       "\ntime_specific_rate", time_specific_rate.shape,
        #       "\na_var", a_var.shape
        #       )
        number_of_localities, locality_rate = self.draw_localities(area_rate=area_specific_rate, area_size=area_size,
                                                                   time_rate=time_specific_rate, a_var=a_var,
                                                                   p_gap=r_p_gap)

        if self.carrying_capacity_multiplier is not None:
            # print(self.carrying_capacity_multiplier[0].shape , number_of_localities.shape)
            # area_time_carrying_capacity_multiplier = self.carrying_capacity_multiplier[1][0]
            number_of_localities = self.carrying_capacity_multiplier[1][0].astype(int) * number_of_localities


        # set to 0 the number of localities outside the clade age range
        clade_span = (global_true_trajectory == 0).nonzero()[0]
        number_of_localities[:, clade_span] = 0
        fossils_record, n_localities_w_fossils = self.draw_fossils(species_specific_rate=species_specific_rate,
                                                                   geo_table=geo_table,
                                                                   species_presence_absence_data=species_presence_absence_data,
                                                                   number_of_localities=number_of_localities)

        # print("np.size(a_var == 0) / np.size(a_var == 0)",
        #       np.sum(number_of_localities == 0) / np.size(number_of_localities), r_p_gap)

        if return_sqs_data == True:
            species_names = ["s%d" % i for i in range(self.n_species)]
            # print(species_names)
            collection_names = [i + 1 for i in range(self.n_areas)]
            # n_local = len(n_localities_w_fossils)
            # collection_names = [i+1 for i in range(n_local)]
            sqs_input = []
            for s in range(self.n_species):
                for a in range(self.n_areas):
                    for t in range(self.n_bins):
                        n = fossils_record[s, a, t]
                        for i in range(n):
                            sqs_input.append([species_names[s], collection_names[a], t + 1])
                            # print(sqs_input)
            sqs_input = pd.DataFrame(sqs_input, columns=["genus", "collection_no", "stg"])
            # sqs_data.to_csv('sqs_data.csv', index=False)
        else:
            sqs_input = None

        # ----

        # preservation_sampling_rate = self.einsum_rates(species_specific_rate, area_specific_rate, time_specific_rate,
        # var_a)
        # # the expected number of fossils per species, area and time bin.
        # expected_number_of_fossils = self.einsum_fossil_record(e_species_space_time_table, preservation_sampling_rate)
        # # use this as the mean of a poisson distribution to find actual values (i.e. not the mean)
        # fossils_from_poisson = self._rs.poisson(expected_number_of_fossils)

        global_fossil_trajectory = calculate_global_trajectory(fossils_record)
        local_fossil_trajectory = calculate_local_trajectory(fossils_record)

        occ_sp = np.einsum('sat -> s', fossils_record)
        n_occurrences = np.sum(occ_sp)
        n_sampled_species = np.sum(occ_sp > 0)
        tot_br_length = np.sum(sp_x[:, 0] - sp_x[:, 1])

        # print("\n\n\n")
        # print("n_occurrences", n_occurrences)
        # print("fossils_record", np.sum(fossils_record))
        # # print("target_n_occs_range", target_n_occs_range)
        # print("\n\n\n")

        d = {'sp_x': sp_x,
             'global_true_trajectory': global_true_trajectory,
             'local_true_trajectory': local_true_trajectory,
             'global_fossil_trajectory': global_fossil_trajectory,
             'local_fossil_trajectory': local_fossil_trajectory,
             'fossil_data': fossils_record,
             'geo_table': geo_table,
             'species_presence_absence_data': species_presence_absence_data,
             # [REMOVED 2026-11-19] 'e_species_space_time_table' was never read by any
             # downstream code (feature extraction, or the "keys" list saved by
             # deepdiver_utilities.py). The array is no longer computed at all; see the
             # note in run_simulation(). Keeping the key would raise a NameError.
             'localities': number_of_localities,
             'n_localities_w_fossils': n_localities_w_fossils,
             'time_specific_rate': time_specific_rate,
             'species_specific_rate': species_specific_rate,
             'area_specific_rate': area_specific_rate,
             'disp_rate_mean': disp_rate_mean,
             'disp_rate_var': self.disp_rate_variance,
             'a_var': a_var,
             'n_bins': self.n_bins,
             'area_size': area_size,
             'n_areas': self.n_areas,
             'n_species': self.n_species,
             'n_sampled_species': n_sampled_species,
             'tot_br_length': tot_br_length,
             'n_occurrences': n_occurrences,
             'slope_pr': slope_pr,
             'pr_at_origination': pr_at_origination,
             'time_bins_duration': self.time_bins_duration,
             'eta': r_eta,
             'p_gap': r_p_gap,
             'sp_mean_rate': sp_mean_rate,
             'area_size_concentration_prm': alphas,
             'link_area_size_carrying_capacity': kappa,
             'slope_log_sampling': slope,
             'intercept_initial_sampling': intercept,
             'sd_through_time': sd_through_time,
             'sqs_data': sqs_input,
             'locality_rate': locality_rate,
             'additional_info': self._additional_info,
             }

        return d
