import numpy as np
import neo


def theta_mod_idx(sptr, binsize, time_limit):
    '''Theta modulation index as defined in [3]_

    Parameters
    ----------
    sptr : array
    binsize : float
        Temporal binsize of autocorrelogram
    time_limit : flaot
        Limit of autocorrelogram

    References
    -----------
    .. [3] Cacucci, F., Lever, C., Wills, T. J., Burgess, N., & O'Keefe, J. (2004).
       Theta-modulated place-by-direction cells in the hippocampal formation in the rat.
       The Journal of Neuroscience, 24(38), 8265-8277.
    '''

    count, bins = correlogram(
        t1=sptr, t2=None, binsize=binsize, limit=time_limit,  auto=True)
    th = count[(bins[:-1] >= .05) & (bins[:-1] <= .07)].mean()
    pk = count[(bins[:-1] >= .1) & (bins[:-1] <= .14)].mean()
    return (pk - th)/(pk + th)


def fano_factor(trials, bins=1, return_mean_var=False, return_bins=False):
    """
    Calculate binned fano factor over several trials.

    Parameters
    ----------
    trials : list
        a list with np.arrays of spike times
    bins : np.ndarray or int
        bins of where to calculate fano factor. Default is 1
    return_mean_var : bool
        return mean count rate of trials and variance

    Returns
    -------
    out : float, or optional tuple
        fano factor, or optional (mean, var) if return_mean_var, or optional
        (fano factor, bins).

    Note
    ----
    This is a similar method as in [4]_, however there a sliding window was
    used.
    .. todo::
        Sliding window calculation of the Fano factor
        window = 50 * pq.ms
        step_size = 10 * pq.ms
        t_stop = 1 * pq.s
        bins = []; i = 0
        while i * step_size + window <= t_stop:
        bins.extend([i * step_size, i * step_size + window])
        i += 1

    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> result = fano_factor([t1, t2], bins=3)

    # array([ 0.,  0.,  0.])

    If you want to further work with the means and vars

    >>> result = fano_factor([t1, t2], bins=3, return_mean_var=True)

    # (array([ 2.,  1.,  2.]), array([ 0.,  0.,  0.]))

    The fano factor is 1 for Poisson processes

    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> np.random.seed(12345)
    >>> t1 = [homogeneous_poisson_process(
    ...     10 * pq.Hz, t_start=0.0 * pq.s, t_stop=1 * pq.s) for _ in range(100)]
    >>> result = fano_factor(t1)

    # array([ 0.95394394])

    The Fano factor computed in bins along time can be acheived with including
    `bins` which can be `int`

    >>> ff, bins = fano_factor(t1, bins=4, return_bins=True)

    # TODO fix
    # >>> ff
    # array([ 0.78505226,  1.16330097,  1.00901961,  0.80781457])
    # >>> bins
    # array([ 0.06424358,  0.29186518,  0.51948679,  0.74710839,  0.97472999])

    To specify bins


    # TODO fix
    # >>> bins = np.arange(0, 1, .2)
    # >>> fano_factor(t1, bins=bins)
    # array([ 0.95941748,  1.09      ,  1.05650485,  0.72886256])

    References
    ----------
    .. [4] Churchland, M. M., Byron, M. Y., Cunningham, J. P., Sugrue, L. P.,
       Cohen, M. R., Corrado, G. S., ... & Bradley, D. C. (2010). Stimulus onset
       quenches neural variability: a widespread cortical phenomenon. Nature
       neuroscience, 13(3), 369-378.
    """
    # TODO matching
    assert len(trials) > 0, 'trials cannot be empty'
    if isinstance(bins, int):
        nbins = bins
    else:
        nbins = len(bins) - 1
    hists = np.zeros((len(trials), nbins))
    for trial_num, trial in enumerate(trials):
        hist, _bins = np.histogram(trial, bins)
        hists[trial_num, :] = hist
    if len(trials) == 1:  # calculate fano over one trial
        axis = 1  # cols
    else:
        axis = 0  # rows
    mean = np.mean(hists, axis=axis)
    var = np.var(hists, axis=axis)
    if return_mean_var:
        if return_bins:
            return mean, var, bins
        else:
            return mean, var

    else:
        fano = var / mean
        if return_bins:
            return fano, _bins
        else:
            return fano


def fano_factor_multiunit(unit_trials, bins=1, return_rates=False,
                          return_bins=False):
    '''
    Calculate fano factor over several units with several trials as slopes from
    linear regression relating the variance to the mean of spike counts; see
    [4]_.

    Parameters
    ----------
    unit_trials : list of lists with trials
        That is unit_trials[0] = first unit, unit_trials[0][0] = first trial of
        first unit.
    bins : np.ndarray or int
        bins of where to calculate fano factor. Default is 1

    Returns
    -------
    (slopes, std_errors) : tuple
        Fano factor for each bin with corresponding standard error of the mean.
    .. todo::
        Weighted regression (binsize/1000) and distribution matching as in [4]_.

    See also
    --------
    :func:`exana.statistics.fano_factor` : The function that calcuates mean
        and var.
    :func:`scipy.statistics.linregress` : The function that calcuates slopes
        and standard error.

    Note
    ----
    You need many neurons to get a decent output value as you only have one
    datapoint per neuron. If you have some neurons with many trials consider
    doing a weighted regression.
    To get 95 % confidence interval you may use the standard error of teh mean
    by (fano - 2 * std_err, fano + 2 * std_err)

    Examples
    --------
    The fano factor is 1 for Poisson processes, thus we genereate 100 Poisson
    spiking neurons with each 10 trials.
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> np.random.seed(12345)
    >>> units_trials = [
    ...     [homogeneous_poisson_process(
    ...          10 * pq.Hz, t_start=0.0 * pq.s, t_stop=1 * pq.s)
    ...      for _ in range(10)] for _ in range(100)]
    >>> fano, std_err = fano_factor_multiunit(units_trials)
    >>> print('{:.2}, {:.2}'.format(fano[0], std_err[0]))
    0.92, 0.041
    '''
    from scipy.stats import linregress
    if isinstance(bins, int):
        nbins = bins
    else:
        nbins = len(bins) - 1
    nunits = len(unit_trials)
    means = np.zeros((nunits, nbins))
    varis = np.zeros((nunits, nbins))
    for unit_num, trials in enumerate(unit_trials):
        if len(trials) == 0:
            continue
        mean, var, bins = fano_factor(trials, bins, return_mean_var=True,
                                      return_bins=True)
        means[unit_num, :] = mean
        varis[unit_num, :] = var
    fanos = []
    std_errs = []
    for nb in range(nbins):
        slope, intercept, r_value, p_value, std_err = linregress(means[:, nb],
                                                                 varis[:, nb])
        std_errs.append(std_err / np.sqrt(nunits))
        fanos.append(slope)
    if return_rates:
        rates = np.mean(means, axis=0) / (bins[1] - bins[0])
        if return_bins:
            return fanos, std_errs, rates, bins
        else:
            return fanos, std_errs, rates
    else:
        if return_bins:
            return fanos, std_errs, bins
        else:
            return fanos, std_errs


def coeff_var(trials):
    """
    Calculate the coefficient of variation in inter spike interval (ISI)
    distribution over several trials

    Parameters
    ----------
    trials : list of neo.SpikeTrain or array like

    Returns
    -------
    out : list
        Coefficient of variations for each trial, nan if len(trial) == 0

    Examples
    --------
    >>> np.random.seed(12345)
    >>> trials = [np.arange(10), np.random.random((10))]
    >>> print('{d[0]:.2f}, {d[1]:.2f}'.format(d=coeff_var(trials)))
    0.00, -9.53
    """
    cvs = []
    for trial in trials:
        isi = np.diff(trial)
        if len(isi) > 0:
            cvs.append(np.std(isi) / np.mean(isi))
        else:
            cvs.append(np.nan)
    return cvs


def bootstrap(data, num_samples=10000, statistic=None, alpha=0.05):
    """
    Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic.
    Adapted from http://people.duke.edu/~ccc14/pcfb/analysis.html

    Parameters
    ----------
    data : array like
        1D array like representation of your data.
    num_samples : int
        The number of repetitions of random samples of your data.
    statistic : function(2darray, axis)
        The statistic you want to build the ci. Default is mean
    Returns
    -------
    confidence interval : tuple
        The confidence interval for given statistic.

    Examples
    --------
    Unlike using normal assumptions to calculate 95% CI, the
    results generated by the bootstrap are robust even if the underlying data
    are very far from normal.
    Bimodal data of interest

    >>> import numpy.random as npr
    >>> import numpy as np
    >>> npr.seed(12345)
    >>> x = np.concatenate([npr.normal(3, 1, 100), npr.normal(6, 2, 200)])

    To find the find mean 95% CI by 100 000 bootstrap samples

    >>> low, high = bootstrap(data=x, num_samples=100000, statistic=np.mean,
    ...                       alpha=0.05)
    >>> print('{:.2f}, {:.2f}'.format(low, high))
    4.64, 5.12

    Historgram of the data with corresponding scatter with mean and it's CI

    .. plot::

        import matplotlib.pyplot as plt
        import numpy.random as npr
        import numpy as np
        npr.seed(12345)
        from exana.statistics import bootstrap
        x = np.concatenate([npr.normal(3, 1, 100), npr.normal(6, 2, 200)])
        ci = bootstrap(data=x, num_samples=100000, statistic=np.mean, alpha=0.05)
        plt.figure(figsize=(8,4))
        plt.subplot(121)
        plt.hist(x, 50, histtype='step')
        plt.title('Historgram of skewed data')
        plt.subplot(122)
        plt.plot([-0.03,0.03], [np.mean(x), np.mean(x)], 'k', linewidth=2, label='mean')
        plt.scatter(0.1*(npr.random(len(x)) - 0.5), x)
        plt.plot([0.19,0.21], [ci[0], ci[0]], 'r', linewidth=2, label='95% CI')
        plt.plot([0.19,0.21], [ci[1], ci[1]], 'r', linewidth=2)
        plt.plot([0.2,0.2], [ci[0], ci[1]], 'r', linewidth=2)
        plt.xlim([-0.2, 0.3])
        plt.title('Bootstrap 95% CI for mean')
        plt.legend()
        plt.show()

    The bootstrap function is a higher order function, and will return the
    boostrap CI for any valid statistical function, not just the mean.
    For example, to find the 95% CI for the standard deviation, given
    :func:`np.std` as the statistic:

    >>> low, high = bootstrap(data=x, num_samples=100000, statistic=np.std,
    ...                       alpha=0.05)
    >>> print('{:.2f}, {:.2f}'.format(low, high))
    1.97, 2.26
    """
    data = np.asarray(data)
    if np.ndim(data) != 1:
        raise ValueError('Data must be 1 dimensional.')
    statistic = statistic or np.mean
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, axis=1))
    return (stat[int((alpha / 2.0) * num_samples)],
            stat[int((1 - alpha / 2.0) * num_samples)])


def permutation_resampling(case, control, num_samples=10000, statistic=None):
    """
    Simulation-based statistical calculation of p-value that statistic for case
    is different from statistic for control under the null hypothesis that the
    groups are invariant under label permutation. That is, case and control is
    combined and shuffeled randomly `num_samples` times and given statistic is
    calculated after each shuffle. Given the observed differece as the absulete
    differece between the statistic of the case and control. Then the p-value is
    calculated as the number of occurences where the shuffled statistic is
    greater than the observed differece pluss the number of occurences where
    the shuffled statistic is less than the negative observed differece, divided
    by the number of shuffles.
    For example, in a case-control study, it can be used to find the p-value
    under the hypothesis that the mean of the case group is different from that
    of the control group, and we cannot use the t-test because the distributions
    are highly skewed.
    Adapted from http://people.duke.edu/~ccc14/pcfb/analysis.html

    Parameters
    ----------
    case : 1D array like
        Samples from the case study.
    control : 1D array like
        Samples from the control study.
    num_samples : int
        Number of permutations
    statistic : function(2darray, axis)
        The statistic function to compare case and control. Default is mean

    Returns
    -------
    pval : float
        The calculated p-value.
    observed_diff : float
        Absolute difference between statistic of `case` and statistic of
        `control`.
    diffs : list
        A list of length equal to `num_samples` with differences between
        statistic of permutated case and statistic of permutated control.

    Examples
    --------
    Make up some data
    >>> np.random.seed(12345)
    >>> case = [94, 38, 23, 197, 99, 16, 141]
    >>> control = [52, 10, 40, 104, 51, 27, 146, 30, 46]

    Find the p-value by permutation resampling

    >>> pval, observed_diff, diffs = permutation_resampling(
    ...     case, control, 10000, np.mean)

    .. plot::
        import matplotlib.pylab as plt
        import numpy as np
        from exana.statistics import permutation_resampling
        case = [94, 38, 23, 197, 99, 16, 141]
        control = [52, 10, 40, 104, 51, 27, 146, 30, 46]
        pval, observed_diff, diffs = permutation_resampling(
            case, control, 10000, np.mean)
        plt.title('Empirical null distribution for differences in mean')
        plt.hist(diffs, bins=100, histtype='step', normed=True)
        plt.axvline(observed_diff, c='red', label='diff')
        plt.axvline(-observed_diff, c='green', label='-diff')
        plt.text(60, 0.01, 'p = %.3f' % pval, fontsize=16)
        plt.legend()
        plt.show()
    """
    statistic = statistic or np.mean

    observed_diff = abs(statistic(case) - statistic(control))
    num_case = len(case)

    combined = np.concatenate([case, control])
    diffs = []
    for i in range(num_samples):
        xs = np.random.permutation(combined)
        diff = np.mean(xs[:num_case]) - np.mean(xs[num_case:])
        diffs.append(diff)

    pval = (np.sum(diffs > observed_diff) +
            np.sum(diffs < -observed_diff))/float(num_samples)
    return pval, observed_diff, diffs



def stat_test(tdict, test_func=None, nan_rule='remove', stat_key='statistic'):
    '''
    A very simple function to performes statistic tests between multiple groups
    in `tdict` by given test function.

    Parameters
    ----------
    tdict : dict
        Dictionary where each key represents a 1D dataset of observations
    test_func : statistic, pvalue = function(case, control)
        Function that takes in two 1D arrays and returns desired statistic with
        corresponding p-value.
    nan_rule : str {'remove', None}
        What to do with nans
    stat_key : str
        A textual representation of the returned statistic

    Returns
    -------
    out : pandas.DataFrame
        A dataframe describing statistics and pvalue.

    Example
    -------
    >>> tdict = {'group1': [94, 38, 23, 197, 99, 16, 141],
    ...          'group2': [52, 10, 40, 104, 51, 27, 146, 30, 46],
    ...          'group3': [3, 10, 40, 0, 51, 27, 1, 30, 46]}
    >>> def stat_func(a, b):
    ...     pval, diff, _ = permutation_resampling(a, b, 10000, np.mean)
    ...     return diff, pval
    >>> out = stat_test(tdict, test_func=stat_func, stat_key='abs diff mean')
    '''
    import pandas as pd
    if test_func is None:
        from scipy import stats
        test_func = lambda g1, g2: stats.ttest_ind(g1, g2, equal_var=False)
    ps = {}
    sts ={}
    lib = []
    for key1, item1 in tdict.items():
        for key2, item2 in tdict.items():
            if key1 != key2:
                if set([key1, key2]) in lib:
                    continue
                lib.append(set([key1, key2]))
                one = np.array(item1, dtype=np.float64)
                two = np.array(item2, dtype=np.float64)
                if nan_rule == 'remove':
                    one = one[np.isfinite(one)]
                    two = two[np.isfinite(two)]
                elif nan_rule is None:
                    pass
                else:
                    raise NotImplementedError
                assert len(one) > 0, 'Empty list of values'
                assert len(two) > 0, 'Empty list of values'
                stat, p = test_func(one, two)
                ps[key1+'--'+key2] = p
                sts[key1+'--'+key2] = stat
    return pd.DataFrame([ps, sts], index=['p-value', stat_key])


def poisson_continuity_correction(n, rate):
    """
    n : Likelihood to observe n or more events
    rate : float
        Rate of Poisson process

    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    """
    from scipy.stats import poisson
    assert np.all(n >= 0)
    return_arr = np.zeros(n.shape)
    for i, n_i in enumerate(n):
        if n_i == 0:
            return_arr[i] = 1.
        else:
            rates = [poisson.pmf(j, rate) for j in range(int(n_i))]
            return_arr[i] = 1 - np.sum(rates) - 0.5 * poisson.pmf(n_i, rate)
    return return_arr


def hollow_kernel(kernlen, width, hollow_fraction=0.6, kerntype='gaussian'):
    '''
    Returns a hollow kernel normalized to it's sum

    Parameters
    ----------
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fractoin of the central bin to removed.

    Returns
    -------
    kernel : array
    '''
    if kerntype == 'gaussian':
        from scipy.signal import gaussian
        assert kernlen % 2 == 1
        kernel = gaussian(kernlen, width)
        kernel[int(kernlen / 2.)] *= (1 - hollow_fraction)
    else:
        raise NotImplementedError
    return kernel / sum(kernel)


def ccg_significance(t1, t2, binsize, limit, hollow_fraction, width,
                     kerntype='gaussian'):
    """
    Parameters
    ---------
    t1 : np.array, or neo.SpikeTrain
        First spiketrain, raw spike times in seconds.
    t2 : np.array, or neo.SpikeTrain
        Second spiketrain, raw spike times in seconds.
    binsize : float, or quantities.Quantity
        Width of each bar in histogram in seconds.
    limit : float, or quantities.Quantity
        Positive and negative extent of histogram, in seconds.
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fractoin of the central bin to removed.

    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    English et al. 2017, Neuron, Pyramidal Cell-Interneuron Circuit Architecture
    and Dynamics in Hippocampal Networks
    """
    import scipy.signal as scs
    ccg, bins = correlogram(t1, t2, binsize=binsize, limit=limit,
                            density=False)
    kernlen = len(ccg) - 1
    kernel = hollow_kernel(kernlen, width, hollow_fraction, kerntype)
    # padd edges
    len_padd = int(kernlen / 2.)
    ccg_padded = np.zeros(len(ccg) + 2 * len_padd)
    # "firstW/2 bins (excluding the very first bin) are duplicated,
    # reversed in time, and prepended to the ccg prior to convolving"
    ccg_padded[0:len_padd] = ccg[1:len_padd+1][::-1]
    ccg_padded[len_padd: - len_padd] = ccg
    # # "Likewise, the lastW/2 bins aresymmetrically appended to the ccg."
    ccg_padded[-len_padd:] = ccg[-len_padd-1:-1][::-1]
    # convolve ccg with kernel
    ccg_smoothed = scs.fftconvolve(ccg_padded, kernel, mode='valid')
    pfast = np.zeros(ccg.shape)
    assert len(ccg) == len(ccg_smoothed)
    for m, (val_m, rate_m) in enumerate(zip(ccg, ccg_smoothed)):
        pfast[m] = poisson_continuity_correction(np.array([val_m]), rate_m)
    # pcausal describes the probability of obtaining a peak on one side
    # of the histogram, that is signficantly larger than the largest peak
    # in the anticausal direction. we leave the zero peak empty
    pcausal = np.zeros(ccg.shape)
    ccg_half_len = int(np.floor(len(ccg) / 2.))
    max_pre = np.max(ccg[:ccg_half_len])
    max_post = np.max(ccg[ccg_half_len:])
    for m, val_m in enumerate(ccg):
        if m < ccg_half_len:
            pcausal[m] = poisson_continuity_correction(
                np.array([val_m]), max_post)
        if m > ccg_half_len:
            pcausal[m] = poisson_continuity_correction(
                np.array([val_m]), max_pre)
    return pcausal, pfast, bins, ccg, ccg_smoothed


def correlogram(t1, t2=None, binsize=.001, limit=.02, auto=False,
                density=False):
    """Return crosscorrelogram of two spike trains.
    Essentially, this algorithm subtracts each spike time in `t1`
    from all of `t2` and bins the results with np.histogram, though
    several tweaks were made for efficiency.
    Originally authored by Chris Rodger, copied from OpenElectrophy, licenced
    with CeCill-B. Examples and testing written by exana team.

    Parameters
    ---------
    t1 : np.array, or neo.SpikeTrain
        First spiketrain, raw spike times in seconds.
    t2 : np.array, or neo.SpikeTrain
        Second spiketrain, raw spike times in seconds.
    binsize : float, or quantities.Quantity
        Width of each bar in histogram in seconds.
    limit : float, or quantities.Quantity
        Positive and negative extent of histogram, in seconds.
    auto : bool
        If True, then returns autocorrelogram of `t1` and in
        this case `t2` can be None. Default is False.
    density : bool
        If True, then returns the probability density function.
    See also
    --------
    :func:`numpy.histogram` : The histogram function in use.

    Returns
    -------
    (count, bins) : tuple
        A tuple containing the bin right edges and the
        count/density of spikes in each bin.
    Note
    ----
    `bins` are relative to `t1`. That is, if `t1` leads `t2`, then
    `count` will peak in a positive time bin.

    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> limit = 1
    >>> binsize = .1
    >>> counts, bins = correlogram(t1=t1, t2=t2, binsize=binsize,
    ...                            limit=limit, auto=False)
    >>> counts
    array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0])

    The interpretation of this result is that there are 5 occurences where
    in the bin 0 to 0.1, i.e.

    # TODO fix
    # >>> idx = np.argmax(counts)
    # >>> '%.1f, %.1f' % (abs(bins[idx - 1]), bins[idx])
    # '0.0, 0.1'

    The correlogram algorithm is identical to, but computationally faster than
    the histogram of differences of each timepoint, i.e.

    # TODO Fix the doctest
    # >>> diff = [t2 - t for t in t1]
    # >>> counts2, bins = np.histogram(diff, bins=bins)
    # >>> np.array_equal(counts2, counts)
    # True
    """
    if auto: t2 = t1
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if not int(limit * 1e10) % int(binsize * 1e10) == 0:
        raise ValueError(
            'Time limit {} must be a '.format(limit) +
            'multiple of binsize {}'.format(binsize) +
            ' remainder = {}'.format(limit % binsize))
    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)

    # The numpy.arange method overshoots slightly the edges i.e. binsize + epsilon
    # which leads to inclusion of spikes falling on edges.
    bins = np.arange(-limit, limit + binsize, binsize)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to np.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins, density=density)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] = 0#-= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins[1:]
