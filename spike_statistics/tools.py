import numpy as np


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


def make_spiketrain_trials(spike_train, events, t_start=None, t_stop=None):
    '''
    Makes trials based on an Epoch and given temporal bound

    Parameters
    ----------
    spike_train : array
        seconds
    events : array
        times for trials
    t_start : float, or array
        time before events, default is 0
    t_stop : float, or array
        time after events, default is 0.1

    Returns
    -------
    out : list of lists with spike times
    '''

    if t_start is None:
        t_start = 0
    if t_start.ndim == 0:
        t_starts = t_start * np.ones(len(events))
    else:
        t_starts = t_start
        assert len(epoch.times) == len(t_starts), 'events and t_start have different size'
    if t_stop is None:
        t_stop = .1
    if t_stop.ndim == 0:
        t_stops = t_stop * np.ones(len(events))
    else:
        t_stops = t_stop
        assert len(events) == len(t_stops), 'events and t_stop have different size'

    trials = []
    for j, t in enumerate(events):
        t_start = t_starts[j]
        t_stop = t_stops[j]
        spikes = []
        for spike in sptr[(t + t_start < sptr) & (sptr < t + t_stop)]:
            spikes.append(spike-t)
        trials.append(spikes)
    return trials
