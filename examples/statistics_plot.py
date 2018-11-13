import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
import warnings
from utils import simpleaxis


def plot_spike_histogram(trials, color='b', ax=None, binsize=None, bins=None,
                         output='counts', edgecolor=None, alpha=1., ylabel=None,
                         nbins=None):
    """
    histogram plot of trials

    Parameters
    ----------
    trials : list of neo.SpikeTrains
    color : str
        Color of histogram.
    edgecolor : str
        Color of histogram edges.
    ax : matplotlib axes
    output : str
        Accepts 'counts', 'rate' or 'mean'.
    binsize :
        Binsize of spike rate histogram, default None, if not None then
        bins are overridden.
    nbins : int
        Number of bins, defaults to 100 if binsize is None.
    ylabel : str
        The ylabel of the plot, if None defaults to output type.

    Examples
    --------
    >>> import neo
    >>> from numpy.random import rand
    >>> from exana.stimulus import make_spiketrain_trials
    >>> spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
    >>> epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s,
    ...                   durations=[.5] * 10 * pq.s)
    >>> trials = make_spiketrain_trials(spike_train, epoch)
    >>> ax = plot_spike_histogram(trials, color='r', edgecolor='b',
    ...                           binsize=1 * pq.ms, output='rate', alpha=.5)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        import neo
        from numpy.random import rand
        from exana.stimulus import make_spiketrain_trials
        from statistics_plot import plot_spike_histogram
        spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
        epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s, durations=[.5] * 10 * pq.s)
        trials = make_spiketrain_trials(spike_train, epoch)
        ax = plot_spike_histogram(trials, color='r', edgecolor='b', binsize=1 * pq.ms, output='rate', alpha=.5)
        plt.show()

    Returns
    -------
    out : axes
    """
    ### TODO
    if bins is not None:
        assert isinstance(bins, int)
        warnings.warn('The variable "bins" is deprecated, use nbins in stead.')
        nbins = bins
    ###
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    from elephant.statistics import time_histogram
    dim = trials[0].times.dimensionality
    t_start = trials[0].t_start.rescale(dim)
    t_stop = trials[0].t_stop.rescale(dim)
    if binsize is None:
        if nbins is None:
            nbins = 100
        binsize = (abs(t_start)+abs(t_stop))/float(nbins)
    else:
        binsize = binsize.rescale(dim)
    time_hist = time_histogram(trials, binsize, t_start=t_start,
                               t_stop=t_stop, output=output, binary=False)
    bs = np.arange(t_start.magnitude, t_stop.magnitude, binsize.magnitude)
    if ylabel is None:
        if output == 'counts':
            ax.set_ylabel('count')
        elif output == 'rate':
            time_hist = time_hist.rescale('Hz')
            if ylabel:
                ax.set_ylabel('rate [%s]' % time_hist.dimensionality)
        elif output == 'mean':
            ax.set_ylabel('mean count')
    elif isinstance(ylabel, str):
        ax.set_ylabel(ylabel)
    else:
        raise TypeError('ylabel must be str not "' + str(type(ylabel)) + '"')
    ax.bar(bs[:len(time_hist)], time_hist.magnitude.flatten(), width=bs[1]-bs[0],
           edgecolor=edgecolor, facecolor=color, alpha=alpha, align='edge')
    return ax


def plot_isi_hist(sptr, alpha=1, ax=None, binsize=2*pq.ms,
                  time_limit=100*pq.ms, color='b', edgecolor=None):
    """
    Bar plot of interspike interval (ISI) histogram

    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : str
        color of histogram
    edgecolor : str
        edgecolor of histogram
    ax : matplotlib axes
    alpha : float
        opacity
    binsize : Quantity(s)
        binsize of spike rate histogram, default 2 ms
    time_limit : Quantity(s)
        end time of histogram x limit, default 100 ms

    Examples
    --------
    >>> import neo
    >>> from numpy.random import rand
    >>> spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
    >>> ax = plot_isi_hist(spike_train, alpha=.1, binsize=10*pq.ms,
    ...                    time_limit=100*pq.ms, color='r', edgecolor='r')

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        import neo
        from numpy.random import rand
        from statistics_plot import plot_isi_hist
        spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
        plot_isi_hist(spike_train, alpha=.5, binsize=10*pq.ms, time_limit=100*pq.ms, color='r')
        plt.show()

    Returns
    -------
    out : axes
    """
    edgecolor = edgecolor or color
    if ax is None:
        fig, ax = plt.subplots()
    dim = sptr.times.dimensionality
    spk_isi = np.diff(sorted(sptr.times))
    binsize = binsize.rescale(dim).magnitude
    time_limit = time_limit.rescale(dim).magnitude
    ax.hist(spk_isi, bins=np.arange(0., time_limit, binsize),
            normed=True, alpha=alpha, color=color, edgecolor=edgecolor)
    ax.set_xlabel('Interspike interval $\Delta t$ [{}]'.format(dim))
    ax.set_ylabel('Proportion of intervals in {} [{}]'.format(binsize, dim))
    ax.set_xlim(0, time_limit)
    return ax


def plot_xcorr(spike_trains, colors=None, edgecolors=None, fig=None,
               density=True, alpha=1., gs=None, binsize=1*pq.ms,
               time_limit=1*pq.s, split_colors=True, xcolor='k',
               xedgecolor='k', xticksvisible=True, yticksvisible=True,
               acorr=True, ylim=None, names=None):
    """
    Bar plot of crosscorrelation of multiple spiketrians

    Parameters
    ----------
    spike_trains : list of neo.SpikeTrain or neo.SpikeTrain
    colors : list or str
        colors of histogram
    edgecolors : list or str
        edgecolor of histogram
    ax : matplotlib axes
    alpha : float
        opacity
    binsize : Quantity
        binsize of spike rate histogram, default 2 ms
    time_limit : Quantity
        end time of histogram x limit, default 100 ms
    gs : instance of matplotlib.gridspec
    split_colors : bool
        if True splits crosscorrelations into colors from respective
        autocorrelations
    xcolor : str
        color of crosscorrelations
    xedgecolor : str
        edgecolor of crosscorrelations
    xticksvisible : bool
        show xtics on crosscorrelations, (True by default)
    yticksvisible : bool
        show ytics on crosscorrelations, (True by default)
    acorr : bool
        show autocorrelations, (True by default)

    Examples
    --------
    >>> import neo
    >>> from numpy.random import rand
    >>> sptr1 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
    >>> sptr2 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
    >>> sptr3 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
    >>> fig = plot_xcorr([sptr1, sptr2, sptr3])

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        import neo
        from numpy.random import rand
        from statistics_plot import plot_xcorr
        sptr1 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
        sptr2 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
        sptr3 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
        plot_xcorr([sptr1, sptr2, sptr3])
        plt.show()
    Returns
    -------
    out : fig
    """
    if len(spike_trains) == 1:
        spike_trains = [spike_trains]
    from exana.statistics.tools import correlogram
    import matplotlib.gridspec as gridspec
    if colors is None:
        from matplotlib.pyplot import cm
        colors = cm.rainbow(np.linspace(0, 1, len(spike_trains)))
    elif not isinstance(colors, list):
        colors = [colors] * len(spike_trains)
    if edgecolors is None:
        edgecolors = colors
    elif not isinstance(edgecolors, list):
        edgecolors = [edgecolors] * len(spike_trains)

    def get_name(sptr, idx):
        if hasattr(sptr, 'name'):
            name = sptr.name
        elif names is not None:
            assert len(names) == len(spike_trains)
            name = names[idx]
        if name is None:
            name = 'idx {}'.format(idx)
        return name
    if fig is None:
        fig = plt.figure()

    nrc = len(spike_trains)
    if gs is None:
        gs0 = gridspec.GridSpec(nrc, nrc)
    else:
        gs0 = gridspec.GridSpecFromSubplotSpec(nrc, nrc, subplot_spec=gs)
    axs, cnt = [], 0
    for x in range(nrc):
        for y in range(nrc):
            if (y > x) or (y == x):
                if not acorr and y == x:
                    continue
                prev_ax = None if len(axs) == 0 else axs[cnt-1]
                ax = fig.add_subplot(gs0[x, y], sharex=prev_ax, sharey=prev_ax)
                axs.append(ax)
            if y > x:
                plt.setp(ax.get_xticklabels(), visible=xticksvisible)
                plt.setp(ax.get_yticklabels(), visible=yticksvisible)
    cnt = 0
    ccgs = []
    for x in range(nrc):
        for y in range(nrc):
            if y > x:
                sptr1 = spike_trains[x]
                sptr2 = spike_trains[y]

                count, bins = correlogram(
                    t1=sptr1,
                    t2=sptr2,
                    binsize=binsize, limit=time_limit,  auto=False,
                    density=density)
                ccgs.append(count)
                if split_colors:
                    c1, c2 = colors[x], colors[y]
                    e1, e2 = edgecolors[x], edgecolors[y]
                    c1_n = sum(bins <= 0)
                    c2_n = len(bins) - c1_n
                    cs = [c1] * c1_n + [c2] * c2_n
                    es = [e1] * c1_n + [e2] * c2_n
                else:
                    cs, es = xcolor, xedgecolor
                axs[cnt].bar(bins, count, align='edge',
                             width=-binsize, color=cs,
                             edgecolor=es)
                axs[cnt].set_xlim([-time_limit, time_limit])
                name1 = get_name(sptr1, x)
                name2 = get_name(sptr2, y)
                axs[cnt].set_xlabel(name1 + ' ' + name2)
                cnt += 1
            elif y == x and acorr:
                sptr = spike_trains[x]
                count, bins = correlogram(
                    t1=sptr, t2=None,
                    binsize=binsize, limit=time_limit,
                    auto=True, density=density)
                ccgs.append(count)
                axs[cnt].bar(bins, count, width=-binsize, align='edge',
                             color=colors[x], edgecolor=edgecolors[x])
                axs[cnt].set_xlim([-time_limit, time_limit])
                name = get_name(sptr, x)
                axs[cnt].set_xlabel(name + ' ' + name)
                cnt += 1
    if ylim is not None: axs[0].set_ylim(ylim)
    plt.tight_layout()
    return ccgs, bins


# def plot_autocorr(sptr, title='', color='k', edgecolor='k', ax=None,
#                   density=True, auto=True, **kwargs):
#     """
#     Bar plot of autocorrelation
#
#     Parameters
#     ----------
#     sptr : neo.SpikeTrain
#     color : str
#         color of histogram
#     edgecolor : str
#         edgecolor of histogram
#     ax : matplotlib axes
#     alpha : float
#         opacity
#     binsize : Quantity(s)
#         binsize of spike rate histogram, default 2 ms
#     time_limit : Quantity(s)
#         end time of histogram x limit, default 100 ms
#
#     Examples
#     --------
#     >>> import neo
#     >>> from numpy.random import rand
#     >>> sptr1 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
#     >>> ax = plot_autocorr(sptr1)
#
#     .. plot::
#
#         import matplotlib.pyplot as plt
#         import numpy as np
#         import quantities as pq
#         import neo
#         from numpy.random import rand
#         from statistics_plot import plot_autocorr
#         sptr1 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
#         plot_autocorr(sptr1)
#         plt.show()
#
#     Returns
#     -------
#     out : ax
#     """
#     par = {'corr_bin_width': 0.01*pq.s,
#            'corr_limit': 1.*pq.s}
#     if kwargs:
#         par.update(kwargs)
#     from exana.statistics.tools import correlogram
#     if ax is None:
#         fig, ax = plt.subplots()
#     bin_width = par['corr_bin_width'].rescale('s').magnitude
#     limit = par['corr_limit'].rescale('s').magnitude
#     count, bins = correlogram(t1=sptr.times.magnitude, t2=None,
#                               binsize=bin_width, limit=limit,  auto=True)
#     ax.bar(bins[1:], count, width=bin_width, color=color,
#             edgecolor=edgecolor)
#     ax.set_xlim([-limit, limit])
#     ax.set_title(title)
