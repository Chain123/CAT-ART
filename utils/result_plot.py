import os
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def linePlot(x, y, xlabel, ylabel, fout,
             figsize, xlabel_fontsize, ylabel_fontsize,
             xlabel_sciformat, ylabel_sciformat,
             xticks_fontsize, yticks_fontsize,
             marker, markersize, linestyle, colors,
             is_x_date, major_locator, minor_locator,
             use_message_box, figleft_boxleft_gap, figbottom_boxtop_gap, box_font, legends, line_num = 1):
    # plot
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(111)
    ax = plt.axes()
    for i in range(line_num):
        plt.plot(x[i], y[i], marker=marker[i], markersize=markersize, linestyle=linestyle[i], color=colors[i])
    # figure settings
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    if xlabel_sciformat:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 1))
    if ylabel_sciformat:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
    # ax.set_yscale("log", nonposy='clip')
    if is_x_date:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        fig.autofmt_xdate()
    # info box
    print(legends)
    plt.legend(legends, loc="best", fancybox=True)
    if use_message_box:
        y = np.array(y)
        mu = y.mean()
        sigma = y.std()
        textstr = '$\mathrm{mean}=%.4f$\n$\mathrm{std}=%.4f$' % (mu, sigma)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(figleft_boxleft_gap, figbottom_boxtop_gap, textstr,
                transform=ax.transAxes, fontsize=box_font,
                verticalalignment='top', bbox=props)

    plt.savefig(fout, format='eps',
                dpi=1000, bbox_inches='tight')
    plt.show()
