import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shared import utils
import pandas as pd
from config.Config import Config


def create_subplot(n_rows, n_columns, fig_size=8, font_size=24):
    plt.rcParams["axes.xmargin"] = 0  # remove white space on axes
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams["figure.figsize"] = (
        n_columns * fig_size,
        n_rows * fig_size,
    )

    fig, axs = plt.subplots(n_rows, n_columns)
    return fig, axs


def plot_df_as_lineplot(
    axs,
    df,
    x_label=None,
    y_label=None,
    y_lim_default=None,
    y_lims_cols_dict=None,
    legend=False,
):
    for i, column_name in enumerate(df.columns):
        y_lim = y_lim_default
        if y_lims_cols_dict != None:
            if column_name in y_lims_cols_dict:
                y_lim = y_lims_cols_dict[column_name]

        plot_line_to_ax(
            ax=axs[i],
            df=df[column_name],
            title=column_name,
            x_label=x_label,
            y_label=y_label,
            y_lim=y_lim,
            legend=legend,
        )


def plot_dfs_as_lineplots(
    axs,
    list_dfs,
    x_label=None,
    y_label=None,
    y_lim_default=None,
    legend=False,
    y_lims_cols_dict=None,
):
    single_col = True if len(axs.shape) == 1 else False
    for j, df in enumerate(list_dfs):
        axs_cols = [axs[j]] if single_col else axs[:, j]

        plot_df_as_lineplot(
            axs_cols,
            df,
            x_label=x_label,
            y_label=y_label,
            y_lim_default=y_lim_default,
            legend=legend,
            y_lims_cols_dict=y_lims_cols_dict,
        )


def plot_line_to_ax(
    ax, df, title=None, x_label=None, y_label=None, y_lim=None, legend=False
):
    title = None
    sns.lineplot(df, ax=ax, legend=legend)
    ax.title.set_text(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    # ax.set_yticks([-1, 0, 1])
    # ax.set_xticks([0, 500, 1000])
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if tick == 0:
            label.set_horizontalalignment("left")
        if tick == 1000:
            label.set_horizontalalignment("right")

    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if tick == -1:
            label.set_verticalalignment("bottom")
        if tick == 1:
            label.set_verticalalignment("top")


def plot_heatmaps_to_axs(axs, dfs, titles):
    min_val, max_val = utils.get_min_max_from_dfs(dfs)
    for i, df in enumerate(dfs):
        plot_heatmap_to_ax(axs[i], df, title=titles[i], vmin=min_val, vmax=max_val)


def plot_heatmap_to_ax(
    ax, df, title=None, x_label=None, y_label=None, vmin=None, vmax=None, cbar=True
):
    ax.title.set_text(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    sns.heatmap(df, ax=ax, cbar=True, vmin=vmin, vmax=vmax)


def save_fig_to_path(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    return path


def save_fig(fig, config, name):
    timestamp = utils.get_timestamp()
    plot_path = f"{config.output_path}/{name}-{timestamp}.png"
    return save_fig_to_path(fig, plot_path)
