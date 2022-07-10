#!/usr/bin/python3

import os
import importlib
from turtle import update
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import re
import seaborn as sns

ggPlotColors = importlib.import_module("ggplot_colors").__dict__["ggColorSlice"]
style.use("ggplot")

plt.rcParams["lines.linewidth"] = 1
half_textwidth_font_size = 22
textwidth_font_size = 16


# all paths are relative to this directory
OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../plots/")
)

LATEX_GEN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gen/")
)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if not os.path.exists(LATEX_GEN_PATH):
    os.makedirs(LATEX_GEN_PATH)


REPO_PATH = os.path.normpath(
    r"C:\Users\Max\Documents\KIT\Masterarbeit\code\rust_truck_router"
)

DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
)

BIN_PATH = os.path.normpath(os.path.join(REPO_PATH, "target/release"))

GRAPH_PATH = os.path.normpath(
    r"C:\Users\Max\Documents\KIT\Masterarbeit\code",
)


def read_measurement(bin, *args, **kwargs):
    return pd.read_csv(os.path.join(DATA_PATH, bin + ".txt"), *args, **kwargs)


def write_plt(filename, graph):
    # path = os.path.join(os.path.join(OUTPUT_PATH, graph))
    path = os.path.join(os.path.join(OUTPUT_PATH))

    # if not os.path.exists(path):
    #     os.makedirs(path)

    plt.savefig(
        os.path.join(
            path,
            filename,
        ),
        dpi=1200,
    )
    plt.close()


def write_textfile(filename, content):
    path = os.path.join(OUTPUT_PATH, filename)

    with open(path, "w") as f:
        f.write(content)


def exp2_ticks(x):
    if x == 0:
        return "$0$"

    exponent = int(np.log2(x))
    coeff = x / 2 ** exponent

    if coeff == 1:
        return r"$2^{{ {:2d} }}$".format(exponent)

    return r"${:2.0f} \times 2^{{ {:2d} }}$".format(coeff, exponent)


def exp2_ticks_from_exponent(exponent):
    return r"$2^{{ {:2d} }}$".format(exponent)


def make_dijkstra_rank_tick_labels_from_exponent(ax_axis, exponents):
    ax_axis.set_ticklabels([exp2_ticks_from_exponent(int(i)) for i in exponents])


def make_dijkstra_rank_tick_labels_from_number(ax_axis, exponents):
    ax_axis.set_ticklabels([exp2_ticks(int(i)) for i in exponents])


def get_boxplot_outliers(df, by_column_name):
    q1 = df[by_column_name].quantile(0.25)
    q3 = df[by_column_name].quantile(0.75)
    iqr = q3 - q1
    filter = np.invert(
        (df[by_column_name] >= q1 - 1.5 * iqr) & (df[by_column_name] <= q3 + 1.5 * iqr)
    )
    return df.loc[filter]


def gen_avg_times():
    name_ger = "thesis_avg_all-csp"
    name_ger_2 = "thesis_avg_all-csp_2"
    name_eur = "thesis_avg_mid-csp"
    name_eur_2 = "thesis_avg_mid-csp_2"

    avg_ger = read_measurement(name_ger + "-" + "parking_ger_hgv")
    avg_ger_2 = read_measurement(name_ger_2 + "-" + "parking_ger_hgv")

    avg_eur = read_measurement(name_eur + "-" + "parking_europe_hgv")
    avg_eur_2 = read_measurement(name_eur_2 + "-" + "parking_europe_hgv")

    avg_ger = avg_ger.groupby(["algo"]).mean()
    avg_eur = avg_eur.groupby(["algo"]).mean()
    avg_ger_2 = avg_ger_2.groupby(["algo"]).mean()
    avg_eur_2 = avg_eur_2.groupby(["algo"]).mean()

    avg_all = avg_ger.join(avg_eur, rsuffix="_eur")[["time_ms", "time_ms_eur"]]
    avg_all_2 = avg_ger_2.join(avg_eur_2, rsuffix="_eur")[["time_ms", "time_ms_eur"]]

    avg_all["time_ms_2"] = avg_all_2["time_ms"]
    avg_all["time_ms_eur_2"] = avg_all_2["time_ms_eur"]
    # avg_ger["num_nodes_searched_2"] = avg_ger_2["num_nodes_searched"]

    order = [
        "dijkstra",
        "dijkstra_bidir",
        "astar_chpot",
        "astar_bidir_chpot",
        "core_ch",
        "core_ch_chpot",
    ]

    avg_all = avg_all.reindex(order)

    # HERE ORDER OF COLUMNS
    values = (
        # avg_all[["time_ms", "time_ms_2", "num_nodes_searched", "num_nodes_searched_2"]]
        avg_all[["time_ms", "time_ms_2", "time_ms_eur", "time_ms_eur_2"]]
        .to_numpy()
        .flatten()
    )

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_all-TEMPLATE.tex")) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + name_ger
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_all.tex"), "w") as f:
        f.write(template)


def gen_median_all_times():
    name_ger = "thesis_avg_all-csp"
    name_ger_2 = "thesis_avg_all-csp_2"
    name_eur = "thesis_avg_mid-csp"
    name_eur_2 = "thesis_avg_mid-csp_2"

    avg_ger = read_measurement(name_ger + "-" + "parking_ger_hgv")
    avg_ger_2 = read_measurement(name_ger_2 + "-" + "parking_ger_hgv")

    avg_eur = read_measurement(name_eur + "-" + "parking_europe_hgv")
    avg_eur_2 = read_measurement(name_eur_2 + "-" + "parking_europe_hgv")

    avg_ger = avg_ger.groupby(["algo"]).median()
    avg_eur = avg_eur.groupby(["algo"]).median()
    avg_ger_2 = avg_ger_2.groupby(["algo"]).median()
    avg_eur_2 = avg_eur_2.groupby(["algo"]).median()

    avg_all = avg_ger.join(avg_eur, rsuffix="_eur")[["time_ms", "time_ms_eur"]]
    avg_all_2 = avg_ger_2.join(avg_eur_2, rsuffix="_eur")[["time_ms", "time_ms_eur"]]

    avg_all["time_ms_2"] = avg_all_2["time_ms"]
    avg_all["time_ms_eur_2"] = avg_all_2["time_ms_eur"]
    # avg_ger["num_nodes_searched_2"] = avg_ger_2["num_nodes_searched"]

    order = [
        "dijkstra",
        "dijkstra_bidir",
        "astar_chpot",
        "astar_bidir_chpot",
        "core_ch",
        "core_ch_chpot",
    ]

    avg_all = avg_all.reindex(order)

    # HERE ORDER OF COLUMNS
    values = (
        # avg_all[["time_ms", "time_ms_2", "num_nodes_searched", "num_nodes_searched_2"]]
        avg_all[["time_ms", "time_ms_2", "time_ms_eur", "time_ms_eur_2"]]
        .to_numpy()
        .flatten()
    )

    with open(
        os.path.join(LATEX_GEN_PATH, "eval_median_running_times_all-TEMPLATE.tex")
    ) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + name_ger
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(
        os.path.join(LATEX_GEN_PATH, "eval_median_running_times_all.tex"), "w"
    ) as f:
        f.write(template)


def gen_all_times_no_path():
    name_eur = "thesis_avg_mid-csp"
    name_eur_2 = "thesis_avg_mid-csp_2"

    avg_eur = read_measurement(name_eur + "-" + "parking_europe_hgv")
    avg_eur_2 = read_measurement(name_eur_2 + "-" + "parking_europe_hgv")

    # avg_eur = avg_eur.groupby(["algo"]).median()
    # avg_eur_2 = avg_eur_2.groupby(["algo"]).median()

    # avg_eur["time_ms_eur_2"] = avg_eur_2["time_ms_eur"]
    # avg_ger["num_nodes_searched_2"] = avg_ger_2["num_nodes_searched"]

    longest = pd.DataFrame()
    longest_2 = pd.DataFrame()

    order = [
        # "dijkstra",
        # "dijkstra_bidir",
        "astar_chpot",
        "astar_bidir_chpot",
        "core_ch",
        "core_ch_chpot",
    ]

    for algo in order:
        # num_rows = avg_eur.loc[avg_eur["algo"] == algo].size[0] / 10
        # longest = longest.append(avg_eur.loc[avg_eur["algo"] == algo].sort_values("time_ms")[-num_rows:])
        subset = avg_eur.loc[avg_eur["algo"] == algo]
        longest = longest.append(subset.loc[np.isnan(subset["path_distance"])])

        subset_2 = avg_eur_2.loc[avg_eur_2["algo"] == algo]
        longest_2 = longest_2.append(subset_2.loc[np.isnan(subset_2["path_distance"])])

    longest_mean = longest.groupby("algo").mean()
    longest_median = longest.groupby("algo").median()
    longest_mean = longest_mean.reindex(order)
    longest_median = longest_median.reindex(order)

    longest_2_mean = longest_2.groupby("algo").mean()
    longest_2_median = longest_2.groupby("algo").median()
    longest_2_mean = longest_2_mean.reindex(order)
    longest_2_median = longest_2_median.reindex(order)

    longest_all = pd.DataFrame()
    longest_all["mean"] = longest_mean["time_ms"]
    longest_all["median"] = longest_median["time_ms"]
    longest_all["mean_2"] = longest_2_mean["time_ms"]
    longest_all["median_2"] = longest_2_median["time_ms"]
    longest_all = longest_all.reindex(order)

    # HERE ORDER OF COLUMNS
    values = (
        # avg_eur[["time_ms", "time_ms_2", "num_nodes_searched", "num_nodes_searched_2"]]
        longest_all[["mean", "mean_2", "median", "median_2"]]
        .to_numpy()
        .flatten()
    )

    with open(
        os.path.join(LATEX_GEN_PATH, "eval_running_times_no_path-TEMPLATE.tex")
    ) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + "running_times_no_path"
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_no_path.tex"), "w") as f:
        f.write(template)


def gen_avg_opt():
    name = "thesis_avg_opt-csp"
    name_2 = "thesis_avg_opt-csp_2"

    avg = read_measurement(name + "-" + "parking_europe_hgv")
    avg_2 = read_measurement(name_2 + "-" + "parking_europe_hgv")

    avg_mean = avg.groupby(["algo"]).mean()
    avg_mean_2 = avg_2.groupby(["algo"]).mean()

    avg_med = avg.groupby(["algo"]).median()
    avg_med_2 = avg_2.groupby(["algo"]).median()

    avg_all = avg_mean.join(avg_mean_2, rsuffix="_2")[["time_ms", "time_ms_2"]]
    avg_all["time_ms_med"] = avg_med["time_ms"]
    avg_all["time_ms_med_2"] = avg_med_2["time_ms"]

    # HERE ORDER OF ROWS
    order = [
        "no_bw_no_prune",
        "core_ch_chpot",
        # "no_bw_add_prune",
        # "add_bw_add_prune",
    ]

    avg_all = avg_all.reindex(order)

    # HERE ORDER OF COLUMNS
    values = (
        avg_all[["time_ms", "time_ms_2", "time_ms_med", "time_ms_med_2"]]
        .to_numpy()
        .flatten()
    )

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_opt-TEMPLATE.tex")) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + name
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_opt.tex"), "w") as f:
        f.write(template)


def plot_breaks_running_times(graph):
    problem = "csp"
    name = "thesis_rank_times_all-" + problem + "-" + graph
    name_2 = "thesis_avg_all-" + problem + "-" + graph

    queries_all = read_measurement(name).append(read_measurement(name_2))

    # plot only 2^10 and larger
    queries_all = queries_all.loc[queries_all["dijkstra_rank_exponent"] >= 10]

    colors = ggPlotColors(4)
    to_plot = [
        ("time_ms", "log"),
    ]

    # algos = ["astar_chpot", "astar_bidir_chpot", "core_ch", "core_ch_chpot"]
    algos = ["astar_chpot", "core_ch", "core_ch_chpot"]

    for algo in algos:
        queries = queries_all.loc[queries_all["algo"] == algo]
        # queries = queries.loc[queries["path_distance"] == -1]
        for (column_name, plot_scale) in to_plot:
            if column_name in queries.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                bp = queries.boxplot(ax=ax, by="path_number_pauses", column=column_name)

                bp.get_figure().gca().set_title("")
                fig.suptitle("")

                ax.set_xlabel("number of breaks")
                ax.set_ylabel(column_name)
                ax.set_yscale(plot_scale)

                if plot_scale == "linear":
                    ax.set_ylim(bottom=-0.1)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                plt.title("breaks_running_times" + "-" + algo + ": " + graph)
                fig.tight_layout()
                write_plt(
                    "breaks_running_times" + "-" + algo + "-" + column_name + ".png",
                    graph,
                )


def plot_all_rank_times(problem, graph):
    name = "thesis_rank_times_all-" + problem + "-" + graph

    queries_all = read_measurement(name)

    # plot only 2^10 and larger
    queries_all = queries_all.loc[queries_all["dijkstra_rank_exponent"] >= 10]

    colors = ggPlotColors(4)
    to_plot = [
        ("time_ms", "log"),
    ]

    # algos = ["astar_chpot", "astar_bidir_chpot", "core_ch", "core_ch_chpot"]
    algos = ["astar_chpot", "core_ch", "core_ch_chpot"]

    for algo in algos:
        queries = queries_all.loc[queries_all["algo"] == algo]
        # queries = queries.loc[queries["path_distance"] == -1]
        for (column_name, plot_scale) in to_plot:
            if column_name in queries.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                bp = queries.boxplot(
                    ax=ax, by="dijkstra_rank_exponent", column=column_name
                )

                bp.get_figure().gca().set_title("")
                fig.suptitle("")

                ax.set_xlabel("Dijkstra Rank", fontsize=textwidth_font_size)
                ax.set_ylabel(column_name)
                ax.set_yscale(plot_scale)
                ax.set_ylabel("Running Time [ms]", fontsize=textwidth_font_size)

                if plot_scale == "linear":
                    ax.set_ylim(bottom=-0.1)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                make_dijkstra_rank_tick_labels_from_exponent(
                    ax.xaxis, queries["dijkstra_rank_exponent"].unique()
                )

                # plt.title(name + "-" + algo + ": " + graph)
                plt.title("")
                fig.tight_layout()

                write_plt(name + "-" + algo + "-" + column_name + ".png", graph)  #


def plot_constraint_times(graph):
    name_d = "thesis_driving_times-csp-" + graph
    name_b = "thesis_break_times-csp-" + graph

    queries_d = read_measurement(name_d)
    queries_d["max_driving_time"] = queries_d["max_driving_time"] / 3600000
    queries_b = read_measurement(name_b)
    queries_b["max_break_time"] = queries_b["max_break_time"] / 3600000

    colors = ggPlotColors(4)

    algos = ["core_ch", "core_ch_chpot"]

    # sns.lmplot(
    #     x="max_break_time",
    #     y="time_ms",
    #     data=queries_b[queries_b["algo"] == "core_ch_chpot"],
    #     order=2,
    #     scatter=False,
    #     height=5,
    #     aspect=2,
    # )
    # plt.show()

    # smooth = lambda x: x.rolling(1000).mean()
    smooth = lambda x: x.rolling(10).mean()
    max_b = smooth(queries_b.groupby("max_break_time").median()["time_ms"]).max()
    min_b = smooth(queries_b.groupby("max_break_time").median()["time_ms"]).min()

    iv = max_b - min_b
    padding = iv * 0.1

    for algo in algos:
        queries_d_current = (
            queries_d[queries_d["algo"] == algo].groupby("max_driving_time").median()
        )
        queries_b_current = (
            queries_b[queries_b["algo"] == algo].groupby("max_break_time").median()
        )

        queries_d_rolling = smooth(queries_d_current)
        queries_b_rolling = smooth(queries_b_current)

        min_b = queries_b_rolling["time_ms"].min() - padding
        max_b = min_b + 2 * padding + iv

        fig, ax = plt.subplots(figsize=(10, 5))
        plot = queries_d_rolling.plot(ax=ax)
        plot.get_figure().gca().set_title("")
        fig.suptitle("")

        ax.set_xlabel(
            "Maximum Allowed Driving Time [h]", fontsize=half_textwidth_font_size
        )
        ax.set_ylabel("Running Time [ms]", fontsize=half_textwidth_font_size)
        ax.set_ylim(bottom=-0.1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.get_legend().remove()
        plt.title("")
        fig.tight_layout()
        write_plt(name_d + "-" + algo + "-time_ms.png", graph)

        fig, ax = plt.subplots(figsize=(10, 5))
        plot = queries_b_rolling.plot(ax=ax)
        plot.get_figure().gca().set_title("")
        fig.suptitle("")

        ax.set_xlabel("Break Time [h]", fontsize=half_textwidth_font_size)
        ax.set_ylabel("Running Time [ms]", fontsize=half_textwidth_font_size)
        ax.set_ylim(ymin=min_b, ymax=max_b)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.get_legend().remove()
        plt.title("")
        fig.tight_layout()
        write_plt(name_b + "-" + algo + "-time_ms.png", graph)


def plot_core_sizes_experiment(graph):
    name = "thesis_core_sizes-csp-" + graph
    name_2 = "thesis_core_sizes-csp_2-" + graph

    # rel_core_size,abs_core_size,construction_time_ms,time_ms
    queries = read_measurement(name)
    queries_2 = read_measurement(name_2)

    queries["rel_core_size"] = queries["rel_core_size"] * 100
    queries_2["rel_core_size"] = queries_2["rel_core_size"] * 100

    queries = queries.groupby("rel_core_size").mean()
    queries_2 = queries_2.groupby("rel_core_size").mean()

    colors = ggPlotColors(4)

    # plot construction
    constr = queries["construction_time_ms"] / 60000

    fig, ax = plt.subplots(figsize=(10, 5))
    plot = constr.plot(ax=ax)
    plot.get_figure().gca().set_title("")
    fig.suptitle("")
    ax.set_xlabel("Core Size [%]", fontsize=textwidth_font_size)
    ax.set_ylabel("Construction Time [min]", fontsize=textwidth_font_size)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title("")
    fig.tight_layout()
    write_plt(name + "-constr_time.png", graph)

    # plot queries
    queries_all = queries.join(queries_2, rsuffix="_2")[["time_ms", "time_ms_2"]]
    queries_all = queries_all.rename(columns={"time_ms": "1DTC", "time_ms_2": "2DTC"})

    fig, ax = plt.subplots(figsize=(10, 5))
    plot = queries_all.plot(ax=ax)
    plot.get_figure().gca().set_title("")
    fig.suptitle("")
    ax.set_xlabel("Core Size [%]", fontsize=textwidth_font_size)
    ax.set_ylabel("Running Time Time [ms]", fontsize=textwidth_font_size)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title("")
    fig.tight_layout()
    write_plt(name + "-time_ms.png", graph)


def plot_speed_cap_experiment(graph):
    name = "thesis_speed_cap-csp-" + graph
    name_2 = "thesis_speed_cap-csp_2-" + graph

    # rel_core_size,abs_core_size,construction_time_ms,time_ms
    queries = read_measurement(name)
    queries_2 = read_measurement(name_2)

    queries = queries.groupby("speed_cap").mean()
    queries_2 = queries_2.groupby("speed_cap").mean()

    colors = ggPlotColors(4)

    # # plot construction
    # constr = queries["construction_time_ms"] / 60000

    # fig, ax = plt.subplots(figsize=(10, 5))
    # plot = constr.plot(ax=ax)
    # plot.get_figure().gca().set_title("")
    # fig.suptitle("")
    # ax.set_xlabel("Core Size [%]", fontsize=textwidth_font_size)
    # ax.set_ylabel("Construction Time [min]", fontsize=textwidth_font_size)
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.title("")
    # fig.tight_layout()
    # write_plt(name + "-constr_time.png", graph)

    # plot queries
    queries_all = queries.join(queries_2, rsuffix="_2")[["time_ms", "time_ms_2"]]

    print(queries_all)

    queries_all = queries_all.rename(
        columns={
            "time_ms": "1DTC",
            "time_ms_2": "2DTC",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    plot = queries_all.plot(ax=ax)
    plot.get_figure().gca().set_title("")
    fig.suptitle("")
    ax.set_xlabel("Speed Cap [km/h]", fontsize=textwidth_font_size)
    ax.set_ylabel("Running Time [ms]", fontsize=textwidth_font_size)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title("")
    fig.tight_layout()
    write_plt("thesis_speed_cap.png", graph)


if __name__ == "__main__":
    # gen_avg_times()
    # gen_median_all_times()
    # gen_avg_opt()

    # gen_all_times_no_path()

    # plot_breaks_running_times("parking_europe_hgv")
    # plot_all_rank_times("csp", "parking_europe_hgv")
    # plot_all_rank_times("csp_2", "parking_europe_hgv")

    plot_constraint_times("parking_europe_hgv")
    plot_core_sizes_experiment("parking_europe_hgv")
    plot_speed_cap_experiment("parking_europe_hgv_sc")
