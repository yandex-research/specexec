import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
from scipy.spatial import ConvexHull, QhullError
from matplotlib.colors import LogNorm


class LogReader:
    """
    Reads and processes log data from a JSONL experiment log file.

    Attributes:
        .name (str): Name of the experiment or log.
        .config (dict): Configuration data from the log.
        .df (pandas.DataFrame): DataFrame containing run data.
        .dfe (pandas.DataFrame): DataFrame containing experiment data.
        .dfa (pandas.DataFrame): Aggregated data from the runs.
    """

    def __init__(self, save_name, name=None, logs_dir="../logs", groupby_columns=None, filter_dict=None, n_runs=100500):
        """
        Initializes the LogReader with data from the specified log file.

        Args:
            save_name (str): Base name of the log file to read.
            name (str, optional): Optional custom name for the log data.
        """
        if not isinstance(save_name, (tuple, list)):
            save_name = (save_name,)
        self.name = name or save_name[0]
        self.config = None
        self.groupby_columns = groupby_columns or ["max_n_beams", "max_beam_len"]
        self.filter_dict = filter_dict or {}
        self.n_runs = n_runs

        lines = []
        for sname in save_name:
            file_path = f"{logs_dir}/{sname}.jsonl"
            with open(file_path, "r") as f:
                lines.extend(f.readlines())

        exps, runs = [], []
        for line in lines:
            line = json.loads(line)

            if line.get("msg_type") == "config" and self.config is None:  # use the 1st config line only
                self.config = line
            elif line.get("msg_type") == "exp":
                exps.append(line)
            elif line.get("msg_type") in ("", "summary"):
                runs.append(line)

        self.df = pd.DataFrame(runs)
        try:
            self.df["num_gen"] = (self.df.new_tokens / self.df.iters).round(1)
        except AttributeError:
            return

        # remaning legacy fields(before 12 Dec)
        if "max_n_beams" not in self.df.columns:
            self.df["max_n_beams"] = self.df.n_beams
        if "max_beam_len" not in self.df.columns:
            self.df["max_beam_len"] = self.df.beam_len
        if "tree_h" not in self.df.columns:
            self.df["tree_h"] = self.df.beam_len
        if "tree_w" not in self.df.columns:
            self.df["tree_w"] = self.df.n_beams
        if "tree_size" not in self.df.columns:
            self.df["tree_size"] = self.df.beam_len * self.df.n_beams
        if "speed" not in self.df.columns:
            self.df["speed"] = self.df.new_tokens / (self.df.t0 + self.df.t1)

        # dataframe with experiment summaries pre-aggregated in the log file
        self.dfe = pd.DataFrame(exps)

        self.dfa = (
            self.df[self.df.run < self.n_runs]
            .groupby(self.groupby_columns)
            .agg(
                {
                    "input_0": "mean",
                    "input_1": "mean",
                    "run": "count",
                    "iters": "mean",
                    "new_tokens": "mean",
                    "num_gen": "mean",
                    "tree_h": "mean",
                    "tree_w": "mean",
                    "tree_size": "mean",
                    "speed": "mean",
                    "t0": "mean",
                    "t1": "mean",
                    "mem_use": "max",
                    # "max_branch_width": "mean",
                    # "max_budget": "mean",
                }
            )
            .rename(columns={"ver": "num_tests"})
            .reset_index()
        )

    def get_plot_data(self, param="num_gen", x="max_n_beams", y="max_beam_len"):
        """
        Retrieves plot data for a specified parameter.
        Args:  param (str): The parameter to plot.
        Returns: pandas.DataFrame: DataFrame suitable for plotting.
        Raises: ValueError: If the parameter is not found in the DataFrame.
        """
        if param not in self.dfa.columns:
            raise ValueError(f"wrong param {param}")
        return self.dfa.pivot(columns=x, index=y, values=param)

    def plot_rate(self, param="num_gen", ax=None, x="max_n_beams", y="max_beam_len", title=None, **kwargs):
        """
        Plots the rate of a specified parameter.
        Args:  ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 2.5))
        chart_data = self.get_plot_data(param=param, x=x, y=y)  # .round(1)
        if param == "num_gen":
            kwargs = {**kwargs, "vmin": 0, "vmax": 15, "fmt": ".1f", "cmap": "Spectral", "annot_kws": {"size": 12}}
        elif param == "tree_size":
            kwargs = {"cmap": "terrain", "vmin": 0, "vmax": 4096, "norm": LogNorm(), **kwargs}
        elif param == "t0":
            kwargs = {**kwargs, "vmin": 0.2, "vmax": 3, "fmt": ".2f", "cmap": "terrain", "annot_kws": {"size": 10}}
        elif param == "speed":
            kwargs = {"cmap": "plasma", "vmin": 0, "vmax": 4, "fmt": ".2f", **kwargs}
        kwargs = {"fmt": ".0f", "linewidth": 0.5, "annot_kws": {"size": 10}, "cmap": "viridis", **kwargs}

        sns.heatmap(chart_data, annot=True, ax=ax, cbar=False, **kwargs)
        ax.set_title(title or self.name)

    def get_frontier(self, budget="tree_size", util="num_gen", all_x=True):
        """
        Pareto-frontier of budget vs utility
        returns pd.DataFrame with 2 columns of budget and utility values from Pareto frontier points
        """
        points_0 = self.dfa.loc[:, [budget, util]].values
        try:
            hull = ConvexHull(points_0)
            points_hull = points_0[hull.vertices]
        except QhullError:
            points_hull = points_0

        points_frontier = []
        for pt in points_hull:
            flag = True
            for ref in points_hull:
                if ref[0] <= pt[0] and ref[1] > pt[1]:
                    flag = False
                    break
            if flag:
                points_frontier.append(pt)

        if all_x:  # report best value for powers of 2 even if not on P-frontier
            for i in range(15):
                x = 2**i
                if x not in [q[0] for q in points_frontier]:
                    if x in points_0[:, 0]:
                        y = points_0[points_0[:, 0] == x][:, 1].max()
                        points_frontier.append(np.array([x, y]))

        df_result = np.stack(points_frontier)
        df_result = pd.DataFrame(df_result, columns=[budget, util])
        df_result = df_result.sort_values(budget, ascending=True)
        return df_result

    def plot3(self, title=""):
        xx, yy = self.groupby_columns
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 2.5))
        self.plot_rate(title=title + ": gen rate", x=xx, y=yy, ax=ax0)
        self.plot_rate(title=title + ":  tree size", param="tree_size", x=xx, y=yy, ax=ax1)
        self.plot_rate(title=title + ": time 0", param="t0", x=xx, y=yy, ax=ax2)

    def plot4(self, title=""):
        xx, yy = self.groupby_columns
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(18, 3))
        self.plot_rate(title=title + ": gen rate", x=xx, y=yy, ax=ax0)
        self.plot_rate(title=title + ": speed", param="speed", x=xx, y=yy, ax=ax1)
        self.plot_rate(title=title + ":  tree size", param="tree_size", x=xx, y=yy, ax=ax2)
        self.plot_rate(title=title + ": time 0", param="t0", x=xx, y=yy, ax=ax3)
        plt.tight_layout()
