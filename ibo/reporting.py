"""Figures and tables for the IBO experiments."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as _norm

__all__ = ["plot_grouped_comparison", "plot_mean_ratio_across_functions",
           "plot_paired_differences", "report_final_gap", "paired_comparison"]


# Shared style for every figure in the paper
_STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.4,
    "figure.dpi": 200,
    "savefig.dpi": 400,
}

_GROUPS = ['EI', 'PI', 'LCB', 'TS']
_METHODS = ["Classic BO", "IBO_1", "IBO_4", "IBO_P"]
_COLORS = {
    "Classic BO": "#1f77b4",
    "IBO_1": "#d62728",
    "IBO_4": "#ff7f0e",
    "IBO_P": "#2ca02c",
}

# The keys above index the histories; these are what the legend shows.
_DISPLAY = {
    "Classic BO": "Classic BO",
    "IBO_1": r"$\mathrm{IBO}_1$",
    "IBO_4": r"$\mathrm{IBO}_4$",
    "IBO_P": r"$\mathrm{IBO}_P$",
}


def plot_grouped_comparison(histories_dict, n_seeds=30, save_path=None):
    """
    Plots grouped comparisons of different acquisition functions and methods.

    Variables:
    - histories_dict: maps "<method> - <acq>" to an (n_seeds, n_iter+1) array
      of best-so-far values, in the units of the objective.
    - n_seeds: used for the standard error of the mean curve.
    - save_path: a PDF path, or None to only display the figure.

    One panel per acquisition, the shaded band a 95% interval across seeds.
    """
    mpl.rcParams.update(_STYLE)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True,
                             constrained_layout=False)
    axes = axes.flatten()

    for i, acq in enumerate(_GROUPS):
        ax = axes[i]

        for method in _METHODS:
            label = f"{method} - {acq}"
            data = histories_dict[label]

            mean = np.mean(data, axis=0)
            sem = np.std(data, axis=0) / np.sqrt(n_seeds)

            iters = np.arange(len(mean))

            # ---- Main curve ----
            ax.plot(iters, mean, label=_DISPLAY[method],
                    color=_COLORS[method], zorder=3)

            # ---- 95% Confidence Interval ----
            ax.fill_between(iters, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=_COLORS[method], alpha=0.18, linewidth=0,
                            zorder=2)

        # Clean panel styling
        ax.set_title(acq, fontweight='bold')
        ax.grid(alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_xlabel("Iteration")

        if i == 0:
            ax.set_ylabel("Best value found")
            ax.legend(loc='best', frameon=False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="pdf", dpi=400)

    plt.show()


def plot_mean_ratio_across_functions(all_histories, f_min=1.0, save_path=None):
    """
    Computes the global normalised regret across all benchmark functions.

    Variables:
    - all_histories: maps a function name to a dict of
      "<method> - <acq>" -> (n_seeds, n_iter+1) arrays.
    - f_min: the common minimum the objectives were shifted to.
    - save_path: a PDF path, or None to only display the figure.

    The regret of each function is divided by its own initial value, so every
    curve starts at one and the functions contribute on the same scale.  The
    average is taken over seeds first and then over functions; the shaded band
    is a 95% interval across functions, not across seeds.
    """
    mpl.rcParams.update(_STYLE)

    n_functions = len(all_histories)

    # Global mean error curve and its spread across functions
    global_curves, global_sem = {}, {}

    for method in _METHODS:
        for acq in _GROUPS:
            label = f"{method} - {acq}"
            all_funcs_normalized = []

            for f_name, histories in all_histories.items():
                # histories shape: (n_seeds, n_iter + 1)
                hist = np.asarray(histories[label], float)

                # f(0): best value of the initial design, for each seed
                f_0 = hist[:, 0].reshape(-1, 1)

                # Guard against division by zero when the initial design
                # already attains f_min
                denominator = np.where((f_0 - f_min) == 0, 1e-8, (f_0 - f_min))

                # Normalised regret: (f(t) - f_min) / (f(0) - f_min), so every
                # curve starts at 1.0 and decreases towards 0.0
                f_tilde = (hist - f_min) / denominator

                # First average: over seeds
                all_funcs_normalized.append(np.mean(f_tilde, axis=0))

            # Second average: over benchmark functions
            per_function = np.vstack(all_funcs_normalized)
            global_curves[label] = per_function.mean(axis=0)
            global_sem[label] = (per_function.std(axis=0, ddof=1)
                                 / np.sqrt(n_functions))

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True,
                             constrained_layout=False)
    axes = axes.flatten()

    for i, acq in enumerate(_GROUPS):
        ax = axes[i]

        for method in _METHODS:
            label = f"{method} - {acq}"
            mean = global_curves[label]
            sem = global_sem[label]
            iters = np.arange(len(mean))

            ax.plot(iters, mean, label=_DISPLAY[method],
                    color=_COLORS[method], zorder=3)

            # ---- 95% Confidence Interval across functions ----
            ax.fill_between(iters, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=_COLORS[method], alpha=0.18, linewidth=0,
                            zorder=2)

        ax.set_title(acq, fontweight='bold')
        ax.grid(alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_xlabel("Iteration")

        if i == 0:
            ax.set_ylabel("Global normalised regret")
            ax.legend(loc='best', frameon=False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="pdf", dpi=400)

    plt.show()


def plot_paired_differences(histories_bench, histories_nstat,
                            f_min=1.0, reference="Classic BO", method="IBO_P",
                            save_path=None):
    """
    Per-function paired difference in final normalised regret.

    Variables:
    - histories_bench, histories_nstat: the two suites, in the form
      `plot_mean_ratio_across_functions` expects.
    - f_min: the common minimum the objectives were shifted to.
    - reference, method: the two methods compared, method minus reference.
    - save_path: a PDF path, or None to only display the figure.

    Within a seed the methods share the initial design, so the difference is
    paired and the variability of the design cancels.  For each function the
    mean difference over the seeds is plotted with a 95% interval; a point
    below zero means `method` attains the smaller final regret on that
    function.  The two suites are drawn side by side so that the contrast
    between them can be read directly.
    """
    mpl.rcParams.update(_STYLE)

    suites = [("Stationary", histories_bench, "#1f77b4"),
              ("Inhomogeneous", histories_nstat, "#2ca02c")]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)

    for ax, acq in zip(axes, _GROUPS):
        pos = 0
        ticks, labels, seps = [], [], []
        for suite_name, hist, colour in suites:
            for name in hist:
                h_m = np.asarray(hist[name][f"{method} - {acq}"], float)
                h_r = np.asarray(hist[name][f"{reference} - {acq}"], float)
                # normalised regret, per seed, at the end of the run
                nr_m = (h_m[:, -1] - f_min) / (h_m[:, 0] - f_min)
                nr_r = (h_r[:, -1] - f_min) / (h_r[:, 0] - f_min)
                d = nr_m - nr_r
                mu = d.mean()
                hw = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                ax.errorbar(pos, mu, yerr=hw, fmt='o', ms=4, lw=1.2,
                            color=colour, capsize=2)
                ticks.append(pos)
                labels.append(name)
                pos += 1
            seps.append(pos - 0.5)
            pos += 1

        ax.axhline(0.0, color='k', lw=0.8)
        ax.axvline(seps[0], color='0.7', lw=0.8, ls=':')
        ax.set_title(acq, fontweight='bold')
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_xlim(-1, pos - 1)

    axes[0].set_ylabel(rf"$\Delta$ final normalised regret,"
                       "\n" rf"{_DISPLAY[method]} $-$ {reference}")
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=s)
               for s, _, c in suites]
    axes[0].legend(handles=handles, frameon=False, fontsize=9, loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def report_final_gap(all_histories, f_min=1.0, methods=None, latex=False):
    """
    Final gap above the optimum, f(T) - f_min, for every function and method.

    Variables:
    - all_histories: the suite, in the form the plotting functions expect.
    - f_min: the common minimum the objectives were shifted to.
    - methods: the labels to compare; the four of the paper by default.
    - latex: print a tabular environment instead of a console table.

    The gap is in each function's own units, so it is read within a row and
    not across rows; the last line of each block counts, over the functions,
    how often each method attains the smallest gap.  That count is free of any
    scale assumption and is the statistic the aggregated curve cannot show.
    """
    if methods is None:
        methods = list(_METHODS)
    names = list(all_histories)

    for acq in _GROUPS:
        gaps = {m: np.array([
            np.asarray(all_histories[n][f"{m} - {acq}"], float)[:, -1].mean() - f_min
            for n in names]) for m in methods}

        if latex:
            print("\n% " + acq)
            print(r"\begin{tabular}{l" + "r" * len(methods) + "}")
            print(r"\toprule")
            print("Function & " + " & ".join(methods) + r" \\")
            print(r"\midrule")
            for k, n in enumerate(names):
                row = [gaps[m][k] for m in methods]
                best = int(np.argmin(row))
                cells = [(r"\textbf{" + f"{v:.3g}" + "}") if j == best else f"{v:.3g}"
                         for j, v in enumerate(row)]
                print(f"{n} & " + " & ".join(cells) + r" \\")
            print(r"\midrule")
            wins = [int(sum(np.argmin([gaps[m][k] for m in methods]) == j
                            for k in range(len(names)))) for j in range(len(methods))]
            print("Wins & " + " & ".join(f"{w}/{len(names)}" for w in wins) + r" \\")
            print(r"\bottomrule")
            print(r"\end{tabular}")
        else:
            print("\n" + "=" * 68)
            print(f"  {acq}: final gap  f(T) - f_min")
            print("=" * 68)
            print(f"{'function':>12} " + " ".join(f"{m:>12}" for m in methods)
                  + "   best")
            for k, n in enumerate(names):
                row = [gaps[m][k] for m in methods]
                best = methods[int(np.argmin(row))]
                print(f"{n:>12} " + " ".join(f"{v:12.4g}" for v in row)
                      + f"   {best}")
            wins = {m: 0 for m in methods}
            for k in range(len(names)):
                wins[methods[int(np.argmin([gaps[m][k] for m in methods]))]] += 1
            print(f"{'wins':>12} " + " ".join(
                f"{str(wins[m]) + '/' + str(len(names)):>12}" for m in methods))


def paired_comparison(all_histories, f_min=1.0, methods=None, alpha=0.05,
                      latex=False):
    """
    Paired comparison of the final gap between every pair of methods.

    Variables:
    - all_histories: the suite, in the form the plotting functions expect.
    - f_min: the common minimum the objectives were shifted to.  Pass
      normalised regrets with f_min=0.0 to pool functions of different scales.
    - methods: the labels to compare; the four of the paper by default.
    - alpha: two-sided level of the reported interval.

    Within a seed the four methods share the same initial design, so the
    differences are paired and the seed-to-seed variation of the design
    cancels.  For each function and seed the final gap f(T) - f_min is formed,
    the difference between two methods is taken, and the mean difference over
    the function-seed pairs is reported with a normal interval.  A negative
    difference means the row method attains the smaller gap.

    The interval is uncorrected for the number of comparisons, and the
    function-seed pairs are not independent across functions, so it describes
    the spread of the paired differences rather than a calibrated test.  The
    win count beside it is free of any such assumption.
    """
    z = _norm.ppf(1 - alpha / 2)
    if methods is None:
        methods = list(_METHODS)
    names = list(all_histories)

    out = {}
    for acq in _GROUPS:
        # final gap, one row per function, one column per seed
        g = {m: np.vstack([
                np.asarray(all_histories[n][f"{m} - {acq}"], float)[:, -1] - f_min
                for n in names]) for m in methods}

        print("\n" + "=" * 70)
        print(f"  {acq}: paired difference in final gap, row minus column")
        print("=" * 70)
        print(f"{'':>12} " + " ".join(f"{m:>17}" for m in methods))
        for a in methods:
            cells = []
            for b in methods:
                if a == b:
                    cells.append(f"{'--':>17}")
                    continue
                dvec = (g[a] - g[b]).ravel()
                mu = dvec.mean()
                hw = z * dvec.std(ddof=1) / np.sqrt(len(dvec))
                star = "*" if abs(mu) > hw else " "
                wins = int((g[a] < g[b]).sum())
                tot = g[a].size
                cells.append(f"{mu:+8.4f}{star} {wins:3d}/{tot:<3d}")
                out[(acq, a, b)] = (mu, hw, wins, tot)
            print(f"{a:>12} " + " ".join(cells))
        print("  a star marks an interval excluding zero; the fraction counts")
        print("  the function-seed pairs on which the row method is better.")
    return out


def report_final_regret(all_histories, f_min=1.0, methods=None, latex=False):
    """
    Final normalised regret NR(T), for every function and method.

    Variables:
    - all_histories: the suite, in the form the plotting functions expect.
    - f_min: the common minimum the objectives were shifted to.
    - methods: the labels to compare; the four of the paper by default.  The
      first one is the reference of the median ratios printed at the end.
    - latex: print a tabular environment instead of a console table.

    The regret of each run is divided by its own initial value, so the
    functions are on a common scale and a value of 0.05 means that 5% of the
    initial error remains.  The denominator is shared by every method at a
    given seed, so the ranking within a function is the one the raw gaps give;
    the last line of each block counts, over the functions, how often each
    method attains the smallest value.  That count is free of any scale
    assumption and is the statistic the aggregated curve cannot show.
    """
    groups = ['EI', 'PI', 'LCB', 'TS']
    if methods is None:
        methods = ["Classic BO", "IBO_1", "IBO_4", "IBO_P"]
    names = list(all_histories)

    for acq in groups:
        nr = {}
        for m in methods:
            vals = []
            for n in names:
                h = np.asarray(all_histories[n][f"{m} - {acq}"], float)
                # NR(T) = (f(T) - f_min) / (f(0) - f_min), per seed, then
                # averaged: the denominator varies with the seed, so the
                # division comes first.
                denom = h[:, 0] - f_min
                denom = np.where(denom == 0, 1e-8, denom)
                vals.append(float(np.mean((h[:, -1] - f_min) / denom)))
            nr[m] = np.array(vals)

        if latex:
            print("\n% " + acq)
            print(r"\begin{tabular}{l" + "r" * len(methods) + "}")
            print(r"\toprule")
            print("Function & " + " & ".join(methods) + r" \\")
            print(r"\midrule")
            for k, n in enumerate(names):
                row = [nr[m][k] for m in methods]
                best = int(np.argmin(row))
                cells = [(r"\textbf{" + f"{v:.3g}" + "}") if j == best else f"{v:.3g}"
                         for j, v in enumerate(row)]
                print(f"{n} & " + " & ".join(cells) + r" \\")
            print(r"\midrule")
            wins = [int(sum(np.argmin([nr[m][k] for m in methods]) == j
                            for k in range(len(names)))) for j in range(len(methods))]
            print("Wins & " + " & ".join(f"{w}/{len(names)}" for w in wins) + r" \\")
            print(r"\bottomrule")
            print(r"\end{tabular}")
        else:
            print("\n" + "=" * 68)
            print(f"  {acq}: final normalised regret  NR(T)")
            print("=" * 68)
            print(f"{'function':>12} " + " ".join(f"{m:>12}" for m in methods)
                  + "   best")
            for k, n in enumerate(names):
                row = [nr[m][k] for m in methods]
                best = methods[int(np.argmin(row))]
                print(f"{n:>12} " + " ".join(f"{v:12.4g}" for v in row)
                      + f"   {best}")
            wins = {m: 0 for m in methods}
            for k in range(len(names)):
                wins[methods[int(np.argmin([nr[m][k] for m in methods]))]] += 1
            print(f"{'wins':>12} " + " ".join(
                f"{str(wins[m]) + '/' + str(len(names)):>12}" for m in methods))

    # Median over the function-acquisition pairs of the ratio of the reference
    # to each variant: a value above one favours the variant.
    print("\n" + "=" * 68)
    print(f"  median ratio  NR(T) of {methods[0]}  /  NR(T) of the variant")
    print("=" * 68)
    ref = methods[0]
    for m in methods[1:]:
        ratios = []
        for acq in groups:
            for n in names:
                hr = np.asarray(all_histories[n][f"{ref} - {acq}"], float)
                hm = np.asarray(all_histories[n][f"{m} - {acq}"], float)
                dr = np.where(hr[:, 0] - f_min == 0, 1e-8, hr[:, 0] - f_min)
                dm = np.where(hm[:, 0] - f_min == 0, 1e-8, hm[:, 0] - f_min)
                ratios.append(np.mean((hr[:, -1] - f_min) / dr)
                              / np.mean((hm[:, -1] - f_min) / dm))
        print(f"{m:>12}: {np.median(ratios):.2f}")
