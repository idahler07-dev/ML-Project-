"""Step 7: figures (PNG, sized to be readable on a phone screen).

Every figure is drawn from files in data/clean/ and results/tables/.
Output: results/figures/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

# Colours (checked for colour-blind separation) and chart "ink".
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY = "#898781"
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"

# Countries shown in the country figures: Uzbekistan, its Central Asian
# neighbours in the sample, and (for the inflation history) Turkiye, the
# country with the highest inflation in the sample.
FOCUS = ["UZB", "KAZ", "KGZ", "TJK"]
HISTORY = FOCUS + ["TUR"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 13,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved results/figures/{name}")


def country_names() -> dict:
    panel = pd.read_csv(CLEAN_DIR / "panel.csv")
    return dict(zip(panel["country"], panel["country_name"]))


# ---------------------------------------------------------------------- figures

def plot_commodity_indices() -> None:
    monthly = pd.read_csv(CLEAN_DIR / "commodities_monthly.csv")
    monthly["date"] = pd.PeriodIndex(monthly["date"], freq="M").to_timestamp()
    monthly = monthly[monthly["date"] >= "2000-01-01"]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for column, label, colour in [("food", "Food", BLUE), ("energy", "Energy", ORANGE)]:
        ax.plot(monthly["date"], monthly[column], color=colour, label=label)
        last = monthly.dropna(subset=[column]).iloc[-1]
        ax.annotate(label, (last["date"], last[column]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=12)
    ax.axhline(100, color=AXIS, linewidth=1)
    ax.set_title("World food and energy prices, 2000–2026")
    ax.set_ylabel("Price index (2010 average = 100)")
    ax.set_xlabel("Month")
    ax.legend(loc="upper left")
    last_month = monthly["date"].max().strftime("%b %Y")
    fig.text(0, -0.02, f"Monthly, nominal US dollars, up to {last_month}. "
             "Source: World Bank Pink Sheet.", color=INK_2, fontsize=10)
    save(fig, "01_commodity_indices.png")


def plot_inflation_history() -> None:
    panel = pd.read_csv(CLEAN_DIR / "panel.csv")
    names = country_names()
    median = panel.groupby("year")["inflation"].median()

    fig, axes = plt.subplots(len(HISTORY), 1, figsize=(7, 11), sharex=True)
    for ax, code in zip(axes, HISTORY):
        country = panel[panel["country"] == code]
        ax.plot(median.index, median.values, color=GRAY, linewidth=1.5,
                label="Median of all 28 countries")
        ax.plot(country["year"], country["inflation"], color=BLUE, marker="o",
                markersize=4, label=names[code])
        ax.set_title(names[code], fontsize=13)
        ax.set_ylabel("Inflation (%)")
        ax.set_ylim(bottom=min(0, country["inflation"].min() - 1))
    axes[0].legend(handles=[axes[0].lines[1], axes[0].lines[0]],
                   labels=["This country", "28-country median"],
                   loc="upper left", ncols=1)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Consumer price inflation, 2000–2025", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, -0.01, "Each panel has its own vertical scale. Gaps = no data "
             "(Uzbekistan has no inflation data before 2011).\nSource: World Bank WDI "
             "(FP.CPI.TOTL.ZG).", color=INK_2, fontsize=10)
    fig.tight_layout()
    save(fig, "02_inflation_history.png")


def plot_mae_by_model() -> None:
    overall = pd.read_csv(TABLES_DIR / "oos_metrics_overall.csv")
    labels = {"M0_naive": "M0 Naive (last year's inflation)",
              "M1_baseline": "M1 Baseline (own past inflation)",
              "M1_baseline_FE": "M1 + country fixed effects",
              "M2_commodity": "M2 = M1 + food & energy prices",
              "M3_extended": "M3 = M2 + exchange rate & GDP",
              "M4_ridge": "M4 Ridge (M3 predictors)",
              "M5_boosting": "M5 Gradient boosting (M3 predictors)"}
    overall = overall.iloc[::-1]  # first model at the top
    colours = [BLUE if m == "M2_commodity" else GRAY for m in overall["model"]]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh([labels[m] for m in overall["model"]], overall["MAE"], height=0.6, color=colours)
    for y, value in enumerate(overall["MAE"]):
        ax.text(value + 0.04, y, f"{value:.2f}", va="center", color=INK, fontsize=12)
    ax.set_xlim(0, overall["MAE"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean absolute error (percentage points, lower is better)")
    ax.set_title("Forecast error by model, 2012–2025")
    n = int(overall["n"].iloc[0])
    fig.text(0, -0.03, f"Out-of-sample: each year forecast using only earlier years. "
             f"{n} country-year forecasts.", color=INK_2, fontsize=10)
    save(fig, "03_mae_by_model.png")


def plot_m2_minus_m1_by_country() -> None:
    table = pd.read_csv(TABLES_DIR / "m2_vs_m1_by_country.csv")
    names = country_names()
    table = table.sort_values("MAE_diff_M2_minus_M1", ascending=False)  # best at the bottom
    diff = table["MAE_diff_M2_minus_M1"]
    colours = np.where(diff < 0, BLUE, RED)
    better = int((diff < 0).sum())

    fig, ax = plt.subplots(figsize=(7, 9.5))
    ax.barh([names[c] for c in table["country"]], diff, height=0.65, color=colours)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("M2 error minus M1 error (percentage points)")
    ax.set_title("Do food & energy prices help? By country")
    limit = np.abs(diff).max() * 1.1
    ax.set_xlim(-limit, limit)
    ax.text(-limit * 0.97, -1.3, "◀ M2 better", color=INK, fontsize=12, ha="left")
    ax.text(limit * 0.97, -1.3, "M2 worse ▶", color=INK, fontsize=12, ha="right")
    ax.set_ylim(-1.8, len(table) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)]
    ax.legend(handles, [f"M2 better ({better} countries)",
                        f"M2 worse ({len(table) - better} countries)"],
              loc="lower left", bbox_to_anchor=(0, 0.06))
    fig.text(0, -0.02, "Mean absolute error per country, test years 2012–2025 "
             "(Uzbekistan 2015–2025).\nM1 = own past inflation; M2 = M1 + last year's "
             "food and energy price changes.", color=INK_2, fontsize=10)
    save(fig, "04_m2_minus_m1_by_country.png")


def plot_actual_vs_forecast() -> None:
    predictions = pd.read_csv(TABLES_DIR / "oos_predictions.csv")
    names = country_names()
    fig, axes = plt.subplots(len(FOCUS), 1, figsize=(7, 10.5), sharex=True)
    for ax, code in zip(axes, FOCUS):
        rows = predictions[predictions["country"] == code]
        actual = rows[rows["model"] == "M1_baseline"]
        m1 = rows[rows["model"] == "M1_baseline"]
        m2 = rows[rows["model"] == "M2_commodity"]
        ax.plot(actual["year"], actual["actual"], color=BLUE, marker="o", markersize=6,
                label="Actual inflation")
        ax.plot(m1["year"], m1["forecast"], color=ORANGE, marker="s", markersize=5,
                label="M1 forecast (own past inflation)")
        ax.plot(m2["year"], m2["forecast"], color=AQUA, marker="^", markersize=6,
                label="M2 forecast (+ food & energy)")
        ax.set_title(names[code], fontsize=13)
        ax.set_ylabel("Inflation (%)")
    axes[-1].set_xlabel("Year being forecast")
    fig.suptitle("Actual vs forecast inflation", x=0.02, ha="left",
                 fontsize=15, fontweight="bold", y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.965))
    fig.text(0.02, -0.01, "Each forecast uses only data from earlier years. "
             "Each panel has its own vertical scale.", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "05_actual_vs_forecast.png")


def plot_coefficients() -> None:
    table = pd.read_csv(TABLES_DIR / "coefficients_full_sample.csv")
    labels = {"infl_lag1": "Last year's inflation",
              "food_chg_lag1": "Last year's food price change",
              "energy_chg_lag1": "Last year's energy price change",
              "fx_change_lag1": "Last year's exchange-rate change",
              "gdp_growth_lag1": "Last year's GDP growth"}
    models = [("M2_commodity", "M2", BLUE, "o"), ("M3_extended", "M3", ORANGE, "s")]

    fig, axes = plt.subplots(len(labels), 1, figsize=(7, 9))
    for ax, (variable, label) in zip(axes, labels.items()):
        for y, (model, short, colour, marker) in enumerate(models):
            row = table[(table["model"] == model) & (table["variable"] == variable)]
            if row.empty:  # this variable is not in the model
                ax.text(0, y, f"  not in {short}", va="center", color=GRAY, fontsize=11)
                continue
            row = row.iloc[0]
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2)
            ax.plot(row["coef"], y, marker=marker, color=colour, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                    label=f"{short}")
            ax.annotate(f"{row['coef']:.2f}", (row["coef"], y), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=11, color=INK)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks([0, 1], ["M2", "M3"])
        ax.set_ylim(-0.7, 1.7)
        ax.grid(axis="y", visible=False)
        ax.set_title(label, fontsize=13)
        low, high = ax.get_xlim()
        ax.set_xlim(min(low, -abs(high) * 0.2), high)
    axes[-1].set_xlabel("Coefficient (percentage points of inflation per 1 unit)")
    fig.suptitle("Estimated effects with 95% confidence intervals", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    n = int(table["n_obs"].iloc[0])
    fig.text(0.02, -0.02, f"Fitted on all {n} country-years ({table['years'].iloc[0]}); "
             "standard errors clustered by year.\nAn interval that crosses 0 means the "
             "effect is not statistically clear.", color=INK_2, fontsize=10)
    fig.tight_layout()
    save(fig, "06_coefficients.png")


def main() -> None:
    plot_commodity_indices()
    plot_inflation_history()
    plot_mae_by_model()
    plot_m2_minus_m1_by_country()
    plot_actual_vs_forecast()
    plot_coefficients()


if __name__ == "__main__":
    main()
