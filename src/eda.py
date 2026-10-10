
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_raw


def main():
    # Load original measurements for EDA
    df = load_raw()

    # Remove columns that are completely empty
    df = df.dropna(axis=1, how="all")

    # Required columns
    required = ["Output (S)", "Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
    df = df[required].dropna().reset_index(drop=True)

    # Output folder for plots
    project_dir = Path(__file__).resolve().parent.parent
    figures_dir = project_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Phase currents over first 2000 rows
    sample = df.iloc[:2000]
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(sample.index, sample["Ia"], label="Ia", linewidth=0.8)
    ax.plot(sample.index, sample["Ib"], label="Ib", linewidth=0.8)
    ax.plot(sample.index, sample["Ic"], label="Ic", linewidth=0.8)

    # Shade consecutive regions where Output (S) == 1
    fault = sample["Output (S)"].to_numpy() == 1
    start = None

    for i, is_fault in enumerate(fault):
        if is_fault and start is None:
            start = i
        elif not is_fault and start is not None:
            ax.axvspan(start, i - 1, color="red", alpha=0.12)
            start = None

    if start is not None:
        ax.axvspan(start, len(fault) - 1, color="red", alpha=0.12)

    ax.set_title("Three-Phase Currents: First 2000 Rows")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Current")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "currents_first_2000.png", dpi=150)
    plt.show()
    plt.close(fig)

    # Plot 2: Ia vs Ib, colored by fault label
    fig, ax = plt.subplots(figsize=(7, 6))

    no_fault = df["Output (S)"] == 0
    has_fault = df["Output (S)"] == 1

    ax.scatter(
        df.loc[no_fault, "Ia"],
        df.loc[no_fault, "Ib"],
        label="No fault (0)",
        alpha=0.5,
        s=12
    )
    ax.scatter(
        df.loc[has_fault, "Ia"],
        df.loc[has_fault, "Ib"],
        label="Fault (1)",
        alpha=0.5,
        s=12
    )

    ax.set_title("Phase Current Ia vs Ib")
    ax.set_xlabel("Ia")
    ax.set_ylabel("Ib")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "ia_vs_ib.png", dpi=150)
    plt.show()
    plt.close(fig)

    # Plot 3: Ia distribution for fault and no-fault cases
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df.loc[no_fault, "Ia"],
        bins=40,
        alpha=0.6,
        label="No fault (0)"
    )
    ax.hist(
        df.loc[has_fault, "Ia"],
        bins=40,
        alpha=0.6,
        label="Fault (1)"
    )

    ax.set_title("Distribution of Ia by Fault Status")
    ax.set_xlabel("Ia")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "ia_histogram.png", dpi=150)
    plt.show()
    plt.close(fig)

    print("\nEDA complete. Plots saved in:", figures_dir)


if __name__ == "__main__":
    main()
