
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from logistic import gradient_descent, predict, compute_cost


def main():
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"

    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load the prepared dataset
    data = np.load(results_dir / "prepared_data.npz")

    X_R_train = data["X_R_train"]
    X_R_val = data["X_R_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    X_Q_train = data["X_Q_train"]
    X_Q_val = data["X_Q_val"]

    # --------------------------------------------------
    # Part A: Compare learning rates using Set R
    # --------------------------------------------------
    learning_rates = [0.01, 0.1, 1, 10]
    cost_histories = {}
    valid_rates = []

    for alpha in learning_rates:
        print(f"\nTesting learning rate alpha = {alpha}")

        theta, history = gradient_descent(
            X_R_train, y_train,
            alpha=alpha,
            iterations=2000
        )

        cost_histories[alpha] = history

        # Strictly decreasing means every next cost is smaller
        strictly_decreasing = np.all(np.diff(history) < 0)

        print("Initial cost:", history[0])
        print("Final cost:", history[-1])
        print("Cost strictly decreasing:", strictly_decreasing)

        if strictly_decreasing:
            valid_rates.append(alpha)

    # Plot cost histories
    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha, history in cost_histories.items():
        ax.plot(history, label=f"alpha = {alpha}")

    ax.set_title("Logistic Regression: Learning Rate Comparison")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary cross-entropy cost")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "learning_rate_comparison.png",
        dpi=150
    )
    plt.show()
    plt.close(fig)

    # Select the largest learning rate with strictly decreasing cost
    if not valid_rates:
        raise RuntimeError(
            "No learning rate had strictly decreasing cost. "
            "Inspect the cost histories before continuing."
        )

    best_alpha = max(valid_rates)
    print("\nSelected learning rate:", best_alpha)

    # --------------------------------------------------
    # Part B: Train final Set R and Set Q models
    # --------------------------------------------------
    models = {}

    for set_name, X_train, X_val in [
        ("R", X_R_train, X_R_val),
        ("Q", X_Q_train, X_Q_val)
    ]:
        theta, history = gradient_descent(
            X_train, y_train,
            alpha=best_alpha,
            iterations=2000
        )

        train_predictions = predict(X_train, theta)
        val_predictions = predict(X_val, theta)

        train_accuracy = np.mean(train_predictions == y_train)
        val_accuracy = np.mean(val_predictions == y_val)

        models[set_name] = theta

        print(f"\n--- Logistic Regression Set {set_name} ---")
        print("Theta:", theta)
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("Final cost:", history[-1])

        np.savez(
            models_dir / f"logistic_{set_name}.npz",
            theta=theta,
            alpha=best_alpha,
            iterations=2000
        )

    # Save a record of learning-rate results
    with open(
        results_dir / "learning_rate_results.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha", "initial_cost", "final_cost",
            "strictly_decreasing"
        ])

        for alpha, history in cost_histories.items():
            writer.writerow([
                alpha,
                history[0],
                history[-1],
                bool(np.all(np.diff(history) < 0))
            ])

    print("\nSaved logistic model parameters in:", models_dir)
    print("Saved learning-rate results in:", results_dir)


if __name__ == "__main__":
    main()
