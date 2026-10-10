from pathlib import Path
import csv
import numpy as np
import joblib
from sklearn.svm import SVC


def main():
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load the prepared dataset
    data = np.load(results_dir / "prepared_data.npz")

    X_train = data["X_R_train"]
    X_val = data["X_R_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    C_values = [0.1, 1, 10, 100]
    results = []

    # --------------------------------------------------
    # Linear SVM
    # --------------------------------------------------
    print("\n========== LINEAR SVM ==========")

    best_linear_model = None
    best_linear_accuracy = -1
    best_linear_C = None

    for C in C_values:
        model = SVC(kernel="linear", C=C)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        support_vectors = model.support_vectors_.shape[0]

        print(
            f"C={C:<5} "
            f"Validation accuracy={accuracy:.4f} "
            f"Support vectors={support_vectors}"
        )

        results.append([
            "linear", C, "", accuracy, support_vectors
        ])

        if accuracy > best_linear_accuracy:
            best_linear_accuracy = accuracy
            best_linear_model = model
            best_linear_C = C

    joblib.dump(
        best_linear_model,
        models_dir / "svm_linear.joblib"
    )

    print("\nBest linear SVM C:", best_linear_C)
    print("Best linear validation accuracy:", best_linear_accuracy)

    # --------------------------------------------------
    # RBF SVM
    # --------------------------------------------------
    print("\n========== RBF SVM ==========")

    best_rbf_model = None
    best_rbf_accuracy = -1
    best_rbf_C = None
    best_rbf_gamma = None

    for C in C_values:
        for gamma in [0.1, 1]:
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            accuracy = np.mean(predictions == y_val)
            support_vectors = model.support_vectors_.shape[0]

            print(
                f"C={C:<5} gamma={gamma:<4} "
                f"Validation accuracy={accuracy:.4f} "
                f"Support vectors={support_vectors}"
            )

            results.append([
                "rbf", C, gamma, accuracy, support_vectors
            ])

            if accuracy > best_rbf_accuracy:
                best_rbf_accuracy = accuracy
                best_rbf_model = model
                best_rbf_C = C
                best_rbf_gamma = gamma

    joblib.dump(
        best_rbf_model,
        models_dir / "svm_rbf.joblib"
    )

    print("\nBest RBF SVM C:", best_rbf_C)
    print("Best RBF SVM gamma:", best_rbf_gamma)
    print("Best RBF validation accuracy:", best_rbf_accuracy)

    # Save all validation results
    with open(
        results_dir / "svm_validation_results.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "kernel",
            "C",
            "gamma",
            "validation_accuracy",
            "total_support_vectors"
        ])
        writer.writerows(results)

    print("\nSVM models saved in:", models_dir)
    print("Validation results saved in:", results_dir)


if __name__ == "__main__":
    main()