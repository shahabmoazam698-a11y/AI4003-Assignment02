
from pathlib import Path
import csv
import numpy as np
import joblib


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def logistic_predict(X, theta):
    X_with_intercept = np.column_stack(
        (np.ones(X.shape[0]), X)
    )
    probabilities = sigmoid(X_with_intercept @ theta)
    return (probabilities >= 0.5).astype(int)


def evaluate_model(name, y_true, y_pred):
    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    accuracy = np.mean(y_true == y_pred)

    print(f"\n--- {name} ---")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[tn, fp], [fn, tp]])
    print("True negatives (TN):", tn)
    print("False positives (FP):", fp)
    print("False negatives (FN):", fn)
    print("True positives (TP):", tp)
    print("Missed faults (FN):", fn)

    return [name, accuracy, tn, fp, fn, tp]


def main():
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    models_dir = results_dir / "models"

    # Load the prepared data.
    # The test set is used only for this final evaluation.
    data = np.load(results_dir / "prepared_data.npz")

    X_R_test = data["X_R_test"]
    X_Q_test = data["X_Q_test"]
    y_test = data["y_test"]

    # Load manually trained logistic regression parameters.
    logistic_R_data = np.load(models_dir / "logistic_R.npz")
    logistic_Q_data = np.load(models_dir / "logistic_Q.npz")

    theta_R = logistic_R_data["theta"]
    theta_Q = logistic_Q_data["theta"]

    # Load the selected SVM models.
    linear_svm = joblib.load(models_dir / "svm_linear.joblib")
    rbf_svm = joblib.load(models_dir / "svm_rbf.joblib")

    results = []

    # Evaluate all four final models.
    pred_logistic_R = logistic_predict(X_R_test, theta_R)
    results.append(
        evaluate_model("Logistic Regression Set R", y_test, pred_logistic_R)
    )

    pred_logistic_Q = logistic_predict(X_Q_test, theta_Q)
    results.append(
        evaluate_model("Logistic Regression Set Q", y_test, pred_logistic_Q)
    )

    pred_linear_svm = linear_svm.predict(X_R_test)
    results.append(
        evaluate_model("Best Linear SVM", y_test, pred_linear_svm)
    )

    pred_rbf_svm = rbf_svm.predict(X_R_test)
    results.append(
        evaluate_model("Best RBF SVM", y_test, pred_rbf_svm)
    )

    # Save the final test results.
    output_path = results_dir / "test_results.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "test_accuracy",
            "TN",
            "FP",
            "FN",
            "TP"
        ])
        writer.writerows(results)

    print("\nFinal test results saved to:", output_path)


if __name__ == "__main__":
    main()
