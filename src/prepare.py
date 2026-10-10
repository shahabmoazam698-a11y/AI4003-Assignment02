
from pathlib import Path
import numpy as np
from load_data import load_raw


def prepare_data():
    # 1. Load the original dataset
    df = load_raw()

    # 2. Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # 3. Select the required columns
    feature_names = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
    target_name = "Output (S)"
    required_columns = feature_names + [target_name]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    # 4. Keep only the required columns and remove incomplete rows
    df = df[required_columns].dropna().copy()

    # 5. Separate input features (X) and target (y)
    X = df[feature_names].to_numpy(dtype=float)
    y = df[target_name].to_numpy(dtype=int)

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    # 6. Create the exact required random split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))

    n_train = int(0.6 * len(df))
    n_val = int(0.2 * len(df))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train_raw = X[train_idx]
    X_val_raw = X[val_idx]
    X_test_raw = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # 7. Set R: six original features
    X_R = X.copy()

    # 8. Set Q: six original features and their squares
    # Squares are calculated before standardization.
    X_Q = np.column_stack([X, X ** 2])

    # 9. Standardize using TRAINING statistics only
    def standardize(train, val, test):
        mean = train.mean(axis=0)
        std = train.std(axis=0, ddof=0)

        # Avoid division by zero if any feature is constant
        std[std == 0] = 1.0

        train_scaled = (train - mean) / std
        val_scaled = (val - mean) / std
        test_scaled = (test - mean) / std

        return train_scaled, val_scaled, test_scaled, mean, std

    # Standardize Set R
    X_R_train, X_R_val, X_R_test, mean_R, std_R = standardize(
        X_R[train_idx], X_R[val_idx], X_R[test_idx]
    )

    # Standardize Set Q
    X_Q_train, X_Q_val, X_Q_test, mean_Q, std_Q = standardize(
        X_Q[train_idx], X_Q[val_idx], X_Q[test_idx]
    )

    # 10. Save prepared arrays and preprocessing statistics
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / "prepared_data.npz",
        X_R_train=X_R_train,
        X_R_val=X_R_val,
        X_R_test=X_R_test,
        X_Q_train=X_Q_train,
        X_Q_val=X_Q_val,
        X_Q_test=X_Q_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        mean_R=mean_R,
        std_R=std_R,
        mean_Q=mean_Q,
        std_Q=std_Q,
        feature_names=np.array(feature_names)
    )

    # 11. Display a summary
    print("\n--- DATA PREPARATION SUMMARY ---")
    print("Rows retained:", len(df))
    print("Empty columns removed:", ["Unnamed: 7", "Unnamed: 8"])
    print("Set R features:", X_R.shape[1])
    print("Set Q features:", X_Q.shape[1])
    print("Training rows:", len(train_idx))
    print("Validation rows:", len(val_idx))
    print("Testing rows:", len(test_idx))
    print("Training fault counts:", np.unique(y_train, return_counts=True))
    print("Validation fault counts:", np.unique(y_val, return_counts=True))
    print("Testing fault counts:", np.unique(y_test, return_counts=True))
    print("Saved to:", results_dir / "prepared_data.npz")

    return {
        "X_R_train": X_R_train,
        "X_R_val": X_R_val,
        "X_R_test": X_R_test,
        "X_Q_train": X_Q_train,
        "X_Q_val": X_Q_val,
        "X_Q_test": X_Q_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "mean_R": mean_R,
        "std_R": std_R,
        "mean_Q": mean_Q,
        "std_Q": std_Q
    }


if __name__ == "__main__":
    prepare_data()
