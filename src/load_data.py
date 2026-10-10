
from pathlib import Path
import pandas as pd


def load_raw():
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "data" / "detect_dataset.csv"
 

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Check the CSV filename and location."
        )

    df = pd.read_csv(data_path)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    load_raw()
