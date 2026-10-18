"""Download and normalize the UCI Heart Disease dataset."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
DATA_PATH = RAW_DIR / "heart_disease.csv"


def normalize_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert UCI target values 0..4 to binary 0/1 and normalize column names."""
    df = frame.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    target_candidates = [c for c in df.columns if c in {"num", "target", "heart_disease", "diagnosis"}]
    if not target_candidates:
        raise ValueError("Could not find target column (expected num, target, heart_disease, or diagnosis).")
    target_col = target_candidates[0]
    df = df.rename(columns={target_col: "target"})
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"])
    df["target"] = (df["target"] > 0).astype(int)
    return df


def download_data(output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Fetch dataset ID 45 from UCI using ucimlrepo and write a CSV."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features
    targets = dataset.data.targets
    if targets is None or targets.shape[1] == 0:
        raise ValueError("UCI response did not include a target column.")
    combined = pd.concat([features, targets.iloc[:, 0].rename("target")], axis=1)
    combined = normalize_target(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
    print(combined["target"].value_counts(dropna=False).sort_index())
    return combined


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return download_data(path)
    return normalize_target(pd.read_csv(path))


if __name__ == "__main__":
    download_data()
