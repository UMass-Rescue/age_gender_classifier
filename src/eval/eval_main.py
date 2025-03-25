import pandas as pd
from src.eval.analyze_labeled_data_raw import main as raw_main
from src.eval.score_labeled_data import main as process_scores
from eval.transform_scores import main as transform_outputs


def main() -> pd.DataFrame:
    """Run full evaluation pipeline."""
    # raw_main()
    # ts = process_scores()
    ts = "2025-03-24T20:25:56.093675" # TEST
    df = transform_outputs(ts)
    
    # chart scores and true/predicted labels
    return df


if __name__ == "__main__":
    main()
