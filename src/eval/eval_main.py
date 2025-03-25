from src.eval.analyze_labeled_data_raw import main as raw_main
from src.eval.score_labeled_data import main as process_scores
from src.eval.analyze_labeled_data_scored import main as scored_main


def main():
    """Run all evaluation scripts."""
    raw_main()
    # process_scores()
    # scored_main()


if __name__ == "__main__":
    main()
