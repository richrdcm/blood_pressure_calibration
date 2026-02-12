from utils.data_loader import DataLoader
from src.visualizator.dashboard import Visualizator


def main():
    loader = DataLoader()
    file_type = "mcs"
    file_type = "uci"
    file_path = "datasets/raw/mcs/2026_02_05_09_26_44.csv"
    file_path = "datasets/raw/uci/Part_1.csv"

    # 1. Load Data
    try:
        samples = loader.load_from_csv(file_path, file_type, max_duration_sec=10000)

        if samples:
            target_sample = samples[0]

            # 2. Visualize with interactive selector
            # Visualizator.plot_signals(target_sample)
            Visualizator.plot_morphology(target_sample)

    except Exception as e:
        print(f"Error loading data: {e}")


if __name__ == "__main__":
    main()