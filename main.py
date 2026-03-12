from utils.data_loader import DataLoader
from src.visualizator.dashboard import Visualizator
from src.analysis.evaluator import HemodynamicEvaluator
from utils.preprocessing import Preprocessor
from src.extractor.morphology import MorphologyExtractor
from src.extractor.times import TimeExtractor
from utils.preprocessing import remap_fiducial_indices


def main():
    loader = DataLoader()
    file_type = "mcs"
    #file_type = "uci"
    file_path = "datasets/raw/mcs/2026_02_05_09_26_44.csv"
    #file_path = "datasets/raw/uci/Part_1.csv"

    # 1. Load Data
    samples = loader.load_from_csv(file_path, file_type, max_duration_sec=100)

    if samples:
        processor = Preprocessor()
        cleaned_samples = processor.clean_signals(samples)

        morphology = MorphologyExtractor()
        ppg_waves = morphology.extract_samples_waves(cleaned_samples)

        times = TimeExtractor()
        time_features = times.compute_ptts(ppg_derivatives=ppg_waves, samples=cleaned_samples)

        # Take the first
        sample = samples[0]
        clean_sample = cleaned_samples[0]
        ppg_wave = ppg_waves[0]
        time_feature = time_features[0]

        # 2. Visualize with interactive selector
        #Visualizator.plot_signals(sample=sample,
        #                          clean_sample=clean_sample)
        #Visualizator.plot_morphology_from_ecg(cleaned_sample=clean_sample,
        #                                      ppg_wave=ppg_wave,
        #                                      time_feature=time_feature)
        #Visualizator.plot_morphology_from_apg(clean_sample= clean_sample,
        #                                      ppg_wave=ppg_wave)

        #Visualizator.plot_fiducial_comparison(sample=sample,
        #                                      clean_sample=clean_sample,
        #                                      ppg_wave=ppg_wave,
        #                                      colleague_csv_path='datasets/processed/mcs/2026_02_05_09_26_44_fiducials.csv',
        #                                      colleague_fs=sample.ppg_fs)


        # Calibration
        # Visualizator.plot_estimation_performance(sample, ppg_wave, time_feature)


        # Remap the feducial point at 25 to indexes in 125hz
        new_ppg_wave = remap_fiducial_indices(colleague_csv_path='datasets/processed/mcs/2026_02_05_09_26_44_fiducials.csv',
                                              raw_timestamps=sample.ppg_timestamps,
                                              clean_timestamps=clean_sample.ppg_timestamps,
                                              old_ppg_wave=ppg_wave
                                              )
        new_time_feature = times.compute_ptts(ppg_derivatives=[new_ppg_wave], samples=[clean_sample])

        Visualizator.plot_morphology_from_apg(clean_sample= clean_sample,
                                              ppg_wave=new_ppg_wave)
        Visualizator.plot_estimation_performance(sample, new_ppg_wave, new_time_feature[0])

        # A. Process all patients and save CSV
        df_results = HemodynamicEvaluator.process_population(samples=samples,
                                                             ppg_waves=[new_ppg_wave],
                                                             time_features=new_time_feature)

        # B. Calculate Accuracy Metrics
#        kpis = HemodynamicEvaluator.compute_kpis(df_results)

#        print("\n=== SBP Performance ===")
#        print(kpis['sbp'])
#        print("\n=== DBP Performance ===")
#        print(kpis['dbp'])

        # C. Visualize Population Dashboards
#        Visualizator.plot_population_accuracy(df_results, kpis)
#        Visualizator.plot_parameter_histograms(df_results)


if __name__ == "__main__":
    main()