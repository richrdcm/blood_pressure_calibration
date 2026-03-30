from sympy import intervals
from utils.data_loader import DataLoader
from src.visualizator.dashboard import Visualizator
from src.analysis.evaluator import HemodynamicEvaluator
from utils.preprocessing import Preprocessor
from src.extractor.morphology import MorphologyExtractor
from src.extractor.times import TimeExtractor
from utils.preprocessing import remap_fiducial_indices
from src.calibrator.SubjectCalibrator import SubjectCalibrator, FeatureExporter
import os


def main():
    loader = DataLoader()
    file_type = "mcs"
    #file_type = "uci"
    #file_path = "datasets/raw/mcs/fihmi-2026-03-03T09-30.csv"
    #file_path = "datasets/raw/mcs/2026_02_05_09_26_44.csv"
    file_path = "datasets/raw/mcs/ECG_Calib_64hz/fihmi/fihmi_calib_macAddress_6c1deb04a9ce_pt"
    bp_ref_path = "datasets/raw/mcs/ECG_Calib_64hz/fihmi/bp_ref.csv"
    #file_path = "datasets/raw/uci/Part"
    #test_name = "uci_e_as_diastolic_peak"
    test_name = "fihmi"
    os.makedirs("test/" + test_name, exist_ok=True)


    # 1. Load Data
    #samples = loader.load_from_csv(file_path, file_type, max_duration_msec=None)
    samples = loader.load_from_csv(file_path, file_type,
                                   index_from=14, index_to=528,  #528,
                                   each_file_is_own_patient=False,
                                   bp_ref_path=bp_ref_path)
    # UCI Dataset
    """
    samples = loader.load_from_csv(
        file_path=file_path,
        dataset_type=file_type,
        index_from=1, index_to=4
    )
    """

    if not samples:
        print("No samples loaded.")
        return

    # 2. Preprocess
    processor = Preprocessor()
    cleaned_samples = processor.clean_signals(samples)

    # 3. Morphology
    morphology = MorphologyExtractor()
    ppg_waves = morphology.extract_samples_waves(samples=cleaned_samples)

    # 4. Timing
    times = TimeExtractor()
    time_features = times.compute_ptts(ppg_derivatives=ppg_waves, samples=cleaned_samples)

    for sample, cleaned_sample, ppg_wave, time_feature in zip(samples, cleaned_samples, ppg_waves, time_features):


        Visualizator.plot_signals(sample=sample,
                                  clean_sample=cleaned_sample,
                                  test_name=test_name,
                                  show=False)

        Visualizator.plot_morphology_from_ecg(cleaned_sample=cleaned_sample,
                                              ppg_wave=ppg_wave,
                                              time_feature=time_feature,
                                              test_name=test_name,
                                              show=False)


    # 5. Calibration
    df_feat = FeatureExporter.extract_training_data(samples=samples,
                                                    ppg_waves=ppg_waves,
                                                    time_features=time_features)

    calibrations = SubjectCalibrator().calibrate_patients(df_feat)

    # 6. Evaluation
    df_results = HemodynamicEvaluator.process_population(samples=cleaned_samples,
                                                         ppg_waves=ppg_waves,
                                                         time_features=time_features,
                                                         calibrations=calibrations)

    # Visualization
    for sample, cleaned_sample, ppg_wave, time_feature, calibration in zip(samples, cleaned_samples, ppg_waves, time_features, calibrations):
        if calibration:
            Visualizator.plot_estimation_performance(sample, calibration,
                                                     test_name=test_name,
                                                     show=False)

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


    # Remap the feducial point at 25 to indexes in 125hz
    #new_ppg_wave = remap_fiducial_indices(colleague_csv_path='datasets/processed/mcs/2026_02_05_09_26_44_fiducials.csv',
    #                                      raw_timestamps=sample.ppg_timestamps,
    #                                      clean_timestamps=clean_sample.ppg_timestamps,
    #                                      old_ppg_wave=ppg_wave
    #                                      )
    #new_time_feature = times.compute_ptts(ppg_derivatives=[new_ppg_wave], samples=[clean_sample])

    #Visualizator.plot_morphology_from_apg(clean_sample= clean_sample,
    #                                      ppg_wave=new_ppg_wave)
    #Visualizator.plot_estimation_performance(sample, new_ppg_wave, new_time_feature[0])



    # B. Calculate Accuracy Metrics
    kpis = HemodynamicEvaluator.compute_kpis(df_results)
    HemodynamicEvaluator.save_results(df_results=df_results, output_dir="test/" + test_name)

    print("\n=== SBP Performance ===")
    print(kpis['sbp'])
    print("\n=== DBP Performance ===")
    print(kpis['dbp'])

    # C. Visualize Population Dashboards
    Visualizator.plot_population_accuracy(df_results, kpis,
                                          test_name=test_name,
                                          show=True)
    Visualizator.plot_parameter_histograms(df_results,
                                          test_name=test_name,
                                          show=True)



if __name__ == "__main__":
    main()