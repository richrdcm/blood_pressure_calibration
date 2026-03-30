import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from pyparsing import originalTextFor

from schemas.bp_schema import BPSample
import neurokit2 as nk
from scipy.stats import norm
# Internal imports from your project
from src.extractor.morphology import MorphologyExtractor
from src.extractor.times import TimeExtractor
from utils.preprocessing import Preprocessor
import os


class Visualizator:
    @staticmethod
    def plot_signals(sample: BPSample, clean_sample: BPSample, test_name, show=True):
        ppg_raw = sample.ppg
        ppg_proc = clean_sample.ppg
        ecg_raw = sample.ecg
        ecg_proc = clean_sample.ecg
        has_ecg = ecg_raw is not None

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("PPG Signal (Green Channel)", "ECG Signal (Lead II)")
        )

        # ── SIGNAL TRACES ─────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=sample.ppg_timestamps, y=ppg_raw, name="Raw PPG"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=clean_sample.ppg_timestamps, y=ppg_proc, name="Processed PPG", visible=False
        ), row=1, col=1)

        if has_ecg:
            fig.add_trace(go.Scatter(
                x=sample.ecg_timestamps, y=ecg_raw, name="Raw ECG",
                line=dict(color='#FF0000', width=1)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=clean_sample.ecg_timestamps, y=ecg_proc, name="Processed ECG",
                line=dict(color='#8B0000', width=1.5), visible=False
            ), row=2, col=1)

        # ── JOINT SQI SHADING (both rows) ─────────────────────────────────────────
        ts = clean_sample.ppg_timestamps
        if clean_sample.joint_sqi:
            for i, (s, e) in enumerate(clean_sample.joint_sqi):
                if s >= len(ts) or e >= len(ts):
                    continue
                for row in ([1, 2] if has_ecg else [1]):
                    fig.add_vrect(
                        x0=ts[s], x1=ts[e],
                        fillcolor='rgba(0, 255, 100, 0.18)',
                        opacity=1.0, layer="below", line_width=0,
                        row=row, col=1,
                    )

        # ── JOINT SQI LEGEND ENTRY ────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=14, color='rgba(0, 255, 100, 0.85)', symbol='square'),
            name="PPG + ECG good (joint SQI)",
        ), row=1, col=1)

        # ── LAYOUT ────────────────────────────────────────────────────────────────
        n_traces = 2 + (2 if has_ecg else 0)  # signal traces count
        raw_vis = [True, False] + ([True, False] if has_ecg else []) + [True]
        proc_vis = [False, True] + ([False, True] if has_ecg else []) + [True]

        fig.update_layout(
            title=dict(
                text=f"BP Calibration Dashboard - Patient {sample.patient_id}",
                y=0.95, x=0.5, xanchor='center', yanchor='top',
                font=dict(size=24),
            ),
            updatemenus=[dict(
                buttons=[
                    dict(label="Show Raw",
                         method="update",
                         args=[{"visible": raw_vis},
                               {"title.text": f"Raw Signals - Patient {sample.patient_id}"}]),
                    dict(label="Show Processed",
                         method="update",
                         args=[{"visible": proc_vis},
                               {"title.text": f"Preprocessed Signals - Patient {sample.patient_id}"}]),
                ],
                direction="down", showactive=True,
                x=0.01, y=1.15,
            )],
            template="plotly_dark",
            height=800,
            showlegend=True,
            xaxis2=dict(
                title="Time (milliseconds)",
                rangeslider=dict(visible=True, thickness=0.05),
                rangeselector=dict(
                    buttons=[
                        dict(count=500, label="500ms", step="all", stepmode="backward"),
                        dict(count=1000, label="1s", step="all", stepmode="backward"),
                        dict(count=5000, label="5s", step="all", stepmode="backward"),
                        dict(step="all", label="Full Window"),
                    ],
                    y=1.02,
                ),
            ),
        )

        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)

        # ── SAVE ──────────────────────────────────────────────────────────────────
        base_name = f"plot_signals_{sample.patient_id}"
        output_path = "test/" + test_name + "/" + f"{base_name}.html"
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.html"
            counter += 1
        fig.write_html(output_path)
        print(f"Saved plot to: {output_path}")
        if show:
            fig.show()

    @staticmethod
    def plot_morphology_from_ecg(cleaned_sample: BPSample, ppg_wave: list, time_feature: list, test_name, show=True):
        ppg_proc = np.asarray(cleaned_sample.ppg)
        ecg_proc = np.asarray(cleaned_sample.ecg)
        target_fs = cleaned_sample.target_fs
        ecg_time = np.asarray(cleaned_sample.ecg_timestamps)
        ppg_time = np.asarray(cleaned_sample.ppg_timestamps)

        # 2. Feature Extraction
        vpg, apg, waves = ppg_wave
        r_peaks = TimeExtractor.extract_ecg_r_peaks(ecg_signal=ecg_proc, fs=target_fs)

        # 3. Create Subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("PPG Morphology & PTT Timing", "ECG Reference")
        )

        # --- ECG Plot ---
        fig.add_trace(go.Scatter(x=ecg_time,
                                 y=cleaned_sample.ecg,
                                 name="ECG", line=dict(color='#FF3E3E', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=ecg_time[r_peaks],
            y=ecg_proc[r_peaks], mode='markers', name='R-Peaks',
            marker=dict(color='white', size=8, symbol='circle-open'),
            hovertemplate="<b>ECG R-Peak</b><br>Reference Time: %{x}ms<extra></extra>"
        ), row=2, col=1)

        # --- PPG Signal Trace ---
        fig.add_trace(go.Scatter(
            x=ecg_time, y=ppg_proc, name="PPG", line=dict(color='#00FF00', width=1.5),
            hoverinfo='skip'
        ), row=1, col=1)

        # 4. Add Annotated Markers
        for f in time_feature:
            t_r = ppg_time[f['r_peak']]
            t_a = ppg_time[f['a_idx']]
            print(f"DEBUG: R-time: {t_r} | a-time: {t_a} | Diff: {t_a - t_r}")

        for f in time_feature:
            t_r = ppg_time[f['r_peak']]

            # Vertical Anchor Projection (Electrical Start)
            fig.add_vline(x=t_r, line_dash="dot", line_color="rgba(255, 255, 255, 0.2)", row="all")

            w = next((x for x in waves if x['a'] == f['a_idx']), None)
            if not w: continue

            # A. Morphological Points (a, b, c, d, e) - Permanent Labels
            for key in ['a', 'b', 'c', 'd', 'e']:
                idx = w[key]
                if idx is None: continue

                # Determine specific PTT if applicable (a and e)
                ptt_val = f['ptt_a'] if key == 'a' else (f['ptt_e'] if key == 'e' else None)

                fig.add_trace(go.Scatter(
                    x=[ppg_time[idx]], y=[ppg_proc[idx]],
                    mode='markers+text',
                    text=[key], textposition="top center",
                    marker=dict(size=7, color='yellow' if key != 'e' else 'magenta'),
                    name=f"Wave {key}",
                    hovertemplate=f"<b>Wave {key}</b><br>PTT from R: {ptt_val}ms<br>Global: %{{x}}ms<extra></extra>" if ptt_val else f"<b>Wave {key}</b><br>Global: %{{x}}ms<extra></extra>",
                    showlegend=False
                ), row=1, col=1)

            # B. Systolic Peak (White Star)
            fig.add_trace(go.Scatter(
                x=[ppg_time[w['ppg_peak']]], y=[ppg_proc[w['ppg_peak']]],
                mode='markers',
                marker=dict(symbol='star', size=12, color='white', line=dict(width=1, color='black')),
                name='Systolic Peak',
                hovertemplate=f"<b>Systolic Peak</b><br>PTT_sys: {f['ptt_sys']}ms<br>Global: %{{x}}ms<extra></extra>",
                showlegend=False
            ), row=1, col=1)

            # C. Diastolic Peak (Cyan Star) — use dp index directly from wave dict
            w = next((x for x in waves if x.get('a') == f.get('a_idx')), None)
            if w is not None and w.get('dp') is not None:
                dp_idx = w['dp']
                if dp_idx < len(ppg_time):
                    fig.add_trace(go.Scatter(
                        x=[ppg_time[dp_idx]], y=[ppg_proc[dp_idx]],
                        mode='markers',
                        marker=dict(symbol='star', size=12, color='cyan',
                                    line=dict(width=1, color='black')),
                        name='Diastolic Peak',
                        hovertemplate=(
                            f"<b>Diastolic Peak</b><br>"
                            f"PTT_dia: {f.get('ptt_dia')}ms<br>"
                            f"Global: %{{x}}ms<extra></extra>"
                        ),
                        showlegend=False
                    ), row=1, col=1)

        # 5. Layout Update
        fig.update_layout(
            title=dict(text=f"Hemodynamic Mapping: Patient {cleaned_sample.patient_id}", x=0.5, font=dict(size=22)),
            template="plotly_dark",
            height=800,
            hovermode="closest",
            xaxis2=dict(title="Time (ms)", rangeslider=dict(visible=True, thickness=0.05))
        )
        # ── SAVE ──────────────────────────────────────────────────────────────────
        base_name = f"morphology_{cleaned_sample.patient_id}"
        output_path = "test/" + test_name + "/" + f"{base_name}.html"
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.html"
            counter += 1
        fig.write_html(output_path)
        print(f"Saved plot to: {output_path}")
        if show:
            fig.show()

    @staticmethod
    def plot_morphology_from_apg(clean_sample: BPSample, ppg_wave: list, test_name, show=True):
        fs = clean_sample.target_fs
        time = np.array(clean_sample.ecg_timestamps)
        ppg_proc = clean_sample.ppg
        vpg, apg, waves = ppg_wave

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("PPG (Volume)", "VPG (Velocity)", "APG (Acceleration)")
        )

        # Standard traces
        fig.add_trace(go.Scatter(x=time, y=ppg_proc, name="PPG", line=dict(color='#00FF00')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=vpg, name="VPG", line=dict(color='#FFA500')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=apg, name="APG", line=dict(color='#0074D9')), row=3, col=1)

        # Legend Helpers
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='white', dash='dash'), name='Volume Peak'),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='yellow'), name='Accel. Peak (a)'),
                      row=1, col=1)
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='magenta', dash='dot'), name='Dicrotic Notch (e)'),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='cyan', dash='dot'),
                       name='Diastolic Peak'),
            row=1, col=1)

        for i, w in enumerate(waves):
            # Calculate AGI for this specific beat
            agi = MorphologyExtractor.calculate_aging_index(apg, w)

            # Find the Diastolic Peak
            dp_idx = MorphologyExtractor.find_diastolic_peak(ppg_proc, vpg, w, fs)

            # Vertical lines for key events
            fig.add_vline(x=time[w['ppg_peak']], line_dash="dash", line_color="white", opacity=0.3, row="all")
            fig.add_vline(x=time[w['a']], line_color="yellow", opacity=0.4, row="all")
            if w['e']:
                fig.add_vline(x=time[w['e']], line_dash="dot", line_color="magenta", opacity=0.6, row="all")
            if dp_idx:
                fig.add_vline(x=time[dp_idx], line_dash="dot", line_color="cyan", opacity=0.6, row="all")

                # 3. Calculate Reflection Index (RI) = (H_dp / H_sys) * 100
                h_sys = ppg_proc[w['ppg_peak']]
                h_dp = ppg_proc[dp_idx]
                ri = round((h_dp / h_sys) * 100, 1)

                fig.add_annotation(
                    x=time[dp_idx], y=h_dp,
                    text=f"DP (RI: {ri}%)",
                    showarrow=True, arrowhead=2, row=1, col=1
                )

            if agi is not None:
                fig.add_annotation(
                    x=time[w['ppg_peak']],
                    y=ppg_proc[w['ppg_peak']] + 0.1,  # Offset above peak
                    text=f"AGI: {agi}",
                    showarrow=False,
                    font=dict(color="cyan", size=12),
                    row=1, col=1
                )

            # Annotations for a,b,c,d,e on the APG Row
            for key in ['a', 'b', 'c', 'd', 'e']:
                if w[key] is not None:
                    fig.add_annotation(x=time[w[key]], y=apg[w[key]], text=key, row=3, col=1, showarrow=True)

        for i, w in enumerate(waves):
            try:
                # 1. IDENTIFY BOUNDARIES
                # Start of this beat (a-wave)
                t_start = time[w['a']]
                # Peak of this beat
                t_sys = time[w['ppg_peak']]
                # Notch of this beat (Magenta line)
                t_notch = time[w['e']]

                # Determine the start of the NEXT beat (The "Wall")
                next_w = waves[i + 1] if i + 1 < len(waves) else None
                # Use next beat's start time, or the very end of the signal if it's the last beat
                t_next_beat_start = time[next_w['a']] if next_w else time[-1]

                # 2. FIND DIASTOLIC PEAK (Now strictly bounded by t_next_beat_start)
                dp_idx = MorphologyExtractor.find_diastolic_peak(ppg_proc, w, next_w, fs)

                # 3. DRAW BACKGROUND ZONES
                # Yellow: Systole (From a-wave to Systolic Peak)
                fig.add_vrect(x0=t_start, x1=t_sys, fillcolor="yellow", opacity=0.1, layer="below", line_width=0, row=1,
                              col=1)

                # Magenta: Transition (From Systolic Peak to Dicrotic Notch)
                fig.add_vrect(x0=t_sys, x1=t_notch, fillcolor="magenta", opacity=0.1, layer="below", line_width=0,
                              row=1, col=1)

                # Cyan: Diastole (From Dicrotic Notch to THE START OF THE NEXT BEAT)
                # This is the fix: t_next_beat_start acts as the boundary
                fig.add_vrect(x0=t_notch, x1=t_next_beat_start, fillcolor="cyan", opacity=0.1, layer="below",
                              line_width=0, row=1, col=1)

                # 4. DRAW THE DIASTOLIC PEAK MARKER (If found)
                if dp_idx:
                    fig.add_vline(x=time[dp_idx], line_dash="dot", line_color="cyan", row="all")

            except Exception as e:
                print(f"Skipping beat {i} due to error: {e}")
                continue

            # --- ADD LEGEND HELPERS ---
            # Adding these so the user knows what the colors mean
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color='rgba(255, 255, 0, 0.3)', symbol='square'),
                                 name='Systolic Phase'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color='rgba(255, 0, 255, 0.3)', symbol='square'),
                                 name='Notch/Transition'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color='rgba(0, 255, 255, 0.3)', symbol='square'),
                                 name='Diastolic Phase'), row=1, col=1)


        fig.update_layout(
            title={
                'text': f"PPG Morphology Analysis - Patient {clean_sample.patient_id}",
                'x': 0.5, 'xanchor': 'center', 'font': dict(size=20)
            },
            height=1000, template="plotly_dark", showlegend=True,
                          xaxis3_rangeslider_visible=True)
        # ── SAVE ──────────────────────────────────────────────────────────────────
        base_name = f"morphology_{clean_sample.patient_id}"
        output_path = "test/" + test_name + "/" + f"{base_name}.html"
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.html"
            counter += 1
        fig.write_html(output_path)
        print(f"Saved plot to: {output_path}")
        if show:
            fig.show()

    @staticmethod
    def plot_estimation_performance(sample, calib, test_name, show=True):
        """
        Visualizes the pre-calculated Bayesian Estimation.
        No math performed here.
        """
        # Retrieve Pre-Calculated Stats (No Simulation Here)
        mu_s = calib['sbp_est_mean']
        std_s = calib['sbp_est_std']
        mu_d = calib['dbp_est_mean']
        std_d = calib['dbp_est_std']

        # 3. Create Layout (1 Row, 2 Cols)
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.65, 0.35],
            specs=[[{"type": "xy"}, {"type": "table"}]],
            horizontal_spacing=0.03,
            subplot_titles=("Bayesian Probability Density", "Biophysical Profile")
        )

        # --- COL 1: PROBABILITY CURVES ---
        # Generate X-axis based on the calculated stats
        x_bp = np.linspace(min(40, mu_d - 4 * std_d), max(200, mu_s + 4 * std_s), 500)
        y_sbp = norm.pdf(x_bp, mu_s, std_s)
        y_dbp = norm.pdf(x_bp, mu_d, std_d)

        # SBP Trace (Green)
        fig.add_trace(go.Scatter(
            x=x_bp, y=y_sbp, mode='lines',
            line=dict(color='#00FF00', width=3),
            fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.2)',
            showlegend=False
        ), row=1, col=1)

        # DBP Trace (Blue)
        fig.add_trace(go.Scatter(
            x=x_bp, y=y_dbp, mode='lines',
            line=dict(color='#0074D9', width=3),
            fill='tozeroy', fillcolor='rgba(0, 116, 217, 0.2)',
            showlegend=False
        ), row=1, col=1)

        # Reference Lines
        if sample.bps is not None:
            fig.add_vline(x=sample.bps, line_dash="dash", line_color="white", row=1, col=1,
                          annotation_text="Ref SBP", annotation_position="top right")
        if sample.bpd is not None:
            fig.add_vline(x=sample.bpd, line_dash="dash", line_color="cyan", row=1, col=1,
                          annotation_text="Ref DBP", annotation_position="top right")

        # --- COL 2: METRIC TABLE ---
        def fmt(val, err, unit=""):
            return f"{val:.4f} ± {err:.4f} {unit}"

        # We define 3 columns: Metric Name, Reference Value, Estimated Value
        ref_sbp = f"{sample.bps:.2f} mmHg" if sample.bps is not None else "N/A"
        ref_dbp = f"{sample.bpd:.2f} mmHg" if sample.bpd is not None else "N/A"
        fig.add_trace(go.Table(
            header=dict(values=['<b>Metric</b>', '<b>Reference</b>', '<b>Bayesian Est. (μ ± σ)</b>'],
                        fill_color='#4B0082', font=dict(color='white', size=12), align='left'),
            cells=dict(
                values=[
                    # COL 1: NAMES
                    ['<b>Systolic BP</b>', '<b>Diastolic BP</b>', '---',
                     'Stiffness (α)', 'Elasticity (E0)', 'Wall Ratio (h/d)', 'Intercept (β0)'],

                    # COL 2: REFERENCE (Ground Truth)
                    [ref_sbp, ref_dbp, "", "-", "-", "-", "-"],

                    # COL 3: ESTIMATED
                    [f"<b>{fmt(mu_s, std_s, 'mmHg')}</b>",
                     f"<b>{fmt(mu_d, std_d, 'mmHg')}</b>",
                     "",
                     fmt(calib['alpha_mean'], calib['alpha_std']),
                     fmt(calib['e0_mean_kPa'], calib['e0_std_kPa'], "kPa"),
                     fmt(calib['hd_ratio_mean'], calib['hd_ratio_std']),
                     fmt(calib['beta0_mean'], calib['beta0_std'])]
                ],
                # Coloring: Green row, Blue row, then Grey rows
                fill_color=[['rgba(0,255,0,0.1)', 'rgba(0,116,217,0.1)'] + ['#2c3e50'] * 5],
                font=dict(color='white', size=11), align='left', height=30
            )
        ), row=1, col=2)

        fig.update_layout(
            title=dict(text=f"Patient {sample.patient_id}: Hemodynamic Probabilities", x=0.02),
            template="plotly_dark", height=600, showlegend=False,
            xaxis_title="Blood Pressure (mmHg)", yaxis_title="Probability Density",
            margin=dict(l=50, r=20, t=80, b=50)
        )

        fig.update_xaxes(range=[20, 240], row=1, col=1)
        # Save HTML without overwriting
        base_name = f"plot_estimation_{sample.patient_id}"
        output_path = "test/" + test_name + "/" + f"{base_name}.html"
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.html"
            counter += 1
        fig.write_html(output_path)
        print(f"Saved plot to: {output_path}")
        if show:
            fig.show()

    @staticmethod
    def plot_population_accuracy(df_results, kpis, test_name, show=True):
        """
        Plots the Clinical Validation Dashboard:
        - Row 1: Correlation Plots (Estimated vs Reference)
        - Row 2: Bland-Altman Plots (Difference vs Mean)
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"SBP Correlation (r={kpis['sbp']['Correlation_r']})",
                f"DBP Correlation (r={kpis['dbp']['Correlation_r']})",
                f"SBP Bland-Altman (Bias={kpis['sbp']['Bias']})",
                f"DBP Bland-Altman (Bias={kpis['dbp']['Bias']})"
            ),
            vertical_spacing=0.12, horizontal_spacing=0.1
        )

        # --- ROW 1: CORRELATION PLOTS (Identity Line) ---
        for i, mode in enumerate(['sbp', 'dbp']):
            col = i + 1
            est = df_results[f'{mode}_est_mean']
            ref = df_results[f'ref_{mode}']

            # Scatter Points
            fig.add_trace(go.Scatter(
                x=ref, y=est, mode='markers', name=f'{mode.upper()} Data',
                marker=dict(color='#00FF00' if mode == 'sbp' else '#0074D9', opacity=0.6, size=7),
                showlegend=False
            ), row=1, col=col)

            # Ideal Identity Line (y=x)
            min_val = min(min(ref), min(est))
            max_val = max(max(ref), max(est))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                line=dict(color='white', dash='dash', width=1), name='Ideal', showlegend=False
            ), row=1, col=col)

        # --- ROW 2: BLAND-ALTMAN PLOTS ---
        # Plot (Estimated - Reference) vs (Average of Est and Ref)
        for i, mode in enumerate(['sbp', 'dbp']):
            col = i + 1
            est = df_results[f'{mode}_est_mean']
            ref = df_results[f'ref_{mode}']

            diff = est - ref
            mean = (est + ref) / 2

            bias = kpis[mode]['Bias']
            sd = kpis[mode]['Precision_SD']
            upper_loa = bias + 1.96 * sd
            lower_loa = bias - 1.96 * sd

            # Scatter Difference
            fig.add_trace(go.Scatter(
                x=mean, y=diff, mode='markers',
                marker=dict(color='#FFA500', opacity=0.6, size=6),
                name='Difference', showlegend=False
            ), row=2, col=col)

            # Bias Line
            fig.add_hline(y=bias, line_color='white', line_width=2, row=2, col=col,
                          annotation_text=f"Bias: {bias} mmHg")

            # Limits of Agreement (95% CI)
            fig.add_hline(y=upper_loa, line_dash="dot", line_color="red", row=2, col=col,
                          annotation_text="+1.96 SD")
            fig.add_hline(y=lower_loa, line_dash="dot", line_color="red", row=2, col=col,
                          annotation_text="-1.96 SD")

        # Update Axes Labels
        fig.update_xaxes(title_text="Reference BP (mmHg)", row=1, col=1)
        fig.update_xaxes(title_text="Reference BP (mmHg)", row=1, col=2)
        fig.update_yaxes(title_text="Estimated BP (mmHg)", row=1, col=1)

        fig.update_xaxes(title_text="Mean BP (mmHg)", row=2, col=1)
        fig.update_xaxes(title_text="Mean BP (mmHg)", row=2, col=2)
        fig.update_yaxes(title_text="Error (Est - Ref)", row=2, col=1)

        fig.update_layout(
            title="Population Hemodynamic Accuracy Assessment (AAMI Standard)",
            template="plotly_dark", height=800, width=1200
        )
        base_name = "accuracy_assessment"
        output_path = "test/" + test_name + "/" + f"{base_name}.html"
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.html"
            counter += 1
        fig.write_html(output_path)
        print(f"Saved plot to: {output_path}")
        if show:
            fig.show()

    @staticmethod
    def plot_parameter_histograms(df_results, test_name, show=True):
        """
        Visualizes the distribution of the HIDDEN parameters (Alpha, E0, h/d)
        to ensure they match physiological expectations.
        """
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Stiffness (α)", "Elasticity (E0)", "Wall Ratio (h/d)"))

        params = [
            ('alpha_mean', 'Stiffness', 'green'),
            ('e0_mean_kPa', 'Elasticity (kPa)', 'cyan'),
            ('hd_ratio_mean', 'h/d Ratio', 'magenta')
        ]

        for i, (col, name, color) in enumerate(params):
            fig.add_trace(go.Histogram(
                x=df_results[col], name=name, nbinsx=20,
                marker_color=color, opacity=0.75
            ), row=1, col=i + 1)

        fig.update_layout(
            title="Biophysical Parameter Distribution (Population)",
            template="plotly_dark", height=400
        )
        base_name = "Biophysical_Parameter_Distribution"
        output_path = "test/" + test_name + "/" + f"{base_name}.html"
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.html"
            counter += 1
        fig.write_html(output_path)
        print(f"Saved plot to: {output_path}")
        if show:
            fig.show()

    @staticmethod
    def plot_fiducial_comparison(sample, clean_sample, ppg_wave, colleague_csv_path, colleague_fs=25.0):
        """
        Visually compares fiducial points from a colleague (CSV) with your own.
        Automatically handles Sampling Rate mismatches and time-array regeneration.
        """
        import os
        import pandas as pd
        from utils.preprocessing import Preprocessor

        fs = clean_sample.target_fs

        # Define our arrays clearly
        raw_timestamps = np.array(sample.ppg_timestamps)
        clean_timestamps = np.array(clean_sample.ppg_timestamps)
        ppg_proc = np.array(clean_sample.ppg)
        ppg_time_2, ppg_proc_2 = Preprocessor.clean_ppg_signal_cheby(sample)

        vpg, apg, waves = ppg_wave

        # 4. Load Colleague's Points
        if not os.path.exists(colleague_csv_path):
            print(f"Error: Colleague CSV not found at {colleague_csv_path}")
            return

        df_coll = pd.read_csv(colleague_csv_path)

        # 5. Create Subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(f"Your Fiducial Points ({fs} Hz)",
                            f"Colleague's Fiducial Points (No resampling)")
        )

        # Plot your cleaned signal on BOTH rows so we can compare apples to apples
        fig.add_trace(
            go.Scatter(x=clean_timestamps, y=ppg_proc, name="Processed PPG", line=dict(color='#00FF00', width=1.5)),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=ppg_time_2, y=ppg_proc_2, name="Processed PPG", line=dict(color='#00FF00', width=1.5),
                       showlegend=False), row=2, col=1)

        # --- ROW 1: YOUR POINTS ---
        for w in waves:
            for key in ['a', 'b', 'c', 'd', 'e']:
                idx = w.get(key)
                if idx is not None and idx < len(ppg_proc):
                    fig.add_trace(go.Scatter(
                        x=[clean_timestamps[idx]], y=[ppg_proc[idx]], mode='markers+text',
                        text=[f"<b>{key}</b>"], textposition="top center",
                        marker=dict(size=8, color='yellow' if key != 'e' else 'magenta'),
                        name=f"User {key}", showlegend=False
                    ), row=1, col=1)

            sys_peak = w.get('ppg_peak')
            if sys_peak is not None and sys_peak < len(ppg_proc):
                fig.add_trace(go.Scatter(
                    x=[clean_timestamps[sys_peak]], y=[ppg_proc[sys_peak]], mode='markers+text',
                    text=["<b>sys_peak</b>"], textposition="top center",
                    marker=dict(symbol='star', size=12, color='white', line=dict(width=1, color='black')),
                    showlegend=False
                ), row=1, col=1)

        # --- ROW 2: COLLEAGUE'S POINTS ---
        colleague_cols = ['sp', 'dn', 'dp', 'a', 'b', 'c', 'd', 'e', 'f']

        # We will use a color map to differentiate his major points
        color_map = {
            'on': 'yellow', 'sp': 'white', 'dn': 'magenta', 'dp': 'cyan', 'off': 'red'
        }

        for _, row in df_coll.iterrows():
            for col in colleague_cols:
                if col in row and pd.notna(row[col]):
                    raw_idx = int(row[col])

                    # 1. Ensure index is within the bounds of the RAW sample
                    if 0 <= raw_idx < len(raw_timestamps):
                        # 2. Get the actual real-world time of their point
                        true_time = raw_timestamps[raw_idx]

                        # 3. MAP IT: Find the exact closest index in our new resampled grid
                        clean_idx = (np.abs(clean_timestamps - true_time)).argmin()

                        marker_color = color_map.get(col, 'orange')
                        marker_symbol = 'star' if col in ['sp', 'dp'] else 'circle'
                        marker_size = 10 if col in ['sp', 'dp', 'on', 'dn'] else 7

                        # Plot using the newly mapped clean_idx
                        fig.add_trace(go.Scatter(
                            #x=[clean_timestamps[clean_idx]], y=[ppg_proc[clean_idx]], mode='markers+text',
                            x=[ppg_time_2[raw_idx]], y=[ppg_proc_2[raw_idx]], mode='markers+text',
                            text=[f"<b>{col}</b>"], textposition="bottom center",
                            marker=dict(symbol=marker_symbol, size=marker_size, color=marker_color,
                                        line=dict(width=1, color='black')),
                            name=f"Colleague {col}", showlegend=False
                        ), row=2, col=1)

        # --- LAYOUT FINISHING ---
        fig.update_layout(
            title=dict(text=f"Fiducial Point Alignment Analysis - Patient {sample.patient_id}", x=0.5),
            template="plotly_dark", height=800, hovermode="x unified",
            xaxis2=dict(title="Time (ms)", rangeslider=dict(visible=True, thickness=0.05))
        )

        fig.show()