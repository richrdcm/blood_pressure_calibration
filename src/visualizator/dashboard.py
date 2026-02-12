import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from schemas.bp_schema import BPSample
from utils.preprocessing import Preprocessor
from src.extractor.morphology import MorphologyExtractor


class Visualizator:
    @staticmethod
    def plot_signals(sample: BPSample):
        fs = sample.fs

        # --- SAFETY CHECK: Force identical lengths ---
        # We use the length of the timestamp as the "Master" length
        master_len = len(sample.timestamp)

        # Prepare PPG (ensure length matches timestamp)
        ppg_raw = sample.ppg[:master_len]
        ppg_proc = Preprocessor.clean_signal(ppg_raw, fs)

        # Prepare ECG (ensure length matches timestamp)
        ecg_raw = None
        ecg_proc = None
        if sample.ecg:
            # Trim or pad ECG to match timestamp length
            ecg_raw = sample.ecg[:master_len]
            ecg_proc = Preprocessor.process_ecg(ecg_raw, fs)

        # Create Figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=("PPG Signal (Green Channel)", "ECG Signal (Lead II)"))

        # --- PPG TRACES (Row 1) ---
        # Use master_len to ensure x and y match
        fig.add_trace(go.Scatter(x=sample.timestamp[:master_len], y=ppg_raw, name="Raw PPG"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sample.timestamp[:master_len], y=ppg_proc, name="Processed PPG", visible=False),
                      row=1, col=1)

        # --- ECG TRACES (Row 2) ---
        if ecg_raw is not None:
            fig.add_trace(go.Scatter(x=sample.timestamp[:master_len], y=ecg_raw, name="Raw ECG",
                                     line=dict(color='#FF0000', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=sample.timestamp[:master_len], y=ecg_proc, name="Processed ECG",
                                     line=dict(color='#8B0000', width=1.5), visible=False), row=2, col=1)

        # --- DROP-DOWN SELECTOR (Raw vs Processed) ---
        dropdown_buttons = [
            dict(label="Show Raw",
                 method="update",
                 args=[{"visible": [True, False, True, False]},
                       {"title": f"Raw Signal Analysis - Patient {sample.patient_id}"}]),
            dict(label="Show Processed",
                 method="update",
                 args=[{"visible": [False, True, False, True]},
                       {"title": f"Filtered/Normalized Analysis - Patient {sample.patient_id}"}]),
        ]

        # --- UPDATE LAYOUT WITH SLIDER ---
        fig.update_layout(
            # 1. SET GLOBAL TITLE
            title={
                'text': f"BP Calibration Dashboard - Patient {sample.patient_id}",
                'y': 0.95,  # Vertical position (0 to 1)
                'x': 0.5,  # Horizontal position (0.5 is center)
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },

            # 2. UPDATEMENUS (Ensure 'title' here also updates the global title)
            updatemenus=[dict(
                buttons=[
                    dict(label="Show Raw",
                         method="update",
                         args=[{"visible": [True, False, True, False]},
                               {"title.text": f"Raw Signal Analysis - Patient {sample.patient_id}"}]),
                    # Updates the global title text
                    dict(label="Show Processed",
                         method="update",
                         args=[{"visible": [False, True, False, True]},
                               {"title.text": f"Filtered/Normalized Analysis - Patient {sample.patient_id}"}]),
                ],
                direction="down",
                showactive=True,
                x=0.01,  # Moves the menu to the top left
                y=1.15
            )],

            template="plotly_dark",
            height=800,
            showlegend=True,

            # X-Axis Slider configuration
            xaxis2=dict(
                title="Time (milliseconds)",  # Updated Label
                rangeslider=dict(visible=True, thickness=0.05),
                rangeselector=dict(
                    buttons=list([
                        dict(count=500, label="500ms", step="all", stepmode="backward"),
                        dict(count=1000, label="1s", step="all", stepmode="backward"),
                        dict(count=5000, label="5s", step="all", stepmode="backward"),
                        dict(step="all", label="Full Window")
                    ]),
                    y=1.02
                )
            )
        )

        # Ensure the subplots don't overlap with the slider
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)

        fig.show()

    @staticmethod
    def plot_morphology(sample: BPSample):
        fs = sample.fs
        time = np.array(sample.timestamp)
        ppg_proc = Preprocessor.clean_signal(sample.ppg, fs)
        vpg, apg, waves = MorphologyExtractor.extract_waves(ppg_proc, fs)


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

        for w in waves:
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


        fig.update_layout(height=1000, template="plotly_dark", showlegend=True,
                          xaxis3_rangeslider_visible=True)
        fig.show()

    @staticmethod
    def plot_physics_validation(sample, calibration_results, waves, ppg_proc, apg):
        time = np.array(sample.timestamp)
        fs = sample.fs

        # Extract Calibration Coefficients
        alpha = calibration_results['alpha_stiffness']
        beta_0 = calibration_results['baseline_log_ptt']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("PPG with Physical Projections", "APG with Belief Windows"))

        # Plot Raw PPG and APG
        fig.add_trace(go.Scatter(x=time, y=ppg_proc, name="PPG", line=dict(color='#00FF00')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=apg, name="APG", line=dict(color='#0074D9')), row=2, col=1)

        for w in waves:
            # 1. Physics Prediction: ln(PTT) = Beta1 * BPS + Beta0
            # Reconstruct PTT from our Beta coefficients
            beta_1 = -alpha / 2
            pred_log_ptt = beta_1 * sample.bps + beta_0
            pred_ptt_ms = np.exp(pred_log_ptt)

            # 2. Calculate the "Predicted Time" relative to ECG R-peak
            t_r = time[w['r_peak']]
            t_pred = t_r + (pred_ptt_ms / 1000)

            # 3. Draw the "Belief Window" on the APG (Search Area)
            # We assume a +/- 30ms window of uncertainty
            fig.add_vrect(
                x0=t_pred - 0.03, x1=t_pred + 0.03,
                fillcolor="white", opacity=0.15, line_width=0,
                layer="below", row=2, col=1, name="Physiological Belief"
            )

            # 4. Draw the Prediction Line (The 'Wall')
            fig.add_vline(x=t_pred, line_dash="dash", line_color="orange",
                          annotation_text="Phys. Pred", row="all")

            # 5. Draw Actual APG Points (a, b, c, d, e)
            for key in ['a', 'b', 'c', 'd', 'e']:
                if w.get(key):
                    fig.add_trace(go.Scatter(
                        x=[time[w[key]]], y=[apg[w[key]]],
                        mode='markers+text', text=key, textposition="top center",
                        marker=dict(color='yellow', size=8), showlegend=False
                    ), row=2, col=1)

        fig.update_layout(height=800, template="plotly_dark",
                          title=f"Patient {sample.patient_id} Validation (Alpha: {alpha})")
        fig.show()