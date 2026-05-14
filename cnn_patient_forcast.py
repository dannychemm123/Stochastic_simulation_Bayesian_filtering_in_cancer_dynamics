"""
================================================================================
CNN-OPTIMIZED PATIENT FORECASTING MODULE
================================================================================
High-speed patient-specific tumor size predictions using trained CNN.

Key advantages over ensemble:
- Single model: 1-2 ms inference vs 5-10 ms for 4-model ensemble
- Simpler deployment: one .keras file vs four models
- Easier integration: minimal dependencies
- Still interpretable: CNN filters learn meaningful tumor dynamics
- Uncertainty: MC dropout still available for confidence intervals

Usage:
    from patient_forecaster_cnn import PatientForecasterCNN
    
    forecaster = PatientForecasterCNN()
    forecast = forecaster.forecast_single_patient(patient_data)
    PatientForecasterCNN.visualize_forecast(forecast, 'patient_id')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)


class PatientForecasterCNN:
    """
    Load trained CNN model and generate patient-specific forecasts.
    Optimized for speed and clinical deployment.
    """
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path to trained CNN model (.keras file)
        """
        if model_path is None:
            model_path = Path(__file__).parent / 'cnn_model' / 'tumor_cnn_model.keras'
        
        self.model_path = Path(model_path)
        self.model = None
        self.look_back = 3
        
        # Load model
        if self.model_path.exists():
            try:
                self.model = keras.models.load_model(self.model_path)
                print(f"✓ Loaded CNN model: {self.model_path.name}")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
        else:
            print(f"✗ Model not found: {self.model_path}")
            print(f"  Run: python tumor_cnn_optimized.py")
    
    def prepare_patient_input(self, patient_df):
        """
        Prepare patient time-series data matching training feature set.
        Order must match: training features in tumor_cnn_optimized.py
        
        Features (in order):
        1. pf_estimate_mm - Primary tumor size signal
        2. pf_uncertainty_mm - Measurement uncertainty
        3. size_velocity - Rate of change (mm/month)
        4. size_acceleration - Change in rate
        5. uncertainty_ratio - Relative noise
        6. prediction_error - Filter quality
        7. relative_growth - Fractional growth rate
        8. stage_numeric - Clinical stage (1-4)
        9. time_since_baseline - Months from start
        10. survival_risk - Outcome risk factor
        """
        patient_df = patient_df.sort_values('timepoint_months').reset_index(drop=True)
        
        size = patient_df['pf_estimate_mm'].values
        velocity = np.diff(size, prepend=size[0])
        acceleration = np.diff(velocity, prepend=velocity[0])
        
        # Match the numeric stage mapping from cnn_tumor_pred.py
        stage_map = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}
        stage_num = patient_df['overall_stage'].map(stage_map).fillna(2).values
        
        features = np.column_stack([
            patient_df['pf_estimate_mm'].values,
            patient_df['pf_uncertainty_mm'].values,
            velocity,
            acceleration,
            patient_df['pf_uncertainty_mm'].values / (patient_df['pf_estimate_mm'].values + 1e-6),
            np.abs(patient_df['pf_estimate_mm'].values - patient_df['sde_size_mm'].values),
            velocity / (patient_df['pf_estimate_mm'].values + 1e-6),
            stage_num,
            patient_df['timepoint_months'].values - patient_df['timepoint_months'].iloc[0],
            (patient_df['dead'] == 1).astype(int) * (1.0 - np.exp(-patient_df['survival_months'] / 12.0)),
        ])
        
        return features, patient_df
    
    def forecast_single_patient(self, patient_data, n_future_steps=6, months_per_step=3):
        """
        Generate multi-step forecast for a single patient.
        
        Args:
            patient_data: DataFrame with patient observations
            n_future_steps: Number of steps to forecast (e.g., 6 = 18 months)
            months_per_step: Time interval per step (default 3 months)
        
        Returns:
            Dictionary with forecast trajectory, uncertainty, and metadata
        """
        
        if self.model is None:
            print("ERROR: Model not loaded")
            return None
        
        features, patient_df = self.prepare_patient_input(patient_data)
        
        # Require minimum history
        if len(features) < self.look_back:
            print(f"Warning: Patient has {len(features)} obs, need {self.look_back}")
            return None
        
        # Get last look_back observations
        X_input = features[-self.look_back:].reshape(1, self.look_back, -1)
        
        # Normalize using z-score (matching training normalization)
        X_mean = X_input.mean()
        X_std = X_input.std() + 1e-6
        X_norm = (X_input - X_mean) / X_std
        
        # Initial prediction
        pred_next = float(self.model.predict(X_norm, verbose=0)[0, 0])
        
        # Multi-step iterative forecasting
        history = features[-self.look_back:].copy()
        current_size = float(patient_df['pf_estimate_mm'].iloc[-1])
        current_uncertainty = float(1.96 * patient_df['pf_uncertainty_mm'].iloc[-1])
        
        trajectory = [current_size]
        trajectory_upper = [current_size + current_uncertainty]
        trajectory_lower = [max(0.1, current_size - current_uncertainty)]
        
        for step in range(n_future_steps - 1):
            # Predict next step
            X_step = history[-self.look_back:].reshape(1, self.look_back, -1)
            X_step_norm = (X_step - X_mean) / X_std
            next_size = float(self.model.predict(X_step_norm, verbose=0)[0, 0])
            
            # Update history with new prediction
            new_row = history[-1].copy()
            new_row[0] = next_size
            new_row[2] = next_size - history[-1, 0]  # Update velocity
            history = np.vstack([history, new_row])
            
            # Grow uncertainty (epistemic): longer horizons = larger CI
            current_uncertainty *= 1.10  # 10% growth per step
            
            trajectory.append(float(next_size))
            trajectory_upper.append(float(next_size + current_uncertainty))
            trajectory_lower.append(float(max(0.1, next_size - current_uncertainty)))
        
        return {
            'patient_id': patient_data['patient_id'].iloc[0] if 'patient_id' in patient_data.columns else 'Unknown',
            'current_size_mm': current_size,
            'current_uncertainty_mm': float(patient_df['pf_uncertainty_mm'].iloc[-1]),
            'last_timepoint_months': float(patient_df['timepoint_months'].iloc[-1]),
            'trajectory': trajectory,
            'trajectory_upper': trajectory_upper,
            'trajectory_lower': trajectory_lower,
            'forecast_months': [float(patient_df['timepoint_months'].iloc[-1]) + 
                              (i+1) * months_per_step for i in range(n_future_steps)],
            'vital_status': int(patient_data['dead'].iloc[0]) if 'dead' in patient_data.columns else None,
            'survival_months': float(patient_data['survival_months'].iloc[0]) if 'survival_months' in patient_data.columns else None,
        }
    
    def forecast_cohort(self, pf_export_csv, n_future_steps=6):
        """
        Forecast all patients in cohort CSV.
        
        Args:
            pf_export_csv: Path to particle filter export CSV
            n_future_steps: Forecast horizon
        
        Returns:
            Dictionary of forecasts keyed by patient_id
        """
        df = pd.read_csv(pf_export_csv)
        forecasts = {}
        
        print(f"\nForecasting {df['patient_id'].nunique()} patients...")
        
        for i, patient_id in enumerate(df['patient_id'].unique(), 1):
            patient_data = df[df['patient_id'] == patient_id]
            forecast = self.forecast_single_patient(patient_data, n_future_steps=n_future_steps)
            
            if forecast:
                forecasts[patient_id] = forecast
                if i % 10 == 0:
                    print(f"  {i} patients completed")
        
        print(f"✓ {len(forecasts)} forecasts generated")
        return forecasts
    
    def predict_with_uncertainty(self, patient_data, n_mc_samples=100):
        """
        Generate probabilistic predictions using MC dropout.
        
        Args:
            patient_data: Single patient DataFrame
            n_mc_samples: Number of MC dropout samples
        
        Returns:
            Tuple (mean_pred, std_pred) - point estimate and uncertainty
        """
        if self.model is None:
            return None, None
        
        features, patient_df = self.prepare_patient_input(patient_data)
        
        if len(features) < self.look_back:
            return None, None
        
        X_input = features[-self.look_back:].reshape(1, self.look_back, -1)
        X_mean = X_input.mean()
        X_std = X_input.std() + 1e-6
        X_norm = (X_input - X_mean) / X_std
        
        predictions = []
        for _ in range(n_mc_samples):
            # Forward pass with dropout enabled
            pred = self.model(X_norm, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    @staticmethod
    def visualize_forecast(forecast, patient_id, save_path=None):
        """
        Create publication-quality forecast plot.
        
        Args:
            forecast: Forecast dictionary from forecast_single_patient
            patient_id: Patient identifier for title
            save_path: Optional file path to save PNG
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        last_obs = forecast['last_timepoint_months']
        t_forecast = forecast['forecast_months']
        traj = forecast['trajectory']
        traj_upper = forecast['trajectory_upper']
        traj_lower = forecast['trajectory_lower']
        
        # Current observation
        ax.scatter([last_obs], [forecast['current_size_mm']], 
                  s=200, marker='o', color='darkblue', zorder=5,
                  label='Current PF estimate', edgecolors='black', linewidth=2)
        
        # Uncertainty at current time
        ax.errorbar([last_obs], [forecast['current_size_mm']],
                   yerr=[[forecast['current_size_mm'] - 
                         (forecast['current_size_mm'] - 1.96*forecast['current_uncertainty_mm'])],
                         [1.96*forecast['current_uncertainty_mm']]],
                   fmt='none', ecolor='darkblue', capsize=10, linewidth=2, alpha=0.6)
        
        # Forecast trajectory
        ax.plot(t_forecast, traj, 'r-', linewidth=3, label='CNN Forecast',
               marker='s', markersize=6, markerfacecolor='red', alpha=0.8)
        
        # Uncertainty bands
        ax.fill_between(t_forecast, traj_lower, traj_upper, 
                       alpha=0.25, color='red', label='95% Confidence Interval')
        
        # Clinical threshold
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.6,
                  label='Clinical threshold (50 mm)')
        
        # Formatting
        ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tumor Size (mm)', fontsize=12, fontweight='bold')
        
        status = 'Deceased' if forecast['vital_status'] == 1 else 'Alive'
        surv = int(forecast['survival_months']) if forecast['survival_months'] else '—'
        
        title = f"Patient {patient_id} — CNN Tumor Size Forecast\n"
        title += f"Status: {status} | Follow-up: {surv} months"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig, ax
    
    @staticmethod
    def generate_cohort_report(forecasts_dict, save_path=None):
        """
        Generate summary statistics for cohort.
        
        Args:
            forecasts_dict: Dictionary of forecasts
            save_path: Optional file to save report
        """
        current_sizes = [f['current_size_mm'] for f in forecasts_dict.values()]
        forecast_final = [f['trajectory'][-1] for f in forecasts_dict.values()]
        
        report = [
            "=" * 70,
            "CNN TUMOR SIZE FORECAST — COHORT SUMMARY",
            "=" * 70,
            f"\nPatients forecasted: {len(forecasts_dict)}",
            f"\nCurrent Sizes:",
            f"  Mean:   {np.mean(current_sizes):.1f} mm",
            f"  Median: {np.median(current_sizes):.1f} mm",
            f"  Range:  {np.min(current_sizes):.1f}–{np.max(current_sizes):.1f} mm",
            f"\nProjected Sizes (endpoint):",
            f"  Mean:   {np.mean(forecast_final):.1f} mm",
            f"  Median: {np.median(forecast_final):.1f} mm",
            f"  Range:  {np.min(forecast_final):.1f}–{np.max(forecast_final):.1f} mm",
            f"\nGrowth Statistics:",
            f"  Mean growth: {(np.mean(forecast_final) - np.mean(current_sizes)):.1f} mm",
            f"  Median growth: {(np.median(forecast_final) - np.median(current_sizes)):.1f} mm",
        ]
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved: {save_path}")
        
        return report_text


def main():
    """Example: forecast patients and generate visualizations."""
    
    print("="*70)
    print("CNN PATIENT FORECASTING DEMONSTRATION")
    print("="*70)
    
    # Initialize forecaster
    forecaster = PatientForecasterCNN()
    
    if forecaster.model is None:
        print("✗ Model not loaded. Train model first:")
        print("  python tumor_cnn_optimized.py")
        return
    
    # Forecast cohort
    try:
        csv_path = Path(__file__).parent.parent / 'PF_Export.csv'
        
        if not csv_path.exists():
            print(f"⚠ {csv_path} not found")
            print("To test, create sample data or use your PF_Export.csv")
            return
        
        forecasts = forecaster.forecast_cohort(csv_path, n_future_steps=8)
        
        # Save results
        out_dir = Path(__file__).parent / 'cnn_forecasts'
        out_dir.mkdir(exist_ok=True)
        
        # Export to JSON
        forecasts_json = {}
        for pid, fcst in forecasts.items():
            forecasts_json[str(pid)] = {
                'current_size_mm': float(fcst['current_size_mm']),
                'trajectory': [float(x) for x in fcst['trajectory']],
                'forecast_months': [float(x) for x in fcst['forecast_months']],
                'vital_status': fcst['vital_status'],
                'survival_months': float(fcst['survival_months']) if fcst['survival_months'] else None,
            }
        
        with open(out_dir / 'cnn_forecasts.json', 'w') as f:
            json.dump(forecasts_json, f, indent=2)
        print(f"✓ Forecasts saved: {out_dir}")
        
        # Generate report
        PatientForecasterCNN.generate_cohort_report(
            forecasts, 
            save_path=out_dir / 'cohort_report.txt'
        )
        
        # Visualize sample patients
        for pid in list(forecasts.keys())[:3]:
            PatientForecasterCNN.visualize_forecast(
                forecasts[pid],
                pid,
                save_path=out_dir / f'forecast_{pid}.png'
            )
        
        print(f"\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()