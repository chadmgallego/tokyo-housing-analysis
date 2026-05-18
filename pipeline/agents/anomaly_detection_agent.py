import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import json

OUTPUT_DIR = Path(__file__).parent / "figs"
OUTPUT_DIR.mkdir(exist_ok=True)

class AnomalyDetectionAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_features = [
            "rent", "area", "floor", "building_age",
            "building_size", "distance_to_nearest_station",
            "avg_distance_to_stations", "management_fee"
        ]
    
    # ── PREPARE DATA ──────────────────────────────────────────
    def prepare_data(self) -> np.ndarray:
        df_model = self.df[self.numeric_features + ["floor_plan"]].copy()
        df_model["management_fee"] = df_model["management_fee"].fillna(
            df_model["management_fee"].median()
        )
        top_plans = df_model["floor_plan"].value_counts().nlargest(8).index
        df_model["floor_plan"] = df_model["floor_plan"].where(
            df_model["floor_plan"].isin(top_plans), "Other"
        )
        df_encoded = pd.get_dummies(df_model, columns=["floor_plan"], drop_first=True)
        scaler = StandardScaler()
        df_encoded[self.numeric_features] = scaler.fit_transform(df_encoded[self.numeric_features])
        return df_encoded.values
    
    # ── ISOLATION FOREST (wrapped for looping) ────────────────
    def run_isolation_forest(self, X: np.ndarray, contamination: float, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        iso = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state
        )
        labels = iso.fit_predict(X)   # 1 = normal, -1 = anomaly
        scores = iso.decision_function(X)
        return labels, scores

    def run(self, contamination=0.05, random_state=42, output_dir=OUTPUT_DIR) -> Dict[str, Any]:
        # ── PREPARE & PCA  ───────────────────────────────────────────────
        X = self.prepare_data()
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        # ── RUN ISOLATION FOREST & PLOT ───────────────────────────
        labels, scores = self.run_isolation_forest(X, contamination=contamination, random_state=random_state)
        normal  = labels == 1
        anomaly = labels == -1
        
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        ax.scatter(X_pca[normal, 0],  X_pca[normal, 1],
           c="steelblue", alpha=0.35, s=14, edgecolors="none", zorder=2)
        ax.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1],
           c="crimson", alpha=0.85, s=22, edgecolors="white",
           linewidths=0.25, zorder=3)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
                label=f"Normal  (n={normal.sum():,})", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                label=f"Anomaly (n={anomaly.sum():,})", markersize=8),
        ]
        
        ax.legend(handles=legend_elements, fontsize=9,
          facecolor="#1e1e2e", edgecolor="#444", labelcolor="white")
        ax.set_title(f"IsolationForest Anomalies — Tokyo Housing  (contamination={contamination})",
             color="white", fontsize=12, pad=8)
        
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})", color="white", fontsize=9)
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")

        fig.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(output_dir / f"anomaly_pca_{timestamp}.png", dpi=150,
            facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        
        final_df = self.df.copy()
        final_df["anomaly_label"] = labels
        final_df["anomaly_score"] = scores
        
        results = {
            "status": "VALID",
            "message": f"Anomaly detection completed with contamination={contamination}.",
            "number_of_anomalies": int(anomaly.sum()),
            "data": final_df
        }
        self._write_log(results, timestamp)
        return results
    
    def _write_log(self, results: Dict[str, Any], timestamp: str) -> None:
        # Metadata-only payload — DataFrame is excluded to keep the log
        # small and human-readable (a full DataFrame could be thousands of rows)
        log = {
            "agent": "AnomalyDetectionAgent",
            "timestamp": timestamp,
            "status": results["status"],
            "message": results.get("message", ""),
            "number_of_anomalies": results.get("number_of_anomalies", 0),
            "output_files": {
                "pca_plot": f"anomaly_pca_{timestamp}.png"
            }
        }
        # Timestamp in the filename gives each run its own log file — preserves history
        log_path = Path(__file__).parent / "logs" / f"anomaly_log_{timestamp}.json"
        # Create the logs/ directory if it doesn't exist yet
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False preserves Japanese characters in station/address fields
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Saved to {log_path}")
