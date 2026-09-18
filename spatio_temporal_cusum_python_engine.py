#!/usr/bin/env python3
"""
Spatio-Temporal CUSUM (Cumulative Sum) Anomaly Detector for USGS Earthquake Data
----------------------------------------------------------------------------------
This script fetches live USGS GeoJSON earthquake feeds, groups events into spatial
grid cells, calculates seismic energy release via the Gutenberg-Richter relation,
and runs sequential CUSUM change detection to flag abnormal seismic swarms.
"""

import argparse
import json
import math
import os
import sys
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

def fetch_usgs_data(feed="all_month"):
    """Fetches real-time USGS GeoJSON earthquake feed."""
    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"
    print(f"[*] Fetching live USGS GeoJSON feed ({feed})...")
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[!] Error fetching data from USGS: {e}")
        sys.exit(1)

    records = []
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"]
        props = feature["properties"]
        
        if props["mag"] is None:
            continue

        records.append({
            "id": feature["id"],
            "title": props.get("title", "Unknown"),
            "lon": coords[0],
            "lat": coords[1],
            "depth": coords[2],
            "mag": float(props["mag"]),
            "time": pd.to_datetime(props["time"], unit="ms"),
            "place": props.get("place", "Unknown")
        })

    df = pd.DataFrame(records)
    print(f"[+] Successfully loaded {len(df)} seismic events from {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}.")
    return df

def run_spatio_temporal_cusum(df, grid_deg=2.0, time_step="1D", k_factor=0.5, h_factor=3.5, min_events=4):
    """
    Executes Spatio-Temporal CUSUM anomaly detection.
    
    Energy Proxy: E ~ 10^(1.5 * M)
    Control Statistic: S_t = max(0, S_{t-1} + (X_t - mu_0 - k))
    Alarm Condition: S_t > h
    """
    # Spatial binning
    df["lat_bin"] = (df["lat"] // grid_deg) * grid_deg
    df["lon_bin"] = (df["lon"] // grid_deg) * grid_deg
    
    # Gutenberg-Richter equivalent energy proxy
    df["energy"] = 10 ** (1.5 * df["mag"])

    anomalies = []
    grid_summaries = []

    for (lat, lon), group in df.groupby(["lat_bin", "lon_bin"]):
        if len(group) < min_events:
            continue

        # Resample energy across uniform temporal steps
        ts = group.set_index("time")["energy"].resample(time_step).sum().fillna(0)
        if len(ts) < 4:
            continue

        mu_0 = ts.mean()
        sigma_0 = ts.std() if ts.std() > 0 else 1e-6

        # Allowance parameter k and decision boundary h
        k = k_factor * sigma_0
        h = h_factor * sigma_0

        S = np.zeros(len(ts))
        for t in range(1, len(ts)):
            S[t] = max(0, S[t - 1] + (ts.iloc[t] - mu_0 - k))
            
            if S[t] > h:
                anomalies.append({
                    "grid_lat": lat + grid_deg / 2,
                    "grid_lon": lon + grid_deg / 2,
                    "timestamp": ts.index[t].strftime('%Y-%m-%d %H:%M:%S'),
                    "energy_spike": float(ts.iloc[t]),
                    "baseline_mean": float(mu_0),
                    "cusum_score": float(S[t]),
                    "threshold_h": float(h),
                    "event_count": len(group),
                    "max_mag": float(group["mag"].max())
                })

        grid_summaries.append({
            "lat": lat,
            "lon": lon,
            "max_cusum": float(S.max()),
            "count": len(group)
        })

    return pd.DataFrame(anomalies), pd.DataFrame(grid_summaries)

def generate_spatial_plot(df, anomalies_df, output_path="cusum_spatial_map.png"):
    """Generates a spatial plot of earthquakes and highlighted CUSUM anomaly zones."""
    plt.figure(figsize=(14, 8))
    
    # Scatter epicenters
    scatter = plt.scatter(
        df["lon"], df["lat"], 
        c=df["mag"], cmap="YlOrRd", 
        s=df["mag"]**2.5 * 1.8, 
        alpha=0.5, edgecolors="none", label="Earthquake Events"
    )
    plt.colorbar(scatter, label="Magnitude (M)")

    # Highlight CUSUM Anomaly Hotspots
    if not anomalies_df.empty:
        plt.scatter(
            anomalies_df["grid_lon"], anomalies_df["grid_lat"], 
            s=350, facecolors="none", edgecolors="cyan", 
            linewidth=2.5, linestyle="--", label="CUSUM Anomaly Zone"
        )

    plt.title("Spatio-Temporal CUSUM Earthquake Swarm & Energy Anomalies", fontsize=14, fontweight="bold")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[+] Diagnostic map saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal CUSUM Earthquake Anomaly Detector")
    parser.add_argument("--feed", default="all_month", help="USGS feed (all_day, all_week, all_month, 4.5_month)")
    parser.add_argument("--grid", type=float, default=2.0, help="Grid size in degrees (default: 2.0)")
    parser.add_argument("--timestep", default="1D", help="Temporal binning step (1D, 12H, 6H)")
    parser.add_argument("--k", type=float, default=0.5, help="Slack factor k (multiplied by std dev)")
    parser.add_argument("--h", type=float, default=3.5, help="Threshold factor h (multiplied by std dev)")
    parser.add_argument("--output", default="cusum_anomalies.csv", help="CSV export filename")

    args = parser.parse_args()

    # 1. Fetch
    df = fetch_usgs_data(args.feed)
    
    # 2. Compute CUSUM
    anomalies_df, grid_df = run_spatio_temporal_cusum(
        df, grid_deg=args.grid, time_step=args.timestep, k_factor=args.k, h_factor=args.h
    )

    # 3. Report Results
    if not anomalies_df.empty:
        print(f"\n[ALERT] Detected {len(anomalies_df)} spatial CUSUM anomalies exceeding threshold h={args.h}*std!")
        print(anomalies_df[["grid_lat", "grid_lon", "timestamp", "cusum_score", "max_mag"]].head(10).to_string(index=False))
        anomalies_df.to_csv(args.output, index=False)
        print(f"\n[+] Full anomaly report exported to '{args.output}'.")
    else:
        print("\n[INFO] No CUSUM anomalies detected matching current criteria.")

    # 4. Save Visualization
    try:
        generate_spatial_plot(df, anomalies_df)
    except Exception as e:
        print(f"[!] Could not generate map image: {e}")

if __name__ == "__main__":
    main()