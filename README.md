# mHM/mRM Explorer: Interactive Hydrological Model Visualization

An interactive web application built with **Shiny for Python** for visualizing and evaluating outputs from the **multiscale Hydrologic Model (mHM)** and **multiscale Routing Model (mRM)**.

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Shiny](https://img.shields.io/badge/Shiny-for%20Python-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Overview

**mHM/mRM Explorer** bridges the gap between complex gridded hydrological model outputs and point-based observation data. It allows researchers and hydrologists to evaluate model performance across large domains through a seamless, reactive interface.

The application handles large-scale datasets efficiently by combining **NetCDF** for spatial visualization and **Zarr** for high-speed time series extraction.

## ✨ Key Features

### 🌍 Interactive Spatial Mapping
- **Gridded Visualization:** Dynamic rendering of hydrological states and fluxes (e.g., discharge, soil moisture, ET).
- **Time Navigation:** Explore model steps through an intuitive time-slider with real-time map updates.
- **Basemap Selection:** Switch between Satellite, Topographic, and Standard OpenStreetMap views.

### 📊 Model Evaluation & Metrics
- **Performance Points:** Measurement gauging stations are displayed as vector points, color-coded by performance metrics:
  - **KGE** (Kling-Gupta Efficiency)
  - **Correlation** ($r$)
  - **Bias** ($\beta$)
  - **Variability** ($\gamma$)
- **Pixel Inspection:** Click anywhere on the raster grid to instantly extract and plot the time series for that specific pixel.

### 📈 Station Diagnostic Dashboard
Selecting a gauging station opens a comprehensive analysis suite:
- **Value Boxes:** Instant view of key statistical performance markers.
- **Seasonal Regime:** Comparison of long-term observed vs. simulated monthly averages.
- **Annual Flows:** Visualization of yearly water balance and trends.
- **Flow Duration Curves (FDC):** Diagnostic plots with logarithmic scaling to assess model accuracy across high and low flow regimes.

## 🛠 Technical Stack

- **Frontend:** [Shiny for Python](https://shiny.posit.co/py/)
- **Mapping:** `ipyleaflet` for interactive GeoJSON and Raster overlays.
- **Data Handling:** `xarray` & `dask` (NetCDF processing), `Zarr` (High-speed random access).
- **Visualization:** `matplotlib` with optimized `Agg` backend.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- [mHM](https://www.ufz.de/mhm/) model outputs in NetCDF/Zarr format.

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/mhm-explorer.git](https://github.com/yourusername/mhm-explorer.git)
   cd mhm-explorer
