import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import io, base64, os, json, threading
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import ipyleaflet as L

# ============================================================
# 1. DATA LOADING & PATHS (Unified v1.0)
# ============================================================
# Automatic environment detection
SERVER_PATH = "/mnt/data/kolar.v/dyje_bakalarka"

if os.path.exists(SERVER_PATH):
    DATA_PATH = SERVER_PATH
    print("Environment: CzechGlobe Server")
else:
    DATA_PATH = os.path.dirname(__file__)
    print(f"Environment: Local PC ({DATA_PATH})")

def load_data():
    def p(name): return os.path.join(DATA_PATH, name)
    
    # Mapping Data (NetCDF) - Lazy loading with chunks for stability
    ds_mhm = xr.open_dataset(p("mHM_Fluxes_States.nc"), chunks={"time": 1}) if os.path.exists(p("mHM_Fluxes_States.nc")) else None
    ds_mrm = xr.open_dataset(p("mRM_Fluxes_States.nc"), chunks={"time": 1}) if os.path.exists(p("mRM_Fluxes_States.nc")) else None
    
    # Time-series Data (Zarr) - Optimized for interactive plots
    ds_mhm_z = None
    if os.path.exists(p("mHM_data.zarr")):
        try:
            ds_mhm_z = xr.open_dataset(p("mHM_data.zarr"), engine="zarr", chunks={})
        except Exception as e:
            print(f"Warning: mHM_data.zarr load failed: {e}")

    ds_mrm_z = None
    if os.path.exists(p("mRM_data.zarr")):
        try:
            ds_mrm_z = xr.open_dataset(p("mRM_data.zarr"), engine="zarr", chunks={})
        except Exception as e:
            print(f"Warning: mRM_data.zarr load failed: {e}")
    
    # Statistics & Metadata
    summary_path = p("evaluated_data/summary_stats.csv")
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    
    return ds_mhm, ds_mrm, ds_mhm_z, ds_mrm_z, summary

# Initialize all datasets
ds_mhm_map, ds_mrm_map, ds_mhm_stats, ds_mrm_stats, STATIONS_SUMMARY = load_data()

# Global UI helpers & Variable extraction
all_vars = {}
if ds_mhm_map: all_vars.update({f"mhm:{v}": f"mHM: {v}" for v in ds_mhm_map.data_vars})
if ds_mrm_map: all_vars.update({f"mrm:{v}": f"mRM: {v}" for v in ds_mrm_map.data_vars})

# Metadata for Map and Slider
time_values = pd.to_datetime(ds_mhm_map.time.values) if ds_mhm_map is not None else []
if ds_mhm_map is not None:
    bounds = ((float(ds_mhm_map.lat.min()), float(ds_mhm_map.lon.min())), 
              (float(ds_mhm_map.lat.max()), float(ds_mhm_map.lon.max())))
else:
    bounds = ((48.5, 15.0), (49.5, 16.5)) # Fallback

# ============================================================
# 2. UI DEFINITION
# ============================================================
app_ui = ui.page_fillable(
    ui.tags.style("""
        html, body { margin: 0; padding: 0; height: 100vh; overflow: hidden; font-family: 'Segoe UI', sans-serif; }
        #map_a { height: 100vh !important; width: 100vw !important; position: absolute; }
        .floating-menu { position: absolute; background: rgba(255,255,255,0.92); padding: 15px; border-radius: 8px; z-index: 1000; border: 1px solid #ccc; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .section-title { font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 12px; font-size: 11px; color: #555; text-transform: uppercase; }
        .legend-row { display: flex; width: 100%; height: 14px; border: 1px solid #666; margin-top: 6px; border-radius: 2px; }
        .legend-tick { flex: 1; height: 100%; }
        .legend-labels { display: flex; justify-content: space-between; font-size: 10px; margin-top: 3px; color: #555; }
    """),
    ui.div(
        output_widget("map_a"),
        # Left Panel (Timeline & Legend)
        ui.div(
            ui.div("mHM/mRM Explorer", style="font-size: 18px; font-weight: bold; margin-bottom: 10px;"),
            ui.div("Timeline Control", class_="section-title"),
            ui.input_slider("time_idx", None, 0, len(time_values)-1, 0, animate=False, width="100%"),
            ui.output_text("current_date"),
            ui.hr(),
            ui.div("Metric Legend (Stations)", class_="section-title"),
            ui.output_ui("legend_ui"), # LEGENDA SEKCE
            class_="floating-menu", style="top: 20px; left: 20px; width: 320px;"
        ),
        # Right Panel (Layer Switcher)
        ui.div(
            ui.div("Layer Settings", class_="section-title"), # PŘEPÍNAČ VRSTEV SEKCE
            ui.input_select("active_layer", "Raster Variable:", all_vars, selected=list(all_vars.keys())[0] if all_vars else None),
            ui.input_checkbox("show_stations", "Show Gauging Stations", True),
            ui.input_select("marker_metric", "Color Stations by:", {"kge": "KGE", "r": "Correlation", "beta": "Bias", "gamma": "Variability"}),
            class_="floating-menu", style="top: 20px; right: 20px; width: 260px;"
        )
    )
)

# ============================================================
# 3. SERVER LOGIC
# ============================================================
def server(input, output, session):
    selected_st_id = reactive.Value(None)
    last_click_coords = reactive.Value(None)
    map_click_blocked = reactive.Value(False)

    # --- 3A. RASTER DATA (Mapping Logic) ---
    @render_widget
    def map_a():
        m = L.Map(center=[48.9, 16.0], zoom=9, scroll_wheel_zoom=True, zoom_control=False)
        m.add_layer(L.TileLayer())
        
        # Raster Layer
        img_overlay = L.ImageOverlay(url="", bounds=bounds, opacity=0.7)
        m.add_layer(img_overlay)
        
        # Station Layer
        geo_layer = L.GeoJSON(data={}, point_style={"radius": 8, "fillOpacity": 1, "color": "white", "weight": 1.5})
        m.add_layer(geo_layer)

        @reactive.Effect
        def _update_map():
            # Raster update
            try:
                prefix, var = input.active_layer().split(":")
                ds = ds_mhm_map if prefix == "mhm" else ds_mrm_map
                data = ds[var].isel(time=input.time_idx())
                
                fig = plt.figure(figsize=(6, 6), dpi=80); ax = fig.add_axes([0, 0, 1, 1])
                is_q = "q" in var.lower()
                v_min, v_max = float(data.min()), float(data.max())
                norm = colors.LogNorm(vmin=max(v_min, 0.001), vmax=v_max) if is_q else colors.Normalize(vmin=v_min, vmax=v_max)
                
                data.plot.imshow(ax=ax, cmap="Blues" if is_q else "viridis", norm=norm, add_colorbar=False, add_labels=False)
                ax.axis('off')
                buf = io.BytesIO(); plt.savefig(buf, format='png', transparent=True); plt.close(fig)
                img_overlay.url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            except Exception as e: print(f"Raster error: {e}")

            # Vector update (Stations)
            stations_path = os.path.join(DATA_PATH, "evaluated_data/stations.json")
            if input.show_stations() and os.path.exists(stations_path):
                with open(stations_path, "r") as f: geo_data = json.load(f)
                metric = input.marker_metric()
                for feat in geo_data['features']:
                    val = feat['properties'].get(metric, 0)
                    feat['properties']['style'] = {"fillColor": _get_color(metric, val)}
                geo_layer.data = geo_data
                geo_layer.visible = True
            else: geo_layer.visible = False

        # Interaction Handlers
        def on_st_click(feature, **kwargs):
            map_click_blocked.set(True)
            selected_st_id.set(str(int(float(feature['properties']['id']))))
            ui.modal_show(ui.modal(ui.output_ui("st_dashboard"), size="xl", easy_close=True))
            threading.Timer(0.5, lambda: map_click_blocked.set(False)).start()

        geo_layer.on_click(on_st_click)

        def on_map_click(**kwargs):
            if kwargs.get('type') == 'click' and not map_click_blocked():
                last_click_coords.set(kwargs.get('coordinates'))
                ui.modal_show(ui.modal(ui.output_plot("raster_ts_plot"), size="l", easy_close=True))
        
        m.on_interaction(on_map_click)
        return m

    # --- 3B. RASTER DATA (Time-series Plots) ---
    @render.plot
    def raster_ts_plot():
        if not last_click_coords(): return None
        lat, lon = last_click_coords()
        prefix, var = input.active_layer().split(":")
        ds = ds_mhm_stats if prefix == "mhm" else ds_mrm_stats
        data = ds[var].sel(lat=lat, lon=lon, method="nearest").compute()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        data.plot(ax=ax, color="#2c3e50")
        ax.set_title(f"Grid Cell Analysis: {lat:.3f}N, {lon:.3f}E")
        return fig

    # --- 3C. VECTOR DATA (Station Dashboard & Plots) ---
    @render.ui
    def st_dashboard():
        sid = selected_st_id()
        row = STATIONS_SUMMARY[STATIONS_SUMMARY['ID'].astype(str) == sid].iloc[0]
        return ui.div(
            ui.h3(f"Gauging Station: {sid}"),
            ui.layout_column_wrap(
                ui.value_box("KGE", f"{row['KGE']:.3f}"),
                ui.value_box("Correlation", f"{row['Correlation']:.3f}"),
                ui.value_box("Bias", f"{row['Bias']:.3f}"),
                ui.value_box("Variability", f"{row['Variability']:.3f}"),
                width=1/4, fill=False
            ),
            ui.hr(),
            ui.layout_column_wrap(
                ui.card(ui.card_header("Seasonal Cycle"), ui.output_plot("plot_seasonal")),
                ui.card(ui.card_header("FDC Curve"), ui.output_plot("plot_fdc")),
                width=1/2
            )
        )

    @render.plot
    def plot_seasonal():
        path = os.path.join(DATA_PATH, "evaluated_data/plots", f"{selected_st_id()}_seasonal.csv")
        df = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df['Qobs'], 'k-o', label="Observed")
        ax.plot(df.index, df['Qrouted'], 'r--o', label="Simulated")
        ax.legend(); ax.grid(True, alpha=0.3); return fig

    @render.plot
    def plot_fdc():
        path = os.path.join(DATA_PATH, "evaluated_data/plots", f"{selected_st_id()}_fdc.csv")
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['exceedance'], df['Qobs'], 'k-', label="Observed")
        ax.plot(df['exceedance'], df['Qsim'], 'r--', label="Simulated")
        ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3); return fig

    # --- 3D. LEGEND & HELPERS ---
    def _get_color(metric, val):
        pal = ["#7f0000", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#1a9850"]
        if metric in ["kge", "r"]: return pal[int(np.clip(val * 7, 0, 7))] if val > 0 else pal[0]
        return "#1a9850" if 0.85 <= val <= 1.15 else ("#cc00ff" if val < 0.85 else "#d7191c")

    @render.ui
    def legend_ui():
        metric = input.marker_metric()
        pal = ["#7f0000", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#1a9850"] if metric in ["kge", "r"] else ["#cc00ff", "#1a9850", "#d7191c"]
        return ui.div(
            ui.div(*[ui.div(class_="legend-tick", style=f"background: {c};") for c in pal], class_="legend-row"),
            ui.div(ui.span("Poor / Low"), ui.span("Medium"), ui.span("High / Perfect", style="float: right;"), class_="legend-labels")
        )

    @render.text
    def current_date():
        return f"Current Date: {time_values[input.time_idx()].strftime('%Y-%m-%d')}"

app = App(app_ui, server)