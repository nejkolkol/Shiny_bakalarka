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
# 1. DATA LOADING & PATHS
# ============================================================
SERVER_PATH = "/mnt/data/kolar.v/dyje_bakalarka"
if os.path.exists(SERVER_PATH):
    DATA_PATH = SERVER_PATH
    print("Environment: CzechGlobe Server")
else:
    DATA_PATH = os.path.dirname(__file__)
    print(f"Environment: Local PC ({DATA_PATH})")

def load_data():
    def p(name): return os.path.join(DATA_PATH, name)
    ds_mhm = xr.open_dataset(p("mHM_Fluxes_States.nc"), chunks={"time": 1}) if os.path.exists(p("mHM_Fluxes_States.nc")) else None
    ds_mrm = xr.open_dataset(p("mRM_Fluxes_States.nc"), chunks={"time": 1}) if os.path.exists(p("mRM_Fluxes_States.nc")) else None
    ds_mhm_z = xr.open_dataset(p("mHM_data.zarr"), engine="zarr", chunks={}) if os.path.exists(p("mHM_data.zarr")) else None
    ds_mrm_z = xr.open_dataset(p("mRM_data.zarr"), engine="zarr", chunks={}) if os.path.exists(p("mRM_data.zarr")) else None
    summary_path = p("evaluated_data/summary_stats.csv")
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    return ds_mhm, ds_mrm, ds_mhm_z, ds_mrm_z, summary

ds_mhm_map, ds_mrm_map, ds_mhm_stats, ds_mrm_stats, STATIONS_SUMMARY = load_data()

all_vars = {}
if ds_mhm_map: all_vars.update({f"mhm:{v}": f"mHM: {v}" for v in ds_mhm_map.data_vars})
if ds_mrm_map: all_vars.update({f"mrm:{v}": f"mRM: {v}" for v in ds_mrm_map.data_vars})

time_values = pd.to_datetime(ds_mhm_map.time.values) if ds_mhm_map is not None else []
bounds = ((float(ds_mhm_map.lat.min()), float(ds_mhm_map.lon.min())), (float(ds_mhm_map.lat.max()), float(ds_mhm_map.lon.max()))) if ds_mhm_map is not None else ((48.5, 15.0), (49.5, 16.5))

# ============================================================
# 2. UI DEFINITION (Oprava pozicování a mezer)
# ============================================================
app_ui = ui.page_fillable(
    ui.tags.style("""
        html, body { margin: 0; padding: 0; height: 100vh; overflow: hidden; font-family: 'Segoe UI', sans-serif; }
        #map_a { height: 100vh !important; width: 100vw !important; position: absolute; top: 0; left: 0; z-index: 1; }
        
        .floating-menu { 
            position: absolute; 
            background: rgba(255,255,255,0.95); 
            padding: 15px; 
            border-radius: 8px; 
            z-index: 1000; 
            border: 1px solid #ccc; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); 
        }
        
        .section-title { font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 12px; font-size: 11px; color: #555; text-transform: uppercase; }

        /* Sblížení legend (gap: 5px) */
        .legend-wrapper { display: flex; gap: 5px; align-items: flex-start; margin-top: 10px; }
        .legend-column { flex: 1; }
        
        .legend-container { display: flex; flex-direction: column; gap: 4px; }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 10px; color: #333; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.3); flex-shrink: 0; }
        .metric-title { font-weight: bold; font-size: 11px; color: #2c3e50; margin-bottom: 5px; border-bottom: 1px solid #eee; }

        .raster-legend-container { display: flex; gap: 8px; align-items: stretch; height: 150px; margin-top: 5px; }
        .raster-legend-bar { width: 12px; border: 1px solid #444; border-radius: 2px; }
        .raster-labels { display: flex; flex-direction: column; justify-content: space-between; font-size: 9px; color: #333; font-weight: bold; }
    """),
    ui.div(
        output_widget("map_a"),
        ui.div(
            ui.div("mHM/mRM Explorer", style="font-size: 18px; font-weight: bold; margin-bottom: 10px;"),
            ui.div("Timeline Control", class_="section-title"),
            ui.input_slider("time_idx", None, 0, len(time_values)-1, 0, animate=False, width="100%"),
            ui.output_text("current_date"),
            ui.hr(),
            ui.div(
                ui.div(ui.output_ui("legend_ui"), class_="legend-column"),
                ui.div(ui.output_ui("raster_legend_ui"), class_="legend-column"),
                class_="legend-wrapper"
            ),
            # Zúženo na 340px
            class_="floating-menu", style="top: 20px; left: 20px; width: 340px;"
        ),
        ui.div(
            ui.div("Layer Settings", class_="section-title"),
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

    # --- HELPERS ---
    def get_metric_config(metric):
        # Přesné mapování pro zobrazení v legendě
        m_info = {
            "kge": "Kling-Gupta Eff.",
            "r": "Correlation",
            "beta": "Bias (Beta)",
            "gamma": "Variability (Gamma)"
        }
        curr_name = m_info.get(metric, "Metric")
        
        if metric in ["kge", "r"]:
            return {
                "name": curr_name,
                "bins": [
                    (0.1, "#7f0000", "0.0-0.1"), (0.2, "#a50026", "0.1-0.2"), (0.3, "#d73027", "0.2-0.3"),
                    (0.4, "#f46d43", "0.3-0.4"), (0.5, "#fdae61", "0.4-0.5"), (0.6, "#fee08b", "0.5-0.6"),
                    (0.7, "#d9ef8b", "0.6-0.7"), (0.8, "#a6d96a", "0.7-0.8"), (0.9, "#66bd63", "0.8-0.9"), (1.1, "#1a9850", "0.9-1.0")
                ]
            }
        else:
            return {
                "name": curr_name,
                "bins": [
                    (0.1, "#cc00ff", "< 0.1"), (0.4, "#0000ff", "0.1-0.4"), (0.7, "#00ffff", "0.4-0.7"),
                    (0.9, "#00ffcc", "0.7-0.9"), (1.1, "#00ff00", "0.9-1.1"), (1.3, "#ffff00", "1.1-1.3"),
                    (1.6, "#ff9900", "1.3-1.6"), (9.9, "#ff0000", "> 1.6")
                ]
            }

    def _get_color(metric, val):
        cfg = get_metric_config(metric)
        for threshold, color, label in cfg["bins"]:
            if val <= threshold: return color
        return cfg["bins"][-1][1]

    # --- MAP ---
    @render_widget
    def map_a():
        m = L.Map(center=[48.9, 16.0], zoom=9, scroll_wheel_zoom=True, zoom_control=False)
        m.add_layer(L.TileLayer())
        img_overlay = L.ImageOverlay(url="", bounds=bounds, opacity=0.7)
        m.add_layer(img_overlay)
        geo_layer = L.GeoJSON(data={}, point_style={"radius": 8, "fillOpacity": 1, "color": "white", "weight": 1.5})
        m.add_layer(geo_layer)

        @reactive.Effect
        def _update_map():
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
            except: pass

            if input.show_stations():
                stations_path = os.path.join(DATA_PATH, "evaluated_data/stations.json")
                if os.path.exists(stations_path):
                    with open(stations_path, "r") as f: geo_data = json.load(f)
                    metric = input.marker_metric()
                    for feat in geo_data['features']:
                        val = feat['properties'].get(metric, 0)
                        feat['properties']['style'] = {"fillColor": _get_color(metric, val)}
                    geo_layer.data = geo_data
                    geo_layer.visible = True
            else: geo_layer.visible = False

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

    # --- LEGENDS ---
    @render.ui
    def legend_ui():
        metric = input.marker_metric()
        cfg = get_metric_config(metric)
        items = [ui.div(cfg["name"], class_="metric-title")]
        for _, color, label in cfg["bins"]:
            items.append(ui.div(
                ui.div(class_="legend-dot", style=f"background-color: {color} !important;"),
                ui.span(label),
                class_="legend-item"
            ))
        return ui.div(*items, class_="legend-container")

    @render.ui
    def raster_legend_ui():
        try:
            prefix, var = input.active_layer().split(":")
            ds = ds_mhm_map if prefix == "mhm" else ds_mrm_map
            unit = ds[var].attrs.get("units", "-")
            data = ds[var].isel(time=input.time_idx())
            v_min, v_max = float(data.min()), float(data.max())
            is_q = "q" in var.lower()
            gradient = "linear-gradient(to top, #f7fbff, #08306b)" if is_q else "linear-gradient(to top, #440154, #21918c, #fde725)"
            return ui.div(
                ui.div(f"{var} [{unit}]", class_="metric-title"),
                ui.div(
                    ui.div(class_="raster-legend-bar", style=f"background: {gradient};"),
                    ui.div(ui.span(f"{v_max:.1f}"), ui.span(f"{(v_min+v_max)/2:.1f}"), ui.span(f"{v_min:.1f}"), class_="raster-labels"),
                    class_="raster-legend-container"
                )
            )
        except: return ui.div("Select a layer")

    @render.text
    def current_date():
        return f"Date: {time_values[input.time_idx()].strftime('%Y-%m-%d')}"

    # --- DASHBOARDS & PLOTS (Omitted for brevity, keep your original ones) ---
    @render.ui
    def st_dashboard():
        sid = selected_st_id()
        row = STATIONS_SUMMARY[STATIONS_SUMMARY['ID'].astype(str) == sid].iloc[0]
        return ui.div(ui.h3(f"Station: {sid}"), ui.value_box("KGE", f"{row['KGE']:.3f}"))

app = App(app_ui, server)