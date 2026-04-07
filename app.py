import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import io, base64, os, json, threading, asyncio
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import ipyleaflet as L

# ============================================================
# 1. DATA LOADING & PATHS
# ============================================================
SERVER_PATH = "/mnt/data/kolar.v/dyje_bakalarka"
DATA_PATH = SERVER_PATH if os.path.exists(SERVER_PATH) else os.path.dirname(__file__)

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

all_time_values = pd.to_datetime(ds_mhm_map.time.values) if ds_mhm_map is not None else []
min_date = all_time_values.min().date() if len(all_time_values) > 0 else None
max_date = all_time_values.max().date() if len(all_time_values) > 0 else None

# Calculate grid bounds including half-pixel offset for correct alignment
if ds_mhm_map is not None:
    dlat = float(abs(ds_mhm_map.lat[1] - ds_mhm_map.lat[0]))
    dlon = float(abs(ds_mhm_map.lon[1] - ds_mhm_map.lon[0]))
    bounds = (
        (float(ds_mhm_map.lat.min()) - dlat/2, float(ds_mhm_map.lon.min()) - dlon/2), 
        (float(ds_mhm_map.lat.max()) + dlat/2, float(ds_mhm_map.lon.max()) + dlon/2)
    )
else:
    bounds = ((48.5, 15.0), (49.5, 16.5))

all_vars = {}
if ds_mhm_map: all_vars.update({f"mhm:{v}": f"mHM: {v}" for v in ds_mhm_map.data_vars})
if ds_mrm_map: all_vars.update({f"mrm:{v}": f"mRM: {v}" for v in ds_mrm_map.data_vars})

# ============================================================
# 2. UI DEFINITION
# ============================================================
app_ui = ui.page_fillable(
    ui.tags.style("""
        html, body { margin: 0; padding: 0; height: 100vh; overflow: hidden; font-family: 'Segoe UI', sans-serif; }
        #map_a { height: 100vh !important; width: 100vw !important; position: absolute; top: 0; left: 0; z-index: 1; }
        .floating-menu { position: absolute; background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; z-index: 1000; border: 1px solid #ccc; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .leaflet-image-layer { image-rendering: pixelated !important; image-rendering: crisp-edges !important; }
        .section-title { font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 12px; font-size: 11px; color: #555; text-transform: uppercase; }
        .legend-wrapper { display: flex; gap: 5px; align-items: flex-start; margin-top: 5px; }
        .legend-column { flex: 1; }
        .legend-container { display: flex; flex-direction: column; gap: 4px; }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 10px; color: #333; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.3); flex-shrink: 0; }
        .metric-title { font-weight: bold; font-size: 11px; color: #2c3e50; margin-bottom: 5px; border-bottom: 1px solid #eee; }
        .raster-legend-container { display: flex; gap: 8px; align-items: stretch; height: 120px; margin-top: 5px; }
        .raster-legend-bar { width: 12px; border: 1px solid #444; border-radius: 2px; }
        .raster-labels { display: flex; flex-direction: column; justify-content: space-between; font-size: 9px; color: #333; font-weight: bold; }
        .anim-controls { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
        .accordion { --bs-accordion-bg: transparent; --bs-accordion-border-width: 0; }
        .accordion-button { padding: 10px; font-size: 14px; font-weight: bold; }
        .accordion-body { padding: 10px 5px; overflow: hidden; }

        /* Fix height and width for date display to prevent jumping */
        .date-display-box {
            height: 40px; width: 100%; margin-top: 10px;
            display: flex; align-items: center; justify-content: center;
            background: #f8f9fa; border: 1px solid #eee; border-radius: 4px;
            font-family: 'Courier New', Courier, monospace; 
            font-size: 18px; font-weight: bold; color: #2c3e50;
        }
        
        /* Ensure the inner span has a fixed minimum width for stable alignment */
        .date-display-box span {
            display: inline-block;
            min-width: 120px;
            text-align: center;
            letter-spacing: 1px;
        }
    """),
    ui.div(
        output_widget("map_a"),
        # LEFT PANEL: Dynamics & Interpretation
        ui.div(
            ui.div("mHM/mRM Explorer", style="font-size: 18px; font-weight: bold; margin-bottom: 10px; text-align: center;"),
            ui.accordion(
                ui.accordion_panel(
                    "Time Control",
                    ui.input_date_range("date_range", "Select Period:", start=min_date, end=max_date, min=min_date, max=max_date),
                    ui.output_ui("dynamic_slider"),
                    ui.div(
                        ui.input_action_button("play", "▶", class_="btn-success btn-sm"),
                        ui.input_action_button("stop", "◼", class_="btn-danger btn-sm"),
                        ui.input_select("anim_speed", "Speed:", {"0.5": "Slow", "0.2": "Normal", "0.1": "Fast"}, selected="0.2", width="100px"),
                        class_="anim-controls"
                    ),
                    # Stabilized date container
                    ui.div(ui.output_ui("current_date_display"), class_="date-display-box"),
                ),
                ui.accordion_panel(
                    "Legends",
                    ui.div(
                        ui.div(ui.output_ui("legend_ui"), class_="legend-column"),
                        ui.div(ui.output_ui("raster_legend_ui"), class_="legend-column"),
                        class_="legend-wrapper"
                    ),
                ),
                id="acc_left",
                multiple=True,
            ),
            class_="floating-menu", style="top: 20px; left: 20px; width: 340px;"
        ),
        # RIGHT PANEL: Data Source & Simulation Settings
        ui.div(
            ui.div("Layer Configuration", style="font-size: 16px; font-weight: bold; margin-bottom: 10px;"),
            ui.div("Variable Selection", class_="section-title"),
            ui.input_select("active_layer", "Raster Variable:", all_vars, selected=list(all_vars.keys())[0] if all_vars else None),
            ui.hr(),
            ui.div("Gauging Stations", class_="section-title"),
            ui.input_checkbox("show_stations", "Show Stations", True),
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
    is_playing = reactive.Value(False)

    @reactive.Calc
    def filtered_time_info():
        start, end = input.date_range()
        mask = (all_time_values.date >= start) & (all_time_values.date <= end)
        indices = np.where(mask)[0]
        dates = all_time_values[mask]
        return indices, dates

    @output
    @render.ui
    def dynamic_slider():
        indices, _ = filtered_time_info()
        return ui.input_slider("time_idx", None, 0, len(indices) - 1, 0, animate=False, width="100%")

    @reactive.Effect
    @reactive.event(input.play)
    def _play(): is_playing.set(True)

    @reactive.Effect
    @reactive.event(input.stop)
    def _stop(): is_playing.set(False)

    @reactive.Effect
    def _animation_step():
        if is_playing():
            reactive.invalidate_later(float(input.anim_speed()))
            indices, _ = filtered_time_info()
            cur = input.time_idx()
            nxt = (cur + 1) if cur < (len(indices) - 1) else 0
            ui.update_slider("time_idx", value=nxt)

    def get_metric_config(metric):
        m_info = {"kge": "Kling-Gupta Eff.", "r": "Correlation", "beta": "Bias (Beta)", "gamma": "Variability (Gamma)"}
        curr_name = m_info.get(metric, "Metric")
        if metric in ["kge", "r"]:
            return {"name": curr_name, "bins": [(0.1, "#7f0000", "0.0-0.1"), (0.2, "#a50026", "0.1-0.2"), (0.3, "#d73027", "0.2-0.3"), (0.4, "#f46d43", "0.3-0.4"), (0.5, "#fdae61", "0.4-0.5"), (0.6, "#fee08b", "0.5-0.6"), (0.7, "#d9ef8b", "0.6-0.7"), (0.8, "#a6d96a", "0.7-0.8"), (0.9, "#66bd63", "0.8-0.9"), (1.1, "#1a9850", "0.9-1.0")]}
        else:
            return {"name": curr_name, "bins": [(0.1, "#cc00ff", "< 0.1"), (0.4, "#0000ff", "0.1-0.4"), (0.7, "#00ffff", "0.4-0.7"), (0.9, "#00ffcc", "0.7-0.9"), (1.1, "#00ff00", "0.9-1.1"), (1.3, "#ffff00", "1.1-1.3"), (1.6, "#ff9900", "1.3-1.6"), (9.9, "#ff0000", "> 1.6")]}

    def _get_color(metric, val):
        cfg = get_metric_config(metric)
        for threshold, color, label in cfg["bins"]:
            if val <= threshold: return color
        return cfg["bins"][-1][1]

    @render_widget
    def map_a():
        m = L.Map(center=[48.9, 16.0], zoom=9, scroll_wheel_zoom=True, zoom_control=False)
        m.add_layer(L.TileLayer())
        img_overlay = L.ImageOverlay(url="", bounds=bounds, opacity=0.8)
        geo_layer = L.GeoJSON(data={}, point_style={"radius": 8, "fillOpacity": 1, "color": "white", "weight": 1.5})
        m.add_layer(img_overlay); m.add_layer(geo_layer)

        @reactive.Effect
        def _update_map():
            try:
                indices, _ = filtered_time_info()
                real_idx = indices[input.time_idx()]
                prefix, var = input.active_layer().split(":")
                ds = ds_mhm_map if prefix == "mhm" else ds_mrm_map
                data = ds[var].isel(time=real_idx)
                
                fig = plt.figure(figsize=(8, 8), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
                is_q = "q" in var.lower()
                v_min, v_max = float(data.min()), float(data.max())
                norm = colors.LogNorm(vmin=max(v_min, 0.001), vmax=v_max) if is_q else colors.Normalize(vmin=v_min, vmax=v_max)
                data.plot.imshow(ax=ax, cmap="Blues" if is_q else "viridis", norm=norm, add_colorbar=False, add_labels=False, interpolation="nearest")
                ax.axis('off')
                buf = io.BytesIO(); plt.savefig(buf, format='png', transparent=True, pad_inches=0); plt.close(fig)
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

    @render.ui
    def legend_ui():
        metric = input.marker_metric()
        cfg = get_metric_config(metric)
        items = [ui.div(cfg["name"], class_="metric-title")]
        for _, color, label in cfg["bins"]:
            items.append(ui.div(ui.div(class_="legend-dot", style=f"background-color: {color} !important;"), ui.span(label), class_="legend-item"))
        return ui.div(*items, class_="legend-container")

    @render.ui
    def raster_legend_ui():
        try:
            prefix, var = input.active_layer().split(":")
            indices, _ = filtered_time_info()
            ds = ds_mhm_map if prefix == "mhm" else ds_mrm_map
            da = ds[var]
            
            # Fetch units from metadata
            unit = da.attrs.get("units", "-")
            
            data_frame = da.isel(time=indices[input.time_idx()])
            v_min, v_max = float(data_frame.min()), float(data_frame.max())
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

    @render.ui
    def current_date_display():
        _, dates = filtered_time_info()
        date_str = dates[input.time_idx()].strftime('%Y-%m-%d')
        return ui.HTML(f"<span>{date_str}</span>")

    @render.ui
    def st_dashboard():
        sid = selected_st_id()
        if not sid: return ui.div("No station selected")
        df_sub = STATIONS_SUMMARY[STATIONS_SUMMARY['ID'].astype(str) == sid]
        if df_sub.empty: return ui.div(f"No data for station {sid}")
        row = df_sub.iloc[0]
        c_kge, c_corr, c_bias, c_var = _get_color("kge", row['KGE']), _get_color("r", row['Correlation']), _get_color("beta", row['Bias']), _get_color("gamma", row['Variability'])
        def box_style(color): return f"background-color: {color} !important; color: black !important;"
        return ui.div(ui.h3(f"Station Analysis: {sid}", style="font-weight: bold; margin-bottom: 15px;"), ui.layout_column_wrap(ui.value_box("KGE", f"{row['KGE']:.3f}", style=box_style(c_kge)), ui.value_box("Correlation", f"{row['Correlation']:.3f}", style=box_style(c_corr)), ui.value_box("Bias", f"{row['Bias']:.3f}", style=box_style(c_bias)), ui.value_box("Variability", f"{row['Variability']:.3f}", style=box_style(c_var)), width=1/4, fill=False), ui.hr(), ui.layout_column_wrap(ui.card(ui.card_header("Seasonal Cycle"), ui.output_plot("plot_seasonal")), ui.card(ui.card_header("Flow Duration Curve"), ui.output_plot("plot_fdc")), width=1/2))

    @render.plot
    def plot_seasonal():
        sid = selected_st_id(); path = os.path.join(DATA_PATH, "evaluated_data/plots", f"{sid}_seasonal.csv")
        if not os.path.exists(path): return None
        df = pd.read_csv(path, index_col=0); fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df['Qobs'], 'k-o', label="Observed"); ax.plot(df.index, df['Qrouted'], 'r--o', label="Simulated")
        ax.set_ylabel("Discharge [m3/s]"); ax.set_xlabel("Month"); ax.legend(); ax.grid(True, alpha=0.3); return fig

    @render.plot
    def plot_fdc():
        sid = selected_st_id(); path = os.path.join(DATA_PATH, "evaluated_data/plots", f"{sid}_fdc.csv")
        if not os.path.exists(path): return None
        df = pd.read_csv(path); fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['exceedance'], df['Qobs'], 'k-'); ax.plot(df['exceedance'], df['Qsim'], 'r--'); ax.set_yscale('log'); ax.set_ylabel("Discharge [m3/s]"); ax.set_xlabel("Exceedance Probability"); ax.legend(); ax.grid(True, alpha=0.3); return fig

    @render.plot
    def raster_ts_plot():
        if not last_click_coords(): return None
        lat, lon = last_click_coords(); prefix, var = input.active_layer().split(":")
        ds = ds_mhm_stats if prefix == "mhm" else ds_mrm_stats
        data = ds[var].sel(lat=lat, lon=lon, method="nearest").compute()
        fig, ax = plt.subplots(figsize=(10, 4)); data.plot(ax=ax, color="#2c3e50"); ax.set_title(f"Grid Cell: {lat:.3f}N, {lon:.3f}E"); return fig

app = App(app_ui, server)
