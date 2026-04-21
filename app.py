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
    """Loads NetCDF, Zarr, and summary statistics from defined paths."""
    def p(name): return os.path.join(DATA_PATH, name)
    ds_mhm = xr.open_dataset(p("mHM_Fluxes_States.nc"), chunks={"time": 1}) if os.path.exists(p("mHM_Fluxes_States.nc")) else None
    ds_mrm = xr.open_dataset(p("mRM_Fluxes_States.nc"), chunks={"time": 1}) if os.path.exists(p("mRM_Fluxes_States.nc")) else None
    ds_mhm_z = xr.open_dataset(p("mHM_data.zarr"), engine="zarr", chunks={}) if os.path.exists(p("mHM_data.zarr")) else None
    ds_mrm_z = xr.open_dataset(p("mRM_data.zarr"), engine="zarr", chunks={}) if os.path.exists(p("mRM_data.zarr")) else None
    summary_path = p("evaluated_data/summary_stats.csv")
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    return ds_mhm, ds_mrm, ds_mhm_z, ds_mrm_z, summary

ds_mhm_map, ds_mrm_map, ds_mhm_stats, ds_mrm_stats, STATIONS_SUMMARY = load_data()

# Temporal range setup
all_time_values = pd.to_datetime(ds_mhm_map.time.values) if ds_mhm_map is not None else []
min_date = all_time_values.min().date() if len(all_time_values) > 0 else None
max_date = all_time_values.max().date() if len(all_time_values) > 0 else None

# Grid bounds calculation
if ds_mhm_map is not None:
    dlat = float(abs(ds_mhm_map.lat[1] - ds_mhm_map.lat[0]))
    dlon = float(abs(ds_mhm_map.lon[1] - ds_mhm_map.lon[0]))
    bounds = (
        (float(ds_mhm_map.lat.min()) - dlat/2, float(ds_mhm_map.lon.min()) - dlon/2), 
        (float(ds_mhm_map.lat.max()) + dlat/2, float(ds_mhm_map.lon.max()) + dlon/2)
    )
else:
    bounds = ((48.5, 15.0), (49.5, 16.5))

# Variables available for selection
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
        
        .floating-menu { 
            position: absolute; background: rgba(255,255,255,0.92); 
            padding: 15px; border-radius: 8px; z-index: 1000; border: 1px solid #ccc; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); max-height: 90vh; overflow-y: auto; 
        }

        /* Bottom panel alignment between side menus with standardized gaps */
        .bottom-panel {
            position: absolute; 
            bottom: 20px; 
            left: 380px;  /* Left menu width (340) + margin (20) + gap (20) */
            right: 300px; /* Right menu width (260) + margin (20) + gap (20) */
            z-index: 900;
        }

        /* Analysis card with relative height (60% of viewport) */
        .analysis-card {
            background: rgba(255,255,255,0.96); 
            border: 1px solid #ccc;
            border-radius: 8px; 
            box-shadow: 0 -4px 15px rgba(0,0,0,0.15);
            padding: 20px; 
            height: 60vh; 
            display: flex; 
            flex-direction: column;
        }

        .leaflet-image-layer { image-rendering: pixelated !important; image-rendering: crisp-edges !important; }
        .section-title { font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 12px; font-size: 11px; color: #555; text-transform: uppercase; }
        .legend-wrapper { display: flex; gap: 10px; align-items: flex-start; margin-top: 5px; }
        .legend-column { flex: 1; }
        .legend-container { display: flex; flex-direction: column; gap: 4px; }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 10px; color: #333; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.3); flex-shrink: 0; }
        .metric-title { font-weight: bold; font-size: 11px; color: #2c3e50; margin-bottom: 5px; border-bottom: 1px solid #eee; }
        .raster-legend-container { display: flex; gap: 8px; align-items: stretch; height: 120px; margin-top: 5px; }
        .raster-legend-bar { width: 15px; border: 1px solid #444; border-radius: 2px; }
        .raster-labels { display: flex; flex-direction: column; justify-content: space-between; font-size: 10px; color: #333; font-weight: bold; }
        .anim-controls { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
        
        .date-display-box {
            height: 40px; width: 100%; margin-top: 10px;
            display: flex; align-items: center; justify-content: center;
            background: #f8f9fa; border: 1px solid #eee; border-radius: 4px;
            font-family: monospace; font-size: 18px; font-weight: bold; color: #2c3e50;
        }
    """),
    ui.div(
        output_widget("map_a"),
        # LEFT PANEL: Controls and Legends
        ui.div(
            ui.div("mHM/mRM Explorer", style="font-size: 18px; font-weight: bold; margin-bottom: 10px; text-align: center;"),
            ui.accordion(
                ui.accordion_panel(
                    "Time & Aggregation",
                    ui.input_date_range("date_range", "Select Period:", start=min_date, end=max_date, min=min_date, max=max_date),
                    ui.input_select("agg_type", "Aggregation Type:", {
                        "none": "No Aggregation (Daily)",
                        "clim_mean": "Long-term Monthly Mean",
                        "clim_median": "Long-term Monthly Median",
                        "ts_month_mean": "Monthly Mean Series",
                        "ts_year_mean": "Annual Mean Series"
                    }),
                    ui.input_checkbox("apply_agg_to_map", "Apply aggregation to Map (Slower)", False),
                    ui.output_ui("dynamic_slider"),
                    ui.div(
                        ui.output_ui("play_pause_button"),
                        ui.input_select("anim_speed", "Speed:", {"0.5": "Slow", "0.2": "Normal", "0.1": "Fast"}, selected="0.2", width="100px"),
                        class_="anim-controls"
                    ),
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
                id="acc_left", multiple=True
            ),
            class_="floating-menu", style="top: 20px; left: 20px; width: 340px;"
        ),
        # RIGHT PANEL: Layer Settings
        ui.div(
            ui.div("Layer Configuration", style="font-size: 16px; font-weight: bold; margin-bottom: 10px;"),
            ui.div("Variable Selection", class_="section-title"),
            ui.input_select("active_layer", "Raster Variable:", all_vars),
            ui.hr(),
            ui.div("Gauging Stations", class_="section-title"),
            ui.input_checkbox("show_stations", "Show Gauging Stations", True),
            ui.input_select("marker_metric", "Color Stations by:", {"kge": "KGE", "r": "Correlation", "beta": "Bias", "gamma": "Variability"}),
            class_="floating-menu", style="top: 20px; right: 20px; width: 260px;"
        ),
        # BOTTOM PANEL: Chart Area
        ui.div(
            ui.output_ui("bottom_analysis_card"),
            class_="bottom-panel"
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
    def current_ds():
        """Returns the variable subset based on selected date range."""
        prefix, var = input.active_layer().split(":")
        ds = ds_mhm_map if prefix == "mhm" else ds_mrm_map
        return ds[var].sel(time=slice(input.date_range()[0], input.date_range()[1]))

    @reactive.Calc
    def aggregated_data():
        """Calculates temporal aggregation for the map view only if enabled."""
        da = current_ds()
        agg = input.agg_type()
        if agg == "none" or not input.apply_agg_to_map(): return da
        if agg == "clim_mean": return da.groupby("time.month").mean("time")
        if agg == "clim_median": return da.load().groupby("time.month").median("time")
        if agg == "ts_month_mean": return da.resample(time="1MS").mean("time")
        if agg == "ts_year_mean": return da.resample(time="1YS").mean("time")
        return da

    @output
    @render.ui
    def play_pause_button():
        """Renders the Play/Pause action button."""
        lbl, cls = ("⏸ Pause", "btn-warning") if is_playing() else ("▶ Play", "btn-success")
        return ui.input_action_button("toggle_play", lbl, class_=f"{cls} btn-sm", style="width: 80px;")

    @reactive.Effect
    @reactive.event(input.toggle_play)
    def _toggle_play():
        """Toggles the playback state."""
        is_playing.set(not is_playing())

    @output
    @render.ui
    def dynamic_slider():
        """Adjusts the slider range based on current data dimension."""
        da = aggregated_data()
        n = da.sizes[da.dims[0]]
        return ui.input_slider("time_idx", None, 0, n - 1, 0, animate=False, width="100%")

    @reactive.Effect
    def _animation_step():
        """Controls temporal stepping for animation."""
        reactive.invalidate_later(float(input.anim_speed()))
        if is_playing():
            with reactive.isolate():
                da = aggregated_data()
                n = da.sizes[da.dims[0]]
                cur = input.time_idx()
                nxt = (cur + 1) if cur < (n - 1) else 0
                ui.update_slider("time_idx", value=nxt)

    @render_widget
    def map_a():
        """Initializes and updates the main map widget."""
        m = L.Map(center=[48.9, 16.0], zoom=9, scroll_wheel_zoom=True, zoom_control=False)
        m.add_layer(L.TileLayer())
        img_overlay = L.ImageOverlay(url="", bounds=bounds, opacity=0.8)
        geo_layer = L.GeoJSON(data={}, point_style={"radius": 8, "fillOpacity": 1, "color": "white", "weight": 1.5})
        m.add_layer(img_overlay); m.add_layer(geo_layer)

        @reactive.Effect
        def _update_map():
            """Updates the raster image layer on the map."""
            try:
                da_full = aggregated_data()
                da = da_full.isel({da_full.dims[0]: input.time_idx()})
                v_min, v_max = float(da.min()), float(da.max())
                is_q = "q" in input.active_layer().lower()
                norm = colors.LogNorm(vmin=max(v_min, 0.001), vmax=v_max) if is_q else colors.Normalize(vmin=v_min, vmax=v_max)
                fig = plt.figure(figsize=(8, 8), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
                da.plot.imshow(ax=ax, cmap="Blues" if is_q else "viridis", norm=norm, add_colorbar=False, add_labels=False, interpolation="nearest")
                ax.axis('off'); buf = io.BytesIO(); plt.savefig(buf, format='png', transparent=True, pad_inches=0); plt.close(fig)
                img_overlay.url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            except: pass

            # Gauging station markers logic
            if input.show_stations():
                stations_path = os.path.join(DATA_PATH, "evaluated_data/stations.json")
                if os.path.exists(stations_path):
                    with open(stations_path, "r") as f: geo_data = json.load(f)
                    metric = input.marker_metric()
                    for feat in geo_data['features']:
                        val = feat['properties'].get(metric, 0)
                        feat['properties']['style'] = {"fillColor": _get_color(metric, val)}
                    geo_layer.data = geo_data; geo_layer.visible = True
            else: geo_layer.visible = False

        def on_st_click(feature, **kwargs):
            """Handles click on gauging stations to open dashboard."""
            map_click_blocked.set(True)
            selected_st_id.set(str(int(float(feature['properties']['id']))))
            ui.modal_show(ui.modal(ui.output_ui("st_dashboard"), size="xl", easy_close=True))
            threading.Timer(0.5, lambda: map_click_blocked.set(False)).start()
        geo_layer.on_click(on_st_click)

        def on_map_interaction(**kwargs):
            """Handles map clicks for grid cell analysis."""
            if kwargs.get('type') == 'click' and not map_click_blocked():
                last_click_coords.set(kwargs.get('coordinates'))
        m.on_interaction(on_map_interaction)
        return m

    @output
    @render.ui
    def bottom_analysis_card():
        """Renders the analysis card containing the 60vh time series plot."""
        if not last_click_coords(): return None
        return ui.div(
            ui.div(
                ui.div(f"Analysis at {last_click_coords()[0]:.4f}N, {last_click_coords()[1]:.4f}", 
                       style="font-weight: bold; flex-grow: 1;"),
                ui.input_action_button("close_graph", "×", class_="btn-lg btn-outline-danger", 
                                       style="border:none; font-size: 28px; line-height: 1; padding: 0;"),
                style="display: flex; align-items: center; border-bottom: 2px solid #eee; margin-bottom: 10px;"
            ),
            # Fill the 60vh container height
            ui.output_plot("raster_ts_plot", height="100%"), 
            class_="analysis-card"
        )

    @reactive.Effect
    @reactive.event(input.close_graph)
    def _close_graph():
        """Clears coordinates to hide the analysis panel."""
        last_click_coords.set(None)

    @render.plot
    def raster_ts_plot():
        """Generates time series plot from statistical Zarr files."""
        if not last_click_coords(): return None
        lat, lon = last_click_coords()
        prefix, var = input.active_layer().split(":")
        ds = ds_mhm_stats if prefix == "mhm" else ds_mrm_stats
        da_p = ds[var].sel(lat=lat, lon=lon, method="nearest").sel(time=slice(input.date_range()[0], input.date_range()[1]))
        agg = input.agg_type()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        if agg == "none": 
            da_p.plot(ax=ax, color="#2c3e50")
        elif agg == "clim_mean": 
            da_p.groupby("time.month").mean().plot(ax=ax, marker='o', color='red')
            ax.set_xticks(range(1, 13))
        elif agg == "clim_median": 
            da_p.load().groupby("time.month").median().plot(ax=ax, marker='o', color='green')
            ax.set_xticks(range(1, 13))
        elif "ts_month" in agg: 
            da_p.resample(time="1MS").mean().plot(ax=ax, marker='.')
        elif "ts_year" in agg: 
            da_p.resample(time="1YS").mean().plot(ax=ax, marker='s')
            
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # --- HELPERS AND LEGENDS ---
    def get_metric_config(metric):
        """Metadata for evaluation metrics coloring."""
        m_info = {"kge": "Kling-Gupta Eff.", "r": "Correlation", "beta": "Bias (Beta)", "gamma": "Variability (Gamma)"}
        curr_name = m_info.get(metric, "Metric")
        if metric in ["kge", "r"]:
            return {"name": curr_name, "bins": [(0.1, "#7f0000", "0.0-0.1"), (0.3, "#d73027", "0.2-0.3"), (0.5, "#fdae61", "0.4-0.5"), (0.7, "#d9ef8b", "0.6-0.7"), (0.9, "#66bd63", "0.8-0.9"), (1.1, "#1a9850", "0.9-1.0")]}
        return {"name": curr_name, "bins": [(0.1, "#cc00ff", "< 0.1"), (0.7, "#00ffff", "0.4-0.7"), (1.1, "#00ff00", "0.9-1.1"), (1.6, "#ff9900", "1.3-1.6"), (9.9, "#ff0000", "> 1.6")]}

    def _get_color(metric, val):
        """Determines hex color based on metric value."""
        for threshold, color, label in get_metric_config(metric)["bins"]:
            if val <= threshold: return color
        return "#ff0000"

    @output
    @render.ui
    def legend_ui():
        """Renders the station metrics legend."""
        metric = input.marker_metric()
        cfg = get_metric_config(metric)
        items = [ui.div(cfg["name"], class_="metric-title")]
        for _, color, label in cfg["bins"]:
            items.append(ui.div(ui.div(class_="legend-dot", style=f"background-color: {color} !important;"), ui.span(label), class_="legend-item"))
        return ui.div(*items, class_="legend-container")

    @output
    @render.ui
    def raster_legend_ui():
        """Renders the raster colormap legend."""
        try:
            prefix, var = input.active_layer().split(":")
            ds = ds_mhm_map if prefix == "mhm" else ds_mrm_map
            unit = ds[var].attrs.get("units", "-")
            da_full = aggregated_data()
            da_frame = da_full.isel({da_full.dims[0]: input.time_idx()})
            v_min, v_max = float(da_frame.min()), float(da_frame.max())
            is_q = "q" in var.lower()
            grad = "linear-gradient(to top, #f7fbff, #08306b)" if is_q else "linear-gradient(to top, #440154, #21918c, #fde725)"
            return ui.div(ui.div(f"{var} [{unit}]", class_="metric-title"), 
                          ui.div(ui.div(class_="raster-legend-bar", style=f"background: {grad};"), 
                                 ui.div(ui.span(f"{v_max:.1f}"), ui.span(f"{(v_min+v_max)/2:.1f}"), ui.span(f"{v_min:.1f}"), class_="raster-labels"), 
                                 class_="raster-legend-container"))
        except: return ui.div("Legend unavailable")

    @output
    @render.ui
    def current_date_display():
        """Updates the formatted date display in the sidebar."""
        try:
            da = aggregated_data()
            idx = input.time_idx()
            if 'time' in da.coords: return ui.HTML(f"<span>{pd.to_datetime(da.time.values[idx]).strftime('%Y-%m-%d')}</span>")
            elif 'month' in da.coords: return ui.HTML(f"<span>Month: {int(da.month.values[idx])}</span>")
            elif 'year' in da.coords: return ui.HTML(f"<span>Year: {int(da.year.values[idx])}</span>")
            return ui.HTML("<span>-</span>")
        except: return ui.HTML("<span>Error</span>")

    @render.ui
    def st_dashboard():
        """Renders gauging station analysis dashboard in a modal."""
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
        """Seasonal cycle plot for gauging stations."""
        sid = selected_st_id(); path = os.path.join(DATA_PATH, "evaluated_data/plots", f"{sid}_seasonal.csv")
        if not os.path.exists(path): return None
        df = pd.read_csv(path, index_col=0); fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df['Qobs'], 'k-o', label="Observed"); ax.plot(df.index, df['Qrouted'], 'r--o', label="Simulated")
        ax.set_ylabel("Discharge [m3/s]"); ax.set_xlabel("Month"); ax.legend(); ax.grid(True, alpha=0.3); return fig

    @render.plot
    def plot_fdc():
        """Flow duration curve for gauging stations."""
        sid = selected_st_id(); path = os.path.join(DATA_PATH, "evaluated_data/plots", f"{sid}_fdc.csv")
        if not os.path.exists(path): return None
        df = pd.read_csv(path); fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['exceedance'], df['Qobs'], 'k-'); ax.plot(df['exceedance'], df['Qsim'], 'r--'); ax.set_yscale('log'); ax.set_ylabel("Discharge [m3/s]"); ax.set_xlabel("Exceedance Probability"); ax.legend(); ax.grid(True, alpha=0.3); return fig

app = App(app_ui, server)