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
# 1. NAČTENÍ DAT
# ============================================================
def load_data():
    ds_mhm = xr.open_dataset("mHM_fluxes_states.nc") if os.path.exists("mHM_fluxes_states.nc") else None
    ds_mrm = xr.open_dataset("mRM_fluxes_states.nc") if os.path.exists("mRM_fluxes_states.nc") else None
    ds_mhm_z = xr.open_zarr("mHM_data.zarr") if os.path.exists("mHM_data.zarr") else None
    ds_mrm_z = xr.open_zarr("mRM_data.zarr") if os.path.exists("mRM_data.zarr") else None
    summary = pd.read_csv("evaluated_data/summary_stats.csv") if os.path.exists("evaluated_data/summary_stats.csv") else pd.DataFrame()
    return ds_mhm, ds_mrm, ds_mhm_z, ds_mrm_z, summary

ds_mhm_map, ds_mrm_map, ds_mhm_stats, ds_mrm_stats, STATIONS_SUMMARY = load_data()

all_vars = {}
if ds_mhm_map: all_vars.update({f"mhm:{v}": f"mHM: {v}" for v in ds_mhm_map.data_vars})
if ds_mrm_map: all_vars.update({f"mrm:{v}": f"mRM: {v}" for v in ds_mrm_map.data_vars})

time_values = pd.to_datetime(ds_mhm_map.time.values) if ds_mhm_map is not None else []
bounds = ((float(ds_mhm_map.lat.min()), float(ds_mhm_map.lon.min())), (float(ds_mhm_map.lat.max()), float(ds_mhm_map.lon.max())))
center = [49.5, 15.5]

# ============================================================
# 2. UI
# ============================================================
app_ui = ui.page_fillable(
    ui.tags.style("""
        html, body { margin: 0; padding: 0; height: 100vh; overflow: hidden; font-family: sans-serif; }
        #map_a { height: 100vh !important; width: 100vw !important; position: absolute; }
        .floating-menu { position: absolute; background: rgba(255,255,255,0.95); padding: 12px; border-radius: 5px; z-index: 1000; border: 1px solid #ccc; }
        .section-title { font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #ddd; padding-bottom: 3px; margin-top: 10px; font-size: 13px; }
        .legend-row { display: flex; width: 100%; height: 15px; border: 1px solid #666; margin-top: 5px; }
        .legend-tick { flex: 1; height: 100%; }
        .legend-labels { display: flex; justify-content: space-between; font-size: 10px; margin-top: 2px; color: #444; }
    """),
    ui.div(
        output_widget("map_a"),
        ui.div(
            ui.div("mHM/mRM Explorer", style="font-size: 16px; font-weight: bold;"),
            ui.div("Časová osa", class_="section-title"),
            ui.input_slider("time_idx", None, 0, len(time_values)-1, 0, animate=False, width="100%"),
            ui.output_text("current_date"),
            ui.hr(),
            ui.div("Legenda profilů", class_="section-title"),
            ui.output_ui("legend_vector"),
            class_="floating-menu", style="top: 20px; left: 20px; width: 320px;"
        ),
        ui.div(
            ui.div("Nastavení", class_="section-title"),
            ui.input_select("active_layer", "Rastr:", all_vars, selected="mrm:Qrouted"),
            ui.input_checkbox("show_stations", "Zobrazit profily", True),
            ui.input_select("marker_metric", "Metrika bodů:", {"kge": "KGE", "r": "Korelace", "beta": "Bias", "gamma": "Variabilita"}),
            class_="floating-menu", style="top: 20px; right: 20px; width: 250px;"
        )
    )
)

# ============================================================
# 3. SERVER
# ============================================================
def server(input, output, session):
    selected_st_id = reactive.Value(None)
    last_click_coords = reactive.Value(None)
    map_click_blocked = reactive.Value(False)

    def get_point_color(metric, val):
        pal = ["#7f0000", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#1a9850"]
        if metric in ["kge", "r"]:
            return pal[int(np.clip(val * 7, 0, 7))] if val > 0 else pal[0]
        else:
            if 0.85 <= val <= 1.15: return "#1a9850"
            return "#cc00ff" if val < 0.85 else "#d7191c"

    @render_widget
    def map_a():
        m = L.Map(center=center, zoom=8, scroll_wheel_zoom=True, zoom_control=False)
        m.add_layer(L.TileLayer())
        img_overlay = L.ImageOverlay(url="", bounds=bounds, opacity=0.7); m.add_layer(img_overlay)
        geo_layer = L.GeoJSON(data={}, point_style={"radius": 9, "fillOpacity": 1, "color": "white", "weight": 1.5})
        m.add_layer(geo_layer)

        @reactive.Effect
        def _update_layers():
            prefix, var = input.active_layer().split(":")
            ds = ds_mhm_map if prefix == "mhm" else ds_mrm_map
            data = ds[var].isel(time=input.time_idx())
            
            fig = plt.figure(figsize=(6, 6), dpi=70); ax = fig.add_axes([0, 0, 1, 1])
            is_q = "q" in var.lower()
            norm = colors.LogNorm(vmin=max(float(data.min()), 0.001), vmax=float(data.max())) if is_q else colors.Normalize(vmin=float(data.min()), vmax=float(data.max()))
            data.plot.imshow(ax=ax, cmap="Blues" if is_q else "viridis", norm=norm, add_colorbar=False, add_labels=False, interpolation="none")
            ax.axis('off')
            buf = io.BytesIO(); plt.savefig(buf, format='png', transparent=True); plt.close(fig)
            img_overlay.url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            if input.show_stations() and os.path.exists("evaluated_data/stations.json"):
                with open("evaluated_data/stations.json", "r") as f: geo_data = json.load(f)
                metric = input.marker_metric()
                for feat in geo_data['features']:
                    feat['properties']['style'] = {"fillColor": get_point_color(metric, feat['properties'].get(metric, 0))}
                geo_layer.data = geo_data
                geo_layer.visible = True

        def on_st_click(feature, **kwargs):
            map_click_blocked.set(True)
            sid = str(int(float(feature['properties']['id'])))
            print(f"DEBUG: Vyžádán dashboard pro stanici {sid}")
            selected_st_id.set(sid)
            ui.modal_show(ui.modal(ui.output_ui("st_dashboard"), size="xl", easy_close=True))
            threading.Timer(0.5, lambda: map_click_blocked.set(False)).start()

        geo_layer.on_click(on_st_click)

        def handle_click(**kwargs):
            if kwargs.get('type') == 'click' and not map_click_blocked():
                last_click_coords.set(kwargs.get('coordinates'))
                ui.modal_show(ui.modal(ui.output_plot("raster_ts_plot"), size="l", easy_close=True))
        m.on_interaction(handle_click)
        return m

    @render.ui
    def st_dashboard():
        sid = selected_st_id()
        if not sid: return ui.div("Chyba načítání ID.")
        print(f"DEBUG: Vytvářím HTML dashboardu pro {sid}")
        
        try:
            row = STATIONS_SUMMARY[STATIONS_SUMMARY['ID'].astype(str) == sid].iloc[0]
            stats = ui.layout_column_wrap(
                ui.value_box("KGE", f"{row['KGE']:.3f}"),
                ui.value_box("r", f"{row['Correlation']:.3f}"),
                ui.value_box("Bias", f"{row['Bias']:.3f}"),
                ui.value_box("Var", f"{row['Variability']:.3f}"),
                width=1/4, fill=False
            )
        except: stats = ui.div("Statistiky nenalezeny.")

        return ui.div(
            ui.h3(f"Analýza profilu {sid}"),
            stats,
            ui.hr(),
            ui.layout_column_wrap(
                ui.card(ui.card_header("Sezónní režim"), ui.output_plot("plot_seasonal")),
                ui.card(ui.card_header("Čára překročení"), ui.output_plot("plot_fdc")),
                width=1/2
            )
        )

    @render.plot
    def plot_seasonal():
        sid = selected_st_id()
        print(f"DEBUG: Volám plot_seasonal pro {sid}")
        path = f"evaluated_data/plots/{sid}_seasonal.csv"
        if not os.path.exists(path): 
            print(f"DEBUG ERROR: {path} nenalezen"); return None
        df = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df['Qobs'], 'k-o', label="Obs")
        ax.plot(df.index, df['Qrouted'], 'r--o', label="Sim")
        ax.set_ylabel(r"$Q [m^3s^{-1}]$"); ax.set_xlabel("Měsíc"); ax.legend(); ax.grid(True, alpha=0.3)
        return fig

    @render.plot
    def plot_fdc():
        sid = selected_st_id()
        print(f"DEBUG: Volám plot_fdc pro {sid}")
        path = f"evaluated_data/plots/{sid}_fdc.csv"
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['exceedance'], df['Qobs'], 'k-', label="Obs")
        ax.plot(df['exceedance'], df['Qsim'], 'r--', label="Sim")
        ax.set_yscale('log'); ax.set_ylabel(r"$Q [m^3s^{-1}]$"); ax.set_xlabel("% překročení")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        return fig

    @render.plot
    def raster_ts_plot():
        lat, lon = last_click_coords()
        prefix, var = input.active_layer().split(":")
        ds = ds_mhm_stats if prefix == "mhm" else ds_mrm_stats
        data = ds[var].sel(lat=lat, lon=lon, method="nearest").compute()
        fig, ax = plt.subplots(figsize=(10, 4))
        data.plot(ax=ax, color="#2c3e50"); ax.set_title(f"Bod: {lat:.3f}, {lon:.3f}")
        return fig

    @render.ui
    def legend_vector():
        metric = input.marker_metric()
        pal = ["#7f0000", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#1a9850"] if metric in ["kge", "r"] else ["#cc00ff", "#1a9850", "#d7191c"]
        return ui.div(
            ui.div(*[ui.div(class_="legend-tick", style=f"background: {c};") for c in pal], class_="legend-row"),
            ui.div(ui.span("Špatné/Nízké"), ui.span("Ideál"), ui.span("Vysoké", style="float: right;"), class_="legend-labels")
        )

    @render.text
    def current_date():
        return f"Datum: {time_values[input.time_idx()].strftime('%d. %m. %Y')}"

app = App(app_ui, server)