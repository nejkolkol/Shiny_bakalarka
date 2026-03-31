import pandas as pd
import xarray as xr
import numpy as np
import os
import json

# Vytvoření adresářů pro mezivýpočty
os.makedirs("evaluated_data/plots", exist_ok=True)

def calculate_kge_components(sim, obs):
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    s, o = sim[mask], obs[mask]
    
    if len(s) < 30:
        return None
        
    # 1. Korelace (Timing)
    r = np.corrcoef(s, o)[0, 1]
    
    # 2. Bias (Volume/Mean)
    beta = s.mean() / o.mean()
    
    # 3. Variabilita (Flow variability)
    # gamma = (CV_sim / CV_obs)
    gamma = (s.std() / s.mean()) / (o.std() / o.mean())
    
    # 4. Celkové KGE
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    return {"kge": kge, "r": r, "beta": beta, "gamma": gamma}

# 1. Načtení modelu a přehledu stanic
print("Načítám modelová data (Zarr)...")
ds_stats = xr.open_zarr("mRM_data.zarr")

# Načtení přehledu (oddělovač středník, ošetření NaN v ID)
df_st = pd.read_csv("pozorovani/prehled_profilu.csv", sep=';', header=None, names=['LON', 'LAT', 'ID']).dropna(subset=['ID'])

summary = []
print(f"Začínám zpracování {len(df_st)} stanic...")

for _, row in df_st.iterrows():
    try:
        st_id = str(int(row['ID']))
        st_file = f"pozorovani/profily/QD_{st_id}_Data.csv"
        
        if os.path.exists(st_file):
            # Načtení pozorování
            df_obs = pd.read_csv(st_file, header=None, names=['ID', 'VAR', 'Y', 'M', 'D', 'Qobs'], skipinitialspace=True)
            
            # Čištění a konverze
            for col in ['Y', 'M', 'D', 'Qobs']:
                df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce')
            df_obs = df_obs.dropna(subset=['Y', 'M', 'D', 'Qobs'])
            
            # Tvorba data a normalizace na půlnoc
            df_obs['time'] = pd.to_datetime(df_obs[['Y', 'M', 'D']].rename(columns={'Y':'year', 'M':'month', 'D':'day'}))
            df_obs['time'] = df_obs['time'].dt.normalize()
            
            # Načtení modelu pro daný bod
            sim_ts = ds_stats['Qrouted'].sel(lat=row['LAT'], lon=row['LON'], method='nearest').compute()
            df_sim = sim_ts.to_dataframe().reset_index()[['time', 'Qrouted']]
            df_sim['time'] = pd.to_datetime(df_sim['time']).dt.tz_localize(None).dt.normalize()
            
            # Spojení (Inner Join)
            m = pd.merge(df_obs[['time', 'Qobs']], df_sim, on='time', how='inner').dropna()
            
            if len(m) > 100: # Minimálně 100 společných dní pro validní statistiku
                metrics = calculate_kge_components(m['Qrouted'].values, m['Qobs'].values)
                
                if metrics:
                    # --- PŘEDVÝPOČET PRO GRAFY ---
                    # 1. Sezónnost (Měsíční průměry)
                    m.groupby(m['time'].dt.month)[['Qobs', 'Qrouted']].mean().to_csv(f"evaluated_data/plots/{st_id}_seasonal.csv")
                    
                    # 2. Roční průměry
                    m.groupby(m['time'].dt.year)[['Qobs', 'Qrouted']].mean().to_csv(f"evaluated_data/plots/{st_id}_yearly.csv")
                    
                    # 3. Čára překročení (FDC)
                    fdc_obs = np.sort(m['Qobs'].values)[::-1]
                    fdc_sim = np.sort(m['Qrouted'].values)[::-1]
                    percentiles = np.linspace(0, 100, 100)
                    pd.DataFrame({
                        'exceedance': percentiles,
                        'Qobs': np.interp(percentiles, np.linspace(0, 100, len(fdc_obs)), fdc_obs),
                        'Qsim': np.interp(percentiles, np.linspace(0, 100, len(fdc_sim)), fdc_sim)
                    }).to_csv(f"evaluated_data/plots/{st_id}_fdc.csv", index=False)

                    # Uložení do souhrnu
                    summary.append({
                        'ID': st_id, 
                        'LAT': row['LAT'], 
                        'LON': row['LON'], 
                        'KGE': metrics['kge'],
                        'Correlation': metrics['r'],
                        'Bias': metrics['beta'],
                        'Variability': metrics['gamma'],
                        'Count': len(m)
                    })
                    print(f"Hotovo: {st_id} | KGE: {metrics['kge']:.3f} | r: {metrics['r']:.2f}")
            else:
                print(f"Přeskočeno {st_id}: Nedostatečný překryv ({len(m)} dní)")
        else:
            print(f"Soubor nenalezen: {st_file}")
            
    except Exception as e:
        print(f"Chyba u ID {row.get('ID')}: {e}")

# Uložení souhrnného souboru
if summary:
    pd.DataFrame(summary).to_csv("evaluated_data/summary_stats.csv", index=False)
    print("\n=== Všechny mezivýpočty včetně KGE komponent uloženy ===")
else:
    print("\n!!! Žádná data nebyla vygenerována !!!")

def create_geojson(df, output_path):
    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "id": str(int(row['ID'])),
                "kge": float(row['KGE']),
                "r": float(row['Correlation']),
                "beta": float(row['Bias']),
                "gamma": float(row['Variability'])
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['LON']), float(row['LAT'])] # GeoJSON má [LON, LAT]
            }
        }
        features.append(feature)
    
    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)

if summary:
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("evaluated_data/summary_stats.csv", index=False)
    # Vytvoření GeoJSONu
    create_geojson(summary_df, "evaluated_data/stations.json")
    print("GeoJSON soubor stations.json byl vytvořen.")