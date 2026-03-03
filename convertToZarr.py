import xarray as xr
import os

def convert_nc_to_zarr(nc_path, zarr_path):
    if not os.path.exists(nc_path):
        print(f"Chyba: Soubor {nc_path} nebyl nalezen.")
        return

    print(f"Načítám {nc_path}...")
    # Otevíráme původní NetCDF
    ds = xr.open_dataset(nc_path)

    # KLÍČOVÝ KROK: Chunking
    # 'time': -1 znamená, že celá časová osa pro jeden pixel je v jednom kuse.
    # To je důvod, proč grafy budou vyskakovat okamžitě.
    ds_optimized = ds.chunk({
        'time': -1, 
        'lat': 20, 
        'lon': 20
    })

    print(f"Převádím na Zarr (vytvářím složku {zarr_path})...")
    # consolidated=True uloží metadata do jednoho souboru pro rychlý start
    ds_optimized.to_zarr(zarr_path, mode='w', consolidated=True)
    print(f"Hotovo: {zarr_path} je připraven.")

if __name__ == "__main__":
    # Spustíme převod pro oba tvoje soubory
    convert_nc_to_zarr("mHM_Fluxes_States.nc", "mHM_data.zarr")
    convert_nc_to_zarr("mRM_Fluxes_States.nc", "mRM_data.zarr")