from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
g=gpd.read_file(next(R.glob('（6）*/*/*.shp'))).to_crs(32645)
rows=[]
for name in ['elevation_utm45n.tif','slope_degree.tif']:
 with rasterio.open(next(R.rglob(name))) as s:
  gg=g.to_crs(s.crs);values=[]
  for geom in gg.geometry:
   try:
    a,_=mask(s,[geom],crop=True,filled=False);v=a.compressed();v=v[np.isfinite(v)];values.append(float(v.mean()) if len(v) else np.nan)
   except ValueError:values.append(np.nan)
  g[name]=values
ice=gpd.read_file(next(R.rglob('RGI_V7*.shp'))).to_crs(32645).geometry.union_all()
g['distance_to_glacier_m']=g.geometry.distance(ice)
pd.DataFrame(g.drop(columns='geometry')).to_csv(R/'question2/spatial_features.csv',index=False)
print(g[['elevation_utm45n.tif','slope_degree.tif','distance_to_glacier_m']].describe().to_string())
