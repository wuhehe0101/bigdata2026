# -*- coding: utf-8 -*-
"""从附件实际点位shp校核EPSG:32645坐标，并检查LUCD在源区/口岸的覆盖状态。
只读取坐标，不用土地覆盖NoData=0推断摩擦系数。"""
from pathlib import Path
import struct,json
import numpy as np
import rasterio
from pyproj import Transformer
ROOT=Path(__file__).resolve().parent
G=ROOT/'data'

def shp_single_point(path):
    b=Path(path).read_bytes()
    # shapefile fixed-length header (100B), record header (8B), point shapeType and XY
    if len(b)<128:raise ValueError('SHP文件过短')
    t=struct.unpack_from('<i',b,108)[0]
    if t!=1:raise ValueError(f'需要Point形状，得到{t}')
    lon,lat=struct.unpack_from('<dd',b,112)
    return float(lon),float(lat)

def check():
    tr=Transformer.from_crs(4326,32645,always_xy=True)
    locations={};details={}
    for key,filename in [('port','吉隆口岸.shp'),('incident','点位.shp')]:
        p=next(G.rglob(filename),None)
        if p is None:raise FileNotFoundError(f'缺少{filename}')
        lon,lat=shp_single_point(p)
        x,y=tr.transform(lon,lat)
        locations[key]={'x':x,'y':y,'longitude':lon,'latitude':lat,'origin':str(p.relative_to(ROOT))}
    lon,lat=85.55,28+20/60+9/3600
    x,y=tr.transform(lon,lat)
    locations['reference_source']={'x':x,'y':y,'longitude':lon,'latitude':lat,'origin':'殷跃平等(2026)滑震初步反演；非释放多边形'}
    dem=G/'elevation_utm45n.tif'
    with rasterio.open(dem) as ds:
        details['DEM']={'EPSG':ds.crs.to_epsg(),'像元分辨率_m':[abs(x) for x in ds.res],
             '尺寸_行列':[ds.height,ds.width],'高程采样_m':{}}
        for key,loc in locations.items():
            details['DEM']['高程采样_m'][key]=float(next(ds.sample([(loc['x'],loc['y'])]))[0])
    land=G/'jilong_LUCD_2023.tif'
    if land.exists():
        with rasterio.open(land) as ds:
            details['土地覆盖2023']={'EPSG':ds.crs.to_epsg(),'NoData':ds.nodata,'关键位置栅格类别':{},'备注':'0表示无数据，不得当作任何土地覆盖类型或摩擦分区'}
            for key,loc in locations.items():
                details['土地覆盖2023']['关键位置栅格类别'][key]=int(next(ds.sample([(loc['longitude'],loc['latitude'])]))[0])
    all_info={'地理坐标与数据出处':locations,'数据集核查':details,'方法约束':'土地覆盖在关键源区与口岸像元可能为NoData；未用于反演μ、ξ或k'}
    (G/'locations.json').write_text(json.dumps(locations,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'results'/'数据集核查.json').write_text(json.dumps(all_info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(all_info,ensure_ascii=False,indent=2))

if __name__=='__main__':check()
