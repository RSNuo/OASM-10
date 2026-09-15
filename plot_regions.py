"""Geographically proportioned regional panels with explicit print geometry."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import patheffects
import rasterio
from rasterio.transform import array_bounds,Affine

REV=Path(__file__).resolve().parent
NAMES={'US_drought':'Central United States','EastEuropeanPlain':'East European Plain','China':'China','Australia':'Australia',
       'India':'India','SouthAfrica':'South Africa','Pampas':'Pampas','centralvalley':'California Central Valley'}

def render_regions(save,layout_probe=False):
    source='validation/map_pipeline/region_requests.json' if layout_probe else 'product/example_geotiffs/manifest.json'
    requests=json.loads((REV/source).read_text());records=[]
    for request in requests:
        folder=REV/'product/example_geotiffs'/request['key']
        if layout_probe:
            raw=REV/'validation/map_pipeline/regions'/request['key'];grid=json.loads((raw/'grid.json').read_text())
            f=pd.read_parquet(raw/'features.parquet');shape=(grid['height'],grid['width'])
            rgb=np.stack([f[c].to_numpy().reshape(shape) for c in ['Sentinel2_B4','Sentinel2_B3','Sentinel2_B2']])
            bounds=array_bounds(*shape,Affine(*grid['transform']))
        else:
            with rasterio.open(folder/'S2_true_color_reflectance.tif') as src:rgb=src.read();bounds=tuple(src.bounds)
        left,bottom,right,top=bounds;latitude=(bottom+top)/2
        dx=(right-left)*111319.49*np.cos(np.deg2rad(latitude));dy=(top-bottom)*111319.49
        rgb=np.moveaxis(np.clip(rgb/.30,0,1)**(1/1.5),0,-1);rgb[~np.isfinite(rgb).all(axis=2)]=1
        matrices=[rgb]
        for depth in [5,20,50]:
            if layout_probe:matrices.append(None)
            else:
                with rasterio.open(next(folder.glob(f'OASM10_{depth}cm_*.tif'))) as src:matrices.append(src.read(1))
        records.append({**request,'bounds':bounds,'dx':dx,'dy':dy,'matrices':matrices})
    # Fixed physical panel widths avoid the large empty columns introduced by
    # constrained layout when mixing spanning headings with equal-aspect maps.
    width=7.1;left_margin=.52;right_margin=.12;column_gap=.12
    panel_width=(width-left_margin-right_margin-3*column_gap)/4
    for part in [0,1]:
        rows=records[4*part:4*(part+1)]
        heights=[panel_width*r['dy']/r['dx'] for r in rows]
        header_height=.24;bottom_gap=.18;column_titles=.24;footer=.57;top_margin=.1
        height=sum(heights)+len(rows)*(header_height+bottom_gap)+column_titles+footer+top_margin
        fig=plt.figure(figsize=(width,height));cursor=height-top_margin
        sm=plt.cm.ScalarMappable(norm=Normalize(0,.6),cmap='viridis')
        for row,(r,panel_height) in enumerate(zip(rows,heights)):
            label=f'({chr(97+4*part+row)}) {NAMES[r["key"]]} · {r["target_time"][:10]}'
            fig.text(left_margin/width,(cursor-.11)/height,label,fontsize=8.7,fontweight='bold',va='center')
            cursor-=header_height
            if row==0:
                for col,title in enumerate(['S2 true color','5 cm','20 cm','50 cm']):
                    center=left_margin+col*(panel_width+column_gap)+panel_width/2
                    fig.text(center/width,(cursor-.09)/height,title,fontsize=8.7,fontweight='bold',ha='center',va='center')
                cursor-=column_titles
            bottom=cursor-panel_height;dx=r['dx'];dy=r['dy'];west,south,east,north=r['bounds']
            for col,matrix in enumerate(r['matrices']):
                x=left_margin+col*(panel_width+column_gap)
                ax=fig.add_axes([x/width,bottom/height,panel_width/width,panel_height/height])
                if matrix is None:
                    ax.set_facecolor('#eeeeee');ax.set_xlim(0,dx);ax.set_ylim(0,dy)
                    ax.text(.5,.5,'Layout only\nAwaiting\nfitted output',transform=ax.transAxes,ha='center',va='center',fontsize=7.5,color='#777777')
                else:
                    kwargs={} if col==0 else {'cmap':'viridis','vmin':0,'vmax':.6}
                    ax.imshow(matrix,extent=[0,dx,0,dy],origin='upper',**kwargs)
                ax.set_aspect('equal')
                if col==0:
                    ax.set_xticks([0,dx],[f'{west:.3f}°',f'{east:.3f}°'],fontsize=7.2)
                    ax.set_yticks([0,dy],[f'{south:.3f}°',f'{north:.3f}°'],fontsize=7.2)
                else:ax.set_xticks([]);ax.set_yticks([])
                ax.tick_params(length=2,pad=1)
                scale=1000 if dx>2200 else 500;bar_x=.08*dx;bar_y=.1*dy
                line=ax.plot([bar_x,bar_x+scale],[bar_y,bar_y],color='white',lw=1.6)[0]
                line.set_path_effects([patheffects.Stroke(linewidth=2.8,foreground='black'),patheffects.Normal()])
                label=ax.text(bar_x+scale/2,bar_y+.04*dy,f'{scale} m',ha='center',fontsize=7.2,color='white')
                label.set_path_effects([patheffects.withStroke(linewidth=1,foreground='black')])
                for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.4)
            cursor=bottom-bottom_gap
        colorbar_ax=fig.add_axes([.21,.29/height,.64,.095/height])
        cb=fig.colorbar(sm,cax=colorbar_ax,orientation='horizontal')
        cb.ax.tick_params(labelsize=7.2,length=2,pad=2)
        cb.set_label('Volumetric soil moisture (m³/m³); shared display range',fontsize=8)
        save(fig,('_layout_regions' if layout_probe else 'scidata_fig3_regions')+str(part+1))
