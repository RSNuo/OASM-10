"""Scientific figures from the corrected protocol only; journal roles stay distinct."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm,Normalize,TwoSlopeNorm
from matplotlib import patheffects
import cartopy.crs as ccrs
import cartopy.feature as cf
from analyze_results import statistics,read,site_statistics
from compare_products import product_frame

REV=Path(__file__).resolve().parent;FIG=REV/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
  'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
  'savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
DEPTHS=[5,20,50];COLORS={'ma':'#176b89','rf':'#ba5a26','xgb':'#768731','ft':'#855e9e'}

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.08)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.08);plt.close(fig)

def map_background(ax):
    ax.set_extent([-175,180,-58,78],ccrs.PlateCarree())
    ax.add_feature(cf.LAND.with_scale('110m'),facecolor='#f1f2f2',edgecolor='none',zorder=0)
    ax.coastlines(resolution='110m',linewidth=.3,color='#aeb6ba')
    ax.spines['geo'].set_visible(False)

def stations():
    fig=plt.figure(figsize=(7.2,3.7));gs=fig.add_gridspec(2,2,hspace=.23,wspace=.12)
    allsets={}
    for j,d in enumerate(DEPTHS):
        f=pd.read_parquet(REV/f'data/all_{d}cm.parquet');g=f.groupby(['physical_site_id','split'])[['longitude','latitude']].median().reset_index()
        allsets[d]=set(g.physical_site_id)
        ax=fig.add_subplot(gs[j//2,j%2],projection=ccrs.PlateCarree());map_background(ax)
        for role,marker,color,size in [('development','o','#6a8599',5),('test','^','#b54531',8)]:
            z=g[g.split==role];ax.scatter(z.longitude,z.latitude,s=size,marker=marker,color=color,alpha=.65 if role=='development' else .9,
                  linewidths=.25,edgecolors='white',transform=ccrs.PlateCarree(),label=f'{role.capitalize()} ({len(z)})',zorder=2)
        ax.set_title(f'({chr(97+j)}) {d} cm',loc='left',fontweight='bold');ax.legend(loc='lower left',frameon=False,fontsize=6.8,handletextpad=.25)
    ax=fig.add_subplot(gs[1,1]);ax.set_title('(d) Spatial-site depth coverage',loc='left',fontweight='bold',fontsize=8)
    labels=[];values=[]
    union=set.union(*allsets.values())
    for combo in [(5,),(20,),(50,),(5,20),(5,50),(20,50),(5,20,50)]:
        c=sum(tuple(d for d in DEPTHS if s in allsets[d])==combo for s in union)
        labels.append('+'.join(map(str,combo)));values.append(c)
    x=np.arange(len(labels));ax.bar(x,values,color='#738da0',width=.65)
    ax.set_xticks(x,labels,fontsize=5.8);ax.set_xlabel('Depths present (cm)',fontsize=7);ax.set_ylabel('Sites',fontsize=7)
    ax.set_ylim(0,max(values)*1.22);ax.set_box_aspect(.42)
    for i,v in enumerate(values):ax.text(i,v+max(values)*.03,str(v),ha='center',fontsize=6.4)
    save(fig,'scidata_fig1_sites')

def region_maps(layout_probe=False):
    from plot_regions import render_regions
    render_regions(save,layout_probe=layout_probe)


def scatters():
    edges=np.linspace(0,.6,61);norm=LogNorm(1e-5,1e-2)
    for d in DEPTHS:
        fig,axes=plt.subplots(1,2,figsize=(6.6,2.1),layout='constrained')
        for ax,split in zip(axes,['oof','test']):
            f=product_frame(d,split);z=f[f.output_eligible];s=statistics(z)
            counts,_,_=np.histogram2d(z.y,z.prediction,bins=(edges,edges));prop=counts/len(z)
            im=ax.pcolormesh(edges,edges,np.ma.masked_equal(prop.T,0),cmap='viridis',norm=norm,rasterized=True)
            ax.plot([0,.6],[0,.6],ls='--',c='#7c7c7c',lw=.7);ax.set(xlim=(0,.6),ylim=(0,.6),aspect='equal')
            ax.set_title(f'{d} cm · {"Outer OOF" if split=="oof" else "Held-out test"}',loc='left',fontsize=8,fontweight='bold')
            ax.set_xlabel('Observed SM (m³/m³)',fontsize=7);ax.set_ylabel('Predicted SM (m³/m³)',fontsize=7)
            ax.set_xticks([0,.2,.4,.6]);ax.set_yticks([0,.2,.4,.6])
            ax.text(.035,.965,f'n = {len(z):,}\nRMSE = {s["rmse"]:.4f}\nR² = {s["r2"]:.3f}',transform=ax.transAxes,va='top',fontsize=6.5,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1.5})
        cb=fig.colorbar(im,ax=axes,fraction=.025,pad=.02);cb.set_label('Fraction per bin',fontsize=7);cb.ax.tick_params(labelsize=6)
        save(fig,f'scidata_fig4_{d}cm')

def quality_maps():
    fig,axes=plt.subplots(3,2,figsize=(7.1,6.0),subplot_kw={'projection':ccrs.PlateCarree()},layout='constrained')
    for row,d in enumerate(DEPTHS):
        s=pd.read_csv(REV/f'analysis/product_{d}cm_site_quality.csv')
        for col,(metric,cmap,norm) in enumerate([('rmse','viridis',Normalize(0,.2)),('bias','RdBu_r',TwoSlopeNorm(0,-.15,.15))]):
            ax=axes[row,col];map_background(ax)
            im=ax.scatter(s.longitude,s.latitude,c=s[metric],s=12,cmap=cmap,norm=norm,linewidths=.2,edgecolors='#777777',zorder=2)
            ax.set_title(f'({chr(97+2*row+col)}) {d} cm · {"RMSE" if col==0 else "Bias"}',loc='left',fontweight='bold')
            if row==2:
                cb=fig.colorbar(im,ax=axes[:,col],orientation='horizontal',fraction=.03,pad=.04,extend='both' if col else 'max')
                cb.set_label(('RMSE' if col==0 else 'Prediction − observation')+' (m³/m³)')
    save(fig,'scidata_fig5_site_quality')

def temporal():
    sel=json.loads((REV/'analysis/quality_selection.json').read_text())['temporal_example']
    fig,axes=plt.subplots(3,1,figsize=(6.8,5.5),sharex=True,sharey=True,layout='constrained')
    for ax,d in zip(axes,DEPTHS):
        f=pd.read_parquet(REV/f'analysis/temporal_example_{d}cm.parquet');f['date']=pd.to_datetime(f.target_time).dt.normalize()
        z=f.groupby('date')[['y','prediction']].mean().sort_index()
        # NaN rows interrupt line segments across gaps exceeding14days.
        breaks=z.index.to_series().diff().gt(pd.Timedelta('14d'))
        for date in z.index[breaks]:z.loc[date-pd.Timedelta('1ns')]=np.nan
        z=z.sort_index()
        ax.plot(z.index,z.y,'o-',markersize=2,lw=.7,color='#333333',label='Observed')
        ax.plot(z.index,z.prediction,'s-',markersize=2,lw=.8,color=COLORS['ma'],label='Reference prediction')
        ax.set_ylabel('SM (m³/m³)');ax.set_title(f'{d} cm · {sel["sources"][str(d)]["source_station"]}',loc='left',fontweight='bold')
        ax.grid(axis='y',alpha=.2);ax.set_ylim(0,.65)
        if z.y.dropna().nunique()==1:
            ax.text(.02,.91,'Constant-zero reference series; temporal r undefined',transform=ax.transAxes,fontsize=7,color='#9c402f',va='top')
    axes[0].legend(loc='upper right',frameon=False,ncol=2,fontsize=7)
    axes[-1].set_xlabel(f'Target date ({sel["year"]}); daily display averages, identical records')
    save(fig,'scidata_supp_fig1_temporal')

def common_site_quality():
    f=pd.read_csv(REV/'analysis/common_site_quality.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.65),layout='constrained')
    for ax,col,label in zip(axes,['r','ubrmse','amplitude_ratio'],['(a) Within-site correlation','(b) Site ubRMSE (m³/m³)','(c) Predicted / observed SD']):
        for d,color in zip(DEPTHS,['#176b89','#b76129','#79519b']):
            vals=np.sort(f.loc[f.depth_cm==d,col].dropna().to_numpy())
            ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',lw=1.3,color=color,label=f'{d} cm (n={len(vals)})')
        ax.set_title(label,loc='left',fontsize=8,fontweight='bold');ax.set_ylim(0,1.03);ax.grid(alpha=.15)
    axes[0].set_xlim(-1,1);axes[0].set_ylabel('Fraction of common sites');axes[0].legend(frameon=False,fontsize=7,loc='upper left')
    axes[1].set_xlim(left=0);axes[2].set_xscale('log');axes[2].axvline(1,color='#aaa',ls='--',lw=.7)
    axes[2].set_xticks([.1,.25,.5,1,2,4,8],['0.1','0.25','0.5','1','2','4','8']);axes[2].set_xlim(.1,8)
    save(fig,'scidata_fig6_common_quality')

def paired():
    a=json.loads((REV/'analysis/results.json').read_text())['depths']
    fig,axes=plt.subplots(1,3,figsize=(7.3,3.5),layout='constrained')
    for ax,d in zip(axes,DEPTHS):
        variants=['concat','ft','no_modality','static_forcing'] if d==5 else ['concat','ft','no_cascade']
        labels={'concat':'Concatenation','ft':'FT-style','no_modality':'No grouping','static_forcing':'Static + forcing','no_cascade':'No cascade'}
        for i,v in enumerate(variants):
            s=a[str(d)]['paired_controls'][v+'_minus_ma_test'];lo,hi=s['rmse_ci95'];m=s['mean_rmse_difference']
            ax.plot([lo,hi],[i,i],color=COLORS['ma'],lw=2);ax.scatter([m],[i],s=35,marker='D',color=COLORS['ma'],zorder=3)
            for j,value in enumerate(s['per_seed_rmse_differences']):ax.scatter(value,i+.15+(j-1)*.065,s=14,marker=['o','s','^'][j],color=['#333333','#777777','#bbbbbb'][j],zorder=3)
        ax.axvline(0,color='#b0b0b0',ls='--',lw=.8);ax.set_yticks(range(len(variants)),[labels[v] for v in variants]);ax.invert_yaxis()
        ax.set_xlabel('Control − reference RMSE\n(m³/m³)');ax.set_title(f'{d} cm',fontweight='bold');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.tick_params(axis='x',labelsize=7)
        ax.grid(axis='x',alpha=.2)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='',label=f'Seed {s}',ms=4) for s,m,c in zip([42,7,555],['o','s','^'],['#333333','#777777','#bbbbbb'])]+[Line2D([],[],color=COLORS['ma'],marker='D',label='Mean and 95% CI',ms=4)],loc='outside lower center',ncol=4,frameon=False,fontsize=7)
    save(fig,'isprs_fig3_paired')

def error_structure():
    a=json.loads((REV/'analysis/results.json').read_text())['depths']
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
    ax=axes[0,0];bins=np.arange(0,.651,.05)
    for variant in ['ma','rf','xgb']:
        z=read(5,variant);z['bin']=pd.cut(z.y,bins,right=False)
        g=z.groupby('bin',observed=True).agg(x=('y','median'),p=('prediction','median'),n=('y','size'));g=g[g.n>=30]
        ax.plot(g.x,g.p,'o-',color=COLORS[variant],label={'ma':'Reference','rf':'RF','xgb':'XGBoost'}[variant],ms=3,lw=1)
    ax.plot([0,.6],[0,.6],'--',color='#999999',lw=.7);ax.set(xlabel='Observed SM (m³/m³)',ylabel='Conditional median prediction',xlim=(0,.6),ylim=(0,.6));ax.set_title('(a) Surface conditional response',loc='left');ax.legend(frameon=False,fontsize=7)
    ax=axes[0,1]
    for j,v in enumerate(['ma','rf','xgb']):
        vals=[]
        for d in DEPTHS:
            s=site_statistics(read(d,v));vals.append(s.amplitude_ratio.dropna().to_numpy())
        bp=ax.boxplot(vals,positions=np.arange(3)+(j-1)*.23,widths=.18,showfliers=False,patch_artist=True,manage_ticks=False)
        for b in bp['boxes']:b.set_facecolor(COLORS[v]);b.set_alpha(.6)
    ax.axhline(1,color='#999999',ls='--',lw=.7);ax.set_xticks(range(3),['5 cm','20 cm','50 cm']);ax.set_ylabel('Predicted / observed within-site SD');ax.set_title('(b) Within-site amplitude',loc='left')
    ax=axes[1,0];x=np.arange(4);labs=[];bias=[];center=[]
    for d in [20,50]:
        for v in ['rf','xgb']:
            z=a[str(d)]['paired_controls']['ma_minus_'+v+'_test']['mse_decomposition'];labs.append(f'{d} cm\n{v.upper()}');bias.append(z['bias_squared_reduction']);center.append(z['centered_mse_reduction'])
    ax.bar(x-.18,bias,width=.34,color='#957a55',label='Bias² reduction');ax.bar(x+.18,center,width=.34,color='#447c91',label='Centered MSE reduction');ax.axhline(0,color='#777777',lw=.7)
    ax.set_xticks(x,labs);ax.set_ylabel('Tree MSE − reference MSE');ax.set_title('(c) Deep error decomposition',loc='left');ax.legend(frameon=False,fontsize=7)
    ax=axes[1,1]
    for j,v in enumerate(['ma','rf','xgb']):
        vals=[a[str(d)]['models'][v+'_seed42']['test']['pooled_within_site_anomaly_r'] for d in DEPTHS]
        ax.plot(range(3),vals,'o-',color=COLORS[v],ms=4,lw=1,label=v)
    ax.set_xticks(range(3),['5 cm','20 cm','50 cm']);ax.set_ylabel('Within-site anomaly correlation');ax.set_title('(d) Temporal association',loc='left');ax.grid(axis='y',alpha=.2)
    save(fig,'isprs_fig4_error_structure')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--static-only',action='store_true');parser.add_argument('--region-layout-only',action='store_true');args=parser.parse_args()
    if args.region_layout_only:region_maps(layout_probe=True);return
    stations()
    if not args.static_only:region_maps();scatters();quality_maps();temporal();common_site_quality();paired();error_structure()
    print('V6 FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()
