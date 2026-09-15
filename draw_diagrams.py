"""Editable vector schematics with explicit non-overlapping node geometry."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})

def box(ax,x,y,w,h,text,color='#EDF2F7',fs=10,edge='#314558'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.09',lw=1,ec=edge,fc=color))
    renderer=ax.figure.canvas.get_renderer()
    width=ax.transData.transform((x+w-.14,y))[0]-ax.transData.transform((x+.14,y))[0]
    height=ax.transData.transform((x,y+h-.10))[1]-ax.transData.transform((x,y+.10))[1]
    for size in [fs-i*.2 for i in range(21) if fs-i*.2>=7.8]:
        prop=FontProperties(family='DejaVu Sans',size=size);lines=[]
        for original in text.split('\n'):
            line=''
            for word in original.split():
                test=(line+' '+word).strip()
                if line and renderer.get_text_width_height_descent(test,prop,False)[0]>width:lines.append(line);line=word
                else:line=test
            lines.append(line)
        if len(lines)*size*ax.figure.dpi/72*1.16<=height:break
    ax.text(x+w/2,y+h/2,'\n'.join(lines),ha='center',va='center',fontsize=size,linespacing=1.16)

def arrow(ax,points,label=None,at=None):
    for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color='#44566A',lw=1.15)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,lw=1.15,color='#44566A'))
    if label:ax.text(*at,label,fontsize=8,ha='center',va='center',bbox=dict(fc='white',ec='none',pad=1.5))

def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def product():
    fig,ax=plt.subplots(figsize=(6.2,7.0));ax.set(xlim=(0,13.4),ylim=(0,11.0));ax.axis('off')
    box(ax,.5,9.4,3.7,1.1,'Sentinel-1\nVV, VH and angle','#E5EEF8')
    box(ax,4.85,9.4,3.7,1.1,'Sentinel-2 / Landsat\nOptical and thermal','#E6F1ED')
    box(ax,9.2,9.4,3.7,1.1,'Environment\nTerrain, soil, climate\nRainfall','#F7EFE0')
    for x in [2.35,6.7,11.05]:arrow(ax,[(x,9.35),(x,8.7)])
    box(ax,.5,7.55,12.4,1.1,'Common feature definitions and timestamps\nTraining-fitted transforms and missing-value rules\nNo fitting within the mapped region',fs=9.5)
    arrow(ax,[(6.7,7.5),(6.7,6.95)])
    panel=FancyBboxPatch((.5,2.4),12.4,4.5,boxstyle='round,pad=.05',lw=1.0,linestyle='--',ec='#637A91',fc='white');ax.add_patch(panel)
    ax.text(.8,6.65,'MATCHED MODELS WITHIN OUTER FOLD  k',fontsize=9,weight='bold',va='center')
    box(ax,4.6,5.05,4.2,1.0,'5 cm reference model\nRainfall descriptors\nSoilGrids 0–5 cm','#E5EEF8',fs=9.5)
    box(ax,1.0,3.15,3.4,1.05,'20 cm reference model\n15–30 cm soil + predicted 5 cm','#E6F1ED',fs=8.5)
    box(ax,5.25,3.15,2.9,1.05,'5 cm output\nfrom fold k','#E5EEF8',fs=9)
    box(ax,9.0,3.15,3.4,1.05,'50 cm reference model\n30–60 cm soil + predicted 5 cm','#F7EFE0',fs=8.5)
    arrow(ax,[(6.7,5.0),(6.7,4.58),(2.7,4.58),(2.7,4.23)])
    arrow(ax,[(6.7,5.0),(6.7,4.58),(10.7,4.58),(10.7,4.23)])
    arrow(ax,[(6.7,5.0),(6.7,4.23)])
    ax.text(6.7,2.73,'Depth models also receive satellite, static and timing inputs.\nDirect rainfall descriptors enter only the 5 cm model.',ha='center',va='center',fontsize=7.7,linespacing=1.1)
    for x in [2.7,6.7,10.7]:arrow(ax,[(x,2.33),(x,1.96)])
    box(ax,.5,.95,12.4,.95,'Average the five matched fold outputs at each depth\nApply land-cover and availability masks; export 5, 20 and 50 cm GeoTIFFs',fs=10)
    ax.text(6.7,.30,'Pass each upstream prediction to its matching depth model\nbefore averaging across folds.',ha='center',fontsize=8.5)
    save(fig,'scidata_fig2_product_pipeline')

def architecture():
    fig,ax=plt.subplots(figsize=(6.2,7.2));ax.set(xlim=(0,15.7),ylim=(0,11.7));ax.axis('off')
    stream=[(.4,'SAR\n4 features','#E5EEF8'),(3.7,'Optical and thermal\n19 features','#E6F1ED'),(7.0,'Timing and rainfall\n4 features + 3 at 5 cm','#F7EFE0')]
    for x,label,color in stream:
        box(ax,x,10.1,2.9,1.0,label,color,9)
        arrow(ax,[(x+1.45,10.04),(x+1.45,9.42)])
        box(ax,x,8.3,2.9,1.05,'MLP tokenizer\n[CLS] token\nModality ID',color,8.5)
        arrow(ax,[(x+1.45,8.24),(x+1.45,7.57)])
    box(ax,10.5,10.1,4.4,1.0,'Static numeric features\nand categorical embeddings','#F1E8F4',9)
    arrow(ax,[(12.7,10.04),(12.7,9.42)])
    box(ax,10.5,8.3,4.4,1.05,'Static projection\nDropout → linear → LayerNorm → GELU','#F1E8F4',8.6)
    box(ax,.4,6.75,9.5,.75,'Token concatenation',fs=10)
    arrow(ax,[(5.15,6.69),(5.15,6.10)])
    box(ax,.4,4.95,9.5,1.1,'Shared Transformer encoder\n2 pre-normalized layers  ·  8 heads  ·  token dimension 168',fs=10)
    arrow(ax,[(5.15,4.89),(5.15,3.92)],'Keys and values',(5.9,4.45))
    arrow(ax,[(12.7,8.24),(12.7,7.85),(15.2,7.85),(15.2,3.48),(10.0,3.48)],'Static query',(12.6,3.48))
    box(ax,10.5,5.45,4.4,1.25,'Depth models only\nPredicted 5 cm\nMLP tokens + [CLS]\n+ modality embedding','#F7EFE0',8.5)
    arrow(ax,[(10.44,6.075),(10.18,6.075),(10.18,7.125),(9.96,7.125)])
    box(ax,.4,3.0,9.5,.85,'Static-query cross-attention and residual refinement','#F1E8F4',10)
    arrow(ax,[(5.15,2.94),(5.15,2.36)])
    box(ax,.4,1.3,9.5,1.0,'Prediction head\nLayerNorm → 168 → 84 → 42 → 1',fs=10)
    ax.text(12.7,1.8,'5, 20 or 50 cm\nsoil moisture',fontsize=9,ha='center',va='center',linespacing=1.5)
    ax.text(7.75,.40,'Dropout and SoilGrids layer are depth-specific.\nArchitecture controls use the same input records.',ha='center',fontsize=8.3)
    save(fig,'isprs_fig2_architecture')

def isolation():
    fig,ax=plt.subplots(figsize=(6.2,6.5));ax.set(xlim=(0,14),ylim=(0,9.9));ax.axis('off')
    box(ax,.5,8.35,13,1,'Resolve station aliases and co-located sensor groups across all three depths\nExternal holdout = union of the legacy test groups',fs=10)
    arrow(ax,[(3.5,8.29),(3.5,7.59)]);arrow(ax,[(10.5,8.29),(10.5,7.59)])
    box(ax,.5,6.55,6,1,'Permitted development sites\nShared five-fold assignment',fs=10)
    box(ax,7.5,6.55,6,1,'External holdout sites\nExcluded at every fitting stage','#F7EFE0',10)
    arrow(ax,[(3.5,6.49),(3.5,5.79)])
    box(ax,.5,4.75,6,1,'Outer training sites\nInternal epoch selection, then refit',fs=10)
    box(ax,7.5,4.75,6,1,'Outer validation sites\nExcluded from upstream and depth fitting','#F7EFE0',9.5)
    arrow(ax,[(6.56,7.05),(7.05,7.05),(7.05,5.25),(7.44,5.25)])
    arrow(ax,[(2.0,4.69),(2.0,3.99)]);arrow(ax,[(5.0,4.69),(5.0,3.99)])
    box(ax,.5,2.7,3.0,1.23,'Inner 5 cm fits\nDepth-training\ninputs',fs=8.7)
    box(ax,4.0,2.7,3.0,1.23,'Outer 5 cm fit\nInputs for OOF\nand test',fs=8.7)
    box(ax,8.0,2.7,5.5,1.23,'Depth models\nFit only outer-training sites;\nevaluate on outer/test sites',fs=9)
    arrow(ax,[(2,2.64),(2,2.1),(10.75,2.1),(10.75,2.64)])
    arrow(ax,[(7.06,3.3),(7.94,3.3)])
    ax.text(7,.85,'Preprocessing and epoch selection obey the same site exclusions.\nThe holdout is a previously studied benchmark,\nnot new prospective validation.',ha='center',va='center',fontsize=9,linespacing=1.4)
    save(fig,'isprs_fig1_isolation')

if __name__=='__main__':product();architecture();isolation();print('THREE EDITABLE SCHEMATICS SAVED')
