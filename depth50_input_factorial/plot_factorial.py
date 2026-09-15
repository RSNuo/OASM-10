"""Standalone scientific result figure; no manuscript figures are modified."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    assert json.loads((HERE/'independent_verification.json').read_text())['passed']
    r=json.loads((HERE/'results.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.8,6.3),sharey=True)
    labels=['B: neither input','C: cascade + direct rain','D: direct rain only']
    colors={'oof':'#3579a8','test':'#ce7842'}
    for row,(metric,ci,title) in enumerate([
        ('mean_rmse_difference','rmse_ci95','RMSE difference (m³/m³; lower is better)'),
        ('anomaly_r_difference','anomaly_r_ci95','Anomaly correlation difference (higher is better)')]):
        for col,split in enumerate(['oof','test']):
            ax=axes[row,col];ax.axvline(0,color='#757575',lw=.9,ls='--')
            for y,key in enumerate(['B-A','C-A','D-A']):
                v=r[split]['contrasts'][key];m=v[metric];lo,hi=v[ci]
                ax.hlines(y,lo,hi,color=colors[split],lw=2.5)
                ax.plot(m,y,'o',color=colors[split],ms=6,zorder=3)
                for off,s in zip([-.10,0,.10],['42','7','555']):
                    ax.plot(v['per_seed'][s][metric],y+off,'.',color='#323232',ms=4,zorder=4)
            ax.set_yticks(range(3),labels);ax.set_ylim(2.45,-.45)
            ax.set_xlabel(title);ax.grid(axis='x',alpha=.14)
            if row==0:ax.set_title('Development OOF' if split=='oof' else 'Previously examined test set',fontweight='bold')
    fig.suptitle('50 cm: direct precipitation × predicted 5 cm input',fontsize=15,y=.98)
    fig.text(.04,.032,'All differences are relative to A (5 cm cascade, no direct precipitation). Bars: paired 95% site-bootstrap intervals.\nSmall dark points: individual seeds 42, 7, 555. Larger colored points: mean of paired seed differences.',fontsize=9,color='#444444')
    fig.tight_layout(rect=[0,.11,1,.94])
    fig.savefig(HERE/'factorial_comparison.png',dpi=220,facecolor='white')
    fig.savefig(HERE/'factorial_comparison.pdf',facecolor='white')
    plt.close(fig)
    print('FACTORIAL_PLOT_COMPLETE')

if __name__=='__main__':main()
