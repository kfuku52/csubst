#!/usr/bin/env python3
"""Plot matched resource measurements and completed nested calibration results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

NAMES = ('baseline','joint','bridge')
LABELS = ('Main','Joint','Bridge')
COLORS = ('#667085','#2878B5','#D55E00')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--accuracy',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    records=json.loads(args.benchmark.read_text())['records']
    accuracy=json.loads(args.accuracy.read_text())
    if len(accuracy) != 12 or any(row['n'] != 100 for row in accuracy):
        raise ValueError('This comparison figure requires all 100 null and 100 alternative datasets.')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titlesize':12,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric,title,ylabel in [(axes[0,0],'seconds','PEPC wall time','Seconds'),
                                   (axes[0,1],'peak_tree_rss_bytes','PEPC peak process-tree RSS','GiB')]:
        for i,(name,label,color) in enumerate(zip(NAMES,LABELS,COLORS)):
            medians,lower,upper=[],[],[]
            for calibration in ('none','parametric_bootstrap'):
                values=np.array([r[metric] for r in records if r['label']==name and r['calibration']==calibration and not r['warmup']])
                if len(values) != 3:
                    raise ValueError('Three measured resource runs are required.')
                if metric != 'seconds':
                    values=values/2**30
                median=float(np.median(values))
                medians.append(median)
                lower.append(median-values.min())
                upper.append(values.max()-median)
            bars=ax.bar(np.arange(2)+(i-1)*.23,medians,width=.22,color=color,label=label,
                        yerr=np.array([lower,upper]),capsize=3,error_kw={'elinewidth':1})
            for bar,value in zip(bars,medians):
                ax.annotate(f'{value:.1f}' if metric=='seconds' else f'{value:.2f}',
                            (bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,6),
                            textcoords='offset points',ha='center',va='bottom',fontsize=9)
        ax.set_xticks([0,1],['No calibration','Refit bootstrap (B=3)'])
        ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',fontweight='bold')
        ax.set_ylim(0,ax.get_ylim()[1]*1.15)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    axes[0,0].legend(frameon=False)
    for ax,alternative,title in [(axes[1,0],False,'Complete-null false positives'),
                                  (axes[1,1],True,'Detection at an injected site')]:
        maximum=0
        for i,(name,color) in enumerate(zip(('marginal','joint','bridge'),COLORS)):
            for label,offset,marker in [('fixed',-.09,'o'),('refit',.09,'s')]:
                record=next(row for row in accuracy if row['alternative']==alternative and row['observation']==name and row['calibration']==label)
                result=record['signal'] if alternative else record
                mean=result['rate']*100
                lo,hi=np.array(result['ci95'])*100
                maximum=max(maximum,hi)
                ax.errorbar(i+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt=marker,color=color,
                            markerfacecolor='white' if label=='fixed' else color,markersize=7,
                            capsize=3,elinewidth=1.3)
        ax.set_xticks(range(3),LABELS)
        ax.set_xlim(-.5,2.5)
        ax.set_ylim(-1,min(100,max(20,np.ceil(maximum/10)*10+5)))
        ax.set_ylabel('Datasets (%)')
        ax.set_title(title,loc='left',fontweight='bold')
        ax.grid(axis='y',alpha=.15)
        if not alternative:
            ax.axhline(5,color='#333333',linestyle='--',linewidth=1)
            ax.text(2.45,5.5,'Nominal 5%',ha='right',fontsize=9)
    axes[1,1].legend(handles=[Line2D([],[],marker='o',color='#555555',markerfacecolor='white',linestyle='',label='Fixed Q / lengths'),
                               Line2D([],[],marker='s',color='#555555',linestyle='',label='Refit Q / lengths')],frameon=False)
    fig.suptitle('Joint / bridge observations with fitted scan calibration',fontsize=15,fontweight='bold',y=.99)
    fig.text(.08,.025,'Resources: median and range of 3 runs. Accuracy: 100 datasets per condition, 39 null replicates each.\nAccuracy error bars: exact 95% binomial intervals. Codon model: GY+FQ; topology and foreground fixed.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.07,1,.96],h_pad=3,w_pad=3)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180)
    fig.savefig(args.out.with_suffix('.svg'))
    plt.close(fig)


if __name__=='__main__':
    main()
