#!/usr/bin/env python
import sys,os
import numpy as np

#--matplotlib
import matplotlib
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text',usetex=True)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pylab as py
import matplotlib.gridspec as gridspec

#--from tools
from tools import load,save,checkdir,lprint

def get_xy_steps(xbins,yvalues):
    X,Y=[],[]
    for i in range(len(xbins)-1):
        x1=xbins[i]
        x2=xbins[i+1]
        y=yvalues[i]    
        X.append(x1)
        Y.append(y)
        X.append(x2)
        Y.append(y)
    return np.array(X),np.array(Y) 

def plot_hist(ax1,ax2,obs,bins,range,yscale='log',legend=False,ylabel=False,logbin=False):

    #--load data
    pythia = np.load('data/%s_pythia.npy'%obs)
    gan    = np.load('data/%s_gan.npy'%obs)


    if logbin:
        bins=10**np.linspace(np.log10(range[0]),np.log10(range[1]),bins)


    #--plot normalized dist
    H,E=np.histogram(pythia,bins=bins,density=True,range=range)
    X,Y=get_xy_steps(E,H)
    hp,=ax1.plot(X,Y,'k-')

    samples=[]
    for d in gan: 
        H,E=np.histogram(d,bins=bins,density=True,range=range)
        X,_Y=get_xy_steps(E,H)
        samples.append(_Y)
    T=np.mean(samples,axis=0)
    dT=np.std(samples,axis=0)
    hbg=ax1.fill_between(X,T-dT,T+dT,color='Yellow')
    hg,=ax1.plot(X,T,'r-')
    if yscale=='log': ax1.semilogy()

    #--plot ratios
    ax2.fill_between(X,1-dT/T,1+dT/T,color='Yellow')
    ax2.plot(X,Y/T,'k-')
    ax2.set_ylim(0.8,1.2)

    #--set limits
    ax1.set_xlim(range[0],range[1])
    ax2.set_xlim(range[0],range[1])

    #--add xlabel
    if   obs=='px': label=r'\boldmath{$p_{x}$}'
    elif obs=='py': label=r'\boldmath{$p_{y}$}'
    elif obs=='pz': label=r'\boldmath{$p_{z}$}'
    elif obs=='e':  label=r'\boldmath{$E$}'
    elif obs=='pt':  label=r'\boldmath{$p_{\rm T}$}'
    elif obs=='theta':  label=r'\boldmath{$\theta$}'
    elif obs=='phi':  label=r'\boldmath{$\phi$}'
    elif obs=='xbj':  label=r'\boldmath{$x_{\rm bj}$}'
    elif obs=='Q2':  label=r'\boldmath{$Q^2$}'
    else:           label='.'
    ax2.xaxis.set_label_coords(0.95,-0.05)
    ax2.set_xlabel(label,size=25)

    #--set ticks
    ax1.tick_params(axis='both',which='both',direction='in',pad=4,labelsize=18)
    ax2.tick_params(axis='both',which='both',direction='in',pad=4,labelsize=18)

    #--set legend
    if legend:
        H=[hp,(hbg,hg)]
        L=[r'$\rm Pythia~8$',r'$\rm GAN$']
        ax1.legend(H,L,fontsize=25)

    #--set yrange
    ax1.set_ylim(2e-4,0.9)

    if ylabel:
        #ax1.xaxis.set_label_coords(0.95,-0.05)
        ax1.set_ylabel(r'\boldmath{$\rm Normalized~Yield$}',size=25)

    #--set tick labels
    ax1.set_xticklabels([])

def main1():

    nrows=(6+2)*3+2
    ncols=3
    fig = py.figure(figsize=(ncols*5,nrows*0.5))

    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig
         ,left=0.07, right=0.98,top=0.97,bottom=0.05 ,wspace=0.15,hspace=0.05)


    cnt=0

    bins=100
    range=(-9.9,9.9)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,0])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,0])
    plot_hist(ax1,ax2,'px',bins,range,yscale='log')

    bins=100
    range=(-9.9,9.9)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,1])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,1])
    plot_hist(ax1,ax2,'py',bins,range,yscale='log')

    bins=100
    range=(10.1,49.9)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,2])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,2])
    plot_hist(ax1,ax2,'pz',bins,range,yscale='log',legend=True)

    cnt=6+2+1

    bins=100
    range=(10.1,49.9)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,0])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,0])
    plot_hist(ax1,ax2,'e',bins,range,yscale='log',ylabel=True)

    bins=100
    range=(2.1,9.9)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,1])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,1])
    plot_hist(ax1,ax2,'pt',bins,range,yscale='log')

    bins=100
    range=(0.1,0.99)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,2])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,2])
    plot_hist(ax1,ax2,'theta',bins,range,yscale='log')
    ax1.set_ylim(2e-4,10)

    cnt=(6+2+1)*2

    bins=100
    range=(-3.9,3.9)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,0])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,0])
    plot_hist(ax1,ax2,'phi',bins,range,yscale='lin')
    ax1.set_ylim(0.001,0.2)

    bins=100
    range=(0.01,0.39)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,1])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,1])
    plot_hist(ax1,ax2,'xbj',bins,range,yscale='log')
    ax1.set_ylim(2e-1,10)
    ax1.set_yticks([1,10])
    ax1.set_yticklabels([r'$1$',r'$10$'])

    bins=100
    range=(3,200)
    ax1 = fig.add_subplot(gs[cnt:cnt+6  ,2])
    ax2 = fig.add_subplot(gs[cnt+6:cnt+8,2])
    plot_hist(ax1,ax2,'Q2',bins,range,yscale='log',logbin=True)
    ax1.semilogx()
    ax2.semilogx()
    ax1.set_ylim(2e-5,0.9)
    ax2.set_xticks([10,100])
    ax2.set_xticklabels([r'$10$',r'$100$'])

    ##########################################
    checkdir('gallery')
    py.savefig('gallery/main1.pdf')

def main2():
    from matplotlib.colors import LogNorm
    import  matplotlib as mpl
    from matplotlib import cm

    nrows,ncols=2,1
    fig = py.figure(figsize=(ncols*5,nrows*3))

    #------------------------------------------   
    ax=py.subplot(nrows,ncols,1)
    pyt=load('data/obs-pythia.dat')
    x, y = pyt['x'],pyt['Q2']
    xmin = np.log10(np.amin(x))
    xmax = np.log10(np.amax(x))
    ymin = np.log10(np.amin(y))
    ymax = np.log10(np.amax(y))
    xbins = np.logspace(-3,0, 50) 
    ybins = np.logspace(0, 3, 50) 
    counts,ybins_,xbins_,image = ax.hist2d(x,y,bins=[xbins,ybins]
        ,norm=LogNorm()
        ,cmap=cm.Greys
        )
    counts, xbins_, ybins_ = np.histogram2d(x, y, bins=(xbins, ybins))

    xbins_=0.5*(xbins_[:-1]+xbins_[1:])
    ybins_=0.5*(ybins_[:-1]+ybins_[1:])
    X, Y = np.meshgrid(xbins_, ybins_)
    cmax=np.amax(counts)
    ax.contour(X,Y,counts.transpose()
        ,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
        ,cmap = mpl.cm.gist_rainbow
        ,levels = [cmax*_ for _ in [0.01,0.1,0.2,0.4,0.6]]
        ,linewidths=3)

    ax.set_xscale("log")               # <- Activate log scale on X axis
    ax.set_yscale("log")               # <- Activate log scale on Y axis

    s = 4*50**2
    x=np.logspace(-3,0,10)
    Q2=s*x
    ax.plot(x,Q2,'k-')

    ax.tick_params(axis='x', which='mayor', labelsize=13,direction='in',length=5)
    ax.tick_params(axis='y', which='mayor', labelsize=13,direction='in',length=5)
    ax.set_xticklabels([])
    ax.set_yticks([10,100])
    ax.set_ylabel(r'\boldmath$Q^2$',size=30,rotation=0)
    ax.yaxis.set_label_coords(-0.12, 0.8)
    ax.text(0.1,0.8,r'$\rm Pythia$',size=20,transform=ax.transAxes)

    #------------------------------------------   
    ax=py.subplot(nrows,ncols,2)
    gan=load('data/obs-gan.dat')
    x, y = gan['x'],gan['Q2']
    xmin = np.log10(np.amin(x))
    xmax = np.log10(np.amax(x))
    ymin = np.log10(np.amin(y))
    ymax = np.log10(np.amax(y))
    xbins = np.logspace(-3,0, 50) 
    ybins = np.logspace(0, 3, 50) 
    counts,ybins_,xbins_,image = ax.hist2d(x,y,bins=[xbins,ybins]
        ,norm=LogNorm()
        ,cmap=cm.Greys
        )

    counts, xbins_, ybins_ = np.histogram2d(x, y, bins=(xbins, ybins))

    xbins_=0.5*(xbins_[:-1]+xbins_[1:])
    ybins_=0.5*(ybins_[:-1]+ybins_[1:])
    X, Y = np.meshgrid(xbins_, ybins_)
    #cmax=np.amax(counts)
    ax.contour(X,Y,counts.transpose()
        ,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
        ,cmap = mpl.cm.gist_rainbow
        ,levels = [cmax*_ for _ in [0.01,0.1,0.2,0.4,0.6]]
        ,linewidths=3)

    ax.set_xscale("log")               # <- Activate log scale on X axis
    ax.set_yscale("log")               # <- Activate log scale on Y axis

    s = 4*50**2
    x=np.logspace(-3,0,10)
    Q2=s*x
    ax.plot(x,Q2,'k-')

    ax.tick_params(axis='x', which='mayor', labelsize=13,direction='in',length=5)
    ax.tick_params(axis='y', which='mayor', labelsize=13,direction='in',length=5)
    ax.set_xlim(None,1)
    ax.set_xlabel(r'\boldmath$x_{\rm bj}$',size=30)
    ax.xaxis.set_label_coords(0.95, -0.15)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_xticks([0.001,0.01,0.1,1])
    ax.set_xticklabels([r'$0.001$',r'$0.01$',r'$0.1$',r'$1$'])
    ax.set_yticks([10,100])
    ax.text(0.1,0.8,r'$\rm GAN$',size=20,transform=ax.transAxes)

    #================================
    py.tight_layout()
    py.savefig('gallery/fig3.pdf')



def fig2():
    
    def get_xy_steps(xbins,yvalues):
        X,Y=[],[]
        for i in range(len(xbins)-1):
            x1=xbins[i]
            x2=xbins[i+1]
            y=yvalues[i]    
            X.append(x1)
            Y.append(y)
            X.append(x2)
            Y.append(y)
        return np.array(X),np.array(Y) 

    pythia = np.load("data/pz_pythia.npy")
    gan = np.load("data/pz_pythia.npy")


    beamEnergy = 50.0
    pz = np.log(beamEnergy - pythia)

    fig = py.figure(figsize=(8,6))
    ax = fig.add_subplot()


    H,E=np.histogram(pz, bins=100, density=True)
    
    X,Y=get_xy_steps(E,H)
    hp,=ax.plot(X,Y,'k-')
    ax.set_ylabel(r'\boldmath{$\rm Normalized~Yield$}', size = 24)
    ax.xaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_tick_params(labelsize=17)
    ax.set_xlabel(r'\boldmath${\cal T}(p_z)$', size = 24) 
    py.tight_layout()
    py.savefig('pz.pdf')


def fig4():

    gan1 = np.load("data/FAT-GAN-CARTESIAN.npy")
    gan2 = np.load("data/FAT-GAN-SPHERICAL.npy")
    gan3 = np.load("data/DS-GAN.npy")

    gan1  = gan1[0:199]
    gan2  = gan2[0:199]
    gan3  = gan3[0:199]


    fig, ax = py.subplots(figsize=(12,8))
    
    ax.plot(gan1,  'r-')
    ax.plot(gan2, 'blue')
    ax.plot(gan3,'green')

    ax.legend([r'$\rm FAT-GAN(Cartesian)$',r'$\rm FAT-GAN (Spherical)$', r'$\rm DS-GAN$'],
              fontsize = 25, loc=2, prop={'size': 15})

    ax.set_ylabel(r'\boldmath$x^2$', size=30) 
    ax.set_yscale('log')

    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)

    ax.set_xlabel(r'\boldmath$Epochs$', size = 30)
    ax.xaxis.set_label_coords(0.50,-0.08)
    ax.set_xticklabels([r'$0$',r'$1000$', r'',r'$50000$',r'','$10000$',r'',r'$150000$',r'', r'$200000$'])


    py.tight_layout()
    py.savefig("fig4.pdf")
    

if __name__=="__main__":

    main1()















