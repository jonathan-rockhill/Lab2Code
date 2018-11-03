import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def fitfunc(p, x):
    return p[0]*x + p[1]

def residual(p, x, y, dy):
    return (fitfunc(p, x)-y)/dy

energy = np.array([511,356,81])

channel = [np.array([1340,950,230]),np.array([1350,960,230])]
dchannel = [np.array([40,30,10]),np.array([40,40,10])]
fig = plt.figure()
subplotnum = 121
fig.text(0.5, 0.04, 'Channel Number', ha='center')
fig.text(0.04, 0.54, 'Energy (KeV)', va='center', rotation='vertical')
fmtlist = ['bs','r^']
tlist = ['day one', 'day two']
for i in range(len(channel)):
    ax = fig.add_subplot(subplotnum+i)
    x = energy
    y = channel[i]
    m = (y[-1]-y[1])/(x[-1]-x[1])
    p01 = [m,y[1]-m*x[1]]
    dy = dchannel[i]
    ax.errorbar(y,x,xerr=dy,fmt=fmtlist[i],label=tlist[i]+' data')
    pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
            args = (x, y, dy), full_output=1)
    plotfits = True
    if cov1 is None:
        print('Fit did not converge')
        print('Success code:', success1)
        print(mesg1)
        plotfits = False
    else:
        print('Fit Converged')
        chisq1 = sum(info1['fvec']*info1['fvec'])
        dof1 = len(x)-len(pf1)
        pferr1 = [np.sqrt(cov1[i,i]) for i in range(len(pf1))]
        print('Converged with chi-squared', chisq1)
        print('Number of degrees of freedom, dof =',dof1)
        print('Reduced chi-squared:', chisq1/dof1)
        print('Inital guess values:')
        print('  p0 =', p01)
        print('Best fit values:')
        print('  pf =', pf1)
        print('Uncertainties in the best fit values:')
        print('  pferr =', pferr1)
        print()
    if plotfits:
        X = np.linspace(min(y),max(y),500)
        ax.plot(X, fitfunc([1/pf1[0],-pf1[1]/pf1[0]], X),label=tlist[i]+' fit')
        newberr = pf1[1]/pf1[0]*np.sqrt((pferr1[1]/pf1[1])**2 + (pferr1[0]/pf1[0])**2)
        sqsum = sum((energy-np.average(energy))**2)
        fits = fitfunc([1/pf1[0],-pf1[1]/pf1[0]], y) 
        residsum = sum((energy - fits)**2)
        print(1-residsum/sqsum)
        textfit = '$f(x) = Ax + B$ \n' \
          '$A = %.2f \pm %.2f$ \n KeV/Channel\n' \
          '$B = %.0f \pm %.0f$ KeV\n' \
          '$\chi^2= %.2f$ \n' \
          '$N = %i$ (dof) \n' \
          '$\chi^2/N = % .2f$' \
          % (1/pf1[0], 1/pf1[0]*pferr1[0]/pf1[0], -pf1[1]/pf1[0], newberr,
            chisq1, dof1, chisq1/dof1)
        ax.text(0.99, .4, (textfit), transform=ax.transAxes, fontsize=12,
             verticalalignment='top',horizontalalignment='right')
    plt.legend(loc='upper left')
plt.suptitle('Figures 2 and 3 - Calibration Data for Both Days')
plt.savefig('Calibration_plots.pdf')
plt.show()
           