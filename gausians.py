import numpy as np
import matplotlib.pyplot as plt
from scipy import loadtxt, optimize
import csv
import os

def linearea(m, b, x0, x1):
    return (m*x1**2+b*x1) - (m*x0**2+b*x0)

def fitfunc(p, x):
    return (p[0]/np.sqrt(2*np.pi*p[1]**2)*np.exp(-(x-p[2])**2/(2*p[1]**2))
        +p[3]*x+p[4])
def residual(p, x, y, dy):
    return (fitfunc(p, x)-y)/(dy)

dirlist = ['JE-JR_Compton_Scattering_Lab/Data-Collection-Day1/',
    'JE-JR_Compton_Scattering_Lab/Data-Collection-Day2/']
otherdirlist = ['JE-JR_Compton_Scattering_Lab/Callibration-Day1/',
    'JE-JR_Compton_Scattering_Lab/Callibration-Day1/']
filelist = []

for folder in dirlist:
    for item in os.listdir(folder):
        if item.endswith('.tsv'):
            filelist.append(folder+item)

def getTheta(pathname):
    return int(pathname.split('/')[-1].split('.')[0].split('a')[-1])



oldroiDict = {15:(1300,2048), 5:(1400, 2048), 30:(1200,1825), 40:(1000,1700),
    50:(950,1425), 55:(875,1385), 60:(850,1300), 65:(750,1300), 70:(650,1250),
    75:(625,1200), 80:(550,1125)}

roiDict = {15:(1300,2048), 5:(1575, 1875), 30:(1250,1700), 40:(1150,1600),
    50:(950,1425), 55:(875,1385), 60:(850,1300), 65:(750,1300), 70:(650,1250),
    75:(625,1200), 80:(650,1050)}


callibration = True

datalist = []
for item in filelist:
    channel, energy, counts = loadtxt(item, unpack=True, skiprows=30)
    theta = getTheta(item)
    lbound, ubound = roiDict[theta]
    lbound -= int(channel[0])
    ubound -= int(channel[0])
    channel, energy, counts = (channel[lbound:ubound], energy[lbound:ubound],
     counts[lbound:ubound])
    counts += 1
    if theta == theta:
        print(item)
        dcount = np.sqrt(counts)
        #energy = channel
        mu = channel[counts.argmax()]
        sigma = (channel[-1] - channel[0])/(4*np.sqrt(2*np.log(2)))
        slope = (counts[-1] - counts[0])/(channel[-1] - channel[0])
        yintercept = counts[0] - slope * channel[0]
        N = sum(counts) - linearea(slope, yintercept, channel[-1], channel[0])
        p01 = [N, sigma, mu, slope, yintercept]
        print(p01)
        pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
            args = (channel, counts, dcount), full_output=1)
        plotfits = True
        if cov1 is None:
            print('Fit did not converge')
            print('Success code:', success1)
            print(mesg1)
            plotfits = False
        else:
            print('Fit Converged')
            print(info1)
            chisq1 = sum(info1['fvec']*info1['fvec'])
            dof1 = len(channel)-len(pf1)
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
        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(111)
        ax.errorbar(channel,counts,yerr=dcount,fmt='k.',label='data')
        if plotfits:
            X = np.linspace(channel[0], channel[-1], channel[-1]-channel[0])
            textfit = '$g(x) = \\dfrac{N}{\\sigma\\sqrt{2\\pi}}\\exp[\\frac{-(x-\\mu)^2}{2\\sigma^2}]$'\
                  '$+Ax+B$\n' \
                  '$\chi^2= %.2f$ \n' \
                  '$N = %i$ (dof) \n' \
                  '$\chi^2/N = % .2f$' \
                   % (chisq1, dof1, chisq1/dof1)
            text2 = '$N = %.0f\pm %.0f $ counts\n' \
                '$\sigma = %.1f \pm %.1f$ Channels \n' \
                '$\mu = %.1f \pm %.1f$ Channels\n' \
                '$A = %.3f \pm %.3f$ counts/Channel\n' \
                '$B = %.1f \pm %.1f$ counts\n' \
                 % (pf1[0], pferr1[0],pf1[1], pferr1[1],pf1[2],
                  pferr1[2],pf1[3], pferr1[3],pf1[4], pferr1[4])
            datalist.append((theta, pf1[2],pf1[1]))
            ax.plot(X, fitfunc(pf1,X),'b-',label='fit')
            ax.text(0.01, .99, textfit, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top')
            ax.text(0.7, .8, text2, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top')
        ax.set_xlabel('Channel (Channel Number)')
        ax.set_ylabel('Counts (number of counts)')
        addstr = ''
        # if theta == 40:
        #     addstr = 'Figure 5 - '
        plt.title(addstr+'Full Energy Peak for Cs-137 at $\\theta = $'+str(theta)+'$\\degree$')
        ax.legend()
        plt.savefig('Theta'+str(theta)+'gaussian_fit.pdf')
        print(sum(counts))
        plt.show()

with open('results.csv', 'w+') as results:
    writer = csv.writer(results)
    for datum in datalist:
        writer.writerow(datum)


