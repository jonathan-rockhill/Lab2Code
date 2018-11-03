import numpy as np
import matplotlib.pyplot as plt
from scipy import loadtxt, optimize

def scattereqn(p, x):
    return p[0]/(1+(p[0]/p[1])*(1 - np.cos(x*(np.pi/180))))
def residual(p, x, y, dy):
    return (scattereqn(p, x)-y)/dy

theta, energy, denergy = loadtxt('results.csv', unpack=True, delimiter=',')

p01 = [662,511]

pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
            args = (theta, energy, denergy), full_output=1)

plotfit = True
if cov1 is None:
    print('Fit did not converge')
    print('Success code:', success1)
    print(mesg1)
    plotfit = False
else:
    print('Fit Converged')
    chisq1 = sum(info1['fvec']*info1['fvec'])
    dof1 = len(theta)-len(pf1)
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

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(min(theta),max(theta),500)
if plotfit:
    ax.plot(X,scattereqn(pf1,X),'b-',label='fit')
ax.errorbar(theta,energy,denergy,[2]*len(theta),fmt='bs',label='data',capsize=4)
ax.plot(X, scattereqn(p01,X),'g-',label='expected relationship')
plt.legend(loc='lower left')
ax.set_xlabel('Scattering Angle (Degrees)')
ax.set_ylabel('Energy of Scattered $\\gamma$ (KeV)')

text = "$E'(x) = \\dfrac{E}{1+\\frac{E}{mc^2}(1+\\cos\\theta)}$\n" \
    "$E = %.0f \\pm %.0f $KeV\n $mc^2 = %.0f \\pm %.0f$ KeV\n $\chi^2= %.2f$ \n" \
    '$N = %i$ (dof) \n $\chi^2/N = % .2f$' % (pf1[0],pferr1[0],pf1[1],pferr1[1],
        chisq1, dof1, chisq1/dof1)
ax.text(.99, .99, text, transform=ax.transAxes, fontsize=12,
    verticalalignment='top',horizontalalignment='right')

plt.title("Figure 6 - Energy of Scattered Photon vs. Scattering Angle")
plt.savefig('EvsTh.pdf')
# for angle in theta:
#     ax.plot(angle, scattereqn(p01,angle),'ko')
#     print('theta = ',angle)
#     print(scattereqn(p01,angle))
#plt.show()