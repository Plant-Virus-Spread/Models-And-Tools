import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy import stats
from lmfit import minimize, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt
import os
#==============================================================================
''' Make new data matrix, same as csv except infected cells are one total for convience '''
DataFrame = pd.read_csv('Cell_count_data_Tromas_2014.csv') # Read data from file

TROMAS_DATA = np.empty((DataFrame.shape[0], 5), int)
for i in range(DataFrame.shape[0]): # Number of rows
    for j in range(5):              # Number of columns desired
        TROMAS_DATA[i][j] = DataFrame.iloc[i, j]
        if j == 4:
            TROMAS_DATA[i][j] = DataFrame.ix[i, 'Venus_only'] + DataFrame.ix[i, 'BFP_only'] + DataFrame.ix[i, 'Mixed']
'''
Col 0: Days post infection
Col 1: Leaf number
Col 2: Replicate plant number
Col 3: Number of unifected cells
Col 4: Number of total infected cells
'''
#==============================================================================
''' Create subplot array '''
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
fig.subplots_adjust(hspace = .25, wspace = .2)

fig_res, ax_res = plt.subplots(2, 2, figsize = (8, 8))
fig_res.subplots_adjust(hspace = .25, wspace = .2)
#==============================================================================
''' Make axis for data points '''
DAYS_AXIS = [3, 5, 7, 10]
#==============================================================================
''' Make axis for negative log likelihood '''
ZERO_DAYS_AXIS = [0, 3, 5, 7, 10]
#==============================================================================
''' Time axis for models '''
t = np.linspace(0, 10, 1000)
#==============================================================================
''' ABCS '''
ABCS = ['A', 'B', 'C', 'D']
#==============================================================================
''' Leafy bois '''
LEAVES = ['Leaf 3', 'Leaf 5', 'Leaf 6', 'Leaf 7']
#==============================================================================
''' Parameter list (All initialized with guesses) '''
likeParams = Parameters()
likeParams.add('I0', value = .0002, min = .0000001, max = 1.0000, vary=True)
likeParams.add('b', value = .950, min = .0001, max = 2.0000, vary=True)
likeParams.add('x5', value = .167, min = .0001, max = 2.0000, vary=True)
likeParams.add('x6', value = 1.046, min = .0001, max = 2.0000, vary=True)
likeParams.add('x7', value = .026, min = .0001, max = 2.0000, vary=True)
likeParams.add('psi3', value = .080, min = .0001, max = 1.0000, vary=True)
likeParams.add('psi5', value = .016, min = .0001, max = 1.0000, vary=True)
likeParams.add('psi6', value = .224, min = .0001, max = 1.0000, vary=True)
likeParams.add('psi7', value = .269, min = .268, max = .270, vary=True)

lsqParams = Parameters()
lsqParams.add('I0', value = .0005, min = .0000001, max = 1.0000)
lsqParams.add('b', value = .5, min = .0001, max = 1.5000)
lsqParams.add('x5', value =.5, min = .01, max = 10.0000)
lsqParams.add('x6', value = .5, min = .01, max = 10.0000)
lsqParams.add('x7', value = .5, min = .01, max = 10.0000)
lsqParams.add('psi3', value = .070, min = .0001, max = 1.0000)
lsqParams.add('psi5', value = .005, min = .005, max = 1.0000)
lsqParams.add('psi6', value = .204, min = .0001, max = 1.0000)
lsqParams.add('psi7', value = .210, min = .0001, max = 1.0000)
#==============================================================================
'''  Initial conditions '''
I0 = .000503
Ik0 = [I0, 0, 0, 0]
#==============================================================================
''' Define the model, Eq. (1) on pg. 3 '''
def original(Ik, t):
    I0 = .000503  # Original 3.72 * 10**-3
    b = .889            # Original .871
    x5 = .070           # Original .724
    x6 = .686           # Original 1.38
    x7 = .029           # Original .107
    psi3 = .074         # Original .083
    psi5 = .007         # Original .018
    psi6 = .205         # Original .233
    psi7 = .227         # Original .286
    
    I3 = Ik[0]
    I5 = Ik[1]
    I6 = Ik[2]
    I7 = Ik[3]

    if (I3 < psi3):
        S3 = (1 - (I3 / psi3))
    else:
        S3 = 0
    if (I5 < psi5):
        S5 = (1 - (I5 / psi5))
    else:
        S5 = 0
    if (I6 < psi6):
        S6 = (1 - (I6 / psi6))
    else:
        S6 = 0
    if (I7 < psi7):
        S7 = (1 - (I7 / psi7))
    else:
        S7 = 0

    dI3dt = b * I3 * S3
    dI5dt = b * I5 * S5 + x5 * S5 * I3
    dI6dt = b * I6 * S6 + x6 * S6 * (I3 + I5)
    dI7dt = b * I7 * S7 + x7 * S7 * (I3 + I5 + I6)

    return [dI3dt, dI5dt, dI6dt, dI7dt]
#==============================================================================
def model(Mk, t, parameters):
    M3 = Mk[0]
    M5 = Mk[1]
    M6 = Mk[2]
    M7 = Mk[3]

    try: # Get parameters
        b = parameters['b'].value
        x5 = parameters['x5'].value
        x6 = parameters['x6'].value
        x7 = parameters['x7'].value
        psi3 = parameters['psi3'].value
        psi5 = parameters['psi5'].value
        psi6 = parameters['psi6'].value
        psi7 = parameters['psi7'].value
    except KeyError:
        b, x5, x6, x7, psi3, psi5, psi6, psi7 = parameters

    if (M3 < psi3):
        S3 = (1 - (M3 / psi3))
    else:
        S3 = 0
    if (M5 < psi5):
        S5 = (1 - (M5 / psi5))
    else:
        S5 = 0
    if (M6 < psi6):
        S6 = (1 - (M6 / psi6))
    else:
        S6 = 0
    if (M7 < psi7):
        S7 = (1 - (M7 / psi7))
    else:
        S7 = 0

    dM3dt = b * M3 * S3
    dM5dt = b * M5 * S5 + x5 * S5 * M3
    dM6dt = b * M6 * S6 + x6 * S6 * (M3 + M5)
    dM7dt = b * M7 * S7 + x7 * S7 * (M3 + M5 + M6)

    return [dM3dt, dM5dt, dM6dt, dM7dt]
#==============================================================================
''' Compute negative log likelihood of Tromas' data given the model, see eq. (3) pg. 11 '''
def negLogLike(parameters):
    # Solve ODE system to get model values; parameters are not yet fitted
    Lk0 = [parameters['I0'].value, 0, 0, 0]
    MM = odeint(model, Lk0, ZERO_DAYS_AXIS, args=(parameters,))

    nll = 0
    epsilon = 10**-10
    for t in range(4):          # Iterate through days
        for p in range(5):      # Iterate through replicates
            for k in range(4):  # Iterate through leaves
                Vktp = TROMAS_DATA[20 * t + 4 * p + k][4]          # Number of infected cells
                Aktp = TROMAS_DATA[20 * t + 4 * p + k][3] + Vktp   # Total number of cells observed
                Iktp = MM[t + 1][k]                                # Frequency of cellular infection

                if (Iktp <= 0):
                    Iktp = epsilon
                    #print("AHHHHHHHHH")
                elif (Iktp >= 1):
                    Iktp = 1 - epsilon
                    #print("AHHHHHHHHH")

                nll += Vktp * np.log(Iktp) + (Aktp - Vktp) * np.log(1 - Iktp)
    
    return -nll
#==============================================================================
''' Generate residuals for least squares fitting '''
def residuals(parameters):
    residuals = np.empty((len(TROMAS_DATA), 1), float)

    # Solve ODE system to get model values; parameters are not yet fitted
    Lsk0 = [parameters['I0'].value, 0, 0, 0]
    MM = odeint(model, Lsk0, ZERO_DAYS_AXIS, args=(parameters,))

    for t in range(4):
        for p in range(5):          # Iterate through days
            for k in range(4):      # Iterate through replicates
                Vktp = TROMAS_DATA[20 * t + 4 * p + k][4]          # Number of infected cells
                Aktp = TROMAS_DATA[20 * t + 4 * p + k][3] + Vktp   # Number of total cells
                Iktp = MM[t + 1][k]                                # Frequency of cellular infection
                
                residuals[20 * t + 4 * p + k] = abs((Vktp / Aktp) - Iktp)
    
    return residuals
#==============================================================================
''' Generate residuals for the Shapiro test '''
def residualsTest(parameters):
    RES_RAW = np.empty(len(TROMAS_DATA), float)

    # Solve ODE system to get model values; parameters are not yet fitted
    Lsk0 = [parameters['I0'].value, 0, 0, 0]
    MM = odeint(model, Lsk0, ZERO_DAYS_AXIS, args=(parameters,))

    for t in range(4):
        for p in range(5):          # Iterate through days
            for k in range(4):      # Iterate through replicates
                Vktp = TROMAS_DATA[20 * t + 4 * p + k][4]          # Number of infected cells
                Aktp = TROMAS_DATA[20 * t + 4 * p + k][3] + Vktp   # Number of total cells
                Iktp = MM[t + 1][k]                                # Frequency of cellular infection
                
                RES_RAW[20 * t + 4 * p + k] = (Vktp / Aktp) - Iktp
    
    return RES_RAW
#==============================================================================
''' Generate residuals for least squares fitting '''
def logResiduals(parameters):
    residuals = []

    # Solve ODE system to get model values; parameters are not yet fitted
    Lsk0 = [parameters['I0'].value, 0, 0, 0]
    MM = odeint(model, Lsk0, ZERO_DAYS_AXIS, args=(parameters,))

    epsilon = 10**-5
    for t in range(4):
        for p in range(5):          # Iterate through days
            for k in range(4):      # Iterate through replicates
                Vktp = TROMAS_DATA[20 * t + 4 * p + k][4]  # Number of infected cells
                Aktp = TROMAS_DATA[20 * t + 4 * p + k][3] + Vktp   # Number of total cells
                Iktp = MM[t + 1][k]                                # Frequency of cellular infection
                
                if (Iktp <= 0):
                    Iktp = epsilon
                    #print("AHHHHHHHHH")
                elif (Iktp >= 1):
                    Iktp = 1 - epsilon

                if not (Vktp == 0):
                    residuals.append(np.log(Vktp / Aktp) - np.log(Iktp))
    
    return residuals
#==============================================================================
''' Miminize negative log likelihood with differential evolution algorithm '''
result_likelihood = minimize(negLogLike, likeParams, method = 'differential_evolution')
#==============================================================================
''' Miminize residuals with least squares method '''
result_leastsq = minimize(logResiduals, lsqParams, method = 'least_squares')

result_leastsq_raw = minimize(residuals, lsqParams, method = 'least_squares')
#==============================================================================
''' Get residuals for Shapiro test '''
RES_RAW = residualsTest(result_leastsq_raw.params)
#==============================================================================
''' Solve original system of differential equations'''
II = odeint(original, Ik0, t)
I3 = II[:, 0]
I5 = II[:, 1]
I6 = II[:, 2]
I7 = II[:, 3]
#==============================================================================
''' Compare fitted values with those in Tromas' paper '''
print("======================================================================")
print("======================================================================")
report_fit(result_likelihood)
print("======================================================================")
print("======================================================================")
report_fit(result_leastsq)
print("======================================================================")
print("======================================================================")
print('LS Shapiro = ', stats.shapiro(RES_RAW), 'LS Shapiro (log) = ', stats.shapiro(logResiduals(result_leastsq.params)))
print("======================================================================")
print("======================================================================")
print("nll (Like params) = ", round(negLogLike(result_likelihood.params)),  
      ' | nll (LS params) = ', round(negLogLike(result_leastsq.params)), 
      ' | SSR - raw (Like params) = ', sum(np.power(residuals(result_likelihood.params), 2)), 
      ' | SSR - raw (LS params) = ', sum(np.power(residuals(result_leastsq.params), 2)), 
      ' | SSR - log (LS params (log)) = ', sum(np.power(logResiduals(result_leastsq.params), 2)))

print("AIC_nll (Like params) = ", round(2 * negLogLike(result_likelihood.params) + 2 * result_likelihood.nvarys), 
      ' | AIC_nll (LS params) = ',  round(2 * negLogLike(result_leastsq.params) + 2 * result_leastsq.nvarys), 
      ' | AIC_SSR - raw (Like params) = ', round(result_likelihood.ndata * np.log(sum(np.power(residuals(result_likelihood.params), 2))[0] / result_likelihood.ndata) + 2 * result_likelihood.nvarys), 
      ' | AIC_SSR - raw (LS params) = ', round(result_leastsq.ndata * np.log(sum(np.power(residuals(result_leastsq.params), 2))[0] / result_leastsq.ndata) + 2 * result_leastsq.nvarys),
      ' | AIC_SSR - log (LS params (log)) = ', round(result_leastsq.ndata * np.log(sum(np.power(logResiduals(result_leastsq.params), 2)) / result_leastsq.ndata) + 2 * result_leastsq.nvarys))
print("======================================================================")
print("======================================================================")
print("NLL I0     = ", round(result_likelihood.params['I0'].value, 6), ", LS I0     = ", round(result_leastsq_raw.params['I0'].value, 6), ", LS I0 (log)    = ", round(result_leastsq.params['I0'].value, 6))
print("NLL b     = ", round(result_likelihood.params['b'].value, 6), ", LS b     = ", round(result_leastsq_raw.params['b'].value, 6), ", LS b (log)     = ", round(result_leastsq.params['b'].value, 6))
print("NLL x5     = ", round(result_likelihood.params['x5'].value, 6), ", LS x5     = ", round(result_leastsq_raw.params['x5'].value, 6), ", LS x5 (log)     = ", round(result_leastsq.params['x5'].value, 6))
print("NLL x6     = ", round(result_likelihood.params['x6'].value, 6), ", LS x6     = ", round(result_leastsq_raw.params['x6'].value, 6), ", LS x6 (log)    = ", round(result_leastsq.params['x6'].value, 6))
print("NLL x7     = ", round(result_likelihood.params['x7'].value, 6), ", LS x7     = ", round(result_leastsq_raw.params['x7'].value, 6), ", LS x7 (log)    = ", round(result_leastsq.params['x7'].value, 6))
print("NLL psi3   = ", round(result_likelihood.params['psi3'].value, 6), ", LS psi3  = ", round(result_leastsq_raw.params['psi3'].value, 6), ", LS psi3 (log)  = ", round(result_leastsq.params['psi3'].value, 6))
print("NLL psi5   = ", round(result_likelihood.params['psi5'].value, 6), ", LS psi5  = ", round(result_leastsq_raw.params['psi5'].value, 6), ", LS psi5 (log)  = ", round(result_leastsq.params['psi5'].value, 6))
print("NLL psi6   = ", round(result_likelihood.params['psi6'].value, 6), ", LS psi6  = ", round(result_leastsq_raw.params['psi6'].value, 6), ", LS psi6 (log)  = ", round(result_leastsq.params['psi6'].value, 6))
print("NLL psi7   = ", round(result_likelihood.params['psi7'].value, 6), ", LS psi7  = ", round(result_leastsq.params['psi7'].value, 6), ", LS psi7 (log)  = ", round(result_leastsq.params['psi7'].value, 6))
print("======================================================================")
print("======================================================================")
#==============================================================================
''' Solve ODE system with fitted parameters from likelihood '''
Lk0 = [result_likelihood.params['I0'].value, 0, 0, 0]
LL = odeint(model, Lk0, t, args=(result_likelihood.params,))
LL3 = LL[:, 0]
LL5 = LL[:, 1]
LL6 = LL[:, 2]
LL7 = LL[:, 3]
#==============================================================================
''' Solve ODE system with fitted parameters from least squares with log transformation '''
LLsk0 = [result_leastsq.params['I0'].value, 0, 0, 0]
LLS = odeint(model, LLsk0, t, args=(result_leastsq.params,))
LLS3 = LLS[:, 0]
LLS5 = LLS[:, 1]
LLS6 = LLS[:, 2]
LLS7 = LLS[:, 3]
#==============================================================================
''' Solve ODE system with fitted parameters from least squares '''
Lsk0 = [result_leastsq_raw.params['I0'].value, 0, 0, 0]
LS = odeint(model, Lsk0, t, args=(result_leastsq_raw.params,))
LS3 = LS[:, 0]
LS5 = LS[:, 1]
LS6 = LS[:, 2]
LS7 = LS[:, 3]
#==============================================================================
''' Compute residuals '''
FLAT_RES_LS   = abs(residuals(result_leastsq.params))
FLAT_RES_LIKE = abs(residuals(result_likelihood.params))
#==============================================================================
''' Plot experimental data and model curves '''
for leaf_iterator in range(4):
#==============================================================================
    ''' Set up matricies '''
    RATIOS = []
    LEAF_DATA = np.empty((5, 4), float)  # Set up matrix
    AVERAGES = []

    PRE_RES_LS = np.empty(20, float)
    PRE_RES_LIKE = np.empty(20, float)
    RES_LS = np.empty((5, 4), float)
    RES_LIKE = np.empty((5, 4), float)
    #==========================================================================
    ''' Compute data from csv '''
    for i in range(round(len(TROMAS_DATA) / 4)):  # Iterate through all rows
        non_infected = TROMAS_DATA[(4 * i + leaf_iterator)][3]
        infected = TROMAS_DATA[(4 * i + leaf_iterator)][4]
        ratio = infected / (non_infected + infected)
        RATIOS.append(ratio)
        PRE_RES_LS[i] = FLAT_RES_LS[4 * i + leaf_iterator]
        PRE_RES_LIKE[i] = FLAT_RES_LIKE[4 * i + leaf_iterator]
    #==========================================================================
    ''' Build a matrix of ratios with each row being a replicate '''
    for i in range(5):
        for j in range(4):
            LEAF_DATA[i][j] = RATIOS[i + 5 * j]
            RES_LS[i][j]    = PRE_RES_LS[i + 5 * j]
            RES_LIKE[i][j]  = PRE_RES_LIKE[i + 5 * j]
    #==========================================================================
    ''' Calculate average cellular infection per day '''
    for i in range(4):
        sum = 0
        for j in range(5):
            sum += LEAF_DATA[j][i]
            mean = sum / 5
        AVERAGES.append(mean)
    #==========================================================================
    ''' Plot results '''
    if leaf_iterator == 0:
        for i in range(5):
            ax[0][0].scatter(DAYS_AXIS, LEAF_DATA[i, :], s = 80, facecolors = 'none', edgecolors = 'b')
            ax_res[0][0].scatter(DAYS_AXIS, RES_LS[i, :], s = 80, facecolors = 'none', edgecolors = 'y')
            ax_res[0][0].scatter(DAYS_AXIS, RES_LIKE[i, ], s = 80, facecolors = 'none', edgecolors = 'g')

        ax[0][0].scatter(DAYS_AXIS, LEAF_DATA[0, :], s = 80, facecolors = 'none', edgecolors = 'b', label = "Data") # Plot copy to get one label
        ax_res[0][0].scatter(DAYS_AXIS, RES_LS[i, :], s = 80, facecolors = 'none', edgecolors = 'y', label = 'Least squares')
        ax_res[0][0].scatter(DAYS_AXIS, RES_LIKE[i, ], s = 80, facecolors = 'none', edgecolors = 'g', label = 'Likelihood')

        #ax[0][0].plot(t, I3, color = 'black', label = "Ganusov")   # Plot differential equation
        ax[0][0].plot(t, LL3, color = 'green', label = "NegLogLike")
        #ax[0][0].plot(t, LS3, '--', color = 'orange', label = "Least Squares (raw)")
        ax[0][0].plot(t, LLS3, '-.', color = 'gold', label = "Least Squares (log)")

        ax[0][0].scatter(DAYS_AXIS, AVERAGES, s = 120, color = 'red', marker = '_', label = "Average")
    
    elif leaf_iterator == 1:
        for i in range(5):
            ax[0][1].scatter(DAYS_AXIS, LEAF_DATA[i, :], s = 80, facecolors = 'none', edgecolors = 'b')
            ax_res[0][1].scatter(DAYS_AXIS, RES_LS[i, :], s = 80, facecolors = 'none', edgecolors = 'y')
            ax_res[0][1].scatter(DAYS_AXIS, RES_LIKE[i, ], s = 80, facecolors = 'none', edgecolors = 'g')

        ax[0][1].scatter(DAYS_AXIS, LEAF_DATA[0, :], s = 80, facecolors = 'none', edgecolors = 'b', label = "Data") # Plot copy to get one label
        ax_res[0][1].scatter(DAYS_AXIS, RES_LS[0, :], s = 80, facecolors = 'none', edgecolors = 'y', label = 'Least squares')
        ax_res[0][1].scatter(DAYS_AXIS, RES_LIKE[0, :], s = 80, facecolors = 'none', edgecolors = 'g', label = 'Likelihood')

        #ax[0][1].plot(t, I5, color = 'black', label = "Ganusov")   # Plot differential equation
        ax[0][1].plot(t, LL5, color = 'green', label = "NegLogLike")
        #ax[0][1].plot(t, LS5, '--', color = 'orange', label = "Least Squares (raw)")
        ax[0][1].plot(t, LLS5, '-.', color = 'gold', label = "Least Squares (log)")

        ax[0][1].scatter(DAYS_AXIS, AVERAGES, s = 120, color = 'red', marker = '_', label = "Average")

    elif leaf_iterator == 2:
        for i in range(5):
            ax[1][0].scatter(DAYS_AXIS, LEAF_DATA[i, :], s = 80, facecolors = 'none', edgecolors = 'b')
            ax_res[1][0].scatter(DAYS_AXIS, RES_LS[i, :], s = 80, facecolors = 'none', edgecolors = 'y')
            ax_res[1][0].scatter(DAYS_AXIS, RES_LIKE[i, ], s = 80, facecolors = 'none', edgecolors = 'g')

        ax[1][0].scatter(DAYS_AXIS, LEAF_DATA[0, :], s = 80, facecolors = 'none', edgecolors = 'b', label = "Data") # Plot copy to get one label
        ax_res[1][0].scatter(DAYS_AXIS, RES_LS[i, :], s = 80, facecolors = 'none', edgecolors = 'y', label = 'Least squares')
        ax_res[1][0].scatter(DAYS_AXIS, RES_LIKE[i, ], s = 80, facecolors = 'none', edgecolors = 'g', label = 'Likelihood')

        #ax[1][0].plot(t, I6, color = 'black', label = "Ganusov")   # Plot differential equation
        ax[1][0].plot(t, LL6, color = 'green', label = "NegLogLike")
        #ax[1][0].plot(t, LS6, '--', color = 'orange', label = "Least Squares (raw)")
        ax[1][0].plot(t, LLS6, '-.', color = 'gold', label = "Least Squares (log)")

        ax[1][0].scatter(DAYS_AXIS, AVERAGES, s = 120, color = 'red', marker = '_', label = "Average")

    elif leaf_iterator == 3:
        for i in range(5):
            ax[1][1].scatter(DAYS_AXIS, LEAF_DATA[i, :], s = 80, facecolors = 'none', edgecolors = 'b')
            ax_res[1][1].scatter(DAYS_AXIS, RES_LS[i, :], s = 80, facecolors = 'none', edgecolors = 'y')
            ax_res[1][1].scatter(DAYS_AXIS, RES_LIKE[i, ], s = 80, facecolors = 'none', edgecolors = 'g')

        ax[1][1].scatter(DAYS_AXIS, LEAF_DATA[0, :], s = 80, facecolors = 'none', edgecolors = 'b', label = "Data") # Plot copy to get one label
        ax_res[1][1].scatter(DAYS_AXIS, RES_LS[i, :], s = 80, facecolors = 'none', edgecolors = 'y', label = 'Least squares')
        ax_res[1][1].scatter(DAYS_AXIS, RES_LIKE[i, ], s = 80, facecolors = 'none', edgecolors = 'g', label = 'Likelihood')

        #ax[1][1].plot(t, I7, color = 'black', label = "Ganusov")   # Plot differential equation
        ax[1][1].plot(t, LL7, color = 'green', label = "NegLogLike")
        #ax[1][1].plot(t, LS7, '--', color = 'orange', label = "Least Squares (raw)")
        ax[1][1].plot(t, LLS7, '-.', color = 'gold', label = "Least Squares (log)")

        ax[1][1].scatter(DAYS_AXIS, AVERAGES, s = 120, color = 'red', marker = '_', label = "Average")
#==============================================================================
for i in range(2):
    for j in range(2):
        ax[i][j].set_xlim(0, 12)
        ax[0][0].legend(loc = "upper left")
        ax[i][j].set_xlabel('Days post innoculation')
        ax[i][j].set_ylabel('Frequency of celluar infection')
        ax[i][j].set_ylim(0, .5)
        #ax[i][j].set_yscale('log')
        ax[i][j].text(-.8, .55, ABCS[2 * i + j], fontsize=16, fontweight='bold', va='top', ha='right')
        ax[i][j].text(10.25, .45, LEAVES[2 * i + j])

        ax_res[i][j].set_xlim(0, 12)
        ax_res[i][j].set_ylim(10e-5, 10e-1)
        ax_res[i][j].set_yscale('log')
        ax_res[i][j].set_xlabel('Days post innoculation')
        ax_res[i][j].set_ylabel('abs(residuals)')
        ax_res[i][j].legend(loc = "upper left")
        ax_res[i][j].text(10.25, .09, LEAVES[2 * i + j])
#==============================================================================
plt.show()
#==============================================================================
''' Save figure '''
save_path = r'C:\Users\joshm\OneDrive\Documents\Plant_Gang\Final_Figures' # Path to where you want the file to be saved
filename = "TromasModelLogLSnoRawLinScale.pdf" # Specify filename
fig.savefig(os.path.join(save_path, filename), bbox_inches = 'tight', pad_inches = 0) # Save figure in the new directory

#LS Shapiro =  ShapiroResult(statistic=0.8029271960258484, pvalue=5.7448104051616156e-09) Like Shapiro =  ShapiroResult(statistic=0.8123624324798584, pvalue=1.0740277112120111e-08