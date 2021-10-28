'''
Author: Joshua Miller, 2019
The 2-alpha probabilistic coinfection model: the best-performing coinfection model which was tested.
This is fitted to data using the binomial distribution-based likelihood method.
'''
#==================================================================================================
import pandas as pd
import numpy as np
import random
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
#==================================================================================================
''' Put data into a DataFrame '''
DataFrame = pd.read_csv('Cell_count_data_Tromas_2014.csv')

TOTAL = np.empty((DataFrame.shape[0], 5), int)
VENUS_ONLY = np.empty((DataFrame.shape[0], 5), int)
BFP_ONLY = np.empty((DataFrame.shape[0], 5), int)
MIXED = np.empty((DataFrame.shape[0], 5), int)

for i in range(DataFrame.shape[0]): # Number of rows
    for j in range(5):              # Number of columns desired
        TOTAL[i][j] = DataFrame.iloc[i, j]
        VENUS_ONLY[i][j] = DataFrame.iloc[i, j]
        BFP_ONLY[i][j] = DataFrame.iloc[i, j]
        MIXED[i][j] = DataFrame.iloc[i, j]

        if (j == 4):
            TOTAL[i][j] = DataFrame.ix[i, 'Venus_only'] + DataFrame.ix[i, 'BFP_only'] + DataFrame.ix[i, 'Mixed']
            VENUS_ONLY[i][j] = DataFrame.ix[i, 'Venus_only']
            BFP_ONLY[i][j] = DataFrame.ix[i, 'BFP_only']
            MIXED[i][j] = DataFrame.ix[i, 'Mixed']
#==================================================================================================
''' Make ratios of infected to unifected '''
TOTAL_RATIOS = np.empty((DataFrame.shape[0], 1), float)
VENUS_RATIOS = np.empty((DataFrame.shape[0], 1), float)
BFP_RATIOS = np.empty((DataFrame.shape[0], 1), float)
MIXED_RATIOS = np.empty((DataFrame.shape[0], 1), float)

TOTAL_RATIOS = TOTAL[:, 4] / (TOTAL[:, 3] + TOTAL[:, 4]) 
VENUS_RATIOS = VENUS_ONLY[:, 4] / (VENUS_ONLY[:, 3] + TOTAL[:, 4])
BFP_RATIOS = BFP_ONLY[:, 4] / (BFP_ONLY[:, 3] + TOTAL[:, 4])
MIXED_RATIOS = MIXED[:, 4] / (MIXED[:, 3] + TOTAL[:, 4])
#==============================================================================
''' Create subplot array '''
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
fig.subplots_adjust(hspace = .25, wspace = .2)
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
''' y lim '''
ylim = .35
#==============================================================================
''' ABCS '''
ABCS = ['A', 'B', 'C', 'D']
#==============================================================================
''' Leafy bois '''
LEAVES = ['Leaf 3', 'Leaf 5', 'Leaf 6', 'Leaf 7']
#==============================================================================
''' Markers for plot '''
MARKERS = ['o', '^', 's', '*']
COLORS = ['b', 'y', 'g', 'm']
PLANT_NUM = [1, 2, 3, 4, 5]
LEAF_NUM = ['3', '5', '6', '7']
LABELS = ['Total', 'Venus-Only', 'BFP - Only', 'Mixed']
#==============================================================================
''' Legends for plot '''
legendM = mlines.Line2D([], [], color='g', linestyle='-', markerfacecolor='none', markeredgecolor='g', markerfacecoloralt='none', marker='o',
                          markersize=10, label= 'Mixed')
legendB = mlines.Line2D([], [], color='b', linestyle='-.', markerfacecolor='none', markeredgecolor='b', markerfacecoloralt='none', marker='s',
                          markersize=10, label= 'BFP')
legendV = mlines.Line2D([], [], color='y', linestyle='--', markerfacecolor='none', markeredgecolor='y', markerfacecoloralt='none', marker='^',
                          markersize=10, label = 'Venus')
#==============================================================================
''' Parameter list (Initialized with already predicted values) '''
Params = Parameters()
Params.add('V0', value = .0003, min = 0, max = .001, vary = True)
Params.add('B0', value = .0003, min = 0, max = .001, vary = True)
Params.add('M0', value = .0001, min = 0, max = .001, vary = True)
Params.add('bB', value = .975, min = .0001, max = 10.0000, vary = True)
Params.add('bV', value = .832, min = .0001, max = 10.0000, vary = True)
Params.add('x5', value = .113, min = .0001, max = 10.0000, vary = True)
Params.add('x6', value = .803, min = .0001, max = 10.0000, vary = True)
Params.add('x7', value = .030, min = .0001, max = 10.0000, vary = True)
Params.add('psi3', value = .072, min = .0001, max = 1.0000, vary = True)
Params.add('psi5', value = .016, min = .0001, max = 1.0000, vary = True)
Params.add('psi6', value = .223, min = .0001, max = 1.0000, vary = True)
Params.add('psi7', value = .247, min = .0001, max = 1.0000, vary = True)
Params.add('alphab', value = 10, min = 0, max = 20.0000, vary = True)
Params.add('alphax', value = 1, min = 0, max = 20.0000, vary = True)
#==============================================================================
''' Define the model '''
def model(Mk, t, parameters):
    V3 = Mk[0]
    B3 = Mk[1]
    M3 = Mk[2]
    V5 = Mk[3]
    B5 = Mk[4]
    M5 = Mk[5]
    V6 = Mk[6]
    B6 = Mk[7]
    M6 = Mk[8]
    V7 = Mk[9]
    B7 = Mk[10]
    M7 = Mk[11]

    try: # Get parameters
        bB = parameters['bB'].value
        bV = parameters['bV'].value
        x5 = parameters['x5'].value
        x6 = parameters['x6'].value
        x7 = parameters['x7'].value
        psi3 = parameters['psi3'].value
        psi5 = parameters['psi5'].value
        psi6 = parameters['psi6'].value
        psi7 = parameters['psi7'].value
        alphab = parameters['alphab'].value
        alphax = parameters['alphax'].value
    except KeyError:
        bB, bV, x5, x6, x7, psi3, psi5, psi6, psi7, alphab, alphax = parameters

    if (B3 < psi3):
        S3 = (1 - ((V3 + B3 + M3) / psi3))
    else:
        S3 = 0
    if (V3 < psi3):
        S3 = (1 - ((V3 + B3 + M3) / psi3))
    else:
        S3 = 0
    if (M3 < psi3):
        S3 = (1 - ((V3 + B3 + M3) / psi3))
    else:
        S3 = 0
    #====================================================
    if (B5 < psi5):
        S5 = (1 - ((V5 + B5 + M5) / psi5))
    else:
        S5 = 0
    if (V5 < psi5):
        S5 = (1 - ((V5 + B5 + M5) / psi5))
    else:
        S5 = 0
    if (M5 < psi5):
        S5 = (1 - ((V5 + B5 + M5) / psi5))
    else:
        S5 = 0
    #====================================================
    if (B6 < psi6):
        S6 = (1 - ((V6 + B6 + M6) / psi6))
    else:
        S6 = 0
    if (V6 < psi6):
        S6 = (1 - ((V6 + B6 + M6) / psi6))
    else:
        S6 = 0
    if (M6 < psi6):
        S6 = (1 - ((V6 + B6 + M6) / psi6))
    else:
        S6 = 0
    #====================================================
    if (B7 < psi7):
        S7 = (1 - ((V7 + B7 + M7) / psi7))
    else:
        S7 = 0
    if (V7 < psi7):
        S7 = (1 - ((V7 + B7 + M7) / psi7))
    else:
        S7 = 0
    if (M7 < psi7):
        S7 = (1 - ((V7 + B7 + M7) / psi7))
    else:
        S7 = 0

    dV3dt = bV * V3 * S3
    dB3dt = bB * B3 * S3
    dM3dt = alphab * (S3 * V3 * B3 * (bB + bV))

    dV5dt = bV * V5 * S5 + x5 * S5 * V3
    dB5dt = bB * B5 * S5 + x5 * S5 * B3
    dM5dt = alphab * S5 * V5 * B5 * (bB + bV) + alphax * x5 * S5 * (B5 * V3 + V5 + B3)

    dV6dt = bV * V6 * S6 + x6 * S6 * (V3 + V5)
    dB6dt = bB * B6 * S6 + x6 * S6 * (B3 + B5)
    dM6dt = alphab * S6 * V6 * B6 * (bB + bV) + alphax * x6 * S6 * (B6 * (V3 + V5) + V6 * (B3 + B5))

    dV7dt = bV * V7 * S7 + x7 * S7 * (V3 + V5 + V6)
    dB7dt = bB * B7 * S7 + x7 * S7 * (B3 + B5 + B6)
    dM7dt = alphab * S7 * V7 * B7 * (bB + bV) + alphax * x7 * S7 * (B7 * (V3 + V5 + V6) + V7 * (B3 + B5 + B6))

    return [dV3dt, dB3dt, dM3dt, dV5dt, dB5dt, dM5dt, dV6dt, dB6dt, dM6dt, dV7dt, dB7dt, dM7dt]
#==============================================================================
''' Compute the nll value for a given set of parameter values '''
def negLogLike(parameters):
    # Solve ODE system to get model prediction; parameters are not yet fitted
    init = [parameters['V0'].value, parameters['B0'].value ,parameters['M0'].value, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    MM = odeint(model, init, ZERO_DAYS_AXIS, args=(parameters,))
    V3 = MM[:, 0]
    B3 = MM[:, 1]
    M3 = MM[:, 2]
    V5 = MM[:, 3]
    B5 = MM[:, 4]
    M5 = MM[:, 5]
    V6 = MM[:, 6]
    B6 = MM[:, 7]
    M6 = MM[:, 8]
    V7 = MM[:, 9]
    B7 = MM[:, 10]
    M7 = MM[:, 11]

    nll = 0
    epsilon = 10**-10
    for t in range(4):          # Iterate through days
        for p in range(5):      # Iterate through replicates
            for k in range(4):  # Iterate through leaves
                Vktp_M = MIXED[20 * t + 4 * p + k][4]      # Number of coinfected cells
                Vktp_V = VENUS_ONLY[20 * t + 4 * p + k][4] # Number of Venus-only infected cells
                Vktp_B = BFP_ONLY[20 * t + 4 * p + k][4]   # Number of BFP-only infected cells
                Aktp = TOTAL[20 * t + 4 * p + k][3] + TOTAL[20 * t + 4 * p + k][4] # Total number of cells observed
                Iktp_V3 = V3[t + 1]
                Iktp_B3 = B3[t + 1]
                Iktp_M3 = M3[t + 1]
                Iktp_V5 = V5[t + 1]
                Iktp_B5 = B5[t + 1]
                Iktp_M5 = M5[t + 1]
                Iktp_V6 = V6[t + 1]
                Iktp_B6 = B6[t + 1]
                Iktp_M6 = M6[t + 1]
                Iktp_V7 = V7[t + 1]
                Iktp_B7 = B7[t + 1]
                Iktp_M7 = M7[t + 1]
                
                # Essentially checking for 0s which would cause problems due to the logarithms in the nll equation.
                # Epsilon is user-defined and will replace any of the following quanties with it
                if (Iktp_M3 < epsilon):
                    Iktp_M3 = epsilon
                if (Iktp_M3 > 1 - epsilon):
                    Iktp_M3 = 1 - epsilon
                if (Iktp_V3 < epsilon):
                    Iktp_V3 = epsilon
                if (Iktp_V3 > 1 - epsilon):
                    Iktp_V3 = 1 - epsilon
                if (Iktp_B3 < epsilon):
                    Iktp_B3 = epsilon
                if (Iktp_B3 > 1 - epsilon):
                    Iktp_B3 = 1 - epsilon
                if (Iktp_M5 < epsilon):
                    Iktp_M5 = epsilon
                if (Iktp_M5 > 1 - epsilon):
                    Iktp_M5 = 1 - epsilon
                if (Iktp_V5 < epsilon):
                    Iktp_V5 = epsilon
                if (Iktp_V5 > 1 - epsilon):
                    Iktp_V5 = 1 - epsilon
                if (Iktp_B5 < epsilon):
                    Iktp_B5 = epsilon
                if (Iktp_B5 > 1 - epsilon):
                    Iktp_B5 = 1 - epsilon
                if (Iktp_M6 < epsilon):
                    Iktp_M6 = epsilon
                if (Iktp_M6 > 1 - epsilon):
                    Iktp_M6 = 1 - epsilon
                if (Iktp_V6 < epsilon):
                    Iktp_V6 = epsilon
                if (Iktp_V6 > 1 - epsilon):
                    Iktp_V6 = 1 - epsilon
                if (Iktp_B6 < epsilon):
                    Iktp_B6 = epsilon
                if (Iktp_B6 > 1 - epsilon):
                    Iktp_B6 = 1 - epsilon
                if (Iktp_M7 < epsilon):
                    Iktp_M7 = epsilon
                if (Iktp_M7 > 1 - epsilon):
                    Iktp_M7 = 1 - epsilon
                if (Iktp_V7 < epsilon):
                    Iktp_V7 = epsilon
                if (Iktp_V7 > 1 - epsilon):
                    Iktp_V7 = 1 - epsilon
                if (Iktp_B7 < epsilon):
                    Iktp_B7 = epsilon
                if (Iktp_B7 > 1 - epsilon):
                    Iktp_B7 = 1 - epsilon

                # Calculate the individual nlls for each leaf and virus 'strain'. Sum together to get total nll
                if (k == 0):
                    nll_M3 = Vktp_M * np.log(Iktp_M3) + (Aktp - Vktp_M) * np.log(1 - Iktp_M3)
                    nll_V3 = Vktp_V * np.log(Iktp_V3) + (Aktp - Vktp_V) * np.log(1 - Iktp_V3)
                    nll_B3 = Vktp_B * np.log(Iktp_B3) + (Aktp - Vktp_B) * np.log(1 - Iktp_B3)
                    nll += (nll_M3 + nll_V3 + nll_B3)
                elif (k == 1):
                    nll_M5 = Vktp_M * np.log(Iktp_M5) + (Aktp - Vktp_M) * np.log(1 - Iktp_M5)
                    nll_V5 = Vktp_V * np.log(Iktp_V5) + (Aktp - Vktp_V) * np.log(1 - Iktp_V5)
                    nll_B5 = Vktp_B * np.log(Iktp_B5) + (Aktp - Vktp_B) * np.log(1 - Iktp_B5)
                    nll += (nll_M5 + nll_V5 + nll_B5)
                elif (k == 2):
                    nll_M6 = Vktp_M * np.log(Iktp_M6) + (Aktp - Vktp_M) * np.log(1 - Iktp_M6)
                    nll_V6 = Vktp_V * np.log(Iktp_V6) + (Aktp - Vktp_V) * np.log(1 - Iktp_V6)
                    nll_B6 = Vktp_B * np.log(Iktp_B6) + (Aktp - Vktp_B) * np.log(1 - Iktp_B6)
                    nll += (nll_M6 + nll_V6 + nll_B6)
                elif (k == 3):
                    nll_M7 = Vktp_M * np.log(Iktp_M7) + (Aktp - Vktp_M) * np.log(1 - Iktp_M7)
                    nll_V7 = Vktp_V * np.log(Iktp_V7) + (Aktp - Vktp_V) * np.log(1 - Iktp_V7)
                    nll_B7 = Vktp_B * np.log(Iktp_B7) + (Aktp - Vktp_B) * np.log(1 - Iktp_B7)
                    nll += (nll_M7 + nll_V7 + nll_B7)
    
    return -nll
#==============================================================================
''' Miminize negative log likelihood with differential evolution algorithm '''
result = minimize(negLogLike, Params, method = 'differential_evolution')
nll = negLogLike(result.params)
#==============================================================================
''' See fitted values '''
print("======================================================================")
print("======================================================================")
report_fit(result)
print("======================================================================")
print("======================================================================")
print("nll = ", round(nll, 3), ", AIC  = ", round(2 * result.nvarys + 2 * nll, 0))
print("======================================================================")
print("======================================================================")
print("V0     = ", round(result.params['V0'].value, 10))
print("B0     = ", round(result.params['B0'].value, 10))
print("M0     = ", round(result.params['M0'].value, 10))
print("bV     = ", round(result.params['bV'].value, 6))
print("bB     = ", round(result.params['bB'].value, 6))
print("x5     = ", round(result.params['x5'].value, 6))
print("x6     = ", round(result.params['x6'].value, 6))
print("x7     = ", round(result.params['x7'].value, 6))
print("psi3   = ", round(result.params['psi3'].value, 6))
print("psi5   = ", round(result.params['psi5'].value, 6))
print("psi6   = ", round(result.params['psi6'].value, 6))
print("psi7   = ", round(result.params['psi7'].value, 6))
print("alphab = ", round(result.params['alphab'].value, 6))
print("alphax = ", round(result.params['alphax'].value, 6))
print("======================================================================")
print("======================================================================")
#==============================================================================
''' Solve ODE system with fitted parameters from likelihood method '''
Llk0 = [result.params['V0'].value, result.params['B0'], result.params['M0'], 0, 0, 0, 0, 0, 0, 0, 0, 0]
LL = odeint(model, Llk0, t, args=(result.params,))
V3 = LL[:, 0]
B3 = LL[:, 1]
M3 = LL[:, 2]
V5 = LL[:, 3]
B5 = LL[:, 4]
M5 = LL[:, 5]
V6 = LL[:, 6]
B6 = LL[:, 7]
M6 = LL[:, 8]
V7 = LL[:, 9]
B7 = LL[:, 10]
M7 = LL[:, 11]
#==============================================================================
''' Plot experimental data and model curves '''
for leaf_iterator in range(4):
#==============================================================================
    ''' Initialize arrays which will be used later '''
    MIXED_LEAFK = np.empty(((round(len(MIXED_RATIOS) / 4)), 1), float)
    VENUS_LEAFK = np.empty(((round(len(VENUS_RATIOS) / 4)), 1), float)
    BFP_LEAFK = np.empty(((round(len(BFP_RATIOS) / 4)), 1), float)
    TEMP_MIXED = np.empty((5, 4), float)
    TEMP_VENUS = np.empty((5, 4), float)
    TEMP_BFP = np.empty((5, 4), float)
    MIXED_AV = []
    VENUS_AV = []
    BFP_AV = []
    #==================================================================================================
    ''' Compute data from csv '''
    for i in range(len(MIXED_LEAFK)):
        MIXED_LEAFK[i] = MIXED_RATIOS[4 * i + leaf_iterator]
        VENUS_LEAFK[i] = VENUS_RATIOS[4 * i + leaf_iterator]
        BFP_LEAFK[i] = BFP_RATIOS[4 * i + leaf_iterator]
    #==================================================================================================
    ''' Build a matrix of ratios with each row being a replicate '''
    for i in range(5):  # Iterate through all rows
        for j in range(4):
            TEMP_MIXED[i][j] = MIXED_LEAFK[i + 5 * j]
            TEMP_VENUS[i][j] = VENUS_LEAFK[i + 5 * j]
            TEMP_BFP[i][j] = BFP_LEAFK[i + 5 * j]
    #==================================================================================================
    ''' Calculate average cellular infection per day '''
    for i in range(4):
        mixedSum = 0
        venusSum = 0
        bfpSum = 0
        for j in range(5):
            mixedSum += TEMP_MIXED[j][i]
            venusSum += TEMP_VENUS[j][i]
            bfpSum += TEMP_BFP[j][i]
            mixedMean = mixedSum / 5
            venusMean = venusSum / 5
            bfpMean = bfpSum / 5
        MIXED_AV.append(mixedMean)
        VENUS_AV.append(venusMean)
        BFP_AV.append(bfpMean)
    #==========================================================================
    ''' Plot results '''
    if leaf_iterator == 0:
        for i in range(5):
            ax[0][0].scatter(DAYS_AXIS, TEMP_MIXED[i, :], s = 80, facecolors = 'none', edgecolors = 'g', marker = MARKERS[0])
            ax[0][0].scatter(DAYS_AXIS, TEMP_VENUS[i, :], s = 80, facecolors = 'none', edgecolors = 'y', marker = MARKERS[1])
            ax[0][0].scatter(DAYS_AXIS, TEMP_BFP[i, :], s = 80, facecolors = 'none', edgecolors = 'b', marker = MARKERS[2])
        
        ax[0][0].plot(t, M3, 'g-')
        ax[0][0].plot(t, V3, 'y--')
        ax[0][0].plot(t, B3, 'b-.')

        ax[0][0].scatter(DAYS_AXIS, VENUS_AV, color = COLORS[0], s = 200, marker = '_')
        ax[0][0].scatter(DAYS_AXIS, BFP_AV, color = COLORS[1], s = 200, marker = '_')
        ax[0][0].scatter(DAYS_AXIS, MIXED_AV, color = COLORS[2], s = 200, marker = '_')

    elif leaf_iterator == 1:
        for i in range(5):
            ax[0][1].scatter(DAYS_AXIS, TEMP_MIXED[i, :], s = 80, facecolors = 'none', edgecolors = 'g', marker = MARKERS[0])
            ax[0][1].scatter(DAYS_AXIS, TEMP_VENUS[i, :], s = 80, facecolors = 'none', edgecolors = 'y', marker = MARKERS[1])
            ax[0][1].scatter(DAYS_AXIS, TEMP_BFP[i, :], s = 80, facecolors = 'none', edgecolors = 'b', marker = MARKERS[2])
        
        ax[0][1].plot(t, M5, 'g-')
        ax[0][1].plot(t, V5, 'y--')
        ax[0][1].plot(t, B5, 'b-.')

        ax[0][1].scatter(DAYS_AXIS, VENUS_AV, color = COLORS[0], s = 200, marker = '_')
        ax[0][1].scatter(DAYS_AXIS, BFP_AV, color = COLORS[1], s = 200, marker = '_')
        ax[0][1].scatter(DAYS_AXIS, MIXED_AV, color = COLORS[2], s = 200, marker = '_')

    elif leaf_iterator == 2:
        for i in range(5):
            ax[1][0].scatter(DAYS_AXIS, TEMP_MIXED[i, :], s = 80, facecolors = 'none', edgecolors = 'g', marker = MARKERS[0])
            ax[1][0].scatter(DAYS_AXIS, TEMP_VENUS[i, :], s = 80, facecolors = 'none', edgecolors = 'y', marker = MARKERS[1])
            ax[1][0].scatter(DAYS_AXIS, TEMP_BFP[i, :], s = 80, facecolors = 'none', edgecolors = 'b', marker = MARKERS[2])
        
        ax[1][0].plot(t, M6, 'g-')
        ax[1][0].plot(t, V6, 'y--')
        ax[1][0].plot(t, B6, 'b-.')

        ax[1][0].scatter(DAYS_AXIS, VENUS_AV, color = COLORS[0], s = 200, marker = '_')
        ax[1][0].scatter(DAYS_AXIS, BFP_AV, color = COLORS[1], s = 200, marker = '_')
        ax[1][0].scatter(DAYS_AXIS, MIXED_AV, color = COLORS[2], s = 200, marker = '_')

    elif leaf_iterator == 3:
        for i in range(5):
            ax[1][1].scatter(DAYS_AXIS, TEMP_MIXED[i, :], s = 80, facecolors = 'none', edgecolors = 'g', marker = MARKERS[0])
            ax[1][1].scatter(DAYS_AXIS, TEMP_VENUS[i, :], s = 80, facecolors = 'none', edgecolors = 'y', marker = MARKERS[1])
            ax[1][1].scatter(DAYS_AXIS, TEMP_BFP[i, :], s = 80, facecolors = 'none', edgecolors = 'b', marker = MARKERS[2])
        
        ax[1][1].plot(t, M7, 'g-')
        ax[1][1].plot(t, V7, 'y--')
        ax[1][1].plot(t, B7, 'b-.')

        ax[1][1].scatter(DAYS_AXIS, VENUS_AV, color = COLORS[0], s = 200, marker = '_')
        ax[1][1].scatter(DAYS_AXIS, BFP_AV, color = COLORS[1], s = 200, marker = '_')
        ax[1][1].scatter(DAYS_AXIS, MIXED_AV, color = COLORS[2], s = 200, marker = '_')
#==============================================================================
''' Set bounds, scale, legends, and labels '''
for i in range(2):
    for j in range(2):
        ax[i][j].set_xlim(0, 12)
        ax[i][j].legend(handles = [legendM, legendV, legendB], loc = "upper left")
        ax[i][j].set_xlabel('Days post innoculation')
        ax[i][j].set_ylabel('Frequency of celluar infection')
        ax[i][j].set_ylim(10e-6, 2)
        ax[i][j].set_yscale('log')
        ax[i][j].text(-.5, 4, ABCS[2 * i + j], fontsize=16, fontweight='bold', va='top', ha='right')
        ax[i][j].text(10.25, ylim - .1 * ylim, LEAVES[2 * i + j])
#==============================================================================
plt.show()
#==============================================================================
''' Save figure '''
filename = "FullPlantCoinfectionProb2Alpha.pdf" # Specify filename
fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0) # Save figure in the new directory
