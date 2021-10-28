'''
Author: Joshua Miller, 2020
This code employs a simple UI in which the user can adjust parameters of the 2-alpha probabilistic model
to see how changes in parameter values affect the shape of the model's curves. When the program starts up,
the user will be prompted to enter the leaf number and upper bound for the days post innoculation axis; since
the data collection stops at 10, it is recommended to use a value of 11 or 12 for best visibilty, but to
see the log-term behavior of the models a large value can be used. When the graph first appears, the model
curves will not show. The user must click on one of the sliders first, upon which the model curves will be
drawn. After this, the user can slide the parameter values around to their heart's content.

 - The first panel shows the model curves and the experimental data.
 - The second panel shows the nll value for the model using the current set of parameters. Note that though only
 one leaf at a time can be shown, the total nll depends on all four, even the three not on screen
 - The third panel shows coinfection plotted against Venus-only, both the models and experimental data. The
 best-fit linear approximation of the data is shown in red, along with grey reference lines. The 'm' values
 show the slopes of their respective lines.
 - The fourth panel shows the slope of the tail of the model curve in the third panel.
 - The fifth panel shows coinfection plotted against BFP-only, both the models and experimental data. The
 best-fit linear approximation of the data is shown in red, along with grey reference lines. The 'm' values
 show the slopes of their respective lines.
 - The fourth panel shows the slope of the tail of the model curve in the fifth panel.
'''
#==================================================================================================
import pandas as pd
import numpy as np
import random
from scipy.stats import variation
from matplotlib.widgets import Slider, Button
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#==================================================================================================
''' Set up initial parameters '''
leaf_num = 0
ylim     = 0
epsilon = 10**-20

nll_ymin = 300000
nll_ymax = 600000
V0_init    = .0002
B0_init    = .0002
M0_init    = .0008
bV_init    = .975
bB_init    = .835
x5_init    = .116
x6_init    = .858
x7_init    = .031
psi3_init  = .073
psi5_init  = .016
psi6_init  = .223
psi7_init  = .247
alpha1_init = 10.120
alpha2_init = .814

while(True):
    leaf_num = input("Enter leaf number: ")

    if (leaf_num == '3'):
        leaf_num = 0
        ylim = .07
        break
    elif (leaf_num == '5'):
        leaf_num = 1
        ylim = .05
        break
    elif (leaf_num == '6'):
        leaf_num = 2
        ylim = .3
        break
    elif (leaf_num == '7'):
        leaf_num = 3
        ylim = .4
        break

bound = int(input("Enter upper bound for t: "))
#==================================================================================================
''' Function which updates the plots after a slider's value changes '''
def update_plot(value):
    global V0
    global B0
    global M0
    global bV
    global bB
    global x5
    global x6
    global x7
    global psi3
    global psi5
    global psi6
    global psi7
    global alpha1
    global alpha2

    # Turn slider position into parameter value
    V0 = V0Slider.val
    B0 = B0Slider.val
    M0 = M0Slider.val
    bV = bVSlider.val
    bB = bBSlider.val
    x5 = x5Slider.val
    x6 = x6Slider.val
    x7 = x7Slider.val
    psi3 = psi3Slider.val
    psi5 = psi5Slider.val
    psi6 = psi6Slider.val
    psi7 = psi7Slider.val
    alpha1 = alpha1Slider.val
    alpha2 = alpha2Slider.val

    # Get model predictions using the parameters from the sliders
    init = [V0, B0, M0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    MM = odeint(model, init, t)
    Vk = MM[:, 3 * leaf_num + 0]
    Bk = MM[:, 3 * leaf_num + 1]
    Mk = MM[:, 3 * leaf_num + 2]

    # Remove any zero values
    for i in range(len(Vk)):
        if (Vk[i] < epsilon):
            Vk[i] = epsilon
        if (Bk[i] < epsilon):
            Bk[i] = epsilon
        if (Mk[i] < epsilon):
            Mk[i] = epsilon

    # Plot data on the relevant panels
    Vk_plot.set_ydata(Vk)
    Bk_plot.set_ydata(Bk)
    Mk_plot.set_ydata(Mk)

    logV_plot.set_xdata(np.log10(Vk))
    logV_plot.set_ydata(np.log10(Mk))
    logB_plot.set_xdata(np.log10(Bk))
    logB_plot.set_ydata(np.log10(Mk))

    nll = negLogLike()
    nll_plot.set_height(nll)

    dMdV = getSlope(Vk, Mk)
    secMV_plot.set_height(dMdV)

    dMdB = getSlope(Bk, Mk)
    secMB_plot.set_height(dMdB)
#==================================================================================================
''' Compute the nll value for a given set of parameter values '''
def negLogLike():
    # Solve ODE system to get model predictions; parameters are not yet fitted
    init = [V0, B0, M0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    MM = odeint(model, init, ZERO_DAYS_AXIS)
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

    global nll
    nll = 0
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
#==================================================================================================
''' Function which calculates the best-fit line to the data in the 3rd and 5th panels '''
def bestFitLine(xData, yData):
    new_xData = []
    new_yData = []

    # Exclude 0s from calculations
    for i in range(len(yData)):    
        if ((yData[i][0] != 0) and (xData[i][0] != 0)):
            new_xData.append(xData[i])
            new_yData.append(yData[i])
        else:
            print("Zero infected cells")

    # Perform regression
    regressor = LinearRegression()
    regressor.fit(np.log10(new_xData), np.log10(new_yData))

    # x-axis of best-fit line
    x = np.linspace(xMin0, xMax0)
    return [x, regressor.coef_[0][0] * x + regressor.intercept_[0], str(round(regressor.coef_[0][0], 3)), str(round(regressor.intercept_[0], 3))]
#==================================================================================================
''' Function to calculate the slope of the tails of the curves in the 3rd and 5th panels '''
def getSlope(xData, yData):
    lower = 0 # Leftmost point which is used for the secant line; the other endpoint
              # is the extreme rightmost point on the curve

    # Get rid of negatives and 0s
    for i in range(len(yData)):
        if (yData[i] < epsilon):
            yData[i] = epsilon
        if (xData[i] < epsilon):
            xData[i] = epsilon

    # Set leftmost endpoint at the average coinfection level or the smallest y-value on the graph window
    while((np.log10(yData[lower]) < max([yMin0, .5 * (min(np.log10(yData)) + max(np.log10(yData)))])) and (lower < len(yData) - 1)):
        lower = lower + 1

    # Make sure the x-coordinate at 'lower' is smaller than the rightmost x-coordinate
    while ((xData[-1] <= xData[lower]) and (lower < len(xData) - 1)):
        lower = lower + 1
        
    return ((np.log10(yData[-1]) - np.log10(yData[lower])) / (np.log10(xData[-1]) - np.log10(xData[lower])))
#==================================================================================================
''' Create subplots '''
fig, ax = plt.subplots(1, 6, gridspec_kw={'width_ratios': [1, .1, 1, .1, 1, .1]}, constrained_layout = True)
fig.subplots_adjust(bottom = .5, wspace = .7)
#==================================================================================================
''' Define sliders '''
axV0Slider = plt.axes([.1, .40, .8, .02]) #[left endpoint, height, right endpoint, thickness]
axB0Slider = plt.axes([.1, .37, .8, .02])
axM0Slider = plt.axes([.1, .34, .8, .02])
axbVSlider = plt.axes([.1, .31, .8, .02]) 
axbBSlider = plt.axes([.1, .28, .8, .02])
axx5Slider = plt.axes([.1, .25, .8, .02])
axx6Slider = plt.axes([.1, .22, .8, .02])
axx7Slider = plt.axes([.1, .19, .8, .02])
axpsi3Slider = plt.axes([.1, .16, .8, .02])
axpsi5Slider = plt.axes([.1, .13, .8, .02])
axpsi6Slider = plt.axes([.1, .10, .8, .02])
axpsi7Slider = plt.axes([.1, .07, .8, .02])
axalpha1Slider = plt.axes([.1, .04, .8, .02])
axalpha2Slider = plt.axes([.1, .01, .8, .02])
V0Slider = Slider(axV0Slider, r'$V_{0}$', valmin=epsilon, valmax=.001, valinit=V0_init, valfmt='%1.9f')
B0Slider = Slider(axB0Slider, r'$B_{0}$', valmin=epsilon, valmax=.001, valinit=B0_init, valfmt='%1.9f')
M0Slider = Slider(axM0Slider, r'$M_{0}$', valmin=epsilon, valmax=.001, valinit=M0_init, valfmt='%1.9f')
bVSlider = Slider(axbVSlider, r'$\beta_{V}$', valmin=epsilon, valmax=3, valinit=bV_init, valfmt='%1.4f')
bBSlider = Slider(axbBSlider, r'$\beta_{B}$', valmin=epsilon, valmax=3, valinit=bB_init, valfmt='%1.4f')
x5Slider = Slider(axx5Slider, r'$\chi_{5}$', valmin=epsilon, valmax=3, valinit=x5_init, valfmt='%1.4f')
x6Slider = Slider(axx6Slider, r'$\chi_{6}$', valmin=epsilon, valmax=3, valinit=x6_init, valfmt='%1.4f')
x7Slider = Slider(axx7Slider, r'$\chi_{7}$', valmin=epsilon, valmax=3, valinit=x7_init, valfmt='%1.4f')
psi3Slider = Slider(axpsi3Slider, r'$\psi_{3}$', valmin=epsilon, valmax=.5, valinit=psi3_init, valfmt='%1.4f')
psi5Slider = Slider(axpsi5Slider, r'$\psi_{5}$', valmin=epsilon, valmax=.5, valinit=psi5_init, valfmt='%1.4f')
psi6Slider = Slider(axpsi6Slider, r'$\psi_{6}$', valmin=epsilon, valmax=.5, valinit=psi6_init, valfmt='%1.4f')
psi7Slider = Slider(axpsi7Slider, r'$\psi_{7}$', valmin=epsilon, valmax=.5, valinit=psi7_init, valfmt='%1.4f')
alpha1Slider = Slider(axalpha1Slider, r'$\alpha_1$', valmin=epsilon, valmax=20, valinit=alpha1_init, valfmt='%1.4f')
alpha2Slider = Slider(axalpha2Slider, r'$\alpha_2$', valmin=epsilon, valmax=20, valinit=alpha2_init, valfmt='%1.4f')
#==================================================================================================
''' Limits '''
xMin0 = -4
xMax0 = -0
yMin0 = -4
yMax0 = 0

xMin2 = 0
xMax2 = bound
yMin2 = 0
yMax2 = .07
#==================================================================================================
''' Initial Values for models'''
V_init = np.linspace(2, 2.1, 1000)
B_init = np.linspace(2, 2.1, 1000)
M_init = np.linspace(2, 2.1, 1000)

x0 = np.linspace(xMin0, xMax0, 1000)
y0 = np.linspace(yMin0, yMax0, 1000)
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
#==================================================================================================
''' Axis for plotting '''
DAYS_AXIS = [3, 5, 7, 10]
#==================================================================================================
''' Make axis for negative log likelihood '''
ZERO_DAYS_AXIS = [0, 3, 5, 7, 10]
#==================================================================================================
''' Time axis for models '''
t = np.linspace(0, bound, 1000)
#==================================================================================================
''' Markers for plot '''
MARKERS = ['o', '^', 's', '*']
COLORS = ['b', 'y', 'g', 'm']
PLANT_NUM = [1, 2, 3, 4, 5]
LEAF_NUM = [3, 5, 6, 7]
LABELS = ['Total', 'Venus-Only', 'BFP - Only', 'Mixed']
#==================================================================================================
''' Leafy bois '''
LEAVES = ['Leaf_3', 'Leaf_5', 'Leaf_6', 'Leaf_7']
#==================================================================================================
''' Define the model '''
def model(Mk, t):
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
    dM3dt = alpha1 * (S3 * V3 * B3 * (bB + bV))

    dV5dt = bV * V5 * S5 + x5 * S5 * V3
    dB5dt = bB * B5 * S5 + x5 * S5 * B3
    dM5dt = alpha1 * S5 * V5 * B5 * (bB + bV) + alpha2 * x5 * S5 * (B5 * V3 + V5 + B3)

    dV6dt = bV * V6 * S6 + x6 * S6 * (V3 + V5)
    dB6dt = bB * B6 * S6 + x6 * S6 * (B3 + B5)
    dM6dt = alpha1 * S6 * V6 * B6 * (bB + bV) + alpha2 * x6 * S6 * (B6 * (V3 + V5) + V6 * (B3 + B5))

    dV7dt = bV * V7 * S7 + x7 * S7 * (V3 + V5 + V6)
    dB7dt = bB * B7 * S7 + x7 * S7 * (B3 + B5 + B6)
    dM7dt = alpha1 * S7 * V7 * B7 * (bB + bV) + alpha2 * x7 * S7 * (B7 * (V3 + V5 + V6) + V7 * (B3 + B5 + B6))

    return [dV3dt, dB3dt, dM3dt, dV5dt, dB5dt, dM5dt, dV6dt, dB6dt, dM6dt, dV7dt, dB7dt, dM7dt]
#==================================================================================================
''' Empty arrays to be used for later '''
MIXED_LEAFk = np.empty(((round(len(MIXED_RATIOS) / 4)), 1), float)
VENUS_LEAFk = np.empty(((round(len(VENUS_RATIOS) / 4)), 1), float)
BFP_LEAFk = np.empty(((round(len(BFP_RATIOS) / 4)), 1), float)
TEMP_MIXED = np.empty((5, 4), float)
TEMP_VENUS = np.empty((5, 4), float)
TEMP_BFP = np.empty((5, 4), float)
MIXED_AV = []
VENUS_AV = []
BFP_AV = []
#==================================================================================================
''' Make data for loglog plot '''
logVk = np.empty((round(len(VENUS_RATIOS) / 4), 1), float)
logBk = np.empty((round(len(BFP_RATIOS) / 4), 1), float)
logMk = np.empty((round(len(MIXED_RATIOS) / 4), 1), float)

for d in range(4):
    for p in range(5):
        logVk[5 * d + p] = VENUS_RATIOS[20 * d + 4 * p + leaf_num]
        logBk[5 * d + p] = BFP_RATIOS[20 * d + 4 * p + leaf_num]
        logMk[5 * d + p] = MIXED_RATIOS[20 * d + 4 * p + leaf_num]
#==================================================================================================
''' Compute data from csv '''
for i in range(len(MIXED_LEAFk)):
    MIXED_LEAFk[i] = MIXED_RATIOS[4 * i + leaf_num]
    VENUS_LEAFk[i] = VENUS_RATIOS[4 * i + leaf_num]
    BFP_LEAFk[i] = BFP_RATIOS[4 * i + leaf_num]
#==================================================================================================
''' Build a matrix of ratios with each row being a replicate '''
for i in range(5):  # Iterate through all rows
    for j in range(4):
        TEMP_MIXED[i][j] = MIXED_LEAFk[i + 5 * j]
        TEMP_VENUS[i][j] = VENUS_LEAFk[i + 5 * j]
        TEMP_BFP[i][j] = BFP_LEAFk[i + 5 * j]
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
#==================================================================================================
''' Plot results '''
for d in range(4):
    for p in range(5):  # PLot points
        ax[0].scatter(DAYS_AXIS[d], TEMP_MIXED[p, d], s = 80, facecolors = 'none', edgecolors = 'g', marker = MARKERS[d])
        ax[0].scatter(DAYS_AXIS[d], TEMP_VENUS[p, d], s = 80, facecolors = 'none', edgecolors = 'y', marker = MARKERS[d])
        ax[0].scatter(DAYS_AXIS[d], TEMP_BFP[p, d], s = 80, facecolors = 'none', edgecolors = 'b', marker = MARKERS[d])

Mk_plot, = ax[0].plot(t, M_init, 'g-', label = "Mixed")
Vk_plot, = ax[0].plot(t, V_init, 'y-', label = "Venus")
Bk_plot, = ax[0].plot(t, B_init, 'b-', label = "BFP")

ax[0].plot(DAYS_AXIS, VENUS_AV, color = 'y', linestyle = ':')
ax[0].plot(DAYS_AXIS, BFP_AV, color = 'b', linestyle = ':')
ax[0].plot(DAYS_AXIS, MIXED_AV, color = 'g', linestyle = ':')

ax[0].legend(loc = "upper left")
ax[0].set_xlim(xMin2, xMax2)
ax[0].set_ylim(yMin2, ylim)
ax[0].set_xlabel('Days post innoculation')
ax[0].set_ylabel('Frequency of celluar infection')
ax[0].text(.75 * xMax2, .9 * ylim, LEAVES[leaf_num])
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
nll_plot, = ax[1].bar(0, 1, bottom = 0, align = 'center', color = 'green')
ax[1].set_xticks([])
ax[1].set_xlabel('nll')
ax[1].set_ylim(nll_ymin, nll_ymax)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x, y, m, b = bestFitLine(logVk, logMk)
ax[2].plot(x, y, color = 'red', linestyle = ':', alpha = .5, label = "m = " + m)

ax[2].plot(np.linspace(xMin0, xMax0), np.linspace(yMin0, yMax0), 'k--', alpha = .5, label = "m = 1")
ax[2].plot(np.linspace(xMin0, xMin0 / 2), np.linspace(yMin0, yMax0), 'k-.', alpha = .5, label = "m = 2")

logV_plot, = ax[2].plot(x0, y0, 'y-')

for d in range(4):
    for p in range(5):
        ax[2].scatter(np.log10(logVk[5 * d + p]), np.log10(logMk[5 * d + p]), s = 80, facecolors = 'none', edgecolors = 'y', marker = MARKERS[d])

ax[2].set_xlim(xMin0, xMax0)
ax[2].set_ylim(yMin0, yMax0)
ax[2].legend(loc="upper left")
ax[2].set_xlabel(r'$log_{10}(Venus)$')
ax[2].set_ylabel(r'$log_{10}(Mixed)$')
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
secMV_plot, = ax[3].bar(0, 1, bottom = 0, align = 'center', color = 'yellow')
ax[3].set_xticks([])
ax[3].set_xlabel('secant')
ax[3].set_ylim(0, 5)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x, y, m, b = bestFitLine(logBk, logMk)
ax[4].plot(x, y, color = 'red', linestyle = ':', alpha = .5, label = "m = " + m)

ax[4].plot(np.linspace(xMin0, xMax0), np.linspace(yMin0, yMax0), 'k--', alpha = .5, label = "m = 1")
ax[4].plot(np.linspace(xMin0, xMin0 / 2), np.linspace(yMin0, yMax0), 'k-.', alpha = .5, label = "m = 2")

logB_plot, = ax[4].plot(x0, y0, 'b-')

for d in range(4):
    for p in range(5):
        ax[4].scatter(np.log10(logBk[5 * d + p]), np.log10(logMk[5 * d + p]), s = 80, facecolors = 'none', edgecolors = 'b', marker = MARKERS[d])

ax[4].set_xlim(xMin0, xMax0)
ax[4].set_ylim(yMin0, yMax0)
ax[4].legend(loc="upper left")
ax[4].set_xlabel(r'$log_{10}(BFP)$')
ax[4].set_ylabel(r'$log_{10}(Mixed)$')
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
secMB_plot, = ax[5].bar(0, 1, bottom = 0, align = 'center', color = 'blue')
ax[5].set_xticks([])
ax[5].set_xlabel('secant')
ax[5].set_ylim(0, 5)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#==================================================================================================
''' Check if a slider's value has been changed, if so redraw everything to update the plots '''
V0Slider.on_changed(update_plot)
B0Slider.on_changed(update_plot)
M0Slider.on_changed(update_plot)
bVSlider.on_changed(update_plot)
bBSlider.on_changed(update_plot)
x5Slider.on_changed(update_plot)
x6Slider.on_changed(update_plot)
x7Slider.on_changed(update_plot)
psi3Slider.on_changed(update_plot)
psi5Slider.on_changed(update_plot)
psi6Slider.on_changed(update_plot)
psi7Slider.on_changed(update_plot)
alpha1Slider.on_changed(update_plot)
alpha2Slider.on_changed(update_plot)
#==================================================================================================
plt.show()
#==================================================================================================