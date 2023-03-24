import numpy as np
import xgboost
from numpy import genfromtxt
# -----------------------------------------------------
def ZScoreNorm(X = np.random.random((20, 2))*10):
    # Z score normalization
    MeanX = np.mean(X, axis = 0)
    StdX  = np.std(X, axis = 0)
    nX = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            nX[i,j] = (X[i,j] - MeanX[j])/StdX[j]
    return nX, MeanX, StdX
def ZScoreNormY(X = np.random.random(20)):
    # Z score normalization
    MeanX = np.mean(X)
    StdX  = np.std(X)
    nX = np.zeros(X.shape[0])
    for i in range(X.shape[0]):        
        nX[i] = (X[i] - MeanX)/StdX
    return nX, MeanX, StdX
# -------------------------------------------------
DataLoc = 'D:/SFRC_TensileStrength_173x8.csv'
featureNum = 8
CategorialVar = np.zeros(featureNum)   
dataset	= genfromtxt(DataLoc, delimiter=',')
X0	= dataset[:,0:featureNum]
Y0	= dataset[:,featureNum] 
Nd = len(Y0)
ridx = np.random.permutation(Nd)
X0 = X0[ridx, :]
Y0 = Y0[ridx]
X, meanX, stdX = ZScoreNorm(X0)  
Y, meanY, stdY = ZScoreNormY(Y0)
print('meanX', meanX)
print('stdX', stdX)
print('stdX', stdX)
print('meanY', meanY)
print('stdY', stdY)

Input = [400, 140, 739, 1108, 1, 63.636, 28, 80.232]					
##Actual fspt
T = 5.5223
print('Input', Input)

Input_z = np.zeros((1, featureNum))
for i in range(featureNum):
    Input_z[0, i] = (Input[i] - meanX[i])/stdX[i]
print('Input_z.shape', Input_z.shape)

Input_z_d = xgboost.DMatrix(Input_z)        

# load the model
PredictionModel_1 = xgboost.Booster()
PredictionModel_1.load_model("D:/TrainedModels/Trained_DE_XGBoost_Model_SEL.json")

PredictionModel_2 = xgboost.Booster()
PredictionModel_2.load_model("D:/TrainedModels/Trained_DE_XGBoost_Model_ASEL.json")    

Y_z_SEL = PredictionModel_1.predict(Input_z_d)
Y_z_ASEL = PredictionModel_2.predict(Input_z_d)

Y_SEL = Y_z_SEL*stdY + meanY
Y_ASEL = Y_z_ASEL*stdY + meanY
print('############################')
print('Predicted fspt (DE-XGBoost with SEL) = ', Y_SEL[0], '(MPa)')
print('Predicted fspt (DE-XGBoost with ASEL) = ', Y_ASEL[0], '(MPa)')
print('Actual fspt = ', T, '(MPa)')
