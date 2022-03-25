import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

def plotCVError(y, y_pred, modelo):
  matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

  residual = y - y_pred

  fig, ax = plt.subplots()
  ax.scatter(y_pred, residual)
  ax.set_title("Residual plot in {}. \nMAE: {} \nMSE: {} \nRMSE: {} \nR²: {}".format(modelo, metrics.mean_absolute_error(y, y_pred), 
                                                                                         metrics.mean_squared_error(y, y_pred), np.sqrt(metrics.mean_squared_error(y, y_pred)),
                                                                                         metrics.r2_score(y, y_pred)))
  fig.get_figure().savefig('{}_residual.png'.format(modelo),dpi=600, bbox_inches='tight')
  
  
def getData():
    new_df = pd.read_csv('enade_regression.csv',sep=',',decimal='.')
    features = new_df.loc[:, new_df.columns != 'nota_geral_normalizada']

    # new_df.drop('nota_geral_categoria_2_60', 'nota_geral_categoria_2', 'nota_geral_categoria_3', 'nota_geral_categoria_5', axis=1)

    X = features
    y = new_df['nota_geral_normalizada'] #variavel alvo

    cv = KFold(n_splits=4, random_state=1, shuffle=True) #25% teste

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,stratify=y.tolist())
    
    return X,y,cv, X_train, y_train


def runCV(regressor, X,y,cv, X_train, y_train):    
    # regressor.fit(X_train, y_train)
    y_pred = cross_val_predict(regressor, X, y, cv=cv, n_jobs=-1, verbose=15)
    
    return y_pred


def main():
    X, y, cv, X_train, y_train = getData()

    
    regressores = [
        ['Arvore_regression', DecisionTreeRegressor(ccp_alpha=0.0, max_depth=10, max_leaf_nodes=50)],
        ['Bayes_regression', BayesianRidge(alpha_1=1e-8, alpha_2=1e-5, fit_intercept=False,lambda_1=1e-7, lambda_2=0.1, normalize=True,tol=0.1)],
        ['Linear_regression',  LinearRegression(n_jobs=-1, fit_intercept=True, normalize=False, positive=False)],
        ['Forest_regression', RandomForestRegressor(n_jobs=-1, bootstrap=True, max_depth=50, max_leaf_nodes=100, n_estimators=100)],
        ['Ridge_regression', Ridge(alpha=1.0, fit_intercept=False, normalize=False, tol=0.001)],
        ['SVM_regression', SVR()]
    ]
    
    for i in regressores:
        nome_modelo = i[0]
        print('Executando ', nome_modelo)
        regressor = i[1]
        
        y_pred = runCV(regressor, X,y,cv, X_train, y_train)
        
        plotCVError(y,y_pred, nome_modelo)

        
if __name__ == '__main__':
    main()