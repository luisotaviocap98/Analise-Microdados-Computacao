# http://computacaointeligente.com.br/outros/intro-sklearn-part-3/
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# https://medium.com/@EduardoSaverin/confusion-matrix-614be4ff4c9e
def saveConfusion(matrix, modelo, y, y_pred):
    with open('{}_matriz.txt'.format(modelo),'w') as data:
        
        data.write("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(metrics.accuracy_score(y, y_pred), 
                                                                                         metrics.precision_score(y, y_pred, average='weighted'), metrics.recall_score(y, y_pred, average='weighted'),
                                                                                         metrics.f1_score(y, y_pred,average='weighted')))
        data.write('\n       ___________________________Predito_____________________________\n')
        data.write('         _____________| Baixa | Média | Alta  |\n')
        data.write('         |Baixa       |'+str(matrix[0][0])+'|'+str(matrix[0][1])+'|'+str(matrix[0][2])+'\n')
        data.write('Original |-------------------------------------\n')
        data.write('         |Média       |'+str(matrix[1][0])+'|'+str(matrix[1][1])+'|'+str(matrix[1][2])+'\n')
        data.write('         --------------------------------------\n')
        data.write('         |Alta        |'+str(matrix[2][0])+'|'+str(matrix[2][1])+'|'+str(matrix[2][2])+'\n')


def getData():
    new_df = pd.read_csv('enade_classifier.csv',sep=',',decimal='.')
    features = new_df.loc[:, new_df.columns != 'nt_geral_categoria_3']
    # new_df.drop('nota_geral_normalizada_2_60', 'nota_geral_normalizada_2', 'nota_geral_normalizada_3', 'nota_geral_normalizada_5', axis=1)
    new_df.drop( ['nt_geral_categoria_2', 'nt_geral_categoria_2_60', 'nt_geral_categoria_5'], axis=1)

    X = features
    y = new_df['nt_geral_categoria_3'] #variavel alvo

    cv = KFold(n_splits=4, random_state=1, shuffle=True) #25% teste

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    
    return X,y,cv, X_train, y_train

def runCV(classificador, X,y,cv, X_train, y_train):    
    classificador.fit(X_train, y_train)
    y_pred = cross_val_predict(classificador, X, y, cv=cv, n_jobs=-1, verbose=15)
    
    return y_pred


def main():
    X, y, cv, X_train, y_train = getData()

    
    classificadores = [
        ['Arvore_classifier', DecisionTreeClassifier(ccp_alpha=0.0, criterion='entropy', max_depth=10, max_leaf_nodes=100, min_samples_leaf=2, splitter='random')],
        ['Naive_classifier', GaussianNB(var_smoothing=0.1)],
        ['KNN_classifier', KNeighborsClassifier(algorithm='kd_tree', leaf_size=50, n_neighbors=9, p=2, weights='uniform')],
        ['Forest_classifier', RandomForestClassifier(n_jobs=-1, bootstrap=False, class_weight= 'balanced', criterion='entropy', max_depth=15, max_leaf_nodes=100, min_samples_leaf=2, n_estimators=40)],
        ['Logistica_classifier', LogisticRegression(solver='saga',  max_iter=500, tol=0.01, C=10, multi_class='multinomial')],
        ['SVM_classifier', SVC()]
    ]
    
    for i in classificadores:
        nome_modelo = i[0]
        print('Executando ', nome_modelo)
        classificador = i[1]
        
        y_pred = runCV(classificador, X,y,cv, X_train, y_train)

        saveConfusion(confusion_matrix(y, y_pred, labels=[1,2,3]), nome_modelo, y, y_pred)

        
if __name__ == '__main__':
    main()
    
    
    
# https://towardsdatascience.com/comprehensive-guide-to-multiclass-classification-with-sklearn-127cc500f362
# https://stackoverflow.com/questions/45890328/sklearn-metrics-for-multiclass-classification
# https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
# https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit
# https://www.codegrepper.com/code-examples/python/metrics+for+multiclass+classification+sklearn
# geeksforgeeks.org/multiclass-classification-using-scikit-learn/
# https://www.kaggle.com/nkitgupta/evaluation-metrics-for-multi-class-classification