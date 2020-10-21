import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassiﬁer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


noData                = pd.read_json(r'list_job_tech_dataScience_NO.json')
yesData               = pd.read_json(r'list_job_tech_dataScience_YES.json')
testData              = pd.read_json(r'list_job_to_test.json')

dataF = pd.concat([noData, yesData])
#renommage de classe
def renameClass(data):
    data['clas']= ""
    for i in range (0,len(data)):
        if(data['is_data_scientist'].iloc[i] >0.5):
            data['clas'].iloc[i]= 1
        else :
            data['clas'].iloc[i]=0

    del data['is_data_scientist']
    data['clas'].astype('int')
    
    #return data
renameClass(dataF)
#print(dataF.head(200))  

def dataViz(data):
    
    y = data['clas'].value_counts(normalize=True)
    # Data to plot
    labels = 'Scientist', 'Not a scientist'
    sizes = [y[0],y[1]]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0.1, 0)  # explode 1st slice
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()

#dataViz(dataF)


def overSampling(data):
    ros = RandomOverSampler(random_state=42)
    X = dataF.drop(labels='clas', axis=1)
    y = dataF.loc[:,'clas']
    y = y.astype('int')
    X_over,y_over = ros.fit_resample(X, y)
    return X_over, y_over

X_train, y_train = overSampling(dataF)


def randomForestClsfr(data, X , y):
    pipe_RFC = Pipeline([
        ('vect' , CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf'  , RandomForestClassiﬁer(n_estimators= 200, min_samples_split= 2, min_samples_leaf= 1, max_depth= 10, criterion= 'entropy',n_jobs=-1))
        ])

    pipe_RFC.fit(X['jobTitle'], y)
    Y_predicted = pipe_RFC.predict(data[0])
    return Y_predicted

def logisticReg(data, X , y):
    pipe_logReg = Pipeline([
        ('vect' , CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf'  , LogisticRegression(solver='liblinear')),
    ])

    pipe_logReg.fit(X['jobTitle'], y)
    Y_predicted = pipe_logReg.predict(data[0])
    return Y_predicted
Y_Predicted = randomForestClsfr(testData, X_train, y_train)
#Y_Predicted = logisticReg(testData, X_train, y_train)
#AFFICHAGE DU RESULTAT
for i in range(0, len(testData)):
    print(Y_Predicted[i],testData[0].iloc[i]) 




