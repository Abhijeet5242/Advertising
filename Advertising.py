import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

def MarvellousAdvertisementPrediction(data_path):
    data=pd.read_csv(data_path,index_col=0)
    print("size of Actual dataset",len(data))
    feature_names = ['TV','radio','newspaper']
    print("Names of Features",feature_names)
    x=data[feature_names]
    y=data.sales
    X_train, X_test ,y_train, y_test=train_test_split(x,y,test_size=1/2)
    print("Size of Training dataset", len(X_train))
    print("Size of Testing dataset", len(X_train))
    linreg=LinearRegression()
    linreg.fit(X_train,y_train)
    y_pred=linreg.predict(X_test)
    print("Testing Set")
    print(X_test)
    print("Result of Testing : ")
    print(y_pred)
    print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

def main():
    MarvellousAdvertisementPrediction("Advertisingdata.csv")

if __name__=="__main__":
    main()  
