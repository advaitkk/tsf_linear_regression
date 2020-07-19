"""
Created on Sun Jul 19 17:39:35 2020

@author: Advait
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

df=pd.read_csv("student_scores.csv")
#print(df)
#plt.scatter(df.Hours,df.Scores,color='red')
X=df['Hours'].to_numpy().reshape(-1, 1)
y=df['Scores'].to_numpy().reshape(-1, 1)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#TRAIN
regfunc=LinearRegression()
regfunc.fit(x_train,y_train)
#PLOT
line = regfunc.coef_*X+regfunc.intercept_
plt.plot(X,line)
plt.scatter(df.Hours,df.Scores,color='red')
#PREDICT
myhour=np.array(9.25)
answer=regfunc.predict(myhour.reshape(-1, 1))
print("The predicted score if a student studies for",myhour,"hours is",answer)

