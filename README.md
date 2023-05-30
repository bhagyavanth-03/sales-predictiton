

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import seaborn as sns
df=pd.read_csv('/kaggle/input/advertisingcsv/Advertising.csv')
df.shape
df.isnull().sum()
df.head()
sns.heatmap(df.corr(),annot=True)
sns.lmplot(data=df,x='Radio',y="Sales")
sns.lmplot(data=df,x='TV',y="Sales")
sns.lmplot(data=df,x='Newspaper',y="Sales")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x=df[['TV','Radio','Newspaper']]
y=df['Sales']
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.65, random_state=0)
print(model.intercept_)
print(model.coef_)
act_predict=pd.DataFrame({
    'Actual':y_test.values.flatten(),
    'Predict':y_predict.flatten()
})
act_predict.head(20)
sns.lmplot(data=act_predict,x='Actual',y="Predict")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Mean_absolute_error:",mean_absolute_error(y_test,y_predict))
print("Mean_squared_error:",mean_squared_error(y_test,y_predict))
print("Squre_Mean_absolute_error:",np.sqrt(mean_absolute_error(y_test,y_predict)))
print("r2_score:",r2_score(y_test,y_predict))
