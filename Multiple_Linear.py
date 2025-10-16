import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("/home/nazeel/PycharmProjects/PythonProject/Housing.csv")

print(df.head(5))
print(df.isnull().sum())

df.replace(to_replace="yes",value=1,inplace=True)
df.replace(to_replace="no",value=0,inplace=True)
df.replace(to_replace="furnished",value=2,inplace=True)
df.replace(to_replace="semi-furnished",value=1,inplace=True)
df.replace(to_replace="unfurnished", value=0, inplace=True)


x=df.drop("price",axis=1)
y=df["price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

predict_train=model.predict(x_train)
predict_test=model.predict(x_test)

print("R2 Score of Train set",r2_score(y_train,predict_train))
print("R2 Score of Test set",r2_score(y_test,predict_test))
