
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("/home/nazeel/PycharmProjects/PythonProject/.venv/insurance_dataset (1).csv")

print(df.head(5))


print(df.isnull().sum())

x=df[['age']]
y=df['charges']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.67,random_state=42)

print("Shape of x_train",x_train.shape)
print("Shape of y_train",y_train.shape)
print("Shape of x_test",x_test.shape)
print("Shape of y_test",x_test.shape)

model=LinearRegression()
model.fit(x_train,y_train)

print("Slope coeff",model.coef_[0])
print("Intercer",model.intercept_)

plt.style.use('classic')
plt.figure(figsize=(10,5))
sns.regplot(x='age', y='charges', data=df)
plt.title("Regression plot between Age and Charges")
plt.xlabel("Age")
plt.ylabel("charges")
plt.show()