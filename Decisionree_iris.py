import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



df=pd.read_csv("Iris.csv")
print(df.head(5))
#Display the number of rows & columns in dataframe
print(df.shape)

x=df.drop("Species",axis=1)
y=df["Species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)

model=DecisionTreeClassifier(criterion="entropy",min_samples_split=15)
model.fit(x_train,y_train)

test_predict=model.predict(x_test)

print(accuracy_score(y_test,test_predict))

cn = confusion_matrix(y_test, test_predict)
sns.heatmap(cn, annot=True, fmt='d', cmap='rainbow')
plt.savefig("confusion_matrix1.png")
print("Plot saved as confusion_matrix.png")


plot_tree(model, filled=True, feature_names=x.columns, class_names=model.classes_, rounded=True)
plt.savefig("decision_tree_plot.png")
print("Decision tree saved as 'decision_tree_plot.png'")
