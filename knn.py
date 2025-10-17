import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/home/nazeel/PycharmProjects/PythonProject/Social_Network_Ads.csv")

X = df.drop(['User ID', 'Purchased'], axis=1)
y = df['Purchased']

print("\nFeature DataFrame Info:")
print(X.info())

# Convert categorical 'Gender' to numerical
X = pd.get_dummies(X, drop_first=True)  # drop_first avoids dummy trap

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))


print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix for kNN Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("knn_confusion_matrix.png")
print("\nConfusion matrix saved as 'knn_confusion_matrix.png'")