import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

def main():
	df = pd.read_csv("./KNN_Project_Data.csv")
	sl = StandardScaler()
	sl.fit(df.drop('TARGET CLASS',axis=1))
	sl_features = sl.transform(df.drop('TARGET CLASS',axis=1))
	df1 = pd.DataFrame(data=sl_features,columns=['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC'])
	train_data(df,df1)
	pick_best_k_value(df,df1)

def train_data(df,df1):
	X = df1
	y = df['TARGET CLASS']
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
	kn = KNeighborsClassifier(n_neighbors=1)
	kn.fit(X_train,y_train)
	predict_data(kn,X_test,y_test)

def predict_data(kn,X_test,y_test):
	predictions = kn.predict(X_test)
	print(confusion_matrix(y_test,predictions))
	print(classification_report(y_test,predictions))

def pick_best_k_value(df,df1):
	error_rate = []
	X = df1
	y = df['TARGET CLASS']
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
	
	for i in range(1,40):
	    kn = KNeighborsClassifier(n_neighbors=i)
	    kn.fit(X_train,y_train)
	    predictions_i = kn.predict(X_test)
	    error_rate.append(np.mean(predictions_i != y_test))

	plt.figure(figsize=(10,6))
	plt.plot(range(1,40),error_rate,marker='o',color='blue',linestyle='--',markerfacecolor='red',markersize=10)
	plt.title('Error Rate vs K Value')
	plt.xlabel('K')
	plt.ylabel('Error Rate')
	plt.show()

	#use best k value
	best_k = np.argmin(error_rate)+1S
	print(best_k)
	kn = KNeighborsClassifier(n_neighbors=best_k)
	kn.fit(X_train,y_train)
	predictions = kn.predict(X_test)
	from sklearn.metrics import classification_report,confusion_matrix
	print(" using best k value ",best_k)
	print(confusion_matrix(y_test,predictions))
	print(classification_report(y_test,predictions))

if __name__== "__main__":
  main()