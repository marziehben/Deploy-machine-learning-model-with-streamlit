#import libraries
import matplotlib.pyplot as plt
import numpy
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from itertools import islice
import pickle 
import streamlit as st
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics



# Read a CSV file and determine X features & y target
df=pd.read_csv("C:\\Users\\M\\datatest.csv")
X=df.drop("level",axis=1).values
y=df.level.values


#Build a model & get a Scores
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

k_folds = KFold(n_splits = 5)
scores = cross_val_score(knn, X, y, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
y_true=df
expected=y_true
predict = knn.predict(X)
confusion_matrix(y_true['level'], y_pred=predict)
print(metrics.classification_report(y_true['level'], y_pred=predict))
print(type(X))
print(type(y))

# Visualize the dataframe in the Streamlit app
st.write("""
# Grade of Parts
""")
st.bar_chart(y)

# Title
st.title ("Hello Engineer")

# Header
st.header("Lets know about Quality Status") 


# pickling the model 
pickle_out = open("knn.pkl", "wb") 
pickle.dump(knn, pickle_out) 
pickle_out.close()


# loading in the model to predict(classify) on the data 
pickle_in = open('knn.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

def welcome(): 
	return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(ppm, SR, repeat_of_alarm, not_po_ka_yoke, high_price):
    ppm=int(ppm)
    SR=int(SR)
    repeat_of_alarm=int(repeat_of_alarm)
    not_po_ka_yoke=int(not_po_ka_yoke)
    high_price=int(high_price)
    


    prediction = classifier.predict([[ppm, SR, repeat_of_alarm, not_po_ka_yoke, high_price]])
    print(prediction)
    return prediction 
	

# this is the main function in which we define our webpage
#dictionary for levels
def main():
    idx2label={
		0:"A",
		1:"B",
        2:"C",
        3:"D",
        4:"E",
        5:"F"
	}
	# giving the webpage a title 
    st.title("Classification")
    html_temp = """ 
	<div style ="background-color:yellow;padding:13px"> 
	<h1 style ="color:black;text-align:center;"> Classifier ML App </h1> 
	</div> 
	"""

	# this line allows us to display the front end aspects we have 
    st.markdown(html_temp, unsafe_allow_html = True)
    ppm = st.selectbox("PPM", [0,1])
    SR = st.selectbox("S/R", [0,1])
    repeat_of_alarm = st.selectbox("Repeat of Alarm", [0,1])
    not_po_ka_yoke =st.selectbox("Not Pokayoka", [0,1])
    high_price= st.selectbox("High Price", [0,1])
    level = ([])
	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction(classify)
	# and store it in the variable result \
    if st.button("See a Grade"):
        level=prediction(ppm, SR, repeat_of_alarm, not_po_ka_yoke, high_price)
        st.success((idx2label[level[0]]))
	
if __name__=='__main__': 
	main()