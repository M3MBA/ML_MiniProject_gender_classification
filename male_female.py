import csv
import numpy
from sklearn import tree

gender = []         #Data from kaggle is being used to train the model in order predict more accurately.
height = []         
weight = []

with open('male_female-data.csv','r') as file:  #there are 500 pre recorded data
    reader = csv.reader(file, delimiter = ",")
    header = next(reader)

    for row in reader:
        gender.append(row[0])
        height.append(row[1])
        weight.append(row[2])
combine = numpy.column_stack((height,weight))       #2(1D) array are converted into 1(2D)array using numpy
    
clf = tree.DecisionTreeClassifier()

clf = clf.fit(combine,gender)

#Enter the Height(in cm) and Weight(in kg) respectively below .

prediction = clf.predict([[170, 80]])

print(prediction)
