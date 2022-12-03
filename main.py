import pandas as pd
import math

from matplotlib import pyplot as plt

training_file = pd.read_csv("training.csv")
testing_file = pd.read_csv("testing.csv")

## we have 3 classes these are Iris-setosa, Iris-versicolor, Iris-virginica
## Priors = Number of Class / Number of Instances

Number_Of_Iris_setosa = 0
Number_Of_Iris_versicolor = 0
Number_Of_Iris_virginica = 0

for classType in training_file['Species']:
    if classType == 'Iris-setosa':
        Number_Of_Iris_setosa += 1
    elif classType == 'Iris-versicolor':
        Number_Of_Iris_versicolor += 1
    else:
        Number_Of_Iris_virginica += 1

print("")
print("Number_Of_Iris_setosa = " + str(Number_Of_Iris_setosa) )
print("Number_Of_Iris_versicolor = " + str(Number_Of_Iris_versicolor) )
print("Number_Of_Iris_virginica = " + str(Number_Of_Iris_virginica) )
print("")

Total_Number_Of_Data = Number_Of_Iris_setosa + Number_Of_Iris_versicolor + Number_Of_Iris_virginica

Priors_Of_Iris_setosa = Number_Of_Iris_setosa / Total_Number_Of_Data
Priors_Of_Iris_versicolor = Number_Of_Iris_versicolor / Total_Number_Of_Data
Priors_Of_Iris_virginica = Number_Of_Iris_virginica / Total_Number_Of_Data

print("Priors_Of_Iris_setosa = " + str(Priors_Of_Iris_setosa) )
print("Priors_Of_Iris_versicolor = " + str(Priors_Of_Iris_versicolor) )
print("Priors_Of_Iris_virginica = " + str(Priors_Of_Iris_virginica) )
print("")

## Find Mean of Each Classes

def Find_Total_Number_Of_Petal_Length_Width(n):
    p1 = []
    count_x = 0 ;
    count_y = 0 ;
    for index, row in training_file.iterrows():
        if row['Species'] == n:
            count_x += row['PetalLengthCm']
            count_y += row['PetalWidthCm']
    p1.append(count_x);
    p1.append(count_y);
    return p1;



Find_Total_Number_Of_Petal_Length_Width("Iris-setosa") ## [44.2, 7.400000000000002]
Find_Total_Number_Of_Petal_Length_Width("Iris-versicolor") ## [130.0, 40.6]
Find_Total_Number_Of_Petal_Length_Width("Iris-virginica") ## [168.10000000000002, 60.19999999999998]

Mean_of_Class_Iris_setosa = []
Mean_of_Class_Iris_versicolor = []
Mean_of_Class_Iris_virginica = []

Mean_of_Class_Iris_setosa.append(Find_Total_Number_Of_Petal_Length_Width("Iris-setosa")[0]/Number_Of_Iris_setosa)
Mean_of_Class_Iris_setosa.append(Find_Total_Number_Of_Petal_Length_Width("Iris-setosa")[1]/Number_Of_Iris_setosa)

Mean_of_Class_Iris_versicolor.append(Find_Total_Number_Of_Petal_Length_Width("Iris-versicolor")[0]/Number_Of_Iris_versicolor)
Mean_of_Class_Iris_versicolor.append(Find_Total_Number_Of_Petal_Length_Width("Iris-versicolor")[1]/Number_Of_Iris_versicolor)

Mean_of_Class_Iris_virginica.append(Find_Total_Number_Of_Petal_Length_Width("Iris-virginica")[0]/Number_Of_Iris_virginica)
Mean_of_Class_Iris_virginica.append(Find_Total_Number_Of_Petal_Length_Width("Iris-virginica")[1]/Number_Of_Iris_virginica)

print("Mean_of_Class_Iris_setosa = " + str(Mean_of_Class_Iris_setosa))
print("Mean_of_Class_Iris_versicolor = " + str(Mean_of_Class_Iris_versicolor))
print("Mean_of_Class_Iris_virginica = " + str(Mean_of_Class_Iris_virginica))
print("")

p1 = []
p2 = []
p3 = []


def Euclid_by_means_of_class (x,y):
    Euclid_Iris_setosa = math.pow((x - Mean_of_Class_Iris_setosa[0]), 2) + math.pow((y - Mean_of_Class_Iris_setosa[1]), 2)
    Euclid_Iris_versicolor = math.pow((x - Mean_of_Class_Iris_versicolor[0]), 2) + math.pow((y - Mean_of_Class_Iris_versicolor[1]), 2)
    Euclid_Iris_virginica = math.pow((x - Mean_of_Class_Iris_virginica[0]), 2) + math.pow((y - Mean_of_Class_Iris_virginica[1]), 2)
    if ((Euclid_Iris_setosa < Euclid_Iris_versicolor) and (Euclid_Iris_setosa < Euclid_Iris_versicolor)):
        return 1
    elif ((Euclid_Iris_versicolor < Euclid_Iris_setosa) and (Euclid_Iris_versicolor < Euclid_Iris_virginica)):
        return 2
    else :
        return 3

for index, row in testing_file.iterrows():
    if(Euclid_by_means_of_class(row['PetalLengthCm'], row['PetalWidthCm']) == 1):
        testing_file.loc[index, 'Prediction'] = "Iris-setosa"
    elif (Euclid_by_means_of_class(row['PetalLengthCm'], row['PetalWidthCm']) == 2):
        testing_file.loc[index, 'Prediction']= "Iris-versicolor"
    else:
        testing_file.loc[index, 'Prediction'] = "Iris-virginica"

for index, row in training_file.iterrows():
    if(Euclid_by_means_of_class(row['PetalLengthCm'], row['PetalWidthCm']) == 1):
        training_file.loc[index, 'Prediction'] = "Iris-setosa"
    elif (Euclid_by_means_of_class(row['PetalLengthCm'], row['PetalWidthCm']) == 2):
        training_file.loc[index, 'Prediction']= "Iris-versicolor"
    else:
        training_file.loc[index, 'Prediction'] = "Iris-virginica"



#plt.plot(testing_file[testing_file['Prediction'] == "Iris-setosa"]['PetalLengthCm'] ,testing_file[testing_file['Prediction'] == "Iris-setosa"]["PetalWidthCm"]  , 'x' )
#plt.plot(training_file[training_file['Prediction'] == "Iris-setosa"]['PetalLengthCm'] ,training_file[training_file['Prediction'] == "Iris-setosa"]["PetalWidthCm"]  , 'x' )
#plt.plot(training_file[training_file['Species'] == "Iris-setosa"]['PetalLengthCm'] ,training_file[training_file['Species'] == "Iris-setosa"]["PetalWidthCm"]  , 'x' )
plt.plot(testing_file[testing_file['Species'] == "Iris-setosa"]['PetalLengthCm'] ,testing_file[testing_file['Species'] == "Iris-setosa"]["PetalWidthCm"]  , 'x' )

#plt.plot(testing_file[testing_file['Prediction'] == "Iris-versicolor"]['PetalLengthCm'] ,testing_file[testing_file['Prediction'] == "Iris-versicolor"]["PetalWidthCm"]  , 'o' )
#plt.plot(training_file[training_file['Prediction'] == "Iris-versicolor"]['PetalLengthCm'] ,training_file[training_file['Prediction'] == "Iris-versicolor"]["PetalWidthCm"]  , 'o' )
#plt.plot(training_file[training_file['Species'] == "Iris-versicolor"]['PetalLengthCm'] ,training_file[training_file['Species'] == "Iris-versicolor"]["PetalWidthCm"]  , 'o' )
plt.plot(testing_file[testing_file['Species'] == "Iris-versicolor"]['PetalLengthCm'] ,testing_file[testing_file['Species'] == "Iris-versicolor"]["PetalWidthCm"]  , 'o' )

#plt.plot(testing_file[testing_file['Prediction'] == "Iris-virginica"]['PetalLengthCm'] ,testing_file[testing_file['Prediction'] == "Iris-virginica"]["PetalWidthCm"]  , '+' )
#plt.plot(training_file[training_file['Prediction'] == "Iris-virginica"]['PetalLengthCm'] ,training_file[training_file['Prediction'] == "Iris-virginica"]["PetalWidthCm"]  , '+'  )
#plt.plot(training_file[training_file['Species'] == "Iris-virginica"]['PetalLengthCm'] ,training_file[training_file['Species'] == "Iris-virginica"]["PetalWidthCm"]  , '+' )
plt.plot(testing_file[testing_file['Species'] == "Iris-virginica"]['PetalLengthCm'] ,testing_file[testing_file['Species'] == "Iris-virginica"]["PetalWidthCm"]  , '+' )

plt.plot(Mean_of_Class_Iris_setosa[0],Mean_of_Class_Iris_setosa[1], 'o')
plt.plot(Mean_of_Class_Iris_versicolor[0],Mean_of_Class_Iris_versicolor[1] , 'x')
plt.plot(Mean_of_Class_Iris_virginica[0],Mean_of_Class_Iris_virginica[1] , 'o')
plt.legend(['Iris-setosa','Irıs-versicolor','Iris-virginica','Mean Setosa','Mean Versicolor','Mean Setosa'])
plt.title('IRIS dataset')
plt.xlabel('PetalLength(Cm)')
plt.ylabel('PetalWidth(Cm)')
plt.show()



Iris_setosa_Iris_setosa = 0
Iris_versicolor_Iris_setosa = 0
Iris_virginica_Iris_setosa = 0

Iris_setosa_Iris_versicolor = 0
Iris_versicolor_Iris_versicolor = 0
Iris_virginica_Iris_versicolor = 0

Iris_setosa_Iris_virginica = 0
Iris_versicolor_Iris_virginica = 0
Iris_virginica_Iris_virginica = 0

for index, row in training_file.iterrows():
    if(row['Species'] == "Iris-setosa"):
         if(row['Prediction'] == "Iris-setosa"):
             Iris_setosa_Iris_setosa = Iris_setosa_Iris_setosa + 1
         elif (row['Prediction'] == "Iris-versicolor"):
             Iris_versicolor_Iris_setosa = Iris_versicolor_Iris_setosa + 1
         else:
             Iris_virginica_Iris_setosa = Iris_virginica_Iris_setosa + 1
    elif (row['Species'] == "Iris-versicolor"):
         if (row['Prediction'] == "Iris-setosa"):
             Iris_setosa_Iris_versicolor = Iris_setosa_Iris_versicolor + 1
         elif (row['Prediction'] == "Iris-versicolor"):
             Iris_versicolor_Iris_versicolor = Iris_versicolor_Iris_versicolor + 1
         else:
             Iris_virginica_Iris_versicolor = Iris_virginica_Iris_versicolor + 1
    else :
         if (row['Prediction'] == "Iris-setosa"):
             Iris_setosa_Iris_virginica = Iris_setosa_Iris_virginica + 1
         elif (row['Prediction'] == "Iris-versicolor"):
             Iris_versicolor_Iris_virginica = Iris_versicolor_Iris_virginica + 1
         else:
             Iris_virginica_Iris_virginica = Iris_virginica_Iris_virginica + 1

print("Iris_setosa_Iris_setosa = " + str(Iris_setosa_Iris_setosa))
print("Iris_setosa_Iris_setosa = " + str(Iris_versicolor_Iris_setosa))
print("Iris_virginica_Iris_setosa = " + str(Iris_virginica_Iris_setosa))
print(" ")
print("Iris_setosa_Iris_versicolor = " + str(Iris_setosa_Iris_versicolor))
print("Iris_setosa_Iris_versicolor = " + str(Iris_versicolor_Iris_versicolor))
print("Iris_virginica_Iris_versicolor = " + str(Iris_virginica_Iris_versicolor))
print(" ")
print("Iris_setosa_Iris_virginica = " + str(Iris_setosa_Iris_virginica))
print("Iris_versicolor_Iris_virginica = " + str(Iris_versicolor_Iris_virginica))
print("Iris_virginica_Iris_virginica = " + str(Iris_virginica_Iris_virginica))





##PART B

"""

def k_Nearest_Neighboor_Classfier_For_Training(z, q):
    distance = 0
    x = [] ## [index, distance]
    temp = []
    for index, row in training_file.iterrows():
       distance = math.pow((testing_file.loc[z,'PetalLengthCm'] - row['PetalLengthCm'] ), 2) +  math.pow((testing_file.loc[z,'PetalWidthCm'] - row['PetalWidthCm'] ), 2)
       if (len(x) < q):
            if (len(x) == 0):
                x.append([index,distance])
            else:
                x.append([index,distance])
                for n in range(len(x) - 1, 0, -1):
                    if (x[n][1] < x[n - 1][1]):
                        temp = x[n]
                        x[n] = x[n - 1]
                        x[n - 1] = temp
       else:
            for n in range(len(x) - 1, -1, -1):
                if (distance < x[n][1]):
                    if (len(x) == (n + 1)):
                          x[n] = [index,distance]
                    else:
                        temp = x[n]
                        x[n] = [index,distance]
                        x[n + 1] = temp
                else:
                    break

    Iris_setosa = 0
    Iris_versicolor = 0
    Iris_virginica = 0

    for i in x :
        if(training_file.loc[i[0],'Species'] == 'Iris-setosa'):
            Iris_setosa = Iris_setosa + 1
        elif(training_file.loc[i[0],'Species'] == 'Iris-versicolor'):
            Iris_versicolor =  Iris_versicolor + 1
        else:
             Iris_virginica = Iris_virginica + 1

    if(Iris_setosa > Iris_versicolor and Iris_setosa > Iris_virginica):
        testing_file.loc[z, 'K' + str(q)] = "Iris-setosa"
    elif (Iris_versicolor > Iris_setosa and Iris_versicolor > Iris_virginica):
        testing_file.loc[z, 'K' + str(q)] = "Iris-versicolor"
    else:
        testing_file.loc[z, 'K' + str(q)] = "Iris-virginica"


for index, row in testing_file.iterrows():
    k_Nearest_Neighboor_Classfier_For_Training(index, 1)
    k_Nearest_Neighboor_Classfier_For_Training(index, 3)


print(testing_file.to_string())

print(" ")

T1_T1 = 0
T2_T1 = 0
T3_T1 = 0

T1_T2 = 0
T2_T2 = 0
T3_T2 = 0

T1_T3 = 0
T2_T3 = 0
T3_T3 = 0


for index, row in training_file.iterrows():
    if(row['Species'] == "Iris-setosa"):
         if(row['K1'] == "Iris-setosa"):
             T1_T1 = T1_T1 + 1
         elif (row['K1'] == "Iris-versicolor"):
             T2_T1 = T2_T1 + 1
         else:
             T3_T1 = T3_T1 + 1
    elif (row['Species'] == "Iris-versicolor"):
         if (row['K1']== "Iris-setosa"):
             T1_T2 = T1_T2 + 1
         elif (row['K1'] == "Iris-versicolor"):
             T2_T2 = T2_T2 + 1
         else:
             T3_T2 = T3_T2 + 1
    else :
         if (row['K1'] == "Iris-setosa"):
             T1_T3 = T1_T3 + 1
         elif (row['K1'] == "Iris-versicolor"):
             T2_T3 = T2_T3 + 1
         else:
             T3_T3 = T3_T3 + 1





print("-------------")
print("T1_T1 = " + str(T1_T1))
print("T2_T1 = " + str(T2_T1))
print("T3_T1 = " + str(T3_T1))
print("-------------")
print("T1_T2 = " + str(T1_T2))
print("T2_T2 = " + str(T2_T2))
print("T3_T2 = " + str(T3_T2))
print("-------------")
print("T1_T3 = " + str(T1_T3))
print("T2_T3 = " + str(T2_T3))
print("T3_T3 = " + str(T3_T3))
print("--------------")


"""

def k_Nearest_Neighboor_Classfier_For_Training(z, q):
    distance = 0
    x = [] ## [index, distance]
    temp = []
    for index, row in training_file.iterrows():
       distance = math.pow((training_file.loc[z,'PetalLengthCm'] - row['PetalLengthCm'] ), 2) +  math.pow((training_file.loc[z,'PetalWidthCm'] - row['PetalWidthCm'] ), 2)
       if (len(x) < q):
            if (len(x) == 0):
                x.append([index,distance])
            else:
                x.append([index,distance])
                for n in range(len(x) - 1, 0, -1):
                    if (x[n][1] < x[n - 1][1]):
                        temp = x[n]
                        x[n] = x[n - 1]
                        x[n - 1] = temp
       else:
            for n in range(len(x) - 1, -1, -1):
                if (distance < x[n][1]):
                    if (len(x) == (n + 1)):
                          x[n] = [index,distance]
                    else:
                        temp = x[n]
                        x[n] = [index,distance]
                        x[n + 1] = temp
                else:
                    break

    Iris_setosa = 0
    Iris_versicolor = 0
    Iris_virginica = 0

    for i in x :
        if(training_file.loc[i[0],'Species'] == 'Iris-setosa'):
            Iris_setosa = Iris_setosa + 1
        elif(training_file.loc[i[0],'Species'] == 'Iris-versicolor'):
            Iris_versicolor =  Iris_versicolor + 1
        else:
            Iris_virginica = Iris_virginica + 1

    if(Iris_setosa > Iris_versicolor and Iris_setosa > Iris_virginica):
        training_file.loc[z, 'K = ' + str(q)] = "Iris-setosa"
    elif (Iris_versicolor > Iris_setosa and Iris_versicolor > Iris_virginica):
        training_file.loc[z, 'K = ' + str(q)] = "Iris-versicolor"
    else:
        training_file.loc[z, 'K = ' + str(q)] = "Iris-virginica"


for index, row in training_file.iterrows():
    k_Nearest_Neighboor_Classfier_For_Training(index, 1)
    k_Nearest_Neighboor_Classfier_For_Training(index, 3)
    k_Nearest_Neighboor_Classfier_For_Training(index, 5)
    k_Nearest_Neighboor_Classfier_For_Training(index, 7)
    k_Nearest_Neighboor_Classfier_For_Training(index, 9)

print(training_file.to_string())

print(" ")

T1_T1 = 0
T2_T1 = 0
T3_T1 = 0

T1_T2 = 0
T2_T2 = 0
T3_T2 = 0

T1_T3 = 0
T2_T3 = 0
T3_T3 = 0


for index, row in training_file.iterrows():
    if(row['Species'] == "Iris-setosa"):
         if(row['K = 1'] == "Iris-setosa"):
             T1_T1 = T1_T1 + 1
         elif (row['K = 1'] == "Iris-versicolor"):
             T2_T1 = T2_T1 + 1
         else:
             T3_T1 = T3_T1 + 1
    elif (row['Species'] == "Iris-versicolor"):
         if (row['K = 1']== "Iris-setosa"):
             T1_T2 = T1_T2 + 1
         elif (row['K = 1'] == "Iris-versicolor"):
             T2_T2 = T2_T2 + 1
         else:
             T3_T2 = T3_T2 + 1
    else :
         if (row['K = 1'] == "Iris-setosa"):
             T1_T3 = T1_T3 + 1
         elif (row['K = 1'] == "Iris-versicolor"):
             T2_T3 = T2_T3 + 1
         else:
             T3_T3 = T3_T3 + 1


print("T1_T1 = " + str(T1_T1))
print("T2_T1 = " + str(T2_T1))
print("T3_T1 = " + str(T3_T1))
print("-------------")
print("T1_T2 = " + str(T1_T2))
print("T2_T2 = " + str(T2_T2))
print("T3_T2 = " + str(T3_T2))
print("-------------")
print("T1_T3 = " + str(T1_T3))
print("T2_T3 = " + str(T2_T3))
print("T3_T3 = " + str(T3_T3))
print("--------------")



plt.plot(k_vals, accs_test)
plt.xlabel('k values')
plt.ylabel('accuracy')
plt.title('accuracy with changing k vals on test set')
plt.show()

"""
def k_Nearest_Neighboor_Classfier():
    k_Neighboor = [1,5,2,9,12,3,19]
    print(len(k_Neighboor))
    distance = 0
    temp = 0
    a = 3
    a1 = [3,2,9]
    x = []
    for m in range (len(k_Neighboor)):
        print(m)
        if(len(x) < 3):
            if(len(x) == 0):
                x.append(k_Neighboor[m])
            else:
                x.append(k_Neighboor[m])
                for n in range(len(x) - 1, 0, -1):
                    if(x[n] < x[n-1]):
                         temp = x[n]
                         x[n] = x[n-1]
                         x[n - 1] = temp
        else:
             for n in range(len(x)-1,-1,-1):

                if(k_Neighboor[m] < x[n]):
                    if(len(x) == (n + 1)):
                        x[n] = k_Neighboor[m]
                    else:
                        temp = x[n]
                        x[n] = k_Neighboor[m]
                        x[n+1] = temp
                else:
                    break
    print(x)
"""

"""
k_Nearest_Neighboor_Classfier()


arr = [[2,3],[4,9]]
arr.append([1, 2])
print(arr)
print(arr[1][1])

a = ([12,22],[12,9])
print(a[0])
arr[0] = a[0]
print(arr[0][1])



def k_Nearest_Neighboor_Classfierx():
    XT = [5,6,1,7,9]
    y = 6
    distance = 0
    x = [] ## [index, distance]
    temp = []
    for index in range(len(XT)):
       distance = math.pow((y-XT[index]),2)
       if (len(x) < 3):
            if (len(x) == 0):
                x.append([index,distance])
            else:
                x.append([index,distance])
                for n in range(len(x) - 1, 0, -1):
                    if (x[n][1] < x[n - 1][1]):
                        temp = x[n]
                        x[n] = x[n - 1]
                        x[n - 1] = temp
       else:
            for n in range(len(x) - 1, -1, -1):
                if (distance < x[n][1]):
                    if (len(x) == (n + 1)):
                          x[n] = [index,distance]
                    else:
                        temp = x[n]
                        x[n] = [index,distance]
                        x[n + 1] = temp
                else:
                    break

    print(x)

k_Nearest_Neighboor_Classfier()
"""