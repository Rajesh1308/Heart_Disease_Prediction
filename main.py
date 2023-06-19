import streamlit as st
import pandas as pd
import numpy as np
import pickle 

st.title("Heart Disease Prediction")

st.sidebar.header("Algorithm")

algo = st.sidebar.radio("Choose Algorithm", ("Decision Tree",
                                     "Logistic Regression",
                                     "Random Forest",
                                     "Navie Bayes",
                                     "Support Vector Machine - Linear Kernel",
                                     "Support Vector Machine - RBF Kernel",
                                     "Support Vector Machine - Polynomial Kernel",
                                     "Support Vector Machine - Sigmoidal Kernel",
                                     "K Nearest Neighbour",
                                     "Gradient Boost"))

def result(algo, dataset):
    data = np.array(dataset).reshape(1, -1)
    data = pd.DataFrame(data)
    data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    if algo == "Decision Tree":
        with open("decisiontree.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Logistic Regression":
        with open("logisticregression.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Random Forest":
        with open("randomforest.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Navie Bayes":
        with open("naivebayes.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Support Vector Machine - Linear Kernel":
        with open("svmlinear.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Support Vector Machine - RBF Kernel":
        with open("svmrbf.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Support Vector Machine - Polynomial Kernel":
        with open("svmpoly.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Support Vector Machine - Sigmoidal Kernel":
        with open("svmsigmoid.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "K Nearest Neighbour":
        with open("knn.pkl", "rb") as f:
            model = pickle.load(f)
    elif algo == "Gradient Boost" :
        with open("gradientboost.pkl", "rb") as f:
            model = pickle.load(f)

    ans = model.predict(data)
    res = ans[0]
    st.caption("The " + algo + " Model prediction result : ")
    if  res == 1 :
        st.warning("The Patient don't have disease")
    elif res == 0:
        st.warning("The Patient is having some disease")
    

dataset = []
with st.form("my_form"):
    #1
    age = st.number_input("Patient\'s age : ", max_value=120, min_value=0)
    dataset.append(age)

    #2
    gender = st.radio("Gender : ", ("Male", "Female"))
    if gender == "Male":
        dataset.append(1)
    elif gender == "Female":
        dataset.append(0)

    #3
    cp = st.number_input("Chest pain type (4 values) : ", max_value=4, min_value=0)
    dataset.append(cp)

    #4
    trespbps = st.number_input("Resting blood pressure (mm Hg) : ")
    dataset.append(trespbps)

    #5
    chol = st.number_input("serum cholestoral in mg/dl : ")
    dataset.append(chol)

    #6
    fbs = st.number_input(" Fasting blood sugar : ")
    if fbs > 120:
        dataset.append(1)
    else:
        dataset.append(0)

    #7
    restecg = st.number_input("Resting electrocardiographic results (values 0,1,2) : ", min_value=0, max_value=2)
    dataset.append(restecg)

    #8
    thalach = st.number_input("Maximum heart rate achieved : ", max_value=250, min_value=70)
    dataset.append(thalach)

    #9
    exang = st.radio("Exercise induced angina : ", ("Yes", "No"))
    if exang == "Yes":
        dataset.append(1)
    elif exang == "No":
        dataset.append(0)

    #10
    oldpeak = st.number_input("ST depression induced by exercise relative to rest (From ECG) : ")
    dataset.append(oldpeak)

    #11
    slope = st.radio("The slope of the peak exercise ST segment", ("Downsloping", "Flat", "Upsloping"))
    if slope == "Downsloping" :
        dataset.append(0)
    elif slope == "Flat" :
        dataset.append(1)
    elif slope == "Upsloping" :
        dataset.append(2)

    #12
    ca = st.number_input("The number of major vessels (0–3) : ", min_value=0, max_value=3)
    dataset.append(ca)

    #13
    thal = st.radio("Thalassemia : ", ("Null", 
                                       "Fixed defect", 
                                       "Normal blood flow", 
                                       "Reversible defect"))
    if thal == "Null":
        dataset.append(0)
    elif thal == "Fixed defect":
        dataset.append(1)
    elif thal == "Normal blood flow":
        dataset.append(2)
    elif thal == "Reversible defect":
        dataset.append(3)
    

########################################################################
    submitted = st.form_submit_button("Submit")
    if submitted:
       result(algo, dataset)
       
accuracy_score = []
if algo == "Decision Tree":
    pass
elif algo == "Logistic Regression":
    pass
elif algo == "Random Forest":
    pass    
elif algo == "Navie Bayes":
    pass    
elif algo == "Support Vector Machine - Linear Kernel":
    pass    
elif algo == "Support Vector Machine - RBF Kernel":
    pass    
elif algo == "Support Vector Machine - Polynomial Kernel":
    pass    
elif algo == "Support Vector Machine - Sigmoidal Kernel":
    pass
elif algo == "K Nearest Neighbour":
    pass
elif algo == "Gradient Boost" :
    pass