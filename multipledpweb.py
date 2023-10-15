import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading all the four saved models
diabetes_model = pickle.load(open('C:/Users/Nilanjana/Documents/2021-2025/2022-2025 KSIT/KSIT/2nd to 4th year (2022-2025)/2nd Year (2022-2023)/4th Sem/5) 21BE45 - Bio/Project/Diabetes Model/Diabetes_Prediction_Model.sav','rb'))
heart_disease_model = pickle.load(open('C:/Users/Nilanjana/Documents/2021-2025/2022-2025 KSIT/KSIT/2nd to 4th year (2022-2025)/2nd Year (2022-2023)/4th Sem/5) 21BE45 - Bio/Project/Heart Disease Model/Heart_Disease_Prediction_Model.sav','rb'))
parkinsons_model = pickle.load(open('C:/Users/Nilanjana/Documents/2021-2025/2022-2025 KSIT/KSIT/2nd to 4th year (2022-2025)/2nd Year (2022-2023)/4th Sem/5) 21BE45 - Bio/Project/Parkinsons Model/Parkinsons_Prediction_Model.sav','rb'))
breast_cancer_model = pickle.load(open('C:/Users/Nilanjana/Documents/2021-2025/2022-2025 KSIT/KSIT/2nd to 4th year (2022-2025)/2nd Year (2022-2023)/4th Sem/5) 21BE45 - Bio/Project/Breast Cancer Model/Breast_Cancer_Prediction_Model.sav','rb'))

#sidebar for navigation 
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction'],
                           icons = ['droplet-fill','heart-pulse','universal-access','hospital'],
                           default_index=0)


def parkinsons_prediction(input_data1): 
    #changing input data to a numpy array
    input_data_as_numpy_array1 = np.asarray(input_data1)
    #reshape the numpy array
    input_data_reshaped1 = input_data_as_numpy_array1.reshape(1,-1)
    prediction1 = parkinsons_model.predict(input_data_reshaped1)
    print(prediction1)
    if prediction1[0] == 0: 
        return "Congratulations! You can rest easy. You do not have Parkinson's."
    else:
        return "Very Sorry! You seem to have Parkinson's. Please consult a doctor as soon as possible."


def heart_disease_prediction(input_data2):
    #change the input data to a numpy array
    input_data_as_numpy_array2 = np.asarray(input_data2)
    #reshape the numpy array as we are predicting only for one instance
    input_data_reshaped2 = input_data_as_numpy_array2.reshape(1,-1)
    prediction2 = heart_disease_model.predict(input_data_reshaped2)
    print(prediction2)
    if (prediction2[0]== 0):
        return "The person does not have a Heart Disease."
    else:
        return "The person has a Heart Disease"
 
 
def diabetes_prediction(input_data3):
    # changing the input_data to numpy array
    input_data_as_numpy_array3 = np.asarray(input_data3)
    # reshape the array as we are predicting for one instance
    input_data_reshaped3 = input_data_as_numpy_array3.reshape(1,-1)
    prediction3 = diabetes_model.predict(input_data_reshaped3)
    print(prediction3)
    if (prediction3[0] == 0): 
        return "The person is not diabetic."
    else:
        return "The person is diabetic."
 
    
def breast_cancer_prediction(input_data4):
    # change the input data to a numpy array
    input_data_as_numpy_array4 = np.asarray(input_data4)
    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped4 = input_data_as_numpy_array4.reshape(1,-1)
    prediction4 = breast_cancer_model.predict(input_data_reshaped4)
    print(prediction4)
    if (prediction4[0] == 0): 
        return "The Breast cancer is Malignant"
    else:
        return "The Breast Cancer is Benign"
  
 
    
 
def main():
    #diabetes prediction page
    if (selected == 'Diabetes Prediction'):
        #page title
        st.title('Diabetes Prediction using ML')
        #getting input data from user
        Age = st.slider('Age',0,100,0)
        Pregnancies = st.slider('Pregnancies',0,20,0)
        col1, col2, col3 = st.columns(3)
        #with col1: 
        with col1:
            Glucose = st.text_input('Glucose Level')
        with col2:
            BloodPressure = st.text_input('Blood Pressure')
        with col3:
            SkinThickness = st.text_input('Skin Thickness')
        with col1:
            Insulin = st.text_input('Insulin Level')
        with col2:
            BMI = st.text_input('BMI value')
        with col3:
            DiabetesPedigreeFunction = st.text_input('Diabetes PF value')
        # code for Prediction
        diabetes_diagnosis = ''
        # creating a button for Prediction    
        if st.button("Diabetes Test Result"):
            diabetes_diagnosis = diabetes_prediction([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        st.success(diabetes_diagnosis)
    
    
    #heart disease prediction page
    if (selected == 'Heart Disease Prediction'):
        #page title
        st.title('Heart Disease Prediction using ML')
        #getting input data from user
        age = st.slider('Age',0,100,0)
        col1, col2, col3, col4 = st.columns(4)
        #with col1: 
        with col1:
            sex = st.selectbox('Sex',('0','1'))
        with col2:
            cp = st.selectbox('Chest Pain Type',('0','1','2','3'))
        with col3:
            trtbps = st.number_input('Resting BP')
        with col4:
            chol = st.number_input('Cholestrol')
        with col1:
            fbs = st.selectbox('Blood Sugar',('0','1'))
        with col2:
            restecg = st.selectbox('ECG',('0','1','2'))
        with col3:
            thalachh = st.number_input('Max Heart Rate')
        with col4:
            exng = st.selectbox('EIA',('0','1'))
        with col1:
            oldpeak = st.number_input('Old Peak')
        with col2:
            slp = st.selectbox('Slope of OP',('0','1','2'))
        with col3:
            caa = st.selectbox('Major vessels',('0','1','2','3','4'))
        with col4:
            thall = st.selectbox('Thal',('0','1','2','3'))
        # code for Prediction
        heart_disease_diagnosis = ''
        # creating a button for Prediction    
        if st.button("Heart Disease Test Result"):
            heart_disease_diagnosis = heart_disease_prediction([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
        st.success(heart_disease_diagnosis)
    
    
    #parkinsons prediction page
    if (selected == 'Parkinsons Prediction'):
        #page title
        st.title("Parkinson's Disease Prediction using Machine Learning Model")
        #getting input data from user
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: 
            fo = st.text_input('Average Vocal Freq') 
        with col2:
            fhi = st.text_input('Max Vocal Freq')
        with col3:
            flo = st.text_input('Min Vocal Freq')
        with col4:
            Jitter_percent = st.text_input('Jitter %')
        with col5:
            Jitter_Abs = st.text_input('Absolute Jitter Value')
        with col1:
            RAP = st.text_input('MDVP RAP')
        with col2:
            PPQ = st.text_input('MDVP PPQ')
        with col3:
            DDP = st.text_input('Jitter DDP')
        with col4:
            Shimmer = st.text_input('MDVP Shimmer')
        with col5:
            Shimmer_dB = st.text_input('MDVP Shimmer dB')
        with col1:
            APQ3 = st.text_input('Shimmer APQ3')
        with col2:
            APQ5 = st.text_input('Shimmer APQ5')
        with col3:
            APQ = st.text_input('MDVP APQ')
        with col4:
            DDA = st.text_input('Shimmer DDA')
        with col5:
            NHR = st.text_input('NHR ratio')
        with col1:
            HNR = st.text_input('HNR ratio')
        with col2:
            RPDE = st.text_input('RPDE')
        with col3:
            DFA = st.text_input('Signal FCE')
        with col4:
            spread1 = st.text_input('spread1')
        with col5:
            spread2 = st.text_input('spread2')
        with col1:
            D2 = st.text_input('D2')
        with col2:
            PPE = st.text_input('PPE')
        # code for Prediction
        parkinsons_diagnosis = ''
        # creating a button for Prediction    
        if st.button("Parkinson's Test Result"):
            parkinsons_diagnosis = parkinsons_prediction([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        st.success(parkinsons_diagnosis)
        
        
    #breast cancer prediction page
    if (selected == 'Breast Cancer Prediction'):
        #page title
        st.title('Breast Cancer Prediction using ML')
        #getting input data from user
        col1, col2, col3 = st.columns(3)
        with col1: 
            meanradius = st.number_input('Mean Radius') 
        with col2:
            meantexture = st.number_input('Mean Texture')
        with col3:
            meanperimeter = st.number_input('Mean Perimeter')
        with col1:
            meanarea = st.number_input('Mean Area')
        with col2:
            meansmoothness = st.number_input('Mean Smoothness')
        with col3:
            meancompactness = st.number_input('Mean Compactness')
        with col1:
            meanconcavity = st.number_input('Mean Concavity')
        with col2:
            meanconcavepoints = st.number_input('Mean Concave Points')
        with col3:
            meansymmetry = st.number_input('Mean Symmetry')
        with col1:
            meanfractaldimension = st.number_input('Mean FD')
        with col2:
            radiuserror = st.number_input('Radius Error')
        with col3:
            textureerror = st.number_input('Texture Error')
        with col1:
            perimetererror = st.number_input('Perimeter Error')
        with col2:
            areaerror = st.number_input('Area Error')
        with col3:
            smoothnesserror = st.number_input('Smoothness Error')
        with col1:
            compactnesserror = st.number_input('Compactness Error')
        with col2:
            concavityerror = st.number_input('Concavity Error')
        with col3:
            concavepointserror = st.number_input('Concave Points Error')
        with col1:
            symmetryerror = st.number_input('Symmetry Error')
        with col2:
            fractaldimensionerror = st.number_input('FD Error')
        with col3:
            worstradius = st.number_input('Worst Radius')
        with col1:
            worsttexture = st.number_input('Worst Texture')
        with col2:
            worstperimeter = st.number_input('Worst Perimeter')
        with col3:
            worstarea = st.number_input('Worst Area')
        with col1:
            worstsmoothness = st.number_input('Worst Smoothness')
        with col2:
            worstcompactness = st.number_input('Worst Compactness')
        with col3:
            worstconcavity = st.number_input('Worst Concavity')
        with col1:
            worstconcavepoints = st.number_input('Worst Concave Points')
        with col2:
            worstsymmetry = st.number_input('Worst Symmetry')
        with col3:
            worstfractaldimension = st.number_input('Worst FD')
        # code for Prediction
        breast_cancer_diagnosis = ''
        # creating a button for Prediction    
        if st.button("Breast Cancer Test Result"):
            breast_cancer_diagnosis = breast_cancer_prediction([[meanradius, meantexture, meanperimeter, meanarea, meansmoothness, meancompactness, meanconcavity, meanconcavepoints, meansymmetry, meanfractaldimension, radiuserror, textureerror, perimetererror, areaerror, smoothnesserror, compactnesserror, concavityerror, concavepointserror, symmetryerror, fractaldimensionerror, worstradius, worsttexture, worstperimeter, worstarea, worstsmoothness, worstcompactness, worstconcavity, worstconcavepoints, worstsymmetry, worstfractaldimension]])
        st.success(breast_cancer_diagnosis)
            
if __name__ == '__main__':
    main()