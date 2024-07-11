import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import uuid
import os
import mysql.connector

# Setting page configuration
st.set_page_config(page_title="Diagno Predict", page_icon="🧊", layout="wide", initial_sidebar_state="auto")

# Load datasets
training_data = "dataset/Training.csv"
testing_data = "dataset/Testing.csv"

# Load data
try:
    data = pd.read_csv(training_data).dropna(axis=1)
except Exception as e:
    st.error("NOT_FOUND_ERROR - Could not find the training dataset. Check the path for the training dataset under section 'LOAD DATASETS'")

# Encode target value
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Initializing models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Connect to MySQL
mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="KAch@#$123",
  database="disease_prediction"
)

# Function to save predictions to MySQL
def save_predictions_to_db(symptoms, predictions):
    cursor = mydb.cursor()
    sql = "INSERT INTO predictions (symptoms, rf_prediction, nb_prediction, svm_prediction, final_prediction) VALUES (%s, %s, %s, %s, %s)"
    val = (symptoms, predictions["rf_model_prediction"], predictions["naive_bayes_prediction"], predictions["svm_model_prediction"], predictions["final_prediction"])
    cursor.execute(sql, val)
    mydb.commit()
    cursor.close()

# Streamlit app layout
st.markdown('''<div style="font-size:70px; font-weight: bold;"> 
            Diagno Predict 
            </div>''', unsafe_allow_html=True)
st.divider()

# Display disease counts
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

fig = plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
st.pyplot(fig)

# Cross-validation function
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Cross-validation scores
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)
    st.warning(model_name)
    st.success(f"Scores: {scores}")
    st.success(f"Mean Score: {np.mean(scores)}")

# Model testing functions
@st.cache
def train_test_svm():
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    preds = svm_model.predict(X_test)
    
    st.info("Accuracy by Support Vector Machine Classifier")
    st.success(f"On Train Dataset: {accuracy_score(y_train, svm_model.predict(X_train))*100:.2f}%")
    st.success(f"On Test Dataset: {accuracy_score(y_test, preds)*100:.2f}%")
    cf_matrix = confusion_matrix(y_test, preds)
    svm_plot = plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title("Confusion Matrix for SVM Classifier on Test Data")
    st.pyplot(svm_plot)

@st.cache
def train_test_nb():
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    preds = nb_model.predict(X_test)
    
    st.info("Accuracy by Gaussian Naive Bayes Classifier")
    st.success(f"On Train Dataset: {accuracy_score(y_train, nb_model.predict(X_train))*100:.2f}%")
    st.success(f"On Test Dataset: {accuracy_score(y_test, preds)*100:.2f}%")
    cf_matrix = confusion_matrix(y_test, preds)
    nbc_plot = plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
    st.pyplot(nbc_plot)

@st.cache
def train_test_rfc():
    rf_model = RandomForestClassifier(random_state=18)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)
    
    st.info("Accuracy by Random Forest Classifier")
    st.success(f"On Train Dataset: {accuracy_score(y_train, rf_model.predict(X_train))*100:.2f}%")
    st.success(f"On Test Dataset: {accuracy_score(y_test, preds)*100:.2f}%")
    cf_matrix = confusion_matrix(y_test, preds)
    rfc_plot = plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
    st.pyplot(rfc_plot)

# Model testing section
st.divider()
st.markdown('''<div style="font-size:45px; font-weight: bold; padding-bottom: 20px;"> 
            Model Testing 
            </div>''', unsafe_allow_html=True)
st.write()

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    train_test_svm()

with row1_col2:
    train_test_nb()

with row1_col3:
    train_test_rfc()

# Function to predict disease
@st.cache
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
        
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # making final prediction by taking mode of all predictions
    final_prediction = np.unique([rf_prediction, nb_prediction, svm_prediction])[0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Prediction form
st.divider()
st.markdown('''<div style="font-size:45px; font-weight: bold; padding-bottom: 20px;"> 
            Model Predictions 
            </div>''', unsafe_allow_html=True)
st.write()

# Symptom selection form
row4_col1, row4_col2 = st.columns([1, 1], gap="large")

with row4_col2:
    st.write()
    st.info("The Model is trained for the following Range of Symptoms")
    symptoms_list = list(symptoms)
    symptoms_list.sort()
    symtoms_list_regenerated = []
    for word in symptoms_list:
        if "_" in word:
            regenerated_symptom = " ".join(word.split("_"))
            symtoms_list_regenerated.append(regenerated_symptom)
        else:
            symtoms_list_regenerated.append(word.title())
    
    row4_sub1_col1, row4_sub1_col2 = st.columns(2)
    row4_sub1_col1.write(symtoms_list_regenerated[:66])
    row4_sub1_col2.write(symtoms_list_regenerated[66:])

with row4_col1:
    st.warning("")
    predict_disease_form = st.form("Disease Prediction Form", clear_on_submit=True)
    r1, r2 = predict_disease_form.columns(2)
    symptom_value_range = tuple(["None"] + symtoms_list_regenerated)
    symptom_1 = r1.selectbox("Symptom 1:", symptom_value_range, placeholder="Choose an Option")
    symptom_2 = r2.selectbox("Symptom 2:", symptom_value_range, placeholder="Choose an Option")
    symptom_3 = r1.selectbox("Symptom 3:", symptom_value_range, placeholder="Choose an Option")
    symptom_4 = r2.selectbox("Symptom 4:", symptom_value_range, placeholder="Choose an Option")
    symptom_5 = r1.selectbox("Symptom 5:", symptom_value_range, placeholder="Choose an Option")
    symptom_6 = r2.selectbox("Symptom 6:", symptom_value_range, placeholder="Choose an Option")
    predict_disease_form_submit = predict_disease_form.form_submit_button("Make Predictions")

    if predict_disease_form_submit:
        list_entry = [symptom_1, symptom_2, symptom_3, symptom_4, symptom_5, symptom_6]
        symptoms_group_list = [s for s in list_entry if s != "None"]
        symptoms_group = ",".join(symptoms_group_list)
        user_entry_dict = {"Symptom Number": [f"Symptom {j}" for j in range(1, 7)], "Data": list_entry}
        user_entry_symptoms_df = pd.DataFrame(user_entry_dict, index=[n for n in range(1, 7)])
        st.dataframe(user_entry_symptoms_df, use_container_width=True, hide_index=True)
        
        try:
            st.success("Model Predictions")
            output = predictDisease(symptoms_group)
            st.write(output)
            save_predictions_to_db(symptoms_group, output)
        except Exception as e:
            st.error("There was an error in predicting the disease.")

# Display saved predictions
st.divider()
st.info("View previous predictions by the model")
fetch_file_form = st.form("Fetch File Form", clear_on_submit=True)
file_list = [''] + os.listdir("output")
file_fetch = fetch_file_form.selectbox("Select File", file_list)

if file_fetch:
    filepath = os.path.join("output", file_fetch)
    if os.path.isfile(filepath):
        previous_predictions = pd.read_csv(filepath)
        st.dataframe(previous_predictions)

# End of Streamlit app
st.markdown("---")
st.markdown("Made with ❤️ by [Your Name]")
