
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Project Framingham Heart Disease - Group 7")
st.write(
    "The Framingham Heart Study is a long-term study of cardiovascular disease that identified risks factors and their join effects." \
    " There are 4434 patients, each of them with 3 examination periods every 6 years. The outcome is: angina pectoris, myocardial infraction, Atherothrombotic Infarction or Cerebral Hemorrhage (Stroke) or death."
)

#data loading
cvd = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

#first look at the data
st.dataframe(cvd.head())

#observations about the data
st.write(
    "The dataset contains 39 features inclding: demographic informations, clinical measurements, physiological risks factors, lifestyle, medications, baseline disease status, and follow up outcomes." 

)

#formulation of the research question
st.title(
    'Can we predict wheater a patient dies based on his baseline health indicators?'
)

st.markdown(
    """
    To answer this research question, we created a subset of the original dataframe:

    - only patients from period 1 were selected
    - all the variables related to time were dropped 
    - HDL and LDL cholesterol were dropped as they were not available for period 1


    """
)

#we performed metadata enrichment to add more context to the data and its visualization
#metadata enrichment
cvd['SEX'] = cvd['SEX'].replace({2: 'female', 1:'male'})
cvd['CURSMOKE'] = cvd['CURSMOKE'].replace({0: 'not current smoker', 1:'current smoker'})
cvd['DIABETES'] = cvd['DIABETES'].replace({0: 'not a diabetic', 1:'diabetic'})
cvd['BPMEDS'] = cvd['BPMEDS'].replace({0: 'not currently used', 1:'current use'})
cvd['PREVAP'] = cvd['PREVAP'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVSTRK'] = cvd['PREVSTRK'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVMI'] = cvd['PREVMI'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVCHD'] = cvd['PREVCHD'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVHYP'] = cvd['PREVHYP'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PERIOD'] = cvd['PERIOD'].replace({1: 'period 1', 2:'period 2', 3:'period 3'})
cvd['DEATH'] = cvd['DEATH'].replace({0: 'survived', 1:'died'})
cvd['ANGINA'] = cvd['ANGINA'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['HOSPMI'] = cvd['HOSPMI'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['MI_FCHD'] = cvd['MI_FCHD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['ANYCHD'] = cvd['ANYCHD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['STROKE'] = cvd['STROKE'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['CVD'] = cvd['CVD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['HYPERTEN'] = cvd['HYPERTEN'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['educ'] = cvd['educ'].replace({1: 'higher secondary', 2:'graduation', 3:'post graduation', 4: 'PhD'})

#subset creation
cvd_death = cvd.loc[(cvd['PERIOD'] == 'period 1')]
cvd_death = cvd_death.drop(columns = ['TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK', 'TIMECVD', 'TIMEHYP', 'HDLC', 'LDLC', 'TIME'])
cvd_death['DEATH'] = cvd_death['DEATH'].replace({0: 'survived', 1:'died'})


#final subest 
st.write("The final subset contains the following number of rows and columns (patients and features respectively):")
cvd_death.shape

#displaying the columns of the final subset
st.write("The final subset contains the following features:")
cvd_death.columns

#statistics of numerical variables
st.write("Here are some statistics of the numerical variables in the final subset:")
st.dataframe(cvd_death.describe())

#addressing missing values
st.write(
    """
    **Missing values handling**

    There are few missing values present in this subset:  
    between **0.02% and 9%** across **7 attributes**.

    It was decided to impute the missing values **after the train/test split**, as common practice.

    **Imputation strategy:**
    - Numerical variables were imputed using **KNN imputation (k = 5)**
    - Categorical variables were imputed using the **mode (most frequent value)**
    """
)
#calculating missing values percentage
missing_percentage = cvd_death.isnull().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0]

#visual rapresentation of the missing data
fig, ax = plt.subplots()
ax.set(title="Missing data", xlabel="Percent missing", ylabel="Variable", xlim=[0, 10]);
#orizontal bar
bars = ax.barh(missing_percentage.index, missing_percentage.values, color = 'lightblue', edgecolor = 'black')
ax.bar_label(bars);
st.pyplot(fig)

#outliers
st.write(
    """
    **Outliers handling**
    - 
    """
)

num_variables = ['TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']
var_names = {
    'TOTCHOL' : 'Total Cholesterol (mg/dL)',
    'AGE' : 'Age',
    'SYSBP' : 'Systolic Blood Pressure (mmHg)',
    'DIABP' : 'Diastolic Bloop Pressure (mmHg)',
    'CIGPDAY' : 'Number of cigarettes per day',
    'BMI' : 'Body Mass Index',
    'HEARTRTE' : 'Heart Rate',
    'GLUCOSE' : 'Glucose'

}
selected_variable = st.selectbox(
    "Select a numeric variable to visualize",
    num_variables
)

fig, ax = plt.subplots()

sns.boxplot(
    data=cvd_death[selected_variable],
    orient="v",
    ax=ax       
)

ax.set(
    title=var_names[selected_variable],
    xlabel=var_names[selected_variable],
    ylabel='Value'
)

st.pyplot(fig)  

st.write(
    '''
- colesterol lower than 120 and higher than 500 are probably outliers --> impute!
- there are no outliers for age
- systolic blood pressure, possible in a diseased condition
- dyastolic blood pressure, possible in a diseased condition
- bmi higher than 35 reflects an obese condition
- heart rate is plausible
- very high glucose levels indicate diabetes'''
)

st.title("Distribution of numerical variables")
#distribution of the data
selected_hist = st.selectbox("Select a numeric variable to visualize", num_variables, key="hist_selectbox")

# Histogram
fig2, ax = plt.subplots()

ax.hist(cvd_death[selected_hist], edgecolor='black', bins=20, color='lightblue')

ax.set(title=var_names[selected_hist], xlabel=var_names[selected_hist], ylabel='Count')

st.pyplot(fig2)

#visualization of categorical variables

#identification of categorical variables
categorical_variables = ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS','PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'ANGINA', 'HOSPMI', 'MI_FCHD', 'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN','DEATH', 'educ']

#creation fo a dictionary with metadata description
categorical_names = {
    'SEX': 'Sex',
    'CURSMOKE': 'Current Smoking Status',
    'DIABETES': 'Diabetes Status',
    'BPMEDS': 'Use of Blood Pressure Medications',
    'PREVAP': 'Prevalent Angina Pectoris',
    'PREVMI': 'Prevalent Miocardial Infraction',
    'PREVSTRK': 'Prevalent Stroke',
    'PREVHYP': 'Prevalent Hypertension',
    'ANGINA': 'Presence of Angina at the time of collection',
    'HOSPMI': 'Hospitalized for Myocardial Infraction',
    'MI_FCHD': 'Hospitalized Myocardial Infarction or Fatal Coronary Heart Disease',
    'ANYCHD': 'Any Form of Coronary Heart Disease',
    'STROKE': 'Stroke Event Follow-up',
    'CVD': 'Any Cardiovascular Disease Event Follow-up',
    'HYPERTEN': 'Hypertension Follow-up',
    'educ': 'Education Level',
    'PREVCHD': 'Plevalent Coronary Heart Disease',
    'DEATH': 'Death'
}

#barplot to visualize categorical variables
st.title('Categorical variables')
selected_bar = st.selectbox("Select a categorical variable to visualize", categorical_variables, key="barplot")

fig3, ax = plt.subplots()
counts = cvd_death[selected_bar].value_counts()
ax.bar(counts.index, counts.values, edgecolor='black', color = ['lightblue', 'lightpink'])
ax.bar_label(ax.containers[0])
ax.set(title= categorical_names[selected_bar], xlabel= categorical_names[selected_bar], ylabel= 'Count')
st.pyplot(fig3)

#visualization of the target variable 

# Distribution of DEATH
st.write('**Death distribution**')
death_counts = cvd_death['DEATH'].value_counts()

fig, ax = plt.subplots()

ax.pie(death_counts.values, labels=['Alive', 'Dead'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'pink'])

ax.set(title='Distribution of Death Outcome')

# Show in Streamlit
st.pyplot(fig)



#bivariate analysis
st.title('Bivariate analysis')

#categorical variables vs death
selected_cat = st.selectbox("Select a categorical variable to visualize", categorical_variables, key="categorical_barplot")

counts = cvd_death.groupby(['DEATH', selected_cat]).size().unstack()

# Bar plot
ax = counts.plot(kind='bar', edgecolor='black', color=['pink', 'lightblue', 'lightgreen', 'lightyellow'], rot=0)

ax.set(title=categorical_names[selected_cat], xlabel=categorical_names[selected_cat], ylabel='Count')

st.pyplot(ax.figure)

#gropbyfuction to see the difference between died and survived for the categorical variables

st.write('Difference in numerical variables for death and survived')
mean_table = cvd_death.groupby('DEATH')[num_variables].mean()

st.dataframe(mean_table)

