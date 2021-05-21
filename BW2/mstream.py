import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import stats
import time
import sweetviz as sv

# import category_encoders as ce
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing  # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection  # train_test_split
# accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import metrics
from sklearn import set_config
from sklearn.datasets import make_classification
# from feature_engine.selection import SmartCorrelatedSelection

from PIL import Image
import pickle


# Load Data

@st.cache
def load_data(filename=None):
    filename_default = './data/data.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    return df

def load_df_test(filename=None):
    filename_default = './data/df_test.csv'
    if not filename:
        filename = filename_default

    df_test = pd.read_csv(f"./{filename}")
    return df_test
    


df = load_data()
df_test= load_df_test()
df_mean = df.drop(['android.sensor.accelerometer#min', 'android.sensor.accelerometer#max',
                   'android.sensor.accelerometer#std','android.sensor.game_rotation_vector#min',
                   'android.sensor.game_rotation_vector#max','android.sensor.game_rotation_vector#std',
                   'android.sensor.gravity#min', 'android.sensor.gravity#mean',
                   'android.sensor.gyroscope#std', 'android.sensor.gyroscope#min', 'android.sensor.gyroscope#max',
                   'android.sensor.gyroscope_uncalibrated#min', 'android.sensor.gyroscope_uncalibrated#max',
                   'android.sensor.gyroscope_uncalibrated#max', 'android.sensor.gyroscope_uncalibrated#std',
                   'android.sensor.light#mean', 'android.sensor.light#min',
                   'android.sensor.light#std', 'android.sensor.linear_acceleration#min',
                   'android.sensor.magnetic_field#max', 'android.sensor.magnetic_field#std',
                   'android.sensor.magnetic_field#min','android.sensor.linear_acceleration#max',
                   'android.sensor.linear_acceleration#std', 'android.sensor.magnetic_field_uncalibrated#min',
                   'android.sensor.magnetic_field_uncalibrated#max', 
                   'android.sensor.magnetic_field_uncalibrated#std', 'android.sensor.orientation#min',
                   'android.sensor.orientation#max', 'android.sensor.orientation#std' ,
                   'android.sensor.pressure#min', 'android.sensor.pressure#max', 'android.sensor.pressure#std',
                   'android.sensor.proximity#min', 'android.sensor.proximity#max', 'android.sensor.proximity#std',
                   'android.sensor.rotation_vector#min', 'android.sensor.rotation_vector#max',
                   'android.sensor.rotation_vector#std','android.sensor.step_counter#min', 'Unnamed: 0',
                   'android.sensor.step_counter#max',  'android.sensor.step_counter#std', 'sound#mean' ,
                   'sound#min', 'speed#std','activityrecognition#0', 'id', 'sound#std', 'android.sensor.gravity#std', 'speed#min', 'speed#mean' ], axis=1)
#df_mean = df_mean.drop('Unnamed: 0', axis=1)
#
model = pickle.load(open('./data/model.pkl', "rb"))


##################################################################################

##### Plots #####


##################################################################################


def main():
    menu = ["Home", "About", "Model","Upload files", "Calculator"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":

        ####################################################
        header = st.beta_container()
        team = st.beta_container()
        activities = st.beta_container()
        github = st.beta_container()
        # dataset = st.beta_container()
        # conclusion = st.beta_container()
        # footer = st.beta_container()
        ####################################################
        with header:
            st.title('MIFit model predictions')  # site title h1
            st.markdown("""---""")
            st.header('Welcome to our Machine Learning Project!')
            st.text(' ')
            image = Image.open('./data/logo.jpg')
            st.image(image, caption="")

            with team:
                # meet the team button
                st.sidebar.subheader('Our Team')

                st.sidebar.markdown(
                    '[Madina Zhenisbek](https://github.com/madinach)')
                st.sidebar.markdown(
                    '[Hedaya Ali](https://github.com/)')
                st.sidebar.markdown(
                    '[Thomas](https://github.com/)')
                st.sidebar.markdown(
                    '[Paramveer](https://github.com/)')

                st.sidebar.text(' ')
                st.sidebar.text(' ')

        with github:
            # github section:
            st.header('GitHub / Instructions')
            st.markdown(
                'Check the instruction [here](https://github.com/madinach/Mifit/blob/main/README.md)')
            st.text(' ')
############################################################################
    elif choice == "Model":

         
        st.title('Here is our model')     
        
        df_mean[:100]
        
        st.write("Accuracy= 80%, time= 9s")
        
        #start_time = time.time()
        #model.predict(df_test)
        #pred = model.predict(df_test)
        
        


        
        
        
##########################################################################
    elif choice == "Upload files":
        st.subheader("Our new feature!")
        st.write("Mi Fit - Activity Analysis")
        user_data = st.file_uploader("Upload your stats for the day...")

        st.write("_A design improvement for an actual app would be to either"
            " continuously upload the data or have a button that uploads the most recent batch of info_")
        st.write("_E.g._:")
        st.button("How did I do today?")


        # This is how to make use of an uploaded file
        if user_data is not None:
        # Read in the data as text (txt)
            file_data = str(user_data.read())
        # Split it into lines
            file_contents = file_data.split("\\n")
        # Try out our prepro functions
        #st.write(prep.extract_accelerometer(file_contents))
        #st.write(prep.extract_gyroscope(file_contents))
        #st.write(prep.extract_gravity(file_contents))


        st.write("Todo: Combine these different dataframes into a big one")
        st.write("Add the target to those big dataframes")
        st.write("Train models with those")

        # data = load_data('raw')
        #header = st.beta_container()
        #dataset = st.beta_container()
    
    elif choice == "About":
        st.title('MI Fit 💪 🦶🦶🦶')     
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.title('⚜️ About ⚜️:')
        st.markdown('''
    
    This project will determine the type of activity the person is doing. Whether he/she is:\n
    * in a car\n
    * on a train\n
    * in a bus\n
    * walking\n 
    * or just standing still

    
   
    ''')
        st.text('')
        st.text('')
        st.text('')


        st.write('''
        For the whole week, we were working on the the MIFit project.\n
        1-day: We were learning about the data itself. We found out that there are plenty of types of the androids sensors. Some of the useful data can be extracted from the dataset.\n
        2- day: We plotted the graphs.\n
        3 - day: We collected the data, used Graphext.\n
        4-5 day: Preparation for the presentation.
        ''')
        # data = load_data('raw')
        #header = st.beta_container()
        #dataset = st.beta_container()
        
        

        


    elif choice == "Calculator":
        st.header("BMI Calculator")
        st.write("The formula:\n Calories burned per minute = (0.035  X body weight in kg) + ((Velocity in m/s ^ 2) / Height in m)) X (0.029) X (body weight in kg)")
        selection = st.selectbox("Select your age group",
                                 ["10-17", "18-34", "35-44", "45-54", "55-64", "65-74", "75-79"])
        height = st.slider("Your height(in metres)", 0.55, 2.72)
        weight = st.slider("Your weight(in kilograms)", 20, 120)
        bmi = weight / (height * height)

        velocity = 2
        time = st.slider("How long did you exercise for? (Minutes)",0,120)


        # Calories burned per minute =
        # (0.035  X body weight in kg) + ((Velocity in m / s ^ 2) / height)) X(0.029) X (weight)
        calories_per_min = (0.035*weight) + ((velocity / height)) * (0.029*weight)
        total_calories = calories_per_min * time
        st.write(f"You burned {int(total_calories)} calories in total!")
        if total_calories > 5:
            st.write("Nice!")

# '''
#         with dataset:
#             st.title("About")
#
#             #### Data Correlation ####
#             st.set_option('deprecation.showPyplotGlobalUse', False)
#
#             st.text('Data Correlation ')
#             sns.set(style="white")
#             plt.rcParams['figure.figsize'] = (15, 10)
#             sns.heatmap(data.corr(), annot=True, linewidths=.5, cmap="Blues")
#             plt.title('Corelation Between Variables', fontsize=30)
#             plt.show()
#             st.pyplot()
#
#             #### Box Plot #####
#             st.text('Outlier Detection ')
#             fig = plt.figure(figsize=(15, 10))
#             sns.boxplot(data=data)
#             st.pyplot(fig)
#
#
#
#             st.text(' ')
#             st.header("Variables or features explanations:")
#             st.markdown("""* age (Age in years)""")
#             st.markdown("""* sex : (1 = male, 0 = female)""")
#             st.markdown(
#                 """* cp (Chest Pain Type): [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]""")
#             st.markdown("""* trestbps (Resting Blood Pressure in mm/hg )""")
#             st.markdown("""* chol (Serum Cholesterol in mg/dl)""")
#             st.markdown(
#                 """* fps (Fasting Blood Sugar > 120 mg/dl): [0 = no, 1 = yes]""")
#             st.markdown(
#                 """* restecg (Resting ECG): [0: normal, 1: having ST-T wave abnormality , 2: showing probable or definite left ventricular hypertrophy]""")
#             st.markdown("""* thalach (maximum heart rate achieved)""")
#             st.markdown(
#                 """* exang (Exercise Induced Angina): [1 = yes, 0 = no]""")
#             st.markdown(
#                 """* oldpeak (ST depression induced by exercise relative to rest)""")
#             st.markdown(
#                 """* slope (the slope of the peak exercise ST segment)""")
#             st.markdown("""* ca [number of major vessels (0–3)]""")
#             st.markdown(
#                 """* thal : [1 = normal, 2 = fixed defect, 3 = reversible defect]""")
#             st.markdown("""* target: [0 = disease, 1 = no disease]""")
#
#     elif choice == "ML":
#
#
#         with footer:
#             # Footer
#             st.markdown("""---""")
#             st.markdown("Heart Attack Predictions - Machine Learning Project")
#             st.markdown("")
#             st.markdown(
#                 "If you have any questions, checkout our [documentation](https://github.com/fistadev/starwars_data_project) ")
#             st.text(' ')
#
#         ############################################################################################################################
#     else:
#         st.header("Predictions")
#
#         def xgb_page_builder(data):
#             st.sidebar.header('Heart Attack Predictions')
#             st.sidebar.markdown('You can tune the parameters by siding')
#             st.sidebar.text_input("What's your age?")
#
#             cp = st.sidebar.slider(
#                 'Select max_depth (default = 30)', 0, 1, 2)
#             thalach = st.sidebar.slider(
#                 'Select learning rate (divided by 10) (default = 0.1)', min_value=50 , max_value=300 , value=None , step=5)
#             slope = st.sidebar.slider(
#                 'Select min_child_weight (default = 0.3)', 1, 2, 3)
#
#
#         #st.write(xgb_page_builder(data))
#         st.sidebar.header('Heart Attack Predictions')
#
#         st.text(' ')
#         st.markdown('Model selection')
#         st.text(' ')
#         #image = Image.open('./data/model-selection.png')
#         #st.image(image, caption="")
#         st.text(' ')
#
#         st.text(' ')
#         st.markdown('Selecting the best model with KFold')
#         st.text(' ')
#         #image = Image.open('./data/kfold.png')
#         #st.image(image, caption="")
#         st.text(' ')
#
# """
#
#         set_config(display='diagram')
#         st.write(load_clf)
#         a = [
#             54,
#             1,
#             cp,
#             131,
#             246,
#             0.148,
#             5280,
#             thalch,
#             0.326,
#             1.0396,
#             slope,
#             0.7293,
#             2.313,
#             ]
#         preds = load_clf.predict(a)
#         print(preds)
# '''
main()
