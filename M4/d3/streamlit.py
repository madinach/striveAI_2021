import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import stats

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

from PIL import Image


# Load Data

@st.cache
def load_data(filename=None):
    filename_default = './data/heart.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    return df
    # return df, df.shape[0], df.shape[1], filename


data = load_data()

##### Sidebar #####


def xgb_page_builder(data):
    st.sidebar.header('Heart Attack Predictions')
    st.sidebar.markdown('You can tune the parameters by siding')
    cp = st.sidebar.slider('Select cp (default = 0)', 0, 2)
    thale = st.sidebar.slider('Select Thalch (default = 150)',
                              min_value=50,
                              max_value=300,
                              value=150,
                              step=5)
    slope = st.sidebar.slider('Select slope (default = 1)', 0, 1, 2)



st.write(xgb_page_builder(data))

############


header = st.beta_container()
team = st.beta_container()
dataset = st.beta_container()
footer = st.beta_container()


with header:
    st.title('Heart Attack Predictions')  # site title h1
    st.markdown("""---""")
    st.header('Machine Learning Project')
    st.text(' ')
    # image = Image.open('data/baby-yoda.jpg')
    # st.image(image, caption="This is the way")
    st.text(' ')
    with team:
        # meet the team button
        st.subheader('John Locke Team')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        col1, col2, col3, col4 = st.beta_columns(4)
        with col1:
            # image = Image.open('imgs/fabio.jpeg')
            # st.image(image, caption="")
            st.markdown(
                '[Fabio Fistarol](https://github.com/fistadev)')
        with col2:
            # image = Image.open('imgs/madina.jpeg')
            # st.image(image, caption="")
            st.markdown(
                '[Madina Zhenisbek](https://github.com/madinach)')
        with col3:
            # image = Image.open('imgs/alessio.jpg')
            # st.image(image, caption="")
            st.markdown(
                '[Alessio Recchia](https://github.com/alessiorecchia)')
        with col4:
            # image = Image.open('imgs/shahid.jpg')
            # st.image(image, caption="")
            st.markdown(
                '[Shahid Qureshi](https://github.com/shahidqureshi01)')

        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.markdown("""---""")
        st.text(' ')
        st.text(' ')
        # image = Image.open('data/long_time_ago.jpg')
        # st.image(image, caption="")

        # Add audio
        # audio_file = open('data/star_wars_theme_song.mp3', 'rb')
        # audio_bytes = audio_file.read()
        # st.audio(audio_bytes, format='audio/ogg')

##############

st.header("Variables or features explanations:")
st.markdown("""* age (Age in years)""")
st.markdown("""* sex : (1 = male, 0 = female)""")
st.markdown(
    """* cp (Chest Pain Type): [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]""")
st.markdown("""* trestbps (Resting Blood Pressure in mm/hg )""")
st.markdown("""* chol (Serum Cholesterol in mg/dl)""")
st.markdown("""* fps (Fasting Blood Sugar > 120 mg/dl): [0 = no, 1 = yes]""")
st.markdown(
    """* restecg (Resting ECG): [0: normal, 1: having ST-T wave abnormality , 2: showing probable or definite left ventricular hypertrophy]""")
st.markdown("""* thalach (maximum heart rate achieved)""")
st.markdown("""* exang (Exercise Induced Angina): [1 = yes, 0 = no]""")
st.markdown("""* oldpeak (ST depression induced by exercise relative to rest)""")
st.markdown("""* slope (the slope of the peak exercise ST segment)""")
st.markdown("""* ca [number of major vessels (0–3)]""")
st.markdown(
    """* thal : [1 = normal, 2 = fixed defect, 3 = reversible defect]""")
st.markdown("""* target: [0 = disease, 1 = no disease]""")


###############


with dataset:
    st.header("")
    # st.subheader("Galaxies")
    st.markdown("")
    st.markdown("")

########## k-means ###########
    st.markdown("")
    st.markdown("")
    st.subheader("Pipeline")
    st.text("Pipeline")
    st.markdown("")

    def drop_useless_columns(df, manual_drop_list=None):
        if manual_drop_list:
            return df.drop(columns=manual_drop_list), manual_drop_list

        drop_list = []
        for col in df.columns:
            if df[col].nunique() <= 1 or df[col].nunique() >= df.shape[0] * 0.95:
                drop_list.append(col)
        return df.drop(columns=drop_list), drop_list

    def datetime_processing(df):
        if 'time' not in df.columns:
            return df

        df['weekday'] = [x.weekday() for x in df.time]
        df['hour'] = [int(x.strftime('%H')) for x in df.time]
        max_time = df.time.max()
        min_time = df.time.min()
        min_norm, max_norm = -1, 1
        df['date'] = (df.time - min_time) * (max_norm - min_norm) / \
            (max_time - min_time) + min_norm
        return df

    def data_preprocessing(df, manual_drop_list=None):
        """Return processed data and column list that need to be dropped."""

        df, drop_list = drop_useless_columns(df, manual_drop_list)

        df = datetime_processing(df)

        df.C2 = df.C1 + df.C3
        df = df.drop(columns=['time', 'X'])
        df['sum_X'] = df.iloc[:, 9:33].sum(axis=1)

        return df.replace({'status': {'Approved': 0, 'Declined': 1}}), drop_list


####### Logistic Regression ###############

@st.cache(suppress_st_warning=True)
def logistic_train_metrics(df):
    """Return metrics and model for Logistic Regression."""

    X = df.drop(columns=['status'])
    Y = df.status

    std_scaler = StandardScaler()
    std_scaled_df = std_scaler.fit_transform(X)
    std_scaled_df = pd.DataFrame(std_scaled_df, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        std_scaled_df, Y, random_state=0)

    # Fit model
    model_reg = LogisticRegression(max_iter=1000)
    model_reg.fit(X_train.fillna(0), y_train)

    # Make predictions for test data
    y_pred = model_reg.predict(X_test.fillna(0))

    # Evaluate predictions
    accuracy_reg = accuracy_score(y_test, y_pred)
    f1_reg = f1_score(y_test, y_pred)
    roc_auc_reg = roc_auc_score(y_test, y_pred)
    recall_reg = recall_score(y_test, y_pred)
    precision_reg = precision_score(y_test, y_pred)

    return accuracy_reg, f1_reg, roc_auc_reg, recall_reg, precision_reg, model_reg

##################################################################################

##### Plots #####


##################################################################################

with footer:
    st.markdown("""---""")
    st.subheader("Heart Attack Predictions - Machine Learning Project")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    # image = Image.open('data/ship.jpg')
    # st.image(image, caption="")
    st.text(' ')

    # Footer
    st.markdown("")
    st.markdown("")
    st.markdown(
        "If you have any questions, checkout our [documentation](https://github.com/fistadev/starwars_data_project) ")
    st.text(' ')
