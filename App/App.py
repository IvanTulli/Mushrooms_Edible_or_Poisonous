import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
px.defaults.color_continuous_scale = "Spectral_r"
px.defaults.color_discrete_sequence = px.colors.diverging.Spectral_r
from plotly.subplots import make_subplots
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

st.title("🍄 Which Mushrooms are Edible or Poisonous? 🍄")

@st.cache_data
def load_data():
    BASE = Path(__file__).resolve().parent
    PATH = BASE/ 'data.csv'
    PATHF = BASE / 'dataf.csv'
    data_oh = pd.read_csv(PATHF)
    data = pd.read_csv(PATH)
    data.rename(columns = {'cap-diameter': 'cap-diameter (cm)', 'stem-height': 'stem-height (cm)', 'stem-width': 'stem-width (mm)'}, inplace = True)
    class_df=data['class'].replace({1:'Poisonous', 0: 'Edible'})
    return data, data_oh, class_df

@st.cache_data
def load_class_plots(class_df):
    class_plots = make_subplots(rows =1, cols =2, specs=[[{"type": "xy"}, {"type": "domain"}]], subplot_titles = ("Histogram of classes",
                                                                                                                  "Pie chart of classes"))
    class_hist = px.histogram(class_df, x='class', color='class', title= 'Histogram of the class of mushrooms')
    for tr in class_hist.data:
        tr.showlegend = False
        class_plots.add_trace(tr, row=1, col = 1)
    class_pie = px.pie(class_df, names ='class')
    for tr in class_pie.data:
        class_plots.add_trace(tr, row=1, col =2)
    
    class_plots.update_xaxes(title_text='Class', row=1, col=1)
    class_plots.update_yaxes(title_text='Counts', row=1,col=1)

    return class_plots

@st.cache_data
def load_scatter_plots(data):
    fig = make_subplots(rows=2, cols=2, shared_yaxes=False)

    f1 = px.scatter(data, x='cap-diameter (cm)', y='stem-height (cm)', color ='class',  color_discrete_map={'Edible': "#1f77b4", 'Poisonous': "#ff7f0e"},
                    category_orders={'class': ['Edible', 'Poisonous']})
    f1.update_traces(marker=dict(size=8,
                                line=dict(width=0.5, color="white")))
    f2 = px.scatter(data, x='stem-height (cm)', y='stem-width (mm)', color ='class',  color_discrete_map={"Edible": "#1f77b4", "Poisonous": "#ff7f0e"},
                    category_orders={'class': ['Edible', 'Poisonous']})
    f2.update_traces(marker=dict(size=8,
                                line=dict(width=0.5, color="white")))
    f3 = px.scatter(data, x='stem-width (mm)', y='cap-diameter (cm)', color ='class',  color_discrete_map={"Edible": "#1f77b4", "Poisonous": "#ff7f0e"},
                    category_orders={'class': ['Edible', 'Poisonous']})
    f3.update_traces(marker=dict(size=8,
                                line=dict(width=0.5, color="white")))

    for tr in f1.data:
        tr.legendgroup = tr.name
        fig.add_trace(tr, row=1, col=1)
    for tr in f2.data:
        tr.legendgroup = tr.name
        tr.showlegend = False 
        fig.add_trace(tr, row=1, col=2)
    for tr in f3.data:
        tr.legendgroup = tr.name
        tr.showlegend = False 
        fig.add_trace(tr, row=2, col=1)

    fig.update_xaxes(title_text='Cap Diameter (cm)', row=1, col=1)
    fig.update_yaxes(title_text='Stem Height (cm)', row=1,col=1)

    fig.update_xaxes(title_text='Stem Height (cm)', row=1, col=2)
    fig.update_yaxes(title_text='Stem Width (mm)', row=1,col=2)

    fig.update_xaxes(title_text='Stem Width (mm)', row=2, col=1)
    fig.update_yaxes(title_text='Cap Diameter (cm)', row=2,col=1)

    fig.update_layout(height=1000, width=1200, showlegend=True)
    fig.update_layout(
    title_text="Scatter plots of numerical features",
    title_x=0.3,                 
    title_font=dict(size=18)
    )
    
    return fig

def load_num_hist_plots(data, rng1, rng2, rng3):
    fig = make_subplots(rows=2, cols=2, shared_yaxes=False)

    f1 = px.histogram(data.loc[data['cap-diameter (cm)'].between(rng1[0],rng1[1]),'cap-diameter (cm)'], 
                      x='cap-diameter (cm)', color_discrete_sequence=["#1f77b4"])
    f2 = px.histogram(data.loc[data['stem-height (cm)'].between(rng2[0], rng2[1]), 'stem-height (cm)'], 
                      x='stem-height (cm)',color_discrete_sequence=["#1f77b4"])
    f3 = px.histogram(data.loc[data['stem-width (mm)'].between(rng3[0],rng3[1]), 'stem-width (mm)'], 
                      x='stem-width (mm)',color_discrete_sequence=["#1f77b4"])

    for tr in f1.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in f2.data:
        fig.add_trace(tr, row=1, col=2)
    for tr in f3.data:
        fig.add_trace(tr, row=2, col=1)

    fig.update_xaxes(title_text='Cap Diameter (cm)', range=[rng1[0], rng1[1]], row=1, col=1)
    fig.update_yaxes(title_text='Count',row=1,col=1)

    fig.update_xaxes(title_text='Stem Height (cm)', range=[rng2[0], rng2[1]], row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1,col=2)

    fig.update_xaxes(title_text='Stem Width (cm)', range=[rng3[0], rng3[1]], row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2,col=1)

    fig.update_layout(height=500, width=1100, showlegend=True)
    return fig

@st.cache_data
def load_categorical_plots(data):
    f1=px.histogram(data, x='cap-shape', color='class',
                    color_discrete_map={"Edible": "#1f77b4", "Poisonous": "#ff7f0e"})
    f1.update_layout(showlegend=True)
    f1.update_xaxes(title_text="Cap Shape")
    f1.update_yaxes(title_text="Count")
    f2=px.histogram(data, x='cap-color', color='class', labels={'cap-color': "cap color"},
                    color_discrete_map={"Edible": "#1f77b4", "Poisonous": "#ff7f0e"})
    f2.update_layout(showlegend=True)
    f2.update_xaxes(title_text="Cap Color")
    f2.update_yaxes(title_text="Count")
    f3=px.histogram(data, x='season', color='class',
                    color_discrete_map={"Edible": "#1f77b4", "Poisonous": "#ff7f0e"})
    f3.update_layout(showlegend=True)
    f3.update_xaxes(title_text="Season")
    f3.update_yaxes(title_text="Count")
    return f1,f2,f3


data, data_oh, class_df = load_data()
class_plots = load_class_plots(class_df)
scatter_plots = load_scatter_plots(data)
cat1, cat2, cat3 = load_categorical_plots(data)

st.sidebar.title('Options')
option = st.sidebar.selectbox("Make a choice:", ("-", "General information of the data", "Insights from the data", "Results from models",
                                         "Determine if a mushroom is poisonous"))

if option == "General information of the data":
    st.subheader("General information")
    st.markdown('This app uses a cleaned version of this [dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset).')
    st.markdown("The aim is to determine whether a mushroom is edible or poisonous based on features of the mushroom like cap diameter," \
    " stem height or width, the habitat, the season of growth, etc. The target is called **class**.  A mushroom is **edible** if its class is 0,"
    " and **poisonous** if its class is 1.")
    st.markdown("The cleaned dataset has **61096** entries and **13** columns.")
    if st.checkbox("Show 5 random samples from the dataset" , False, key = 'dataheadcheck'):
        st.write(data.replace({'Poisonous':1,'Edible':0}).sample(5))

if option == 'Insights from the data':
    option2 = st.sidebar.selectbox("What do you want to see from the data:", ('-', "Distribution of classes", 
                                                                             "Scatter plots of the numerical features", 
                                                                             "Distribution of the numerical features",
                                                                             "Distribution of the categorical features"))
    if option2 =="Distribution of classes":
        st.subheader("Distribution of classes")
        st.markdown("The distribution between classes is fairly balanced, with 11% more poisonous mushrooms" \
        " than edible. This means that our models should beat the baseline of 55.5% accuracy, which would be attained by always" \
        " predicting the majority class.")
        st.plotly_chart(class_plots)

    elif option2 == "Scatter plots of the numerical features":
        st.subheader("Scatter plots of the numerical features")
        st.markdown("The dataset has 3 numerical features: the diameter of the cap of the mushroom, and the height and width of the stem. " \
        "We plot the 3 possible scatter plots among the numerical features, colored by the class of the point. ")
        st.markdown("We see several **distinct clusters**, as well as an overall tendency of poisonous mushrooms " \
        "to have **smaller cap diameter and stem height/width** compared to certain edible mushrooms. This indicates that these features " \
        "will do a good job distinguishing between the classes.")
        st.plotly_chart(scatter_plots)

    elif option2 =="Distribution of the numerical features":
        st.subheader("Distribution of the numerical features")
        st.markdown("The distributions of the 3 numerical features is **skewed** rather than approximately normal, and shows the presence of " \
        "**outliers**. For tree-based models like random forests and XGBoost this will be no issue, but for models like " \
        "support vector machines (SVM) with an rbf kernel we will apply a log-transform to improve performance. ")
        st.sidebar.markdown("Range of the plots:")
        rng1 = st.sidebar.slider("Cap diameter range (cm)", min_value=0, max_value=63, value=(0, 63), format="%.0f")
        rng2 = st.sidebar.slider("Stem height range (cm)", min_value=0, max_value = 34, value= (0,34))
        rng3 = st.sidebar.slider("Stem width range (mm)", min_value=0, max_value = 104, value = (0,104))
        num_hist_plots = load_num_hist_plots(data,rng1,rng2,rng3)
        st.plotly_chart(num_hist_plots)
    
    elif option2 =="Distribution of the categorical features":
        st.subheader("Distribution of some categorical features")
        st.markdown("#### Cap shape distribution")
        st.markdown("The most frequent cap shapes (convex and flat) dont allow us to distinguish between poisonous and edible " \
        "mushrooms, but those that have bell shape or are in the 'others' category tend to be poisonous.")
        st.plotly_chart(cat1)
        st.markdown("#### Cap color distribution")
        st.markdown("As before, we can't use the most frequent value (brown) to distinguish the class of mushroom, but certain" \
        " less frequent colors like red, orange, green, pink and purple tend to be poisonous.")
        st.plotly_chart(cat2)
        st.markdown("#### Season distribution")
        st.markdown("We see that most mushrooms grow during autumn and summer, and during these seasons we also have the biggest " \
        "proportion of poisonous mushrooms. During winter and spring we have more edible than poisonous mushrooms.")
        st.plotly_chart(cat3)
    
elif option =="Results from models":
    option3 = st.sidebar.selectbox("Select a model", ( '-', 'Random Forest','Support Vector Machine',  'XGBoost'))

    if option3 =='Random Forest':
        st.subheader('Random Forest Results')
    
    elif option3 == 'Support Vector Machine':
        st.subheader('Support Vector Machine Results')
    
    elif option3 == 'XGBoost':
        st.subheader('XGBoost results')


