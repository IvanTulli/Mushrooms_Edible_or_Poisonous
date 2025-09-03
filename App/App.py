import streamlit as st
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

st.sidebar.title("🍄 Is this Mushroom Edible or Poisonous? 🍄")
@st.cache_data
def load_data():
    BASE = Path(__file__).resolve().parent
    PATH = BASE/ 'data.csv'
    PATHF = BASE / 'dataf.csv'
    PATHFI = BASE / 'feature_importances_rf_f1.csv'
    data_oh = pd.read_csv(PATHF)
    data = pd.read_csv(PATH)
    fi_df = pd.read_csv(PATHFI)
    class_df=data['class'].replace({1:'Poisonous', 0: 'Edible'})
    return data, data_oh, class_df, fi_df

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

    f1 = px.scatter(data, x='cap-diameter', y='stem-height', color ='class',  color_discrete_map={'Edible': "#1f77b4", 'Poisonous': "#ff7f0e"},
                    category_orders={'class': ['Edible', 'Poisonous']})
    f1.update_traces(marker=dict(size=8,
                                line=dict(width=0.5, color="white")))
    f2 = px.scatter(data, x='stem-height', y='stem-width', color ='class',  color_discrete_map={"Edible": "#1f77b4", "Poisonous": "#ff7f0e"},
                    category_orders={'class': ['Edible', 'Poisonous']})
    f2.update_traces(marker=dict(size=8,
                                line=dict(width=0.5, color="white")))
    f3 = px.scatter(data, x='stem-width', y='cap-diameter', color ='class',  color_discrete_map={"Edible": "#1f77b4", "Poisonous": "#ff7f0e"},
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

    f1 = px.histogram(data.loc[data['cap-diameter'].between(rng1[0],rng1[1]),'cap-diameter'], 
                      x='cap-diameter', color_discrete_sequence=["#1f77b4"])
    f2 = px.histogram(data.loc[data['stem-height'].between(rng2[0], rng2[1]), 'stem-height'], 
                      x='stem-height',color_discrete_sequence=["#1f77b4"])
    f3 = px.histogram(data.loc[data['stem-width'].between(rng3[0],rng3[1]), 'stem-width'], 
                      x='stem-width',color_discrete_sequence=["#1f77b4"])

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

    fig.update_xaxes(title_text='Stem Width (mm)', range=[rng3[0], rng3[1]], row=2, col=1)
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

@st.cache_data
def load_model(data):
    BASE = Path(__file__).resolve().parent
    RF_PATH = BASE / "best_rf_model.pkl"
    with RF_PATH.open("rb") as f:
        rf_model = pickle.load(f)
    X=data.iloc[:,1:]
    y=data['class']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y,random_state=123)
    y_pred = rf_model.predict(X_test)
    return rf_model, y_test, y_pred

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=None, title=None):
    """
    normalize: None | 'true' | 'pred' | 'all'
      None   -> raw counts
      'true' -> rows sum to 1 (per true class)
      'pred' -> columns sum to 1 (per predicted class)
      'all'  -> all entries sum to 1
    """
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig = px.imshow(
        cm_df,
        text_auto=(".2f" if normalize else "d"),
        color_continuous_scale="Blues",
        aspect="equal"
    )
    fig.update_xaxes(title_text="Predicted label", side="top")
    fig.update_yaxes(title_text="True label", autorange="reversed")  # conventional layout
    fig.update_layout(
        title=title or ("Confusion matrix (normalized)" if normalize else "Confusion matrix"),
        coloraxis_colorbar=dict(title="Proportion" if normalize else "Count"),
        margin=dict(l=60, r=20, t=60, b=40)

    )
    class_names = ["edible", "poisonous"]
    labels = np.unique(np.r_[y_true, y_pred])              

    fig.update_xaxes(
        showticklabels=True, tickmode="array",
        tickvals=labels,                                    
        ticktext=class_names, side="top"
    )
    fig.update_yaxes(
        showticklabels=True, tickmode="array",
        tickvals=labels,                                    
        ticktext=class_names, autorange="reversed"
    )
    fig.update_yaxes(tickangle=-90)
    return fig

def plot_fi(fi_data):
    fig = px.bar(fi_data[:5:], x='Importance', y='Feature', color='Feature', orientation='h')
    fig.update_layout(showlegend=False)
    return fig


data, data_oh, class_df, fi_df = load_data()
class_plots = load_class_plots(class_df)
scatter_plots = load_scatter_plots(data)
cat1, cat2, cat3 = load_categorical_plots(data)
rf_model, y_test, y_pred = load_model(data_oh)
BASE = Path(__file__).resolve().parent

st.sidebar.title('Options')
option = st.sidebar.selectbox("Make a choice:", ("-", "Insights from the data", "Metrics of the trained model",
                                         "Is your mushroom poisonous?"))

if option =='-':
    st.title("🍄 Is this Mushroom Edible or Poisonous? 🍄")
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        st.image(BASE / "Amanita_muscaria.jpg", caption ='Amanita muscaria, a very famous poisonous mushroom. [Source](https://en.wikipedia.org/wiki/Amanita_muscaria)')
    st.markdown('This app contains a model for determining whether a mushroom is edible or poisonous. To determine this it uses features like ' \
    'stem height, cap diameter, cap color and the season of growth.')
    st.markdown('For the training of the model we used a cleaned ' \
    'version of this [dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset). The target is called **class**, with an **edible** mushroom having class 0,'
    ' and a **poisonous** mushroom having class 1.')
  
    st.markdown("You can take a peek at the dataset by clicking the box below.")
    if st.checkbox("Show 5 random samples from the dataset" , False, key = 'dataheadcheck'):
        st.write(data.replace({'Poisonous':1,'Edible':0}).sample(5))
    
    st.markdown("**To make a prediction with the model**, choose **'Is your mushroom poisonous?'** in the sidebar option. You can also" \
    " check out some insights from the data and some metrics of the model.")

elif option == 'Insights from the data':
    option2 = st.sidebar.selectbox("What do you want to see from the data:", ('-', "Distribution of classes", 
                                                                             "Scatter plots of the numerical features", 
                                                                             "Distribution of the numerical features",
                                                                             "Distribution of the categorical features"))
    if option2 =='-':
        st.subheader("Insights from the data")
        st.markdown("Please select an option from the sidebar.")
        col1,col2,col3=st.columns([1,8,1])
        with col2:
            st.image(BASE / "Fairy_ring.jpg", caption = 'A fairy ring. [Source](https://www.woodlandtrust.org.uk/blog/2019/08/what-is-a-fairy-ring/)')

    elif option2 =="Distribution of classes":
        st.subheader("Distribution of classes")
        st.markdown("The distribution between classes is fairly balanced, with 11% more poisonous mushrooms" \
        " than edible. This means that our model should beat the baseline of 55.5% accuracy, which would be attained by always" \
        " predicting the majority class.")
        st.plotly_chart(class_plots)

    elif option2 == "Scatter plots of the numerical features":
        st.subheader("Scatter plots of the numerical features")
        st.markdown("The dataset has 3 numerical features: the diameter of the cap, and the height and width of the stem. " \
        "We plot the 3 possible scatter plots among the numerical features, colored by the class of the point. ")
        st.markdown("We see several **distinct clusters**, as well as an overall tendency of poisonous mushrooms " \
        "to have **smaller cap diameter and stem height/width** compared to certain edible mushrooms. This indicates that these features " \
        "might do a good job distinguishing between the classes.")
        st.plotly_chart(scatter_plots)

    elif option2 =="Distribution of the numerical features":
        st.subheader("Distribution of the numerical features")
        st.markdown("The distributions of the 3 numerical features is **skewed** rather than approximately normal, and shows the presence of " \
        "**outliers**. This will cause no problems for tree-based models like random forests, but for models like " \
        "support vector machines (SVM) with an rbf kernel one should try to performing a log transform, as well as dealing with outliers. ")
        st.sidebar.markdown("Range of the plots:")
        rng1 = st.sidebar.slider("Cap diameter range (cm)", min_value=0, max_value=63, value=(0, 63), format="%.0f")
        rng2 = st.sidebar.slider("Stem height range (cm)", min_value=0, max_value = 34, value= (0,34))
        rng3 = st.sidebar.slider("Stem width range (mm)", min_value=0, max_value = 104, value = (0,104))
        num_hist_plots = load_num_hist_plots(data,rng1,rng2,rng3)
        st.plotly_chart(num_hist_plots)
    
    elif option2 =="Distribution of the categorical features":
        st.subheader("Distribution of some categorical features")
        st.markdown("We include the distributions of some categorical features to give an idea of what information one can extract from them.")
        st.markdown("#### Cap shape distribution")
        st.markdown("The most frequent cap shapes (convex and flat) dont allow us by themselves to distinguish between poisonous and edible " \
        "mushrooms, but those that have bell shape or are in the 'others' category tend to be poisonous.")
        st.plotly_chart(cat1)
        st.markdown("#### Cap color distribution")
        st.markdown("As before, we can't use the most frequent value (brown) by itself to distinguish the class of mushroom, but certain" \
        " less frequent colors like red, orange, green, pink and purple tend to be poisonous.")
        st.plotly_chart(cat2)
        st.markdown("#### Season distribution")
        st.markdown("We see that most mushrooms grow during autumn and summer, and during these seasons we also have the biggest " \
        "proportion of poisonous mushrooms. During winter and spring we have more edible than poisonous mushrooms.")
        st.plotly_chart(cat3)
    
elif option =="Metrics of the trained model":
    model_options = st.sidebar.selectbox("Which metrics do you want to see?", ('-', 'F1-score and confusion matrix', 'Feature importances'))
    if model_options =='-':
        st.subheader("Metrics from the random forest model 🌲🌳🌲")
        st.markdown("A **random forest model** was trained with the hyperparamers chosen to maximize the **f1-score**. Since the classes are fairly" \
        " balanced, a high f1-score also ensures high accuracy. Furthermore, a high f1-score also ensures high recall, and hence a low number of" \
        " poisonous mushrooms misclassified as edible.")

        st.markdown("**To continue**, please select an option from the sidebar.")
        col1,col2,col3=st.columns([1,4,1])
        with col2:
            st.image(BASE / "Random_forest.avif", caption = "A 'random forest'. [Source](https://canopyplanet.org/forests/how-we-protect-forests)")

    elif model_options =='F1-score and confusion matrix':
        st.subheader("F1-score and confusion matrix")
        st.markdown("Overal the model performed very well. Below we show the f1-score and confusion matrix, as well as metrics like accuracy," \
                    " recall, and precision of the predictions on the test set.")
        f1=f1_score(y_test,y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test,y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**F1-score**: {f1:.1%}")
        st.write(f"**Recall**: {recall: .1%}")
        st.write(f"**Precision**: {precision:.1%}")
        st.write(f"**Accuracy**: {accuracy: .1%}")
        st.plotly_chart(plot_confusion_matrix(y_test,y_pred, title=None))
    elif model_options =='Feature importances':
        st.subheader("Top 5 important features for the training of the model")
        st.markdown("The top 5 features that were the most important for the model while fitting are shown below. These where computed " \
        "using **permutation importance** based on the f1-score: how much randomly permuting a feature affects the f1-score of the model.")
        fi_df['Feature'].replace({'stem-width': 'Stem width', 'stem-color_white': 'White stem color', 'cap-diameter': 'Cap diameter',
                              'gill-color_white': 'White gill color', 'stem-height': 'Stem height'}, inplace= True)
        st.plotly_chart(plot_fi(fi_df))
        st.markdown("We see an appearance of the 3 numerical features (stem height/width, cap diameter), as well as two features obtained from " \
        "the categorical ones via one hot encoding (white stem color, and white gill color). However, one should keep in mind that this" \
        " is computed on the one-hot-encoded categorical features, which dilutes the feature importance of each original categorical feature.")

elif option == "Is your mushroom poisonous?":
    st.subheader("Determine if a mushroom is poisonous 💀 or edible 🍄")
    st.markdown(":red[**Warning! Do not use this app for deciding whether to eat a wild mushroom. This model was trained on a synthetic dataset "
    "which may not represent real world data with sufficient accuracy.**]")
    st.markdown("Please input the following data:")
    cap_diameter = st.number_input("What is the diameter of the cap (in cm)?", min_value=0, max_value= 63)
    cap_shape = st.selectbox("What is the cap shape?", tuple(data['cap-shape'].unique().tolist()))
    cap_color = st.selectbox("What is the cap color?", tuple(data['cap-color'].unique().tolist()))
    does_bruise_or_bleed= st.selectbox("Does the mushroom bruise or bleed?", tuple(data['does-bruise-or-bleed'].unique().tolist()))
    gill_color = st.selectbox("What is the gill color?", tuple(data['gill-color'].unique().tolist()))
    stem_height = st.number_input("What is the stem height (in cm)?", min_value=0, max_value= 34)
    stem_width = st.number_input("What is the stem width (in mm)?", min_value=0, max_value= 104)
    stem_color = st.selectbox("What is the stem color?", tuple(data['stem-color'].unique().tolist()))
    has_ring = st.selectbox("Does it have a ring?", ('yes', 'no'))
    if has_ring=='yes':
        has_ring='ring'
        ring_type_options= data['ring-type'].unique().tolist()
        ring_type_options.remove('none')
        ring_type = st.selectbox("What is the ring type?", tuple(ring_type_options))
    elif has_ring =='no':
        has_ring='none'
        ring_type ='none'
    habitat = st.selectbox("What is the habitat of the mushroom?", tuple(data['habitat'].unique().tolist()))
    season = st.selectbox("What is the season?", tuple(data['season'].unique().tolist()))

    input_df = pd.DataFrame({'cap-diameter': [cap_diameter], 'cap-shape': [cap_shape], 'cap-color':[cap_color],
                             'does-bruise-or-bleed': [does_bruise_or_bleed], 'gill-color': [gill_color],
                             'stem-height': [stem_height], 'stem-width': [stem_width], 'stem-color': [stem_color],
                             'has-ring': [has_ring], 'ring-type': [ring_type], 'habitat': [habitat],
                             'season': [season]})
    
    if st.button("Predict!", type='primary'):
        df_concat = pd.concat([data.iloc[:,1:], input_df], ignore_index=True)
        input_df_oh=pd.get_dummies(df_concat).iloc[-1:,:]
        y_pred= rf_model.predict(input_df_oh)
        if y_pred==1:
            st.subheader('The mushroom is predicted to be **poisonous**! 💀😵💀')
        else:
            st.subheader('The mushroom is predicted to be **edible** 🍄😁🍄!')




