# import librairies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import pickle 

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st 

df= pd.read_csv("train.csv")

st.title("Classification des personnes en tranches de crédit grâce à l'IA")
st.subheader("Auteur: Ouiza MEBARKI")

st.sidebar.title("Sommaire")
pages= ["Contexte du projet", " Exploration de données", "Analyse de données", " Modélisation","Interprétation des models" ,"Application", "Conclusion"]

page= st.sidebar.radio("Aller vers la page: ", pages)

if page== pages[0]: 
    st.write("### Contexte du projet")

    st.write("* Ce projet s'inscrit dans le contexte de crédit scoring, ce dernier consiste à analyser et évaluer le niveau de risque du demandeur de crédit en lui attribuant une note, autrement dit un score. ")
    st.image("credit_score.jpg")
    st.write("* Le score de credit est important pour les emprunteurs, car ceux qui ont des scores plus élevés bénificient de conditions de crédit plus fvorables, qui se traduit par des paiments  moins élevées et des faibles taux d'intérêts.")
    st.write("### Objectif du projt:")
    #st.write ("**Probléme**: La direction de la société fianciére mondiale souhaite construire un systéme intelligent pour séparer les personnes en tranches de cédit.") 
    st.write ("* Nous avons à notre disposition un fichier credit score  qui contient des informations relatives au crédit d'une personne, et à à partir duquel nous devons créer un modèle d'apprentissage automatique capable de classer la cote de crédit. ")
    st.write("### Aproches techniques :")
    st.write (" * Premiérement on explore notre jeux de donées. Puis on visualise afin d'extraire des informations et comprendre mieux notre dataset. Finalement on implémente des modéles de Machine Learning pour prédire la cote crédit .")
   
elif page== pages[1]: 

    st.write("### Exploration de données")
    st.dataframe(df.head())

    if st.checkbox("Dimension du dataframe: "):
        st.write(df.shape)
    if st.checkbox("L'unicité de l'ID:"):   
        st.write(df.value_counts('ID'))

    if st.checkbox("Afficher les valeus manquantes: "):
        st.write(df.isna().sum())

    if st.checkbox("Afficher les doublons: "):
        st.write(df.duplicated().sum())

    if st.checkbox("Discription des variables numériques: "):
        st.write(df.describe())

    if st.checkbox("Discription des variables catégoriques: "):
        st.write(df.describe(include='object'))

   

elif page== pages[2]:
    st.write("### Analyse de données")
    df_sample= pd.read_csv("df_sample.csv")

    st.subheader("*Objectif*: On regarde l'impact de chaque variable sur la tranche du crédit ")

    st.subheader("Les proportions de la variable cible dans notre jeux de données ")

    cérdit_score_count= df_sample['Credit_Score'].value_counts()
    fig_sb, ax_sb= plt.subplots()
    ax_sb= plt.pie(cérdit_score_count, labels=cérdit_score_count.index, autopct='%1.2f%%')
    st.pyplot(fig_sb)
    

    st.subheader("Distribution de crédit score par rapport à l'occupation ")
    fig2,ax= plt.subplots()
    sns.histplot(data=df_sample, x="Occupation", hue="Credit_Score", multiple="dodge",  shrink=.8, ax = ax)
    plt.xticks(rotation=90) 
    st.pyplot(fig2)
    # px.bar(df, x="Occupation", color="Credit_Score")
    
    st.subheader("Distribution de crédit score par rapport à Payment_of_Min_Amount")
    fig_sb1, ax_sb1= plt.subplots()
    ax_sb1= sns.countplot(df_sample, x="Payment_of_Min_Amount", hue="Credit_Score")
    st.pyplot(fig_sb1)


    st.subheader("Distribution de crédit score par rapport aux Crédit_mix")
    fig_sb2, ax_sb2= plt.subplots()
    ax_sb2=sns.countplot(df_sample, x="Credit_Mix", hue="Credit_Score")
    st.pyplot(fig_sb2)

    
    st.subheader("La corrélation entre les variables numeriques ")
    numericals = df_sample.select_dtypes(include='number').columns
    fig3, ax = plt.subplots(figsize=(20,15))
    sns.heatmap(df_sample[numericals].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, ax = ax )
    st.write(fig3)

    st.subheader("La distribution des variables numeriques ")
    fig4, ax = plt.subplots( figsize=(8,20))
    df_sample.hist(bins=21,layout=(-1, 3), edgecolor="black", ax=ax)
    st.write(fig4)

    st.subheader("La répartition da salaires annuels dans notre jeux de données ")
    fig5,ax= plt.subplots()
    sns.boxplot(df_sample, x='Credit_Score', y='Annual_Income', ax=ax)
    st.pyplot(fig5)

    st.subheader("Outstanding_Debt vs credit score")
    fig6,ax= plt.subplots()
    sns.barplot(df_sample, x= "Outstanding_Debt", y="Credit_Score",ax=ax)
    st.pyplot(fig6)
    
    st.subheader(" Interest_Rate vs credit score")
    fig7,ax= plt.subplots()
    sns.barplot(df_sample, x= "Interest_Rate", y="Credit_Score",ax=ax)
    st.pyplot(fig7)
    
    st.subheader("Total_EMI_per_month vs credit score")
    fig8,ax= plt.subplots()
    sns.scatterplot(df_sample, x= "Total_EMI_per_month", y="Credit_Score",ax=ax)
    st.pyplot(fig8)
     
    st.subheader("Credit_History_Age vs credit score")
    fig9,ax= plt.subplots()
    sns.barplot(df_sample, x= "Credit_History_Age", y="Credit_Score")
    st.pyplot(fig9)



    
elif page== pages[3]:
    st.write("### Modeles Machine Learning ")
    df_prep= pd.read_csv("df_preprocessed.csv")  

    X= df_prep.drop(['Credit_Score'],axis = 1)
    y=df_prep['Credit_Score']
    
    # Split 
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

    # scaler les données 
    scaler = StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.fit_transform(X_test)

    lr= pickle.load(open("model_lr.pkl", 'rb'))
    knn= pickle.load(open("model_knn.pkl", 'rb'))
    svm= pickle.load(open("model_svm.pkl", 'rb'))
    rf= pickle.load(open("model_rf.pkl", 'rb'))
    gb= pickle.load(open("model_gb.pkl", 'rb'))
    gbc= pickle.load(open("model_gbc.pkl", 'rb'))

    


    y_pred_lr= lr.predict(X_test)
    y_pred_knn= knn.predict(X_test)
    y_pred= svm.predict(X_test)
    y_pred= rf.predict(X_test)
    y_pred= gb.predict(X_test) 
    y_pred= gbc.predict(X_test)
    
    st.write(" **Note**: La métrique utilisée pour mesurer la précision du modèle est le F1_score")
    model= st.selectbox(label="Modèle ", options= ['Logistic Regression', 'KNeighborsClassifier', 'SVC','RandomForestClassifier','GradientBoosting'])
    
    def train_model(model): 
        if model=='Logistic Regression': 
            y_pred= lr.predict(X_test)
           
        elif model== 'KNeighborsClassifier': 
            y_pred= knn.predict(X_test)

        elif model == 'SVC': 
            y_pred= svm.predict(X_test) 

        elif model =='RandomForestClassifier':
             y_pred= rf.predict(X_test) 

        elif model =='GradientBoosting':
             y_pred= gb.predict(X_test)

        elif model =='XGBClassifier':
             y_pred= gbc.predict(X_test)
        
        
        f1= f1_score(y_test, y_pred, average='micro').round(2)
        
        st.subheader("classification_report")
        target_names = ["class 0", "class 1", "class 2"]   
        st.dataframe(classification_report(y_test, y_pred, target_names=target_names, output_dict=True))

        st.subheader("Matrice de confusion")
        conf_mat= confusion_matrix(y_test,y_pred) 
        figm,ax= plt.subplots()
        sns.heatmap(conf_mat, annot=True, cmap='Blues', ax=ax)
        st.write(figm)
        st.subheader(" La précion du modèle est de : ")

        return f1

        
    st.write(train_model(model))

    
elif page== pages[4]:
    st.write("### Résultats: ")
    st.write( "* Le meilleur score obtenue avec un simple RandomForest: F1_score=0,75")

    st.write("### Approches techniques: ")
    st.write(" * Modèles ML avec PCA")
    st.write(" * Modèles ML avec Smote")
    st.write(" * Random Forest et XGB avec les hyperparamétres ")
    st.write("### Note: Amélioration possibles:")
    st.write("* Hyperparamétres du RandomForest ")
 

elif page== pages[5]:
    st.write("### Application: Prédiction de score de crédit")

    st.write("**Etape1**: Choisir les features importantes ")
    st.image("Features.jpg")
    st.write("**Etape2**: Simple RandomForest (meilleur score) ")
    st.write("**Etape3**: Création de l'application")
    st.write("- ***Flask***")
    st.write("- ***Streamlit***" )
    st.write("**Etape4**: Déploiment de l'application sur le Cloud Streamlit")
    st.write("Lien: https://projectartefactcreditscore-ag5wphxlkdqhigvhapqfmz.streamlit.app/")
                    

elif page== pages[6]:

    st.write(" * Mettre en application et de nous familiariser avec certaines notions d'intelligence artficielle étudiées lors de la formation.")
    st.write(" * Grace à l'utilisation de techniques avancées de l'IA, on a pu construire un systéme intelligent qui sépare les personnes en tranches de crédit.")

   



    
   
    



    





