# importing necessary libraries
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the model
classifier = pickle.load(open('XGBoostModel.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title='Customer Segmentation Web App', layout='centered')
st.title('Customer Segmentation Web App')

# Customer segmentation function
def segment_customers(input_data):
    # List of all features expected by the model
    all_features = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                    'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 
                    'AcceptedCmp2', 'Complain', 'Response', 'Age', 'Total_yearCust', 'Year_Joined', 'Month_Joined', 
                    'Total_Expenses', 'Total_Acc_Cmp', 'TotalNumPurchases', 'children', 'Education1', 'Partner', 'Age_Group']
    
    # Create a DataFrame with all features set to some default value (e.g., 0 or 'unknown')
    df = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)
    
    # Update the DataFrame with the actual input data
    df.update(pd.DataFrame(input_data, columns=['Income', 'Kidhome', 'Teenhome', 'Age', 'Partner', 'Education1', 'Total_Expenses', 'Total_Acc_Cmp', 'TotalNumPurchases']))
    
    # Convert Income and other numeric features to float
    df['Income'] = df['Income'].astype(float)
    df['Total_Expenses'] = df['Total_Expenses'].astype(float)
    df['Total_Acc_Cmp'] = df['Total_Acc_Cmp'].astype(float)
    df['TotalNumPurchases'] = df['TotalNumPurchases'].astype(float)
    
    # Convert categorical columns to category dtype
    df['Kidhome'] = df['Kidhome'].astype('category')
    df['Teenhome'] = df['Teenhome'].astype('category')
    df['Partner'] = df['Partner'].astype('category')
    df['Education1'] = df['Education1'].astype('category')
    
    # Predict using the classifier
    prediction = classifier.predict(df)
    
    pred_1 = 'unknown'
    if prediction == 0:
        pred_1 = 'Cluster 0: Highest no.of customers, Most of the households have atleast one kid, Majority of the households doesnt have any teens, With or with out partner, Most of them have atleast one or two child, Majority of them did thet graduation and postgraduation, very few did their basic education, Low income, Low Expenditure, Very less purchases, Very less accepted campaign, Expenses on different categories are done  more by middle aged people, Higher expenses are made by households having no children, Post graduate people does higher expenses.'
    elif prediction == 1:
        pred_1 = 'Cluster 1: Second highest no. of customers, Most of the households doesn’t have any kids, Majority of the households  have at least one teen With or without partner, Majority of them at least one child, Majority of them did they graduation and postgraduation, Income below 60000, Low Expenditure, Less spending on wines, fruits etc, Less than 20 purchases, Very less accepted campaign, Expenses on different categories are done  more by middle aged people, Higher expenses are made by households having no children, Post graduate people does higher expenses.'
    elif prediction == 2:
        pred_1 = 'Cluster 2: Most of the households doesn’t have any kids, Majority of the households doesn’t have any teens With or without partner, Well Educated, Mostly Middle-aged people, followed by equal no. senior citizens and adults, Cluster with the highest income, High Expenditure, High spending on wines, High spending on meat products when compared to other clusters, More than 20 purchases, Accepted a lot of campaigns.'
    elif prediction == 3:
        pred_1 = 'Cluster 3: Most of the households doesn’t have any kids, Majority of the households have at least one teen, With or without partner, Well Educated, Mostly Middle-aged people and senior citizens, High income, High Expenditure, Highest spending on wines, High spending on meat prods, More than 25 purchases, Accepted a lot of campaigns.'

    return pred_1

def main():
    Income = st.text_input("Type In The Household Income")
    Kidhome = st.radio("Select Number Of Kids In Household", ('0', '1', '2'))
    Teenhome = st.radio("Select Number Of Teens In Household", ('0', '1', '2'))
    Age = st.slider("Select Age", 24, 100)
    Partner = st.radio("Living With Partner?", ('0', '1'))
    Education1 = st.radio("Select Education", ("Basic", "Graduate", "Postgraduate"))
    Total_Expenses = st.text_input("Total Expenses")
    Total_Acc_Cmp = st.text_input("Total Accepted Campaigns")
    TotalNumPurchases = st.text_input("Total Number of Purchases")

    result = ""

    # When 'Segment Customer' is clicked, make the prediction and store it
    if st.button("Segment Customer"):
        result = segment_customers([[Income, Kidhome, Teenhome, Age, Partner, Education1, Total_Expenses, Total_Acc_Cmp, TotalNumPurchases]])

    st.success(result)

if __name__ == '__main__':
    main()
