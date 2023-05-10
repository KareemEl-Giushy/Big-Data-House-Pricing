import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import streamlit as st
# from bokeh.plotting import figure as bokehfig
warnings.filterwarnings('ignore')

df = pd.read_csv('../dataset/train.csv')


st.title("Real State Market - House Pricing")
st.text("This is an interactive report about the findings in the housing dataset")

st.divider()

st.markdown("""
# 1. First things first:
### analysing 'SalePrice'
'SalePrice' is the reason of our quest. It's like when we're going to a party. We always have a reason to be there. Usually, women are that reason. (disclaimer: adapt it to men, dancing or alcohol, according to your preferences)

Using the women analogy, let's build a little story, the story of 'How we met 'SalePrice''.

Everything started in our Kaggle party, when we were looking for a dance partner. After a while searching in the dance floor, we saw a girl, near the bar, using dance shoes. That's a sign that she's there to dance. We spend much time doing predictive modelling and participating in analytics competitions, so talking with girls is not one of our super powers. Even so, we gave it a try:

'Hi, I'm Kaggly! And you? 'SalePrice'? What a beautiful name! You know 'SalePrice', could you give me some data about you? I just developed a model to calculate the probability of a successful relationship between two people. I'd like to apply it to us!'

""")

st.header("The Distribution Of The Home Prices")
fig = plt.figure(figsize=(10, 4))
sns.distplot(df['SalePrice'])
st.pyplot(fig)
st.markdown("""
*'Elegant! I see that you:*

* *<b>Deviate from the normal distribution.</b>*
* *<b>Have appreciable positive skewness.</b>*
* *<b>Show peakedness.</b>*

*This is getting interesting! 'SalePrice', could you give me your body measures?'*
""", unsafe_allow_html=True)

st.divider()

st.header("Count Categorical Values")
st.text("Descussing The Different Classification Of Data Available")
categoriesLocation = ['MSZoning', 'Street', 'LotShape', 'LandContour',]
categoriesUtilties = ['Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',]
categoriesStatus = ['Condition1', 'Condition2', 'BldgType', 'HouseStyle',]
categoriesStyle = ['RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',]
categoriesFoundation = ['ExterQual', 'ExterCond', 'ExterCond', 'Foundation',]
categoriesConditioning = ['Heating', 'HeatingQC', 'CentralAir', 'Electrical',]
otherCategories = ['KitchenQual', 'Functional', 'SaleType', 'SaleCondition']

st.subheader("Locations Available")        
for i, c in enumerate(categoriesLocation):
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x=c, data=df)
    st.pyplot(fig)

st.text("""
From the data it can be seen that the location available are mostly:
1. Have a Pavement
2. Is a Level Appartment
""")


st.divider()

st.header("Relationship with categorical features")

var = 'YearBuilt'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
fig= plt.figure(figsize=(16, 8))
sns.boxplot(x=var, y="SalePrice", data=data)
st.pyplot(fig)

st.markdown("""
Although it's not a strong tendency, I'd say that 'SalePrice' is more prone to spend more money in new stuff than in old relics.

**Note**: we don't know if 'SalePrice' is in constant prices. Constant prices try to remove the effect of inflation. If 'SalePrice' is not in constant prices, it should be, so than prices are comparable over the years.

**In summary**
Stories aside, we can conclude that:

- 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships are positive, which means that as one variable increases, the other also increases. In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.
- 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the overall quality.

We just analysed four variables, but there are many other that we should analyse. The trick here seems to be the choice of the right features (feature selection) and not the definition of complex relationships between them (feature engineering).

That said, let's separate the wheat from the chaff.

""")