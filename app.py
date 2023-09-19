import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go
import plotly.express as px 
import datetime
from datetime import date,timedelta 
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from  statsmodels.tsa.stattools import adfuller

app_name='Stock Market Forecasting App'
st.title(app_name)
st.subheader('This App Forecast the Stock Market Price of Selected Company')
st.image("https://tse4.mm.bing.net/th?id=OIP.BS0ssq41VvHcshPzbOFFcQHaEK&pid=Api&P=0&h=180")

start_date=st.sidebar.date_input('Start date',date(2020,1,1))
end_date=st.sidebar.date_input('End date',date(2020,12,31))

ticker_list=["AAPL","MSFT"]
ticker=st.sidebar.selectbox('Select the company ',ticker_list)

data=yf.download(ticker,start=start_date,end=end_date)

data.insert(0,"date",data.index,True)
data.reset_index(drop=True,inplace=True)

st.write('Data from ',start_date,'to',end_date)

st.write(data)

st.header('Dat Visualixation')
st.subheader('Plot of the data')
fig=px.line(data,x='date',y=data.columns,title='Closing price of the stock',width=1000,height=600)
st.plotly_chart(fig)
column=st.selectbox('Select the col to be used',data.columns[1:])

data=data[['date',column]]
st.write("Selected Data")
st.write(data)

st.header('Is data Stationary')
st.write('if p value is less than 0.05 ,then data is stationary')
st.write(adfuller(data[column])[1]<0.05)

st.header('Decomposition of Data')
decomposition =seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

st.plotly_chart(px.line(x=data['date'],y=decomposition.trend,title='Trend',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data['date'],y=decomposition.seasonal,title='Seasonality',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data['date'],y=decomposition.resid,title='Residuals',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='red',line_dash='dot'))

p=st.slider('Select the value of p',0,5,2)
q=st.slider('Select the value of d',0,5,1)
d=st.slider('Select the value of q',0,5,2)
seasonal_order =st.number_input('Select the value of seasonal p',0,24,12)

model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model=model.fit()

st.header('Model Summary')
st.write(model.summary())
st.write("---")

#predict fut val

forecast_period=st.number_input("Select number of days to forecast",1,365,10)

predictions=model.get_prediction(start=len(data),end=len(data)+forecast_period)
predictions=predictions.predicted_mean
st.write(predictions)

predictions.index=pd.date_range(start=end_date,periods=len(predictions),freq='D')
predictions=pd.DataFrame(predictions)
predictions.insert(0,"date",predictions.index,True)
predictions.reset_index(drop=True,inplace=True)
st.write("Predictions",predictions)
st.write("Actual",data)
st.write("---")

fig=go.Figure()
fig.add_trace(go.Scatter(x=data["date"],y=data[column],mode='lines',name='Actual',line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["date"],y=predictions["predicted_mean"],mode='lines',name='Predicted0',line=dict(color='red')))


fig.update_layout(title='Actual vs Predicted',xaxis_title='date',yaxis_title='Price',width=800,height=400)
st.plotly_chart(fig)
show_plot=False
if st.button("Show separate Plots"):
 if not show_plot:
    st.write(px.line(x=data["date"],y=data[column],title='Actual',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
    st.write(px.line(x=predictions["date"],y=predictions["predicted_mean"],title='Predicted',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='red'))
    show_plot=True
else:
   show_plot=False

hide_plots=False 
if st.button("Hide Plots"):
   if not hide_plots:
      hide_plots=True
   else:
      hide_plots=False
   st.write("---")





























