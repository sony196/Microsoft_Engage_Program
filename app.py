import json
from matplotlib import use                # To load some json files
import requests            #To make requests for animations
import plotly              #For visualizations
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_lottie import st_lottie  #To take animations 
from streamlit_option_menu import option_menu #for navigation bar
import pickle     # To load models and scaled data
import matplotlib.pyplot as plt
import typing_extensions as final

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV

#for web app configuration
st.set_page_config(page_title="Car_DataAnalysis",
    page_icon=":red_car:",
    layout="wide"
)



with open("encoded_items/encodedModel1.json", "r") as f:
    encoded_model= json.load(f)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

df = pd.read_csv('car_data.csv')

with st.sidebar:
    selected = option_menu(
        menu_title = "Main menu",
        options=[
                     "Dashboard","Milege_prediction","Fuel_Capacity_prediction","Horse Power Prediction",
                     "fuel effeciency prediction","curb weight prediction","Performance Factor","Engine_size_prediction","Carprice_predictor",
                     "Resale_price_prediction"],
        icons=["house","book","book","book","book","book","book","book","book","book"],
             )
    
#creating functions to take input from the user for every prediction

def car_Model():
    Model = st.selectbox("Enter the model you prefer",df.Model,index=0)
    return Model

def Engine_Size():
    Engine_size=st.number_input('Enter the engine size of your car',1.5,step=0.5,key="engine_size")
    return Engine_size

def price_of_car():
    price = st.number_input("Enter the price of the car(in lakhs)",5,step=1,key="price_in_thousands")
    return price
def horse_power():
    Horse_power = st.number_input("Enter the horsepower of car",70.0,300.0,step=2.5,key="Horse_power")
    return Horse_power

def wheel_base():
    wheelBase = st.number_input("Enter the wheelbase of car",90.0,300.0,step=0.5,key="Wheel_Base")
    return wheelBase

def WIDTH():
    width = st.number_input("Enter the width of car",50.0,300.0,step=1.5,key="Width")
    return width

def length():
    Length = st.number_input("Enter the length of the car",100,500,step=1,key="length")
    return Length

def curb_weight():
    curbWeight = st.number_input("Enter the curb weight of car",1.000,10.000,step=0.500,key="curbWeight")
    return curbWeight

def performance_factor():
    Performance_Factor = st.number_input("Enter the performance factor of car",50.0,150.5,step=0.5,key="PerformanceFactor")
    return Performance_Factor

def Fuel_Capacity():
    fuel_capacity = st.number_input("Enter the fuel capacity of car",10.0,30.0,step=0.5,key="fuel_capacity")
    return fuel_capacity

def Fuel_Efficiency():
    fuel_efficiency=st.number_input("Enter the fuel effeciency of car",15,50,step=1,key="fuel_efficiency")
    return fuel_efficiency


#If the user want to see the dashboard then this code will be executed
if (selected == "Dashboard"):
    # DashBoard Creation
    st.title(":bar_chart:Dashboard")
    df1 = df[["Sales_in_thousands","Price_in_thousands"]]
    total_sales =(df1['Sales_in_thousands'].sum())
    average_price = round(df1['Price_in_thousands'].mean(),2)
    average_sale = round(df1['Sales_in_thousands'].mean(),2)

    left_column, middle_column, right_column = st.columns(3)
    left_column.metric("Total sales",total_sales)
    middle_column.metric("Average price",average_price,"lakhs")
    right_column.metric("Average sales of car",average_sale)
    st.markdown("---")

    df2 = df[["Model","Sales_in_thousands"]]  #For showing , what the model has high sales
    sales_chart = (
        df2.groupby(by=["Model"]).sum()[["Sales_in_thousands"]].sort_values(by="Sales_in_thousands")
    )
    fig_sales_chart = px.bar(
        sales_chart,
        x=sales_chart.index,
        y="Sales_in_thousands",
        title = "<b>Sales of car</b>",
        color_discrete_sequence=["blue"]*len(sales_chart),
        template = "plotly_white",
        barmode='group',
        height=500,
    )

    def engine_fuel():
        df3 = df[["Engine_size","Fuel_efficiency"]]
        Engine_fuel = (
             df3.groupby(by=["Engine_size"]).sum()[["Fuel_efficiency"]].sort_values(by="Fuel_efficiency")
          )
        return Engine_fuel    #To make a visualization that how engine size impact the fuel efficiency
    Engine_fuel=engine_fuel()
    engine_fuel_chart = px.bar(
        Engine_fuel,
        x=Engine_fuel.index,
        y="Fuel_efficiency",
        title="<b>The effeciency of fuel corresponding its engine size</b>",
        color_discrete_sequence=["blue"],
        template = "plotly_white",
        height = 500,
        barmode = 'group',
    )

    engine_fuel_chart.update_layout(
        xaxis_title = "Engine_size",
        yaxis_title = "Fuel_Efficiency",
    )
    left_column,right_column = st.columns(2)     #making the visualizations side by side
    left_column.plotly_chart(fig_sales_chart, use_container_width=True)
    right_column.plotly_chart(engine_fuel_chart,use_container_width=True)



    # To make second row visualizations
    
    st.title("The price of car corresponding to its milege")
    line_chart_data = df.copy()

    sample_tab = pd.crosstab(line_chart_data['mpg'],line_chart_data['Price_in_thousands'])
    fig2 = px.bar(sample_tab)

    st.write(fig2)

    p1 = df[["Engine_size","cylinders"]]
    st.title("The relation between cylinders and engine size")
    st.line_chart(data=p1, width=0, height=0, use_container_width=True)


elif(selected == "Milege_prediction"): #When user want to see the milege prediction then this code will be executed

    # Milege Prediction
    st.title('Do you want check your milege , why late? go ahead')
    model = pickle.load(open('Models/milege_model.pkl','rb'))  # load the model
    scaler1 = pickle.load(open('scalers/scaler_mpg.pickle','rb'))  #loading the scaled data
    cylinders = st.number_input('Enter numbers of cylinders',step=1,key="cylinders") #Taking cylinders input from user
    displacement = st.number_input('Enter the displacement(in hundreds)',10.0,step=50.0,key="displacement") #taking displacement input from user
    weight = st.number_input('Enter the weight of car(in Thousands)',1000,step=150,key="weight")# taking weight input from user
    acceleration = st.number_input('Enter the Acceleration',10.0,step=0.5,key="acceleration") #taking acceleration input from user
    
    def prediction(cylinders,displacement,weight,acceleration): #This is the function to make the scaling of input data
        X_pred = np.zeros(4)
        X_pred[0]=cylinders
        X_pred[1]=displacement
        X_pred[2]=weight
        X_pred[3]=acceleration
        X_pred = scaler1.transform([X_pred])
        prediction = model.predict(X_pred)
        return prediction

    if st.button("Estimate the milege",key="Predict the milege"):# The button which gives prediction value by clicking it.
        try:
            predictions =prediction(cylinders,displacement,weight,acceleration) #doing prediction
            output = round(predictions[0],2) #taking only two decimal values into consideration
            if output<0:
                st.warning("This car will not give sufficient milege") #If output is less than 0,then we conclude that the car will not give good milege
            else:
                st.success("Your car milege is {} mpg".format(output)) # If output is greater than zero,then we display the value
        except:
            st.success("oops!something went wrong \n Try Again")# If anything wrong happens then we will show this message


# If user want to predict the fuel capacity then this code will be executed
elif(selected=='Fuel_Capacity_prediction'):
    st.title("Do you want to know your car fuel capacity,go ahead")
    scaler = None
    model_2 = None
    with open("scalers/scaler_fuelcapacity.pickle", "rb") as f: # Loading the scaled data
        scaler = pickle.load(f)
    with open("Models/fuelcapacity_prediction_model.pickle", "rb") as f: #Loading the model
        model_2 = pickle.load(f)
    # Taking inputs , there are engine_size,Horseapower,wheelBase,width,Length,curbweight from user
    engine_size = Engine_Size()
    HorsePower = horse_power()
    WheelBase = wheel_base()
    width = WIDTH()
    Length = length()
    curb_weight=curb_weight()

    def prediction(Engine_size,Horsepower,Wheelbase,Width,Length,Curb_weight):#This is a function which scale the input data
        X_pred = np.zeros(6)
        X_pred[0]=Engine_size
        X_pred[1]=Horsepower
        X_pred[2]=Wheelbase
        X_pred[3]=Width
        X_pred[4]=Length
        X_pred[5]=Curb_weight
        X_pred = scaler.transform([X_pred])
        prediction = model_2.predict(X_pred)
        return prediction

    if st.button("Determine the fuel capacity",key="predict the Fuel capacity"):# This is the button which gives results after clicking on it 
        #some conditions and exception handling on the data which is given by user
        try:
            Model_2 = model_2
            predictions = prediction(engine_size,HorsePower,WheelBase,width,Length,curb_weight)
            output = round(predictions[0],2)
            if output<0:
                st.warning("Your car does not has the good fuel capacity")
            else:
                st.success("Your car Fuel capacity is {} l/km".format(output))
        except:
            st.warning("ohnoo!something went wrong \n Try Again")

# If a user wants to predict the Horse Power then this code will be executed
elif(selected=="Horse Power Prediction"):
    st.title("Horse Power Prediction")
    st.markdown("Do you want to know your car's Horse Power,go ahead")
    with open("scalers/scaler_horse_power.pickle","rb") as f: # Loading the scaled data
        scaler2 = pickle.load(f)
    with open("Models/Horse_power_model.pickle","rb") as f: # Loading the model
        Model_3 = pickle.load(f)

    #Taking inputs from user
    Model = car_Model()
    price = price_of_car()
    engine_size = Engine_Size()
    Wheelbase = wheel_base()
    Width = WIDTH()
    Length = length()
    FuelCapacity = Fuel_Capacity()
    performance_Factor = performance_factor()

    #This is a function which scale the user input
    def prediction(Model,Price_in_thousands,Engine_size,Wheelbase,Width,Length,Fuel_capacity,Power_perf_factor):
        X_pred = np.zeros(8)
        X_pred[0]=encoded_model[Model]
        X_pred[1]=Price_in_thousands
        X_pred[2]=Engine_size
        X_pred[3]=Wheelbase
        X_pred[4]=Width
        X_pred[5]=Length
        X_pred[6]=Fuel_capacity
        X_pred[7]=Power_perf_factor
        X_pred = scaler2.transform([X_pred])
        prediction = Model_3.predict(X_pred)
        return prediction

    if st.button("predict",key="predict the HorsePower"):# This is a button which gives results by clicking on it
        try:
            predictions = prediction(Model,price,engine_size,Wheelbase,Width,Length,FuelCapacity,performance_Factor)
            output = round(predictions[0],2)
            if output<0:
                st.warning("Your car does not has the good HorsePower.Please try to give your engine more air and more power with a cold air intake")
            else:
                st.success("Your car Fuel capacity is {} watts".format(output))
        except:
            st.warning("ohnoo!something went wrong \n Try Again")

# If a user wants to predict fuel effeciency then this code will be executed
elif(selected == "fuel effeciency prediction"):
    st.title("Fuel Efficiency Prediction")
    st.markdown("Do you want to know the fuel efficiency of your car ,\n go ahead")
    with open("scalers/scaler_fuel_eff.pickle","rb") as f: #Loading the scaled data
        scaler3 = pickle.load(f)
    with open("Models/fuelEffeciency_prediction_model.pickle","rb") as f: #Loading the model
        model_4 = pickle.load(f)
    # Taking input from user
    Model = car_Model()
    price = price_of_car()
    engine_size = Engine_Size()
    Wheelbase = wheel_base()
    HorsePower = horse_power()
    curb_weight = curb_weight()
    FuelCapacity = Fuel_Capacity()
    performance_Factor = performance_factor()
    
    #The function which scale the input data
    def prediction(Model,Price_in_thousands,Engine_size,Horsepower,Wheelbase,Curb_weight,Fuel_capacity,Power_perf_factor):
        X_pred = np.zeros(8)
        X_pred[0]=encoded_model[Model]
        X_pred[1]=Price_in_thousands
        X_pred[2]=Engine_size
        X_pred[3]=Horsepower
        X_pred[4]=Wheelbase
        X_pred[5]=Curb_weight
        X_pred[6]=Fuel_capacity
        X_pred[7] = Power_perf_factor
        X_pred = scaler3.transform([X_pred])
        prediction = model_4.predict(X_pred)
        return prediction


    if st.button("predict",key="predict the Fuel Efficiency"):# The button which predicts the value by clicking on it
        try:
            predictions = prediction(Model,price,engine_size,HorsePower,Wheelbase,curb_weight,FuelCapacity,performance_Factor)
            output = round(predictions[0],2)
            if output<0:
                st.warning("Your car does not has the good HorsePower.Please try to give your engine more air and more power with a cold air intake")
            else:
                st.success("Your car Fuel Efficiency is  {} km/l".format(output))
        except:
            st.warning("ohnoo!something went wrong \n Try Again")

#curb weight prediction 

elif(selected =="curb weight prediction"):
    st.title("Curb Weight Prediction of Car")
    st.markdown("Do you want to know the curb weight of your car , Go Ahead")
    #Load the Model
    model_5 = pickle.load(open('Models/curb_weight_model.pickle','rb'))
    #Taking the inputs from user
    price = price_of_car()
    engine_size = Engine_Size()
    HorsePower = horse_power()
    Wheelbase = wheel_base()
    Width = WIDTH()
    Length = length()
    FuelCapacity = Fuel_Capacity()

    if st.button("predict",key="Predict the curb weight"):# The button which predicts the results
        try:
            Model = model_5
            prediction = Model.predict([[price,engine_size,Wheelbase,HorsePower,Width,Length,FuelCapacity]])
            output = round(prediction[0],2)
            if output<0:
                st.warning("This car does not have good curb weight")
            else:
                st.success("Your car curb weight will be {} ".format(output))
        except:
            st.success("oops!something went wrong \n Try Again")


#If user want to predict the performance factor then this code will be executed

elif(selected== "Performance Factor"):
    st.title("Performance Factor Prediction")
    st.markdown("The lower the performance Factor,The Higher your car's performance level")
    #loading the model
    model_6 = pickle.load(open('Models/pf_model.pickle','rb'))
    with open("scalers/scaler_pf.pickle","rb") as f:# Loading the scaled data
        scaler4 = pickle.load(f)
    # Taking inputs from user
    Model = car_Model()
    price = price_of_car()
    engine_size = Engine_Size()
    HorsePower = horse_power()
    curb_weight = curb_weight()
    FuelCapacity = Fuel_Capacity()
    Fuel_efficiency = Fuel_Efficiency()
# The function which scale the input data
    def prediction(Model,Price_in_thousands,Engine_size,Horsepower,Curb_weight,Fuel_capacity,Fuel_efficiency):
        X_pred = np.zeros(7)
        X_pred[0]=encoded_model[Model]
        X_pred[1]=Price_in_thousands
        X_pred[2]=Engine_size
        X_pred[3]=Horsepower
        X_pred[4]=Curb_weight
        X_pred[5]=Fuel_capacity
        X_pred[6] = Fuel_efficiency
        X_pred = scaler4.transform([X_pred])
        prediction = model_6.predict(X_pred)
        return prediction

    if st.button("predict",key="predict the Fuel Efficiency"):# The button which gives results by clicking on it
        try:
            predictions = prediction(Model,price,engine_size,HorsePower,curb_weight,FuelCapacity,Fuel_efficiency)
            output = round(predictions[0],2)
            if output<0:
                st.warning("Your car does not has the high performance factor it is not good for your car")
            else:
                st.success("Your car performance factor is  {} pf".format(output))
        except:
            st.warning("ohnoo!something went wrong \n Try Again")

#If a user want to predict the engine size then this code will be executed
elif(selected=="Engine_size_prediction"):
    st.title("Engine Size Prediction")
    model_8 = pickle.load(open('Models/engine_size_model.pkl','rb'))  #loading the model
    scaler6 = pickle.load(open('scalers/scaler_engine_size.pickle','rb')) #loading the scaled data
# Taking required inputs from the user
    Model = car_Model()
    price = price_of_car()
    HorsePower = horse_power()
    wheelbase = wheel_base()
    width = WIDTH()
    Length = length()
    curbweight = curb_weight()
    fuelcapacity = Fuel_Capacity()
    performancefactor = performance_factor()


    def prediction(Model,Price_in_thousands,Horsepower,Wheelbase,Width,Length,Curb_weight,Fuel_capacity,Power_perf_factor):
        X_pred = np.zeros(9)
        X_pred[0]=encoded_model[Model]
        X_pred[1]=Price_in_thousands
        X_pred[2]=Horsepower
        X_pred[3]=Wheelbase
        X_pred[4]=Width
        X_pred[5]=Length
        X_pred[6]=Curb_weight
        X_pred[7]=Fuel_capacity
        X_pred[8] = Power_perf_factor
        X_pred = scaler6.transform([X_pred])
        prediction = model_8.predict(X_pred)
        return prediction
    
    if st.button("predict",key="predict the engine size"):# The button which gives results by clicking on it
        try:
            predictions = prediction(Model,price,HorsePower,wheelbase,width,Length,curbweight,fuelcapacity,performancefactor)
            output = round(predictions[0],2)
            if output<0:
                st.warning("This car does not has the good engine_size")
            else:
                st.success("Your car,s engine size is  {} cc".format(output))
        except:
            st.warning("ohnoo!something went wrong \n Try Again")

# If a user want to predict the price of car then this code will be executed
elif(selected=="Carprice_predictor"):
    st.title("Car Price Prediction")
    st.markdown("Do you want to predict the car price,Go Ahead")

    #loading the model
    model_7 = pickle.load(open('Models/price_prediction_model.pickle','rb'))
    scaler5 = pickle.load(open('scalers/scaler_price.pickle','rb'))#loading the scaled data
   #Taking inputs from user
    Model = car_Model()
    engine_size = Engine_Size()
    HorsePower = horse_power()
    Wheelbase = wheel_base()
    Width = WIDTH()
    Length = length()
    FuelCapacity = Fuel_Capacity()
    Fuel_efficiency = Fuel_Efficiency()

     # 
     #This is a function which scale the user input
    def prediction(Model,Engine_size,Horsepower,Wheelbase,Width,Length,Fuel_capacity,Fuel_efficiency):
        X_pred = np.zeros(8)
        X_pred[0]=encoded_model[Model]
        X_pred[1]=Engine_size
        X_pred[2]=Horsepower
        X_pred[3]=Wheelbase
        X_pred[4]=Width
        X_pred[5]=Length
        X_pred[6]=Fuel_capacity
        X_pred[7]=Fuel_efficiency
        X_pred = scaler5.transform([X_pred])
        prediction = model_7.predict(X_pred)
        return prediction

    if st.button("predict",key="predict the price of car"): # The button which gives the results by clicking on it
       try:
            predictions = prediction(Model,engine_size,HorsePower,Wheelbase,Width,Length ,FuelCapacity,Fuel_efficiency )
            output = round(predictions[0],2)
            if output<0:
               st.warning("Your car does not has the good price")
            else:
                st.success("Your car price is  {} lakhs".format(output))
       except:
            st.warning("ohnoo!something went wrong \n Try Again")
# If a user wants to predict the resale price then this code will be executed
elif(selected=="Resale_price_prediction"):
    st.title("Resale price prediction")
    model_9 = pickle.load(open('Models/resale_prediction_model.pickle','rb'))# loading the model
    #taking some inputs from the user
    price = price_of_car()
    enginesize = Engine_Size()
    horsepower = horse_power()
    fuelcapacity=Fuel_Capacity()
    performancefactor = performance_factor() 
    

    if st.button("predict",key="Predict the curb weight"):# The button which predicts the results
        try:
            Model = model_9
            prediction = Model.predict([[price,enginesize,horsepower,fuelcapacity,performancefactor]])
            output = round(prediction[0],2)
            if output<0:
                st.warning("This car does not have good resale price")
            else:
                st.success("Your car resale price  will be {} thousands".format(output))
        except:
            st.success("oops!something went wrong \n Try Again")




# To remove some streamlite app default features
hide_st_style = """
               <style>
               #MainMenu {visibility: hidden;}
               footer {visibility: hidden;}
               header {visibility: hidden;}
               </style>
               """
st.markdown(hide_st_style, unsafe_allow_html=True)