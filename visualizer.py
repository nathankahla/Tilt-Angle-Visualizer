### Visualizer for comparing Wind+Slope 
# The goal is to combine and compare different models for flame deflection angle as a function of wind and slope
# as found in the literature
# Nathan Kahla USFS RMRS Missoula, MT November 2024

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

G = 9.80665 # gravity m/s^2
RHO = 1.225 #ambient air mass density kg/m^3
CP = 1.005  #constant pressure specific heat kJ/kg*K

levelv_dict = {
        'Fireline Intensity (kW/m)' : [.1, 50.0, 10000.0],
        'Freestream Wind Speed (m/s)' : [0, .5, 15]
}


def Nelson_buoyancy(Intensity):
    # using U_b from Nelson 02 
    t_a = 303.15
    upward_buoyancy = np.power(((2*G*Intensity)/(RHO*CP*t_a)),(1/3))
    return upward_buoyancy

def Balbi_07(input : list ):
    #applies to up-slope conditions only, phi_normal will probably = 0 in all cases for this exercise 
    # v_w = freestream wind speed
    # u_fl = upward gas flow velocity

    slope = input[0]
    v_w = input[1]
    intensity = input[2]
    
    phi_normal = 0
    local_slope_numer = (np.sin(np.deg2rad(slope))*np.cos(np.deg2rad(phi_normal)))
    local_slope_denom = np.sqrt(np.sin(np.deg2rad(slope))**2*np.cos(phi_normal)**2+np.cos(np.deg2rad(slope))**2)
    local_slope = np.arcsin(local_slope_numer/local_slope_denom)
    u_fl = Nelson_buoyancy(intensity)
    buoyancy = np.arctan(v_w/u_fl)

    flame_tilt_angle = local_slope + buoyancy

    return np.rad2deg(flame_tilt_angle)

def Balbi_09(input : list):
    # u_0 = upward gas flow velocity
    # u_normal = windspeed parallel with slope (freestream wind speed *cos(slope))
    
    slope = input[0]
    u_normal = np.cos(np.deg2rad(slope))*input[1]
    intensity = input[2]
    
    u_0 = Nelson_buoyancy(intensity)
    
    hsflame_tilt_angle = np.arctan((u_normal*np.cos(np.deg2rad(slope))**2)/(u_0 + u_normal*np.sin(np.deg2rad(slope))*np.cos(np.deg2rad(slope))))

    lsflame_tilt_angle = np.arctan((np.tan(np.deg2rad(slope))/(1+2*np.tan(np.deg2rad(slope))**2)))

    if abs(u_normal) > u_0*np.tan(abs(np.deg2rad(slope))):
        return np.rad2deg(hsflame_tilt_angle)+slope
    else:
        return np.rad2deg(lsflame_tilt_angle)+slope


def Balbi_20(input : list):
    #U denotes wind velocity parallel with slope, 
    #u_b denotes upward gas velocity from buoyancy
    slope = input[0]
    u_normal = np.cos(np.deg2rad(slope))*input[1]
    intensity = input[2]
    
    u_b = Nelson_buoyancy(intensity)
    flame_tilt_angle = np.arctan((np.tan(np.deg2rad(slope))+u_normal/(u_b)))

    return np.rad2deg(flame_tilt_angle)

def Morandini_etal(input : list):
    #v_w is normal velocity, parallel with inclined surface , v_fl= freestream
    slope = input[0]
    v_w = input[1]
    v_fl = input[1]
    intensity = input[2]
    
    U_b = Nelson_buoyancy(intensity)
    flame_tilt_angle = np.arctan((v_w*np.cos(np.deg2rad(slope))**2)/(U_b +v_w*np.cos(np.deg2rad(slope))*np.sin(np.deg2rad(slope))))

    return np.rad2deg(flame_tilt_angle)+slope

def set_x_axis(num_points, min, max):
    #function to take a number of x points desired and range from the user and return a list of points, just slope for now
    spread = max-min
    inc = spread/num_points
    x_axis_points = list(np.arange(min, max+inc, inc))
    return x_axis_points

def set_level_variables(levelv, value_list):
    #takes list of values from sliders, and then duplicates to a list of 6 lists, where 5 are the range of level variables (default levelv is velocity)
    levelv_list =[]
    if levelv != 'No Level Variables':
        varmax = levelv_dict[levelv][2]
        varmin = levelv_dict[levelv][1]
        levelv_step = (varmax-0)/5  #set number of level variable traces here

        if levelv == 'Freestream Wind Speed (m/s)':
            levelv_list.append([0,value_list[1]])
            for step in np.arange(varmin, varmax, levelv_step):
                levelv_list.append([step, value_list[1]])
        else:
            arange_list = [50, 500, 1000, 2000, 7500]       #hardcoding list for Intensity Values
            for step in arange_list:
                levelv_list.append([value_list[0], step])
    else:
        #for case where no level variable is requested
        levelv_list.append(value_list)

    return levelv_list

def xform_levelv_list(x_axis_points, levelv_list):
    #takes all the combinations of level variables and nests a list for each x point, structured such that I can feed them intelligently into my different models
    #hardcoding that there are only 5 level variables
    ready_list = []
    list_1 = []
    if len(levelv_list) > 1:
        list_2 = []
        list_3 = []
        list_4 = []
        list_5 = []
        for x in x_axis_points:
            list_1.append([x, levelv_list[0][0],levelv_list[0][1]])
            list_2.append([x, levelv_list[1][0],levelv_list[1][1]])
            list_3.append([x, levelv_list[2][0],levelv_list[2][1]])
            list_4.append([x, levelv_list[3][0],levelv_list[3][1]])
            list_5.append([x, levelv_list[4][0],levelv_list[4][1]])
        ready_list.append(list_1)
        ready_list.append(list_2)
        ready_list.append(list_3)
        ready_list.append(list_4)
        ready_list.append(list_5)
    else:
        for x in x_axis_points:
            list_1.append([x, levelv_list[0][0],levelv_list[0][1]])
        ready_list.append(list_1)
    
    return ready_list

def generate_datasets(xaxis_points, ready_list, level_v):
    #for each model, and for each set of variables including the 5 level variables, fill columns of a dataframe, where column 1 is x,
    model_list = [Balbi_07,Balbi_09,Balbi_20,Morandini_etal]
    df = pd.DataFrame(xaxis_points)
    
    if level_v == 'Fireline Intensity (kW/m)':
        lvlv_name = "intensity"
        index = 2
    elif level_v == 'Freestream Wind Speed (m/s)':
        lvlv_name = "windspeed"
        index = 1
    else:
        lvlv_name = "none"

    for model in model_list:
        if lvlv_name == "none":
                col_name = f"{model.__name__}"
                model_results = [model(point) for point in ready_list[0]]
                df[col_name] = model_results
        else:
            for i in range(0,5):
                col_name = f"{model.__name__}_{lvlv_name}_{ready_list[i][0][index]}_{i+1}"
                model_results = [model(point) for point in ready_list[i]]
                df[col_name] = model_results 
            
    df.to_csv("output.csv", index= False)
    
    return df

def create_gui():
    st.set_page_config(page_title="Tilt Angle Visualizer", layout="wide")
    st.title("Tilt Angle Visualizer")

    col0,col1,col2 = st.columns(3, gap="medium")

    model_list = ["Balbi_07", "Balbi_09", "Balbi_20", "Morandini_etal"]
    levelv_list = ['No Level Variables', 'Fireline Intensity (kW/m)', 'Freestream Wind Speed (m/s)']
    val_list = []

    with col0:
        st.write("X-Axis Options")
        num_points = st.slider("How many X points do you want", 10.0, 200.0, 60.0, 10.0, key = 'xi')
        min = st.slider("What is Min slope you are interested in?", 0, 90, 0, 5, key= 'minx')
        max = st.slider("What is Max slope you are interested in?", min, 90, 30, 5, key= 'maxx')

    with col1:
        levelv = st.selectbox("Level Variable", levelv_list, 0, key= "levelv")

        if levelv == 'Fireline Intensity (kW/m)':
            intensity = st.slider("Fireline Intensity (kW/m)", .1, 10000.0, 1000.0, 50.0, disabled = True, key = "i")
        else: 
            intensity = st.slider("Fireline Intensity (kW/m)", .1, 10000.0, 1000.0, 50.0, key = "i")

        if levelv == "Freestream Wind Speed (m/s)":
            fws = st.slider("Freestream Wind Speed (m/s)", 0.0, 12.0, 1.0, 1.0, disabled= True, key= "ws")
        else:
            fws = st.slider("Freestream Wind Speed (m/s)", 0.0, 12.0, 1.0, 1.0, key= "ws")
    with col2:
        st.write("Which models are you interested in seeing side by side?")
        which_model = st.multiselect("Which models?", model_list, default= model_list)
        
    x_axis_points = set_x_axis(num_points, min, max)
    
    val_list= [fws, intensity]

    levelv_list = set_level_variables(levelv, val_list)
    ready_list = xform_levelv_list(x_axis_points, levelv_list)
    data = generate_datasets(x_axis_points, ready_list, levelv)

    #graph traces from selected models
    fig = go.Figure()
    if levelv == 'No Level Variables':
        for model in model_list:
            if model in which_model:
                fig.add_trace((go.Scatter(x= x_axis_points, y = data[f"{model}"], name = f"{model}")))
    else: 
        for model in model_list:
            if model in which_model:
                new_data = data.filter(like= f"{model}")
                for col in new_data.columns:
                    fig.add_trace((go.Scatter(x= x_axis_points, y = new_data[col], name=col)))
    
    fig.update_traces(hovertemplate = "(%{x: .4r}, %{y: .4r})")

    fig.update_layout(title="Flame Tilt Angle vs. Slope",
		xaxis_title="Slope (deg)",
		yaxis_title="Flame Tilt Angle (deg)",
		legend_title="Model",
		height=600, width = 1000
        )
    
    st.plotly_chart(fig)
    

    return val_list
    
if __name__ == "__main__":
    create_gui()
    
