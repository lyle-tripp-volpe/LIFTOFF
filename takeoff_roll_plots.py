#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from os.path import expanduser
import os.path 
import numpy as np
import webbrowser as wb

home = expanduser("~")
filepath = home+'\OneDrive - DOT OST\BADA4_Reduced_Thrust_Sensor_Path_Noise_Comparison_main'#\\FDR_desensitized\\TKOFF'
# filepath = home+'\OneDrive\Documents\takeoff_thrust\FDR_desensitized\TKOFF'
filename = 'takeoff_distance_A320_A330_A340'#_kluge'
file_extension = '.csv'
df = pd.read_csv(os.path.join(filepath, filename + file_extension), index_col=["FLIGHT_ID", "TIME_OFFSET"])
df.rename(
    columns={'HEAD_WIND':'HEAD_WIND_KNOTS'
             ,'FUEL_QUANTITY':'FUEL_QUANTITY_KILOGRAMS'
             ,'P64  Air Temperature (outside) at Start of Event (library) (Deg Celsius)':'AIR_TEMP_OUTSIDE_CELSIUS'
             ,'P64  Gross Weight at Start of Event (library) (kg)':'GROSS_WEIGHT_KILOGRAMS'
             ,'P64  Flap Position at Start of Event (library) (deg+   TED)':'FLAP_POSITION_DEGREES'
            }
    , inplace=True)


# In[2]:


def scatter_plot(df_schedule, color_field = "APT_AIRCRAFT_RUNWAY_STAGE" ):
    data_frame = df_schedule.reset_index(level = ["FLIGHT_ID"])
    title='Liftoff distance from runway end (feet)'
    x='DISTANCE_FROM_RUNWAY_END_liftoff'
    y='DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF'
    labels = {'DISTANCE_FROM_RUNWAY_END_liftoff':'As stated in CFDR',
                        'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF':'As detected from trajectory',
                        'APT_AIRCRAFT_RUNWAY_STAGE': 'Airport / Aicraft Type / Runway / Stage Length ID'
                       }
    hover_data= ['FLIGHT_ID']#, 'DISTANCE_FROM_RUNWAY_END', 'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF']
    opacity = 0.6
    #TODO:make this a function
    x_max = stated_max = df_schedule.DISTANCE_FROM_RUNWAY_END_liftoff.max()
    x_min = stated_min = df_schedule.DISTANCE_FROM_RUNWAY_END_liftoff.min()
    y_max = detected_max = df_schedule.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF.max()
    y_min = detected_min = df_schedule.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF.min()

    perfect_fit_min = x_min if x_min < y_min else y_min
    perfect_fit_max = x_max if x_max > y_max else y_max

    color_field_ordering_dict = {'liftoff_detection_quality_categorized': ['very high', 'high', 'good', 'mediocre', 'poor']}
    
    if color_field == 'liftoff_detection_quality_categorized': #if we are making an overview plot
        fig = px.scatter(
            data_frame=data_frame
            , title=title
            , x=x
            , y=y
            , color=color_field
            , labels=labels
            , hover_data= hover_data
            , opacity=opacity
            , color_discrete_sequence= ['red'] + sorted(px.colors.sequential.Plasma_r[0:2], reverse=True)+px.colors.sequential.Plasma_r[-2:-1] + ['blue']
            , category_orders=color_field_ordering_dict
        )
    else:
            fig = px.scatter(
            data_frame=data_frame
            , title=title
            , x=x
            , y=y
            , color=color_field
            , labels=labels
            , hover_data= hover_data
            , opacity=opacity
        )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min, perfect_fit_max]
            , mode = "lines"
            , name = 'Perfect Fit'
            , line = go.scatter.Line(color = 'black', dash = 'dash')
            , opacity=0.4
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min-275, perfect_fit_max-275]
            , mode = "lines"
            , name = 'Perfect Fit - 275 ft (a.k.a detected liftoff one second before actual)'
            , line = go.scatter.Line(color = 'black', dash = 'longdash')
            , opacity=0.4
        )

    )


    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min-550, perfect_fit_max-550]
            , mode = "lines"
            , name = 'Perfect Fit - 550 ft (a.k.a detected liftoff two seconds before actual)'
            , line = go.scatter.Line(color = 'purple', dash = 'longdashdot')
            , opacity=0.4
        )

    )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min+275, perfect_fit_max+275]
            , mode = "lines"
            , name = 'Perfect Fit + 275 ft (a.k.a detected liftoff one second after actual)'
            , line = go.scatter.Line(color = 'black', dash = 'longdash')
            , opacity=0.4
        )

    )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min+550, perfect_fit_max+550]
            , mode = "lines"
            , name = 'Perfect Fit + 550 ft (a.k.a detected liftoff two seconds after actual)'
            , line = go.scatter.Line(color = 'purple', dash = 'longdashdot')
            , opacity=0.4
        )

    )

    fig.update_yaxes(
    #     scaleanchor = "x",
        scaleratio = 1,
      )
    
    return fig


# In[3]:


def plot_metrics_for_individual_flights(dataframe1, dataframe2 = None, flight_sample_size = 10):
    
    #raise an error if we get sent a series and not a dataframe
    
    metric_maxes = {}
     
    plotbook_filename = 'plotbook' + '_' + str(flight_sample_size) if flight_sample_size != -1 else 'plotbook'
    df = dataframe1
    
    df_liftoff = df[df.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF.notna()]
    df_liftoff.reset_index(level='TIME_OFFSET', inplace=True)
    
    if dataframe2 is not None:
        #add the predicted values
        df_liftoff["ML_predicted_liftoff_point_distance_from_runway_end_(feet)"] = dataframe2.PREDICTED_DISTANCE_FROM_RUNWAY_END
        
    if ('FLIGHT_ID' in df.index.names) & ('FLIGHT_ID' in df.columns):
        df_grouped_by_flight = df.drop(columns = ['FLIGHT_ID'], axis = 1).groupby("FLIGHT_ID")
        df["FLIGHT_ID"] = df["FLIGHT_ID"].astype(str)
    else:
        df_grouped_by_flight = df.groupby("FLIGHT_ID")
    
    fn = plotbook_filename + '.html'
    open( os.path.join(filepath,fn), 'w')
    #     #TODO: write plots to pdf file
    #     fn = plotbook_filename + '.pdf'

    flight_count = 0
    for flight, group in df_grouped_by_flight:
        flight_sample_size = len(group) if flight_sample_size == -1 else flight_sample_size

        if flight_count == flight_sample_size:
            break

#         metrics = ['MSL_ALT']
        metrics = ['AFE_ALT']
        metrics += ['N1', 'TAS_SEGMENT']

        #create Figure with subplots for each metric for a given flight
        fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

        for m in metrics:
            metric_maxes[m]=group[m].max() if m != 'N1' else 100

            #create and add a subplot to the Figure
            sub = go.Scatter(
                    x=group.DISTANCE_FROM_RUNWAY_END
                    , y=group[m]
                    , mode="markers"
                    , name = m
            )
            fig.add_trace(sub, row=metrics.index(m)+1, col=1)

    #         #turn off auto range adjustment
    #         fig.update_layout(
    #             yaxis_autorange = False
    #             )        
    #         fig.update_yaxes({}, row=metrics.index(m)+1, col=1)

            dl_cgtd = detected_liftoff_cumul_grnd_trk_dist = df_liftoff.loc[flight,"DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF"]
            fig.add_trace(
                go.Scatter(
                    x=[dl_cgtd, dl_cgtd]
                    , y=[0,metric_maxes[m]]
                    , mode = "lines"
                    , name = "Detected Liftoff"
                    , line = go.scatter.Line(color = 'gray')
                )
                , row=metrics.index(m)+1, col=1 
            )        

            if dataframe2 is not None:
            #add the predicted values
                pred_liftoff_cgtd = predicted_liftoff_cumul_grnd_trk_dist = df_liftoff.loc[flight,"ML_predicted_liftoff_point_distance_from_runway_end_(feet)"]
                fig.add_trace(
                    go.Scatter(
                        x=[pred_liftoff_cgtd, pred_liftoff_cgtd]
                        , y=[0,metric_maxes[m]]
                        , mode = "lines"
                        , name = "ML v0 Predicted Liftoff"
                        , line = go.scatter.Line(color = 'blue', dash = 'dashdot')
                    )
                    , row=metrics.index(m)+1, col=1 
                )            
            
            rprt_liftoff_cgtd = reported_liftoff_cumul_grnd_trk_dist = df_liftoff.loc[flight,"DISTANCE_FROM_RUNWAY_END"]
            fig.add_trace(
                go.Scatter(
                    x=[rprt_liftoff_cgtd, rprt_liftoff_cgtd]
                    , y=[0,metric_maxes[m]]
                    , mode = "lines"
                    , name = "Reported Liftoff"
                    , line = go.scatter.Line(color = 'orange', dash = 'dash')
                )
                , row=metrics.index(m)+1, col=1 
            )

            fig.layout.yaxis.update(title_text = metrics[0])
            fig.layout.yaxis2.update(title_text = metrics[1], tickvals=[0] + np.arange(70,100,5))
            fig.layout.yaxis3.update(title_text = metrics[2])
            fig.layout.update(title_text = "Flight " + str(flight) + '<br>' + 'Airport / Aicraft Type / Runway / Stage Length ID: ' + 
                              str(df_liftoff.loc[flight,"APT_AIRCRAFT_RUNWAY_STAGE"]))
        
        #write plots to html file         
        html = fig.to_html()
        with open( os.path.join(filepath,fn), 'a') as f:
            f.write(html) 

        flight_count = flight_count + 1
    print(os.path.join(filepath,fn))
        
    return (os.path.join(filepath,fn))


# In[4]:


# #optionally write the desensitized file version if you like
# df_desensitized = pd.read_csv(os.path.join(filepath+'\\FDR_desensitized\\TKOFF', filename+'_desensitized'+file_extension), index_col=["FLIGHT_ID", "TIME_OFFSET"])
# df_kluge = pd.concat([df_desensitized,df[["FUEL_QUANTITY_KILOGRAMS"]]], axis = 1)
# df_kluge.APT_encoded = df_kluge.APT_encoded.astype('str')
# df_kluge.Departure_Runway_end_encoded = df_kluge.Departure_Runway_end_encoded.astype('str')
# df_kluge.STAGE_LENGTH_ID = df_kluge.STAGE_LENGTH_ID.astype('str')
# df_grouping_series = df_kluge[["APT_encoded", "ACTYPE", "Departure_Runway_end_encoded", "STAGE_LENGTH_ID"]].dropna()
# df_kluge["APT_AIRCRAFT_RUNWAY_STAGE"] = df_grouping_series.apply(list, axis=1).str.join(sep='_')
# df_kluge.to_csv(path_or_buf=filepath+'\\FDR_desensitized\\TKOFF\\takeoff_distance_kluge.csv')


# In[19]:


nonsensitive_column_list = [
'FLIGHT_ID'
 ,'P64: Duration of Takeoff (Seconds)'
#  ,'TIME_ON_GROUND_BEFORE_LIFTOFF_(SECONDS)'
#  ,'P64: Duration of Taxi Out (Minutes)'
#  ,'DURATION'
#  ,'TIME_OFFSET'
#  ,'SPEED_SOUND_START_EVENT'
#  ,'HEAD_WIND'
#  ,'AFE_ALT'
#  ,'TAS_START_EVENT'
 ,'P64: True Airspeed at Liftoff (knots)'
#  ,'MACH_NUMBER_START_EVENT'
#  ,'GS_SEGMENT'
#  ,'TAS_SEGMENT'
#  ,'MACH_NUMBER_SEGMENT
#  ,'DRAG
#  ,'LIFT
#  ,'DISTANCE_START_EVENT'
#  ,'DISTANCE_END_EVENT'
#  ,'FUELFLOW_START_EVENT'
#  ,'FUELFLOW_SEGMENT'
 ,'N1_liftoff'
#  ,'THRUST_START_EVENT'
#  ,'THRUST_SEGMENT'
 ,'LIFTOFF'
#  ,'STATED_SEGMENT_START_OF_TAKEOFF'
#  ,'DISTANCE_FROM_RUNWAY_END'
 ,'HEAD_WIND_KNOTS_during_stated_takeoff_ground_roll'
 ,'FLAP_POSITION_DEGREES_during_stated_takeoff_ground_roll'
 ,'FLAP_POSITION_DEGREES_std_during_stated_takeoff_ground_roll' 
#  ,'P64: Average Fuel Flow to all Engines during Takeoff (kg/hr; start --> liftoff)'
 ,'ACTYPE'
 ,'TAKEOFF_GROUND_ROLL_DISTANCE_STATED_(FEET)'
 ,'HEAD_WIND_KNOTS_TWENTY_POINT_LAGGING_AVERAGE'
 ,'RUNWAY_CALCULATED_GRADIENT_degraded'
 ,'GROSS_WEIGHT_KILOGRAMS_rollstart'
 ,'FLAP_POSITION_DEGREES'
 ,'MSL_ALT_degraded'
 ,'AIR_TEMP_OUTSIDE_CELSIUS'
 ,'N1_during_stated_takeoff_ground_roll'
]


# In[6]:


df_liftoff = df[df.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF.notna()]
df_liftoff.reset_index(level='TIME_OFFSET', inplace=True)

df_rollstart = df[df.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF.notna()]
df_rollstart.reset_index(level='TIME_OFFSET', inplace=True)


# In[15]:


#formulate flight schedule
salient_rollstart_col_list = ['DISTANCE_FROM_THRESHOLD_AT_POINT_ONE', 
    'DISTANCE_FROM_POINT_ONE_FEET',
    'GROUNDSPEED_AT_POINT_ONE_KNOTS', 'WELL-BEHAVED_TRAJECTORY',
    #        'LATITUDE_AT_POINT_ONE', 'LONGITUDE_AT_POINT_ONE',
    'STATED_SEGMENT_START_OF_TAKEOFF',
    'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF'
    ]

rollstart_exceptions_to_extraneousness_filter = ['N1', 'DISTANCE_FROM_RUNWAY_END', 'TIME_OFFSET','FUEL_QUANTITY_KILOGRAMS', 'HEAD_WIND_KNOTS', 'GROSS_WEIGHT_KILOGRAMS']

extraneous_schedule_columns_list = [col for col in df_rollstart.columns if col not in salient_rollstart_col_list + rollstart_exceptions_to_extraneousness_filter]

df_schedule = df_liftoff.drop(salient_rollstart_col_list, axis = 1).join(
    df_rollstart.drop(
        columns = extraneous_schedule_columns_list, axis = 1
    )
    , how = 'left'
    , lsuffix = '_liftoff' 
    , rsuffix = '_rollstart'
)

#coalesce the grouping column
# mask = (df_schedule['APT_AIRCRAFT_RUNWAY_STAGE_liftoff'].notna())
# df_schedule.loc[mask, "APT_AIRCRAFT_RUNWAY_STAGE"] = df_schedule['APT_AIRCRAFT_RUNWAY_STAGE_liftoff']
# mask = (df_schedule['APT_AIRCRAFT_RUNWAY_STAGE_rollstart'].notna())
# df_schedule.loc[mask, "APT_AIRCRAFT_RUNWAY_STAGE"] = df_schedule['APT_AIRCRAFT_RUNWAY_STAGE_rollstart']

df_schedule["TAKEOFF_GROUND_ROLL_DISTANCE_STATED_(FEET)"] = df_schedule.DISTANCE_FROM_RUNWAY_END_liftoff - df_schedule.DISTANCE_FROM_RUNWAY_END_rollstart
df_schedule["TAKEOFF_GROUND_ROLL_DISTANCE_DETECTED_(FEET)"] = df_schedule.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF - df_schedule.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF
df_schedule["liftoff_detection_quality_numerical"] = abs(df_liftoff.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF - df_liftoff.DISTANCE_FROM_RUNWAY_END) / df_liftoff.DISTANCE_FROM_RUNWAY_END 
df_schedule["MSL_ALT_degraded"] = round(df_schedule["MSL_ALT"],-2) #add degraded altitude so as to potentially anonymize flights
df_schedule["GROSS_WEIGHT_KILOGRAMS_rollstart"] = df_schedule.GROSS_WEIGHT_KILOGRAMS_liftoff + df_schedule["P64: Fuel Burned by all Engines during Takeoff (kg)"]
df_schedule["RUNWAY_CALCULATED_GRADIENT_degraded"] = round(df_schedule.RUNWAY_CALCULATED_GRADIENT, -4)

#instantiate bins for grouping on the quality of liftoff detection
liftoff_detection_quality_bins = {0.00:"very high", 0.01:"high", 0.05:"good", 0.10:"mediocre", 1.00:"poor"}

labels = []
i=0
for k, v in sorted(liftoff_detection_quality_bins.items()):
    labels.append(v)
    i += 1
labels_as_tuple = tuple(labels)

df_schedule["liftoff_detection_quality_categorized"] = pd.cut(
        df_schedule["liftoff_detection_quality_numerical"]
        , list(liftoff_detection_quality_bins.keys())+[100000]
        , labels = labels_as_tuple
    )


# In[16]:


modes = ['Unstated','Takeoff Ground Roll']

df["Stated_Trajectory_Mode"] = modes.index('Unstated')
df.Stated_Trajectory_Mode = df.Stated_Trajectory_Mode.astype("int")



#def mark_ground_roll_segments(df, flight_id, liftoff_time_offset, rollstart_time_offset, ground_roll_mode_index):
    #return a dataframe of start and stop times by flight
#select rows where TIME_OFFSET is within TIME_OFFSET_rollstart and TIME_OFFSET_liftoff
df_aug = df.loc[:,['N1']]
df_aug = df_aug.join(df_schedule, rsuffix="_sched", how="left").reset_index(level="TIME_OFFSET")
df_aug["Stated_Ground_Roll"] = (df_aug.TIME_OFFSET >= df_aug.TIME_OFFSET_rollstart) & (df_aug.TIME_OFFSET < df_aug.TIME_OFFSET_liftoff)
df_aug.set_index('TIME_OFFSET',append=True, inplace=True)
mask = (df_aug.Stated_Ground_Roll == True)
df.loc[mask, 'Stated_Trajectory_Mode'] = modes.index('Takeoff Ground Roll')

#testing mode output
df.Stated_Trajectory_Mode.describe() 


# In[22]:


#add average measures (e.g. N1) and dimensions (e.g., Flaps) during detected ground roll to the schedule
df_traj_takeoff_ground_roll = df[df.Stated_Trajectory_Mode == modes.index('Takeoff Ground Roll')]
df_traj_tkoff_gd_rl_groupby = df_traj_takeoff_ground_roll.groupby(by=["FLIGHT_ID"])
df_schedule["N1_during_stated_takeoff_ground_roll"] = df_traj_tkoff_gd_rl_groupby.mean().N1
df_schedule["HEAD_WIND_KNOTS_during_stated_takeoff_ground_roll"] = df_traj_tkoff_gd_rl_groupby.mean().HEAD_WIND_KNOTS
df_schedule["FLAP_POSITION_DEGREES_during_stated_takeoff_ground_roll"] = df_traj_tkoff_gd_rl_groupby.mean().FLAP_POSITION_DEGREES
df_schedule["FLAP_POSITION_DEGREES_std_during_stated_takeoff_ground_roll"] = df_traj_tkoff_gd_rl_groupby.std().FLAP_POSITION_DEGREES #- df_traj_tkoff_gd_rl_groupby.min().FLAP_POSITION_DEGREES 


# In[21]:


df_schedule_desensitized = df_schedule.reset_index(level=0)[nonsensitive_column_list].set_index("FLIGHT_ID")
df_schedule_desensitized.to_csv(os.path.join(filepath+'\\FDR_desensitized\\TKOFF', filename+'_desensitized_schedule.csv'))


#  len(df[df["WELL-BEHAVED_TRAJECTORY"] == 1]) rows in the sample are well-behaved

# In[ ]:


inp_vars = ["HEAD_WIND_KNOTS_liftoff", "GROSS_WEIGHT_KILOGRAMS_rollstart", "AIR_TEMP_OUTSIDE_CELSIUS"]#, "FLAP_POSITION_DEGREES"]
head_wind_bin_width = 2.5
#create the categorizations of input vars including numerical bins on the input variables in inp_vars list
df_schedule_groupby = df_schedule.groupby(by="APT_AIRCRAFT_RUNWAY_STAGE") 
df_schedule_bins = pd.DataFrame(
    index=pd.Index(df_schedule_groupby.indices.keys(), name = "APT_AIRCRAFT_RUNWAY_STAGE")
    , columns=inp_vars) #for holding the bin definitions

for i in inp_vars:
    frame = pd.DataFrame()
    num_bins = bin_width = 0
    for group, inp_var in df_schedule_groupby:

        if i == "GROSS_WEIGHT_KILOGRAMS_rollstart":
            bin_width = 0.015*inp_var[i].max()
        elif i == "HEAD_WIND_KNOTS_liftoff":
            bin_width = head_wind_bin_width 
        elif i == "AIR_TEMP_OUTSIDE_CELSIUS":
            bin_sequence = [0,(80-32)*5/9,1000]       
        else:
            num_bins = 10
        
        if bin_width > 0:
            num_bins = 1 + int((inp_var[i].max() - inp_var[i].min()) / bin_width)
        
        bins = num_bins if num_bins > 0 else bin_sequence

        print(group, i, "bins ", bins, sep='-->') #testing
        #pass the group-inp_var combo as a series extracted from the groupby object
        var_group_combo_as_series = pd.Series(inp_var[i], name=i)
        (var_group_combo_binned, bins) = pd.cut(var_group_combo_as_series
                           , bins
                           , labels=False
                           , retbins=True
                          )

        var_group_combo_binned = var_group_combo_binned.to_frame(name=i)
        #store the rows or columns
        frame = pd.concat([frame, var_group_combo_binned], axis = 0)

        #store the category definitions
        df_schedule_bins.loc[group, i] = bins
    
    df_schedule[i+"_BIN"] = frame


df_schedule_bins["GROSS_WEIGHT_KILOGRAMS_rollstart_BIN_WIDTH"] = df_schedule_bins.GROSS_WEIGHT_KILOGRAMS_rollstart.apply(lambda x: x[1] - x[0])

# In[ ]:

#some preparation for group analysis
df_schedule_grouped = pd.DataFrame()
grouping_criteria = ["APT_AIRCRAFT_RUNWAY_STAGE", "HEAD_WIND_KNOTS_liftoff_BIN", "GROSS_WEIGHT_KILOGRAMS_rollstart_BIN", "AIR_TEMP_OUTSIDE_CELSIUS_BIN"]#, "FLAP_POSITION_DEGREES_BIN"] 
df_schedule_grouped = df_schedule.reset_index().set_index(grouping_criteria).groupby(by=grouping_criteria)
min_flight_count = 10

#plot relationship of actual N1 to actual ground roll distance for a given weight group, weather group, runway group, aircraft type
#loop through the groups
df_schedule_grouped_filtered = df_schedule_grouped.filter(lambda x: x['N1_liftoff'].count() >= min_flight_count)

# In[ ]:

x = 'TAKEOFF_GROUND_ROLL_DISTANCE_STATED_(FEET)'
y = 'N1_liftoff'
    
fig = px.scatter(
    df_schedule_grouped_filtered
    , title='CFDR actual N1 vs actual ground roll distance for<br>a given temperature/weight/wind/runway/aircraft group<br>'+ 'min flight count per group: ' + str(min_flight_count)
        + '<br>head winds within: ' + str(head_wind_bin_width) + 'knots'
        + '<br>Temperature Category 0 means below 80 Degrees Farenheit'
    , x=x
    , y=y
    , color=df_schedule_grouped_filtered.index.to_flat_index()
    , trendline = 'ols'
    , labels = 
        {'TAKEOFF_GROUND_ROLL_DISTANCE_STATED_(FEET)':'Distance in feet between rollstart and liftoff as stated in CFDR',
                'N1_liftoff':'Throttle setting at liftoff',
#                 'APT_AIRCRAFT_RUNWAY_STAGE': 'Airport / Aicraft Type / Runway / Stage Length ID'
        }
    #, hover_data= ['FLIGHT_ID']
)
fig.layout.yaxis.update(range=(80,100))
fig.layout.xaxis.update(range=(5000,11000))
fig.layout.legend.update(title = "Airport / Aicraft Type / Runway / Stage Length ID<br>, Headwind Category, Weight Category, Air Temp Category")
file = fig.to_html()
with open(filepath+'\\N1_distance_correlation.html', 'w') as f:
    f.write(file)
N1_distance_fig = fig    

# In[ ]:
resulting_groups = df_schedule_grouped_filtered.reset_index().APT_AIRCRAFT_RUNWAY_STAGE.unique()#.to_list()

#spit out the weight bin width
#TODO: get the actual bin into the legend
pd.DataFrame(df_schedule_bins.GROSS_WEIGHT_KILOGRAMS_rollstart_BIN_WIDTH.loc[list(resulting_groups)])

#spit out the flight IDs that correspond to one group of interest 
group_of_interest = ('KORD_A330-3_4PW067_16_6', 3, 3, 0.0)
flights_of_interest = df_schedule_grouped_filtered.loc[group_of_interest,'FLIGHT_ID'].to_list()
print(flights_of_interest)
# In[ ]:

#pie chart of instantaneous N1
bin_sequence = np.arange(50, 110, 10)
color_discrete_sequence= px.colors.sequential.Plasma_r[0:1] + sorted(px.colors.sequential.Plasma_r[-3:], reverse=True)
df_schedule["N1_liftoff_binned"] = pd.cut(df_schedule.N1_liftoff, bin_sequence,labels=bin_sequence[:-1],retbins=False)
df_schedule["N1_liftoff_bin_definition"] = pd.cut(df_schedule.N1_liftoff, bin_sequence,labels=False,retbins=True)[1].tostring()

px.pie(df_schedule.N1_liftoff_binned, names="N1_liftoff_binned", color_discrete_sequence=color_discrete_sequence)

print(len(df_schedule))

# In[ ]:


detection_overview_plot = scatter_plot(df_schedule[df_schedule.liftoff_detection_quality_categorized.notna() == True], "liftoff_detection_quality_categorized")
detection_overview_plot.show()
plot_html = detection_overview_plot.to_html()
with open(filepath+'\\plot_detection_overview.html', 'w') as f:
    f.write(plot_html)


# In[ ]:


#use group to highlight the groups that merit further investigation
detection_plot_by_group = scatter_plot(df_schedule)
detection_plot_by_group.show()
plot_html = detection_plot_by_group.to_html()
with open(filepath+'\\plot_detection_by_group.html', 'w') as f:
    f.write(plot_html)


# In[ ]:


#plot relationship of actual N1 to detected ground roll distance for a given weight, weather, runway, aircraft type
#plot relationship of actual N1 to detected N1 for a given weight, weather, runway, aircraft type


# In[ ]:


flights_with_very_high_liftoff_detection_quality = df_schedule[df_schedule.liftoff_detection_quality_categorized == "very high"].index.to_series()
df_high_qual_lift = df.join(flights_with_very_high_liftoff_detection_quality,how = 'inner')
flights_with_poor_liftoff_detection_quality = df_schedule[df_schedule.liftoff_detection_quality_categorized == 'poor'].index.to_series()
df_poor_qual_lift = df.join(flights_with_poor_liftoff_detection_quality,how = 'inner')
flights_with_mediocre_liftoff_detection_quality = df_schedule[df_schedule.liftoff_detection_quality_categorized == 'mediocre'].index.to_series()
df_mediocre_qual_lift = df.join(flights_with_mediocre_liftoff_detection_quality,how = 'inner')


# In[ ]:


# =============================================================================
# fp = plot_metrics_for_individual_flights(df_high_qual_lift, flight_sample_size = 5)
# wb.open(url=fp)
# fp = plot_metrics_for_individual_flights(df_mediocre_qual_lift, flight_sample_size = 6)
# wb.open(url=fp)
# fp = plot_metrics_for_individual_flights(df_poor_qual_lift, flight_sample_size = 7)
# wb.open(url=fp)
# 
# =============================================================================

# In[ ]:


df_ml_tkoff = pd.read_csv("C:\\Users\\Lyle.Tripp\\OneDrive - DOT OST\\BADA4_Reduced_Thrust_Sensor_Path_Noise_Comparison_main\\FDR_desensitized\\Takeoff-Throttle-Neural-Net\\Out\\Visualizations\\Takeoff Distance Data Out.csv", index_col='FLIGHT_ID')
one_flight_id_of_interest = 1002395
df_one_flight = df.loc[one_flight_id_of_interest,]
df_one_flight["FLIGHT_ID"] = one_flight_id_of_interest
# plot_metrics_for_individual_flights(df_one_flight.reset_index().set_index(["FLIGHT_ID","TIME_OFFSET"]),df_ml, -1)
# plot_metrics_for_individual_flights(df_mediocre_qual_lift,df_ml_tkoff, -1)

df_ml_throttle = pd.read_csv("C:\\Users\\Lyle.Tripp\\OneDrive - DOT OST\\BADA4_Reduced_Thrust_Sensor_Path_Noise_Comparison_main\\FDR_desensitized\\Takeoff-Throttle-Neural-Net\\Out\\Visualizations\\Throttle Data Out.csv", index_col='FLIGHT_ID')
# df_ml_throttle[(df_ml_throttle.AIRCRAFT_TYPE == 'A330') & (df_ml_throttle.AIRPORT == 'OMDB')]

# In[plot error by airframe and engine combination ]
color_field_ordering_dict = airframe_engine_ordering = {'AIRCRAFT_TYPE':['A330-3_4PW067', 'A330-2_4PW067', 'A340-3_2CM015', 'A320-2_3CM021']}
fig_ml_ac_error_scatter = px.scatter(
    df_ml_throttle
    , x = 'N1'
    , y = 'PREDICTED_N1'
    , color = 'AIRCRAFT_TYPE'
    , opacity = 0.5
    , color_discrete_sequence= ['red'] + ['orange'] + ['blue'] + ['maroon']
    , category_orders = color_field_ordering_dict
    , labels = {'N1':'Actual N1 during takeoff', 'PREDICTED_N1': 'Predicted N1 during takeoff', 'AIRCRAFT_TYPE':'Airframe and Engine'}
    )

fig_ml_ac_error_scatter.add_trace(
    go.Scattergl(
        x=[75, 105],
        y=[75, 105],
        marker=dict(color="black"),
        name="Perfect fit",
        mode="lines"
        )
    )

html_fig_ml_ac_error_scatter = fig_ml_ac_error_scatter.to_html()
with open('aircraft_error_scatter.html', 'w') as file:
    file.write(html_fig_ml_ac_error_scatter)

# In[ ]:
#compare ML to regression
df_ml_compare = df_ml_throttle.loc[flights_of_interest,['TAKEOFF_GROUND_ROLL_DISTANCE_STATED_(FEET)', 'PREDICTED_N1']]
df_ml_compare.rename({'PREDICTED_N1' : 'N1_liftoff'}, axis = 1, inplace = True)
x = 'TAKEOFF_GROUND_ROLL_DISTANCE_STATED_(FEET)'
y = 'N1_liftoff'
df_ml_compare["Source"] = "ML v0 <br> (with regression line)"
df_highlight = pd.DataFrame(df_schedule.loc[flights_of_interest, [x, y] ])
df_highlight["Source"] = "CFDR <br> (with regression line)"
df_compare_sources = pd.concat([df_ml_compare, df_highlight])
# In[]:
comparison_fig = px.scatter(df_compare_sources.reset_index()
                            , x = x
                            , y = y
                            , color = "Source"
                            , color_discrete_sequence = ['rebeccapurple', 'springgreen']
                            , trendline='ols'
                            , symbol = "Source"
                            , symbol_sequence= ['circle', 'circle-open']
                            , opacity = 0.5
                            , hover_data = ['FLIGHT_ID']
                            , labels = {'N1_liftoff':'Throttle setting during takeoff: <br> Actual (green dots) vs Predicted (purple dots)', 'TAKEOFF_GROUND_ROLL_DISTANCE_STATED_(FEET)':'Distance in feet between rollstart and liftoff as stated in CFDR'}
                            )
# In[]:
comparison_fig.update_yaxes(range = [86,91])
comparison_fig.update_traces(marker=dict(size=11,
                              line=dict(width=3)),
                  selector=dict(mode='markers'))
comparison_html = comparison_fig.to_html()
with open('N1_distance_correlation_comparison.html', 'w') as file:
    file.write(comparison_html)

# In[ ]:


df_one_flight.reset_index().set_index(["FLIGHT_ID","TIME_OFFSET"]).to_csv("traj.csv")


# In[ ]:


#two flights with similar stated liftoff point 
flights = [1128101, 1124373]
plot_metrics_for_individual_flights(df.loc[flights,])


# =============================================================================
# # In[ ]:
# 
# 
# fig = px.scatter(
#     df_rollstart
#     , title='Start of takeoff roll, distance from runway end (feet)'
#     , x='DISTANCE_FROM_RUNWAY_END'
#     , y='DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF'
#     , color="APT_AIRCRAFT_RUNWAY_STAGE" 
#     , labels = {'DISTANCE_FROM_RUNWAY_END':'As stated in CFDR',
#                 'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF':'As detected from trajectory',
#                 'APT_AIRCRAFT_RUNWAY_STAGE': 'Airport / Aicraft Type / Runway / Stage Length ID'
#                }
# )
# 
# fig.update_yaxes(
#     scaleanchor = "x",
#     scaleratio = 1,
#   )
# 
# fig.show()
# 
# file = fig.to_html()
# with open(filepath+'\plot_rollstart.html', 'w') as f:
#     f.write(file)
# =============================================================================


# In[ ]:


fig_px = px.scatter(
    df_liftoff.reset_index()
    , title='Liftoff distance from runway end (feet)'
    , x='DISTANCE_FROM_RUNWAY_END'
    ,y='MSL_ALT'
    , color = "FLIGHT_ID"
    , labels = {'DISTANCE_FROM_RUNWAY_END':'Distance from runway end (feet)',
                'MSL_ALT':'Altitude above mean sea level (feet)',
                'APT_AIRCRAFT_RUNWAY_STAGE': 'Airport / Aicraft Type / Runway / Stage Length ID'
               }
)

file = fig_px.to_html()
with open(filepath+'\plot_high_quality_by_flight.html', 'w') as f:
    f.write(file)
    
fig_px.show()





# In[]
