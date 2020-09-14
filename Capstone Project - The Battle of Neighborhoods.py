#!/usr/bin/env python
# coding: utf-8

# # 0. Import necessary libraries

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library


#!conda install -c conda-forge geopandas
#import geopandas as gpd

print('Libraries imported.')


# # 1. Import Dataset

# #### Description: 
# <p> This section is about importing the dataset correctly into a DataFrame.<p>
# 
# #### Important: 
# <p> The data contains 26 Swedish counties near and around the capital of Stockholm with a number of features. The data is from 2019 and compounded from official and open data of; 1. demographic data provided by SCB (Statistics Sweden), 2. crime data provided by BRA (Swedish National Council for Crime Prevention).<p>

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Import the locally stored CSV-file, containing demographic data from SCB and crime data from BRA, 
# for each county in the Stockholm region. Assign the csv-file to a DataFrame. 

county_data = pd.read_csv('/Users/mikaelhagglund/Desktop/Python/Coursera/IBM Data Science Professional Certificate/9. Applied Data Science Capstone/Capstone Project/CountyData2019.csv', delimiter = ';')
df_county = pd.DataFrame(county_data)
df_county.head()


# #### Comments on above findings: 
# <p> As seen in the above output, each county in the dataset has a number of parameters such as; lat/long koordinates, demograpic data (from SCB) and crime data (from BRA). </p>

# In[5]:


# What are the shape of the DataFrame?
df_county.shape


# # 2. Analyzing Individual Feature Patterns using Visualization

# #### Description: 
# <p> This section will begin the journey of understanding the data by using correlations, visualizations and other individual feature patterns. <p>

# In[6]:


# What are the data types of the DataFrame?
print(df_county.dtypes)


# In[8]:


# Calculating the correlation between the "int64" and "float64" parameters with a diagonal of 1.0
df_county.corr()


# #### Comment on above findings: 
# <p> The table above shows the correlation between all variables. There are a number of positive, negative and non-correlated relationships between the independent variables (demographic data) and the dependent variables (crime data). By looking through the correlation result in the table, I will choose a number of paired variables to explore further in the next section. <p> 

# ### 2.1 Individual Feature Exploration

# #### Description
# <p> This sub-section will do some individual feature exploration and visualization on a number of interesting features chosen based on the previous performed correlaion map of the DataFrame. <p>

# #### 2.1.1 "UnemploymentRate" as potential predictor variable of "TotalCrimesPer100000Capita"

# In[9]:


# "UnemploymentRate" as potential predictor variable of "TotalCrimesPer100000Capita"
sns.regplot(x="UnemploymentRate", y="TotalCrimesPer100000Capita", data=df_county)
plt.ylim(0,)


# In[10]:


# How well does "UnemploymentRate" correlate with "TotalCrimesPer100000Capita"?
df_county[["UnemploymentRate", "TotalCrimesPer100000Capita"]].corr()


# #### Comment on above findings: 
# <p> The scatterplot shows a positiv linear relationship between "UnemploymentRate" and "TotalCrimesPer100000Capita". The correlation confirms that. Meaning that when a county's unemployment rate 
# increases the reported total crimes (per 100.000 capita) tends to increase as well. <p> 

# #### 2.1.2 "CostPerCapita" as potential predictor variable of "TheftRobberyEtcPer100000Capita"

# In[11]:


# "CostPerCapita" as potential predictor variable of "TheftRobberyEtcPer100000Capita"
sns.regplot(x="CostsPerCapita", y="TheftRobberyEtcPer100000Capita", data=df_county)
plt.ylim(0,)


# In[16]:


# How well does "CostsPerCapita" correlate with "TheftRobberyEtcPer100000Capita"?
df_county[["CostsPerCapita", "TheftRobberyEtcPer100000Capita"]].corr()


# #### Comment on above findings: 
# <p> The scatterplot shows a negative linear relationship between "CostsPerCapita" and "TheftRobberyEtcPer100000Capita". The correlation confirms that. Meaning that when a county's cost per capita 
# increases the reported thefts, robberies and similar crimes (per 100.000 capita) tends to decrease as well.<p> 

# #### 2.1.3 "SupportedRate" as potential predictor variable of "ViolationLifeDeathPer100000Capita"

# In[17]:


# "SupportedRate" as potential predictor variable of "ViolationFreedomPeacePer100000Capita"
sns.regplot(x="SupportedRate", y="ViolationFreedomPeacePer100000Capita", data=df_county)
plt.ylim(0,)


# In[18]:


# How well does "SupportedRate" correlate with "ViolationLifeDeathPer100000Capita"?
df_county[["SupportedRate","ViolationLifeDeathPer100000Capita"]].corr()


# #### Comment on above findings: 
# <p> The scatterplot shows a strong positive linear relationship between "SupportedRate" and "ViolationLifeDeathPer100000Capita". The correlation confirms that. Meaning that when a county's supported rate
# increases the reported violation of life and death crimes (per 100.000 capita) tends to increase as well.<p> 

# #### 2.1.4 "UnemploymentRate" as potential predictor variable of "ViolationFreedomPeacePer100000Capita"

# In[19]:


# "UnemploymentRate" as potential predictor variable of "ViolationFreedomPeacePer100000Capita"
sns.regplot(x="UnemploymentRate", y="ViolationFreedomPeacePer100000Capita", data=df_county)
plt.ylim(0,)


# In[20]:


# How well does "UnemploymentRate" correlate with "ViolationFreedomPeacePer100000Capita"?
df_county[["UnemploymentRate", "ViolationFreedomPeacePer100000Capita"]].corr()


# #### Comment on above findings: 
# <p> The scatterplot shows a strong positive linear relationship between "UnemploymentRate" and "ViolationFreedomPeacePer100000Capita". The correlation confirms that. Meaning that when a county's unemployment rate increases the reported violation of freedom and peace crimes (per 100.000 capita) tends to increase as well.<p> 

# #### 2.2 Four crime features correlation to each other

# In[22]:


# How well does the five crime features correlate to each other?
df_county[["TotalCrimesPer100000Capita", "ViolationLifeDeathPer100000Capita", "ViolationFreedomPeacePer100000Capita", "TheftRobberyEtcPer100000Capita"]].corr()


# #### Comment on above findings: 
# <p> The correlations between the DataFrame's five crime features; 1. "TotalCrimesPer100000Capita", 2. "ViolationLifeDeathPer100000Capita", 3. "ViolationFreedomPeacePer100000Capita" and 4. "TheftRobberyEtcPer100000Capita" are shown above. 
#     
# The result shows, as can be expected, that there are high positiv correlation between 1. and 2., 3. and 4. respectively. Meaning that if 1. "TotalCrimesPer100000Capita" increases, each of the four other features tends to increase as well. That makes sense.
# 
# Interesting to note is that crime feature 4. has a low positive correlation with 2. and 3. Meaning that an increase of 4. "TheftRobberyEtcPer100000Capita" leads to a slightly increase of 2. and 3. but not as much as one could expect. <p> 

# # 3. Descrpitive Statistical Analysis

# #### Description: 
# <p> This section will begin performe some basic statistics on the DataFrame's variables. <p>

# In[23]:


# Apply the method describe on the DataFrame for some basic statistics.
df_county.describe()


# # 4. Correlation

# #### Description: 
# <p> This section will calculate the Pearson Correlation Coefficient and P-value between a number of chosen and paired variables. <p>
#     
#     
# <b>Pearson Correlation</b>:
# <p>The Pearson Correlation measures the linear dependence between two variables X and Y.</p>
# <p>The resulting coefficient is a value between -1 and 1 inclusive, where:</p>
# <ul>
#     <li><b>1</b>: Total positive linear correlation.</li>
#     <li><b>0</b>: No linear correlation, the two variables most likely do not affect each other.</li>
#     <li><b>-1</b>: Total negative linear correlation.</li>
# </ul>
# 
# 
# <b>P-value</b>: 
# <p>What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.</p>
# 
# By convention, when the
# <ul>
#     <li>p-value is $<$ 0.001: we say there is strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.05: there is moderate evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.1: there is weak evidence that the correlation is significant.</li>
#     <li>the p-value is $>$ 0.1: there is no evidence that the correlation is significant.</li>
# </ul>

# In[24]:


# Importing the necessary stats module in the scipy library.
from scipy import stats


# #### 4.1.1 "UnemploymentRate" vs. "TotalCrimesPer100000Capita"

# In[25]:


pearson_coef, p_value = stats.pearsonr(df_county["UnemploymentRate"], df_county["TotalCrimesPer100000Capita"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# #### Comment on above findings: 
# <p>Since the p-value is $<$ 0.05 it is moderate evidence that the correlation between UnemploymentRate and TotalCrimesPer100000Capita is statistically significant, although the linear relationship isn't extremely strong (~0.558)</p>

# #### 4.1.2 "CostsPerCapita" vs. "TheftRobberyEtcPer100000Capita"

# In[26]:


pearson_coef, p_value = stats.pearsonr(df_county['CostsPerCapita'], df_county['TheftRobberyEtcPer100000Capita'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# #### Comment on above findings: 
# <p>Since the p-value is $<$ 0.001 there is strong evidence that the correlation between CostsPerCapita and TheftRobberyEtcPer100000Capita is statistically significant, the linear relationship is quite strong (~-0.713)</p>

# #### 4.1.3 "SupportedRate" vs. "ViolationLifeDeathPer100000Capita"

# In[27]:


pearson_coef, p_value = stats.pearsonr(df_county['SupportedRate'], df_county['ViolationLifeDeathPer100000Capita'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# #### Comment on above findings: 
# <p>Since the p-value is $<$ 0.001 there is strong evidence that the correlation between SupportedRate and ViolationLifeDeathPer100000Capita is statistically significant, the linear relationship is strong (~0.818)</p>

# #### 4.1.4 "UnemploymentRate" vs. "ViolationFreedomPeacePer100000Capita"

# In[28]:


pearson_coef, p_value = stats.pearsonr(df_county['UnemploymentRate'], df_county['ViolationFreedomPeacePer100000Capita'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# #### Comment on above findings: 
# <p>Since the p-value is $<$ 0.001 there is strong evidence that the correlation between UnemploymentRate and ViolationFreedomPeacePer100000Capita is statistically significant, the linear relationship is strong (~0.759)</p>

# # 5. Folium map visualization

# #### Description: 
# <p> This section will use Folium to visualize the four crime variables to one map each. <p>

# In[29]:


# Import the GeoJson data stored locally.
sthlm_geo = r'/Users/mikaelhagglund/Desktop/Python/Coursera/IBM Data Science Professional Certificate/9. Applied Data Science Capstone/Capstone Project/region-stockholm-kommuner.geojson'


# #### 5.1 Folium map of "TotalCrimesPer100000Capita"

# In[101]:


# Create a Folium map over the 26 counties around Stockholm found in the DataFrame "df_county" 
# and the GeoJson "sthlm_geo". Grade the counties based on Total number of reported crimes 
# in 2019 per 100.000 inhabitants.

sthlm_map_TotalCrimes = folium.Map(location=[59.35,18.066667], zoom_start=9)
sthlm_map_TotalCrimes.choropleth(
    geo_data=sthlm_geo,
    name="Stockholm counties' total crimes in 2019",
    data=df_county,
    columns=['County','TotalCrimesPer100000Capita'],
    key_on='feature.properties.kom_namn',
    fill_color='YlOrRd', # ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’.
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name='Total number of reported crimes in 2019 per 100.000 inhabitants',
    highlight=True
)
sthlm_map_TotalCrimes


# #### Comment on above findings: 
# <p> The Folium map shows that the highest number of reported crimes in 2019 per 100.000 inhabitants are in county Stockholm (central), followed by Södertälje (south) and Sigtuna (north west). <p> 

# #### 5.2 Folium map of "TheftRobberyEtcPer100000Capita"

# In[102]:


# Create a Folium map over the 26 counties around Stockholm found in the DataFrame "df_county" 
# and the GeoJson "sthlm_geo". Grade the counties based on Total number of reported theft, 
# robbery etc. in 2019 per 100.000 inhabitants.

sthlm_map_TheftRobbery = folium.Map(location=[59.35,18.066667], zoom_start=9)
sthlm_map_TheftRobbery.choropleth(
    geo_data=sthlm_geo,
    name="Stockholm counties' total crimes in 2019",
    data=df_county,
    columns=['County','TheftRobberyEtcPer100000Capita'],
    key_on='feature.properties.kom_namn',
    fill_color='YlOrRd', # ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’.
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name='Total number of reported theft, robbery etc. in 2019 per 100.000 inhabitants',
    highlight=True
)
sthlm_map_TheftRobbery


# #### Comment on above findings: 
# <p> The Folium map shows that the highest number of reported thefts, robbery etc. in 2019 per 100.000 inhabitants is in county Stockholm (central). The other counties have relatively low rates within this category of crime. <p> 

# #### 5.3 Folium map of "ViolationLifeDeathPer100000Capita"

# In[103]:


# Create a Folium map over the 26 counties around Stockholm found in the DataFrame "df_county" 
# and the GeoJson "sthlm_geo". Grade the counties based on Total number of reported violation 
# of life and death in 2019 per 100.000 inhabitants.

sthlm_map_ViolationLifeDeath = folium.Map(location=[59.35,18.066667], zoom_start=9)
sthlm_map_ViolationLifeDeath.choropleth(
    geo_data=sthlm_geo,
    name="Stockholm counties' total crimes in 2019",
    data=df_county,
    columns=['County','ViolationLifeDeathPer100000Capita'],
    key_on='feature.properties.kom_namn',
    fill_color='YlOrRd', # ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’.
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name='Total number of reported violation of life and death in 2019 per 100.000 inhabitants',
    highlight=True
)
sthlm_map_ViolationLifeDeath


# #### Comment on above findings: 
# <p> The Folium map shows that the highest number of reported violations of life and death etc. in 2019 per 100.000 inhabitants is in county Södertälje (south) and Sigtuna (north west), closely followed by several other counties. <p> 

# #### 5.4 Folium map of "ViolationFreedomPeacePer100000Capita"

# In[104]:


# Create a Folium map over the 26 counties around Stockholm found in the DataFrame "df_county" 
# and the GeoJson "sthlm_geo". Grade the counties based on Total number of reported violation 
# of freedom and peace in 2019 per 100.000 inhabitants.

sthlm_map_ViolationFreedomPeace = folium.Map(location=[59.35,18.066667], zoom_start=9)
sthlm_map_ViolationFreedomPeace.choropleth(
    geo_data=sthlm_geo,
    name="Stockholm counties' total crimes in 2019",
    data=df_county,
    columns=['County','ViolationFreedomPeacePer100000Capita'],
    key_on='feature.properties.kom_namn',
    fill_color='YlOrRd', # ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’.
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name='Total number of reported violation of freedom and peace in 2019 per 100.000 inhabitants',
    highlight=True
)
sthlm_map_ViolationFreedomPeace


# #### Comment on above findings: 
# <p> The Folium map shows that the highest number of reported violations of freedom and peace etc. in 2019 per 100.000 inhabitants is in county Stockholm (central), Sigtuna (north west) and Norrtälje (north), closely followed by several other counties. <p> 

# # 6. Foursquare for map visualization

# In[49]:


# State the Foursquare credentials
CLIENT_ID = 'TYMBR2J4UJW51Q00HE4CLVLDBKJV1Y2GBOGJFYSGHL3QHKJ4' # my Foursquare ID
CLIENT_SECRET = 'JL5U2MSD5RLVB2TZYSHK0P5NDH5PXUTDBRAERJQUU414ROPN' # my  Foursquare Secret
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[50]:


# Call Foursquare and generate the result basedo n a number of conditions
latitude = 59.35 # latitude coordinate of Stockholm
longitude = 18.066667  # longitude coordinate of Stockholm
LIMIT = 100 # limit of number of venues returned by Foursquare API
categoryId = '4bf58dd8d48988d12e941735' # the Foursquare categoryID for Police Departments
radius = 50000 # define radius
url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&categoryId={}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET,  
    latitude, 
    longitude, 
    VERSION,
    categoryId,
    radius, 
    LIMIT)
url # display URL


# In[51]:


# Display the result received by Foursquare
results = requests.get(url).json()
results


# In[52]:


# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a dataframe
df_policestations_foursquare = pd.json_normalize(venues)
df_policestations_foursquare


# In[53]:


# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in df_policestations_foursquare.columns if col.startswith('location.')] + ['id']
df_policestations_filtered = df_policestations_foursquare.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
df_policestations_filtered['categories'] = df_policestations_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
df_policestations_filtered.columns = [column.split('.')[-1] for column in df_policestations_filtered.columns]

df_policestations_filtered


# In[76]:


# Some manipulating and cleaning of the data.

df_sthlmpolices_cleaned = df_policestations_filtered.drop([0, 1, 3,  7 ,  8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21,  23, 24, 25, 26,  28,   31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,  46, 47, 
                                                        49], axis=0)
df_sthlmpolices_cleaned = df_sthlmpolices_cleaned.reset_index(drop=True)
df_sthlmpolices_cleaned["name"] = df_sthlmpolices_cleaned["name"].replace(["Stockholms City passexpedition"], "Polisen Stockholms City Norrmalm")
df_sthlmpolices_cleaned["name"] = df_sthlmpolices_cleaned["name"].replace(["Polisen Södermalm"], "Stockholms City Södermalm")
df_sthlmpolices_cleaned["name"] = df_sthlmpolices_cleaned["name"].replace(["Polishuset Flemingsberg"], "Polisen Flemingsberg")
df_sthlmpolices_cleaned["name"] = df_sthlmpolices_cleaned["name"].replace(["Polishuset"], "Polisen Sollentuna")
df_sthlmpolices_cleaned


# #### 6.1 Foursquare data of police stations added to "TotalCrimesPer100000Capita" folium map

# In[99]:


# add the Police stations as black circle markers
for lat, lng, label in zip(df_sthlmpolices_cleaned.lat, df_sthlmpolices_cleaned.lng, df_sthlmpolices_cleaned.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color= 'black',
        popup=label,
        fill = True,
        fill_color='black',
        fill_opacity=0.6
    ).add_to(sthlm_map_TotalCrimes)

# display map
sthlm_map_TotalCrimes


# #### 6.2 Foursquare data of police stations added to "TheftRobberyEtcPer100000Capita" folium map

# In[93]:


# add the Police stations as black circle markers
for lat, lng, label in zip(df_sthlmpolices_cleaned.lat, df_sthlmpolices_cleaned.lng, df_sthlmpolices_cleaned.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color= 'black',
        popup=label,
        fill = True,
        fill_color='black',
        fill_opacity=0.6
    ).add_to(sthlm_map_TheftRobbery)

# display map
sthlm_map_TheftRobbery


# #### 6.3 Foursquare data of police stations added to "ViolationLifeDeathPer100000Capita" folium map

# In[94]:


# add the Police stations as black circle markers
for lat, lng, label in zip(df_sthlmpolices_cleaned.lat, df_sthlmpolices_cleaned.lng, df_sthlmpolices_cleaned.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color= 'black',
        popup=label,
        fill = True,
        fill_color='black',
        fill_opacity=0.6
    ).add_to(sthlm_map_ViolationLifeDeath)

# display map
sthlm_map_ViolationLifeDeath


# #### 6.4 Foursquare data of police stations added to "ViolationFreedomPeacePer100000Capita" folium map

# In[95]:


# add the Police stations as black circle markers
for lat, lng, label in zip(df_sthlmpolices_cleaned.lat, df_sthlmpolices_cleaned.lng, df_sthlmpolices_cleaned.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color= 'black',
        popup=label,
        fill = True,
        fill_color='black',
        fill_opacity=0.6
    ).add_to(sthlm_map_ViolationFreedomPeace)

# display map
sthlm_map_ViolationFreedomPeace

