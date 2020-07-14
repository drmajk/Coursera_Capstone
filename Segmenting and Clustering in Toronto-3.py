#!/usr/bin/env python
# coding: utf-8

# ## Assignment 1: Create dataframe by scraping Wikipedia page

# ### Import necessary libraries

# In[28]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
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

print('Libraries imported.')


# ### Open Wikipedia page and assign it to variable

# In[29]:


import urllib.request
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
page = urllib.request.urlopen(url)


# ### Use the library BeautifulSoup to extract the table

# In[30]:


from bs4 import BeautifulSoup
# Use library "BeautifulSoup" and see how the page is divided into sections. 
# The class "wikitable sortable" is the table to be scraped
soup = BeautifulSoup(page, "lxml")
print(soup.prettify())


# ### Find the right table and assign it to a variable

# In[31]:


# Pick the right table from the page and scrap it using BeautifulSoup and then print out the table
right_table = soup.find("table", class_ = "wikitable sortable")
right_table


# ### Append the table value to a list

# In[32]:


# Create three empty lists, one for each column in the scrpaed table. 
#Then for each row, append the value from each column in each one of the lists.

A = []
B = []
C = []

#PostalCode = []
#Borough = []
#Neighborhood = []

for row in right_table.findAll('tr'):
    cells = row.findAll('td') 
    #print(cells)
    if len(cells) == 3:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))
        #print(cells[1])
        #if cells[1] != "Not assigned" or cells[2] != "Not assigned":


# ### Create a DataFrame and clean the values

# In[33]:


#Create a DataFrame with the values from the tree lists, with some cleaninig of each cell.
df = pd.DataFrame(A, columns=["PostalCode"])
df['Borough'] = B
df['Neighborhood'] = C
df['PostalCode'] = df['PostalCode'].str.replace(r'\n', '')
df['Borough'] = df['Borough'].str.replace(r'\n', '')
df['Neighborhood'] = df['Neighborhood'].str.replace(r'\n', '')
df


# ### Do some more cleaning in the Borough column

# In[34]:


# Delete rows containing a not assigned borough. Sort the list on "PostalCode" then reset the index. 
df = df[(df['Borough'] != "Not assigned")]
df_sorted = df.sort_values(by=['PostalCode'])
df_sorted.reset_index(drop=True, inplace=True)
df_sorted


# ### Check the DataFrame shape

# In[35]:


# Print out the shape of the final table.
df_sorted.shape


# ## Assignmet 2: Longitude and Latitude of each Borough

# ### Download the geo coordinates and add them to the right PostalCode

# In[36]:


# Download the csv file into a Dataframe
geo_data = pd.read_csv("http://cocl.us/Geospatial_data")
df_geo = pd.DataFrame(geo_data)
df_geo

# Add the two columns "longitud" and "latitude" to the PostalCode-sorted Dataframe from assignment 1
df_sorted['Latitude'] = geo_data['Latitude']
df_sorted['Longitude'] = geo_data['Longitude']
df_sorted


# ### Check so that it gives the right coordinate values för a certain PostalCode

# In[37]:


# Check that my result pf Postalcode M5G is the same as from the assignment description on Coursera
df_sorted.loc[df_sorted['PostalCode'] == 'M5G']


# ### Check the DataFrame shape

# In[38]:


# Check that the new Dataframe has the same shape as before adding the two columns
df_sorted.shape


# # Assignment 3: Explore and cluster neighborhoods in Toronto

# ## Assignment 3.1: Create Toronto map and add Boroughs' data

# ### Filter the dataset and retreive all Boroughs containing "Toronto"

# In[74]:


toronto_data = df_sorted[df_sorted['Borough'].str.contains("Toronto")].reset_index(drop=True)
#toronto_data.head()

# Get the geographical coordinates of Toronto.
address = 'Toronto'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# ### Create map of Toronto using latitude and longitude values

# In[75]:


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map of Boroughs containing "Toronto" in their name
for lat, lng, borough, neighborhood in zip(toronto_data['Latitude'], toronto_data['Longitude'], toronto_data['Borough'], toronto_data['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# ### Check the first ten rows

# In[41]:


# Show the top 10 Boroughs
toronto_data.head(11)


# ### Check number of unique Boroughs and te shape of the dataset

# In[42]:


toronto_data.Borough.unique()
toronto_data.shape


# ### Add Foursquare credentials

# In[43]:


CLIENT_ID = 'secret information' # your Foursquare ID
CLIENT_SECRET = 'secret information' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version


# ## Assignment 3.2: Explore neighborhood in Toronto

# ### A function to repeatly get tje venues information for all the neighborhoods in "toronto_data"

# In[53]:


import json # library to handle JSON files
import requests # library to handle requests

LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # set a radius


# A function that recall the Foursquare API and gets the venue information for the neighborhoods 
# for each Borough based on the LIMIT and radius set above. 
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# ### Executes the above function for each Borough in the dataset "toronto_data"

# In[54]:


toronto_venues = getNearbyVenues(names=toronto_data['Neighborhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# ### Prints out the top 5 rows of the resulted dataframe

# In[55]:


print(toronto_venues.shape)
toronto_venues.head()


# ### Check how many venues returned for each neighborhood

# In[56]:


toronto_venues.groupby('Neighborhood').count()


# ### Find out how many unique categories can be curated from all the returned venues

# In[57]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ## Assignment 3.3: Analyze Each Neighborhood

# ### Create a dataframe with all venue categories and each neighborhood with 

# In[58]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# ### Check the shape of the new dataframe

# In[59]:


toronto_onehot.shape


# ### Group rows by neighborhood by taking the mean of the frequency of occurrence of each category

# In[60]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# ### Check the new size

# In[61]:


toronto_grouped.shape


# ### Print each neighborhood along with the top 5 most common venues

# In[62]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ### A function to sort the venues in descending order

# In[63]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# ### Create the new dataframe and display the top 10 venues for each neighborhood

# In[64]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# # Assignment 3.4: Cluster Neighborhoods

# ### Run *k*-means to cluster the neighborhood into 5 clusters.

# In[65]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# ### Create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[66]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# ### Visualize the resulting clusters

# In[67]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ### Show each cluster with its associated Borough and venues

# In[69]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[70]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[71]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[72]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[73]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:




