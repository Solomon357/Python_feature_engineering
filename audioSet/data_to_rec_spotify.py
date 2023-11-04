import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import json
import re
import random
import sys
import itertools


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#spotify API *needs downloading*
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util




# p for percentage of records im going to read from file. 0.01 = 1% 
p = 0.01
#this reads all the data from the file 
original_df = pd.read_csv('audioSet/music-data/data.csv')

#FOR NOW LETS ONLY WORK WITH GENRE DATA
genre_df = pd.read_csv("audioSet/music-data/data_w_genres.csv")

# code for random subset below (might not use)
# skiprows= lambda i: i>0 and random.random() > p



#data_frame = data_frame.drop("Unnamed: 0", axis="columns")
#df.tail()

#print(df.head())
#print(genre_df.head())

#check to see what kinda datatypes we're working with

#also is a way to look at all column names since i cant see all of them on terminal
print(genre_df.dtypes)

#from the output we saw that genres and artists was classed as "object"
#checing to see if genres is acc in list format
#print(genre_df['genres'].values[0])


#print(genre_df['genres'].values[0][0]) #output: "["
#based on output this means that genre is a string 
#that is made to look like an object
# (if it was a string "values[0][0]" would of returned " 'show tunes' ")

#we define an updated list by using "regular expressions" or re for short
genre_df['genres_upd'] = genre_df['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])


#now we have a genre column that we can acc use 
print("Updated genre format: " + genre_df['genres_upd'].values[0][0])

## what we're doing next
#1. we're going to get a useful artist column with the same method as the genre column
#2. we're going to merge the data_w_genre dataset with the artist column of the data dataset


original_df['artists_upd_v1'] = original_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))

print("Updated artist format: " + original_df['artists_upd_v1'].values[0][0])

#checking if this works for every case
#print(df[df['artists_upd_v1'].apply(lambda x: not x)].head(5))

#doesn't work for all of them because of other special characters 
# not accounted for (e.g. " ' ")

#another regex to handle this
original_df['artists_upd_v2'] = original_df['artists'].apply(lambda x: re.findall('\"(.*?)\"', x))


original_df['artists_upd'] = np.where(original_df['artists_upd_v1'].apply(lambda x: not x), original_df['artists_upd_v2'], original_df['artists_upd_v1'])

#creating my own song identifier because there are duplicates of the same song with different ids
original_df['artists_song'] = original_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)

original_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)

#get rid of duplicates 
original_df.drop_duplicates('artists_song',inplace = True)

print(original_df[original_df['name']=='Adore You'])

#time to "explode" the artist column
artists_exploded = original_df[['artists_upd','id']].explode('artists_upd')

artists_exploded_enriched = artists_exploded.merge(genre_df, how = 'left', left_on = 'artists_upd',right_on = 'artists')
artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]

#print(artists_exploded_enriched_nonnull[artists_exploded_enriched_nonnull['id'] =='0MJZ4hh60zwsYleWWxT5yW'])

#Group by on the song id and essentially create lists lists
#Consilidate these lists and output the unique values
artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

#print(artists_genres_consolidated.head())

##  what the PROCESSED DATASET looks like (this will probably be what read into firestore)

#   id	                    genres_upd	                                            consolidates_genre_lists

#0	000G1xMMuwxNHmwVsBdtj1	[[candy_pop, dance_rock, new_romantic, new_wav...	    [new_romantic, candy_pop, power_pop, permanent...
#1	000ZxLGm7jDlWCHtcXSeBe	[[boogie-woogie, piano_blues, ragtime, stride]]	        [piano_blues, ragtime, stride, boogie-woogie]
#2	000jBcNljWTnyjB4YO7ojf	[[]]	                                                []
#3	000mGrJNc2GAgQdMESdgEc	[[classical, late_romantic_era], [historic_orc...	    [historic_orchestral_performance, classical, o...
#4	000u1dTg7y1XCDXi80hbBX	[[country, country_road, country_rock]]	                [country_road, country_rock, country]


#### 2. now that we've manipulated the data enough its time for feature engineering  

original_df = original_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')

#print(df.tail())

#### dunno how this changes release date yet
original_df['year'] = original_df['release_date'].apply(lambda x: x.split('-')[0])

float_cols = original_df.dtypes[original_df.dtypes == 'float64'].index.values

ohe_cols = 'popularity'
#### 

#print(df['popularity'].describe())

# create 5 point buckets for popularity 
original_df['popularity_red'] = original_df['popularity'].apply(lambda x: int(x/5))
# tfidf can't handle nulls so fill any null values with an empty list
original_df['consolidates_genre_lists'] = original_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])

#print(df.head())


#simple function to create OHE features
#this gets passed later on
def ohe_prep(df, column, new_name): 
    """ 
    Create One Hot Encoded features of a specific column

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        column (str): Column to be processed
        new_name (str): new column name to be used
        
    Returns: 
        tf_df stands for: term frequency_data frame
        tf_df: One hot encoded features 
    """
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df

#function to build entire feature set
def create_feature_set(df, float_cols):
    """ 
    Process spotify df to create a final set of features that will be used to generate recommendations

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        float_cols (list(str)): List of float columns that will be scaled 
        
    Returns: 
        final: final set of features 
    """
    
    #vectorising songs in tf_df
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(original_df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop = True, inplace=True)

    #explicity_ohe = ohe_prep(df, 'explicit','exp')    
    year_ohe = ohe_prep(df, 'year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15

    #scale float columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
     
    #add song id
    final['id']=df['id'].values
    
    return final

#THIS COMPLETE FEATURE SET IS WHAT WE HAVE WORKED TOWARDS
complete_feature_set = create_feature_set(original_df, float_cols=float_cols)#.mean(axis = 0)
print(complete_feature_set.head())


###########################################################################################

##### CODE FOR GENERATING RECOMMENDER


#client id and secret for my application
#will be unique for any person signing in to spotify 

#redirect_url = 'http://localhost:3000'

#as of writing this i do not know if i have to change ID and Secret for each user
# (im assuming you do have to though)
#SPOTIPY_CLIENT_ID = 'efe3ac326230422785eca822336069d9'
#hardcoding a password is very bad practice, DO NOT UPLOAD TO GITHUB WITHOUT FIXING THIS ISSUE
#SPOTIPY_CLIENT_SECRET= '110bc2e445a44599919b66cd3805a675'

client_id = 'efe3ac326230422785eca822336069d9'
client_secret='110bc2e445a44599919b66cd3805a675'
## below is a Sonarlint suggestion of how to encode the client secret
# def get_client_secret():

#     session = boto3.session.Session()
#     client = session.client(service_name='secretsmanager', region_name='eu-west-1')

#     return client.get_secret_value(SecretId='example_oauth_secret_id')

# client_secret = get_client_secret()

scope = "user-library-read"

#sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

#is this first condition saying that if more than one person is requesting then the username is the last person requesting? no idea
if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s Username" % (sys.argv[0],))
    #get rid of sys.exit() until i figure out why this is a bad move to make
    #sys.exit()

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)
token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://localhost:3000')
sp = spotipy.Spotify(auth=token)

#gather playlist names and images. 
#images aren't going to be used until I start building a UI
id_name = {}
list_photo = {}
for i in sp.current_user_playlists()['items']:

    id_name[i['name']] = i['uri'].split(':')[2]
    list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']

#print(id_name)


def create_necessary_outputs(playlist_name,id_dic, df):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        playlist_name (str): name of the playlist you'd like to pull from the spotify API
        id_dic (dic): dictionary that maps playlist_name to playlist_id
        df (pandas dataframe): spotify dataframe
        
    Returns: 
        playlist: all songs in the playlist THAT ARE AVAILABLE IN THE KAGGLE DATASET
    """
    
    #generate playlist dataframe
    playlist = pd.DataFrame()

    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        #print(i['track']['artists'][0]['name'])
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
        #for some reason below code returns "index out of range" error, comment out for now get to the bottom of this later
        #playlist.loc[ix, 'url'] = i['track']['album']['images'][0]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    
    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added',ascending = False)
    
    return playlist

#this is for a specific playlist in the user account (for now hardcode one of my playlist)
playlist_Throwback = create_necessary_outputs('Throwback', id_name,original_df)

print("This is all the songs that are present in the throwback playlist and the dataset:")
print(playlist_Throwback)

## creating playlist vector

def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    """ 
    Summarize a user's playlist into a single vector

    Parameters: 
        complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
        playlist_df (pandas dataframe): playlist dataframe
        weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 
        
    Returns: 
        playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        complete_feature_set_nonplaylist (pandas dataframe): 
    """
    
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
    
    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)

    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
        
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    #print(playlist_feature_set_weighted.iloc[:,:-4].columns)
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    #playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']
    
    return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist

complete_feature_set_playlist_vector_Throwback, complete_feature_set_nonplaylist_Throwback = generate_playlist_feature(complete_feature_set, playlist_Throwback, 1.09)
#complete_feature_set_playlist_vector_chill, complete_feature_set_nonplaylist_chill = generate_playlist_feature(complete_feature_set, playlist_chill, 1.09)

#to check if the playlist is vectorised
print(complete_feature_set_playlist_vector_Throwback.shape)


# actual song recommendation
def generate_playlist_recos(df, features, nonplaylist_features):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        df (pandas dataframe): spotify dataframe
        features (pandas series): summarized playlist feature
        nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Returns: 
        non_playlist_df_top_40: Top 40 recommendations for that playlist
    """
    
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_10 = non_playlist_df.sort_values('sim',ascending = False).head(10)
    non_playlist_df_top_10['url'] = non_playlist_df_top_10['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return non_playlist_df_top_10

Throwback_top10 = generate_playlist_recos(original_df, complete_feature_set_playlist_vector_Throwback, complete_feature_set_nonplaylist_Throwback)
print(Throwback_top10)