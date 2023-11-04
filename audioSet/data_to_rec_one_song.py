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

#user choice in app will essentially be picking a single row (via filter search) and then in the background
#we pick the 20 most similar songs based on the features of the user song

#this is a series btw rows count as series (and therefore Vectors?)
mock_user_input = original_df.iloc[5934]
#artist name should be: "Xristos Kontopoylos"
print(mock_user_input.values)

# this line of code get the completed feature list of the mock user input (i assume this means all the 0.0s in the feature df will acc have numbers only for the id of mock_user_input) 
complete_feature_set_song = complete_feature_set[complete_feature_set['id'].isin(mock_user_input[['id']].values)]#.drop('id', axis = 1).mean(axis =0)

#we merge with the user input. (i think because user input is a single row )
complete_feature_set_song = complete_feature_set_song.merge(mock_user_input[['id', 'release_date']].to_frame(), on = 'id', how = 'inner')
#complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
print(complete_feature_set_song.head()) 

# iirc "non_playlist_df" is here so i differentiate between songs that are already 
# in spotify playlist compared to songs that are not so it doesnt recommend the same songs.
complete_feature_set_different_songs = complete_feature_set[~complete_feature_set['id'].isin(mock_user_input[['id']].values)]
print(complete_feature_set_different_songs.head())

# AXIS = 1: COLUMNS
# AXIS =0: ROWS (until proven otherwise)

#complete_feature_set['id'] != mock_user_input['id']]#.drop('id', axis = 1)

"""
The whole idea for this work specifically is that we used the spotify API
to get the raw data of a users spotify playlist and turn it into a useful input
in order to plug it into our recommender function and get a list of similar songs 
related to the user playlist

the code i want to make is alot simpler when i think about it, once i can get 
a usable dataframe all i need to do is get the user input (which will have its own vector)
and return 20 songs that have the most similar vector to the users song


"""
#print(original_df)

def generate_song_recos(df, user_input, different_songs_features):
    """ 
    generate df with similar songs to user input.

    Parameters: 
        df (pandas dataframe): spotify dataframe
        user_input (pandas series): summarized song features
        different_song_features (pandas dataframe): feature set of songs that are not the user input
        
    Returns: 
        non_playlist_df_top_40: Top 40 recommendations for that playlist
    """
    
    different_songs_df = df[df['id'].isin(different_songs_features['id'].values)]
    #print(different_songs_df)
    #Value Error issue might be here v
    # i might be missing a line that converts 'artist_upd' to 'artists', in source code the output uses spotify api so artist value can already be converted
    different_songs_df['sim'] = cosine_similarity(different_songs_features.drop(['id', 'artists'], axis = 1).values, user_input.values.reshape(1, -1))[:,0]
    print(different_songs_df['sim'])
    different_songs_df_top_20 = different_songs_df.sort_values('sim',ascending = False).head(20)
    
    return different_songs_df_top_20



#song_rec_top20 = generate_song_recos(original_df, mock_user_input, complete_feature_set_different_songs)

#Value error: can't convert 'string'(artist column is string) to 'float'. the output shouldn't care what type each column is look into it
#print(song_rec_top20)

