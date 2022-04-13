import jams
import json
import pandas as pd
import numpy as np
import os


path = 'DataBase/youtube.csv'
data = pd.read_csv(path)
data = pd.DataFrame(data)

#print(len(data.index))
code = [] # Keeps chord id
music = [] # Keeps music title
classes = [] # Has the chords Symbols
duration = [] # Chord duration 
times = [] # Chord Time
annotator = [] # Annotator id

counter = 0
for pathID in data["id"]:


    jam_path = 'DataBase/jams/{}.jams'.format(pathID)
    jam = jams.load(jam_path)
#num_jams = len(os.listdir(jam_path))
#print(jam['annotations'][0]['annotation_metadata']['annotator']['id'])

    for index in range(len(jam["annotations"])): # Navigate looking for annotations index and data items
        for item in range(len(jam["annotations"][index]["data"])):
            code.append(item) # Chords id
            music.append(data.loc[counter]["title"]) # Titles in jam by pathID
            duration.append(jam["annotations"][index]["data"][item][1]) # Chords duration in annotation x
            classes.append(jam["annotations"][index]["data"][item][2]) # Chords in annotation x
            times.append(jam["annotations"][index]["data"][item][0]) # Chords times
            annotator.append(jam['annotations'][index]['annotation_metadata']['annotator']['id']) # Annotator id
                
    counter+=1

song = pd.DataFrame()
song['id'] = code
song['title'] = music 
song['classes'] = classes
song['start'] = times
song['duration'] = duration
song['annotator'] = annotator
print(song)
song.to_csv('DataBase/chordlibrary.csv', index=True)    
# print(data)