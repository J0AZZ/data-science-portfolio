import jams
import json
import pandas as pd
import numpy as np
import os

times = []
titles = []
lists = []
name = []

jam_path = 'DataBase/jams'
num_jams = len(os.listdir(jam_path))

for j in os.listdir(jam_path):
    jam = jams.load(f'{jam_path}/{j}') # Load Jam Path
    meta = jam['file_metadata']
    title = meta['title']
    metadata = meta['identifiers']
    youtube = metadata['youtube_url']
    duration = meta['duration']
    duration = duration/60 # Calculate duration
    nome = str(j)        
    name.append(nome.split('.')[0]) # Split the archive name
    times.append(duration) # Add duration
    lists.append(youtube) # Add link
    titles.append(title) # Add title
            

data = pd.DataFrame()
data['id'] = name
data['link'] = lists
data['title'] = titles 
data['time'] = times
data.to_csv('DataBase/youtube.csv', index=True)    
print(data)
