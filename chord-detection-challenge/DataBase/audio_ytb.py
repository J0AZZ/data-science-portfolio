from __future__ import unicode_literals
import youtube_dl
import numpy as np
import pandas as pd

dataset = pd.read_csv('DataBase/youtube.csv')

for i in range(0, len(dataset)):
    try:

        titulo = dataset.loc[i]['title']

        ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': f'DataBase/Audio/{titulo}'+'.wav',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
    }],
    'postprocessor_args': [
        '-ar', '16000'
    ],
    'prefer_ffmpeg': True,
    'keepvideo': False
}

        link = dataset.loc[i]['link']
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'{link}'])
    except:
        continue
        