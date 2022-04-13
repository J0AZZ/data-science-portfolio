import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from collections import deque
import pandas as pd

WINDOW_SIZE = 50 #tamanho da janela
WINDOW_DISTANCE = 25 #distancia entre o inicio de uma janela e o inicio da proxima

dataset = pd.read_csv('DataBase/CSV/chordlibrary.csv')

dataset_a1 = pd.DataFrame()
dataset_a2 = pd.DataFrame()
dataset_a3 = pd.DataFrame()
dataset_a4 = pd.DataFrame()


def window(seq, n=WINDOW_SIZE):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

num_musicas = len(os.listdir('DataBase/Audio'))

for i in range(0, num_musicas): #repita para cada musica

    music_name = os.listdir('DataBase/Audio')[i]
    if music_name[num_musicas-5:] == ".wav":
        music_name = music_name[:num_musicas - 5]
    elif music_name[num_musicas-6:] == ".wav":
        music_name = music_name[:num_musicas - 6]

    data, fs = librosa.load(f'DataBase/Audio/{music_name}', sr = None) #transformar musica para array de numpy

    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)))# troca amplitude por decibel

    n = D.shape[0]
    yout = librosa.fft_frequencies(sr=fs ,n_fft=1+(2 * (n - 1)) ) #intervalos do eixo y(frequencia)

    m = D.shape[1]
    hop_length=512

    xout = librosa.frames_to_time(np.arange(m+1), sr=fs, hop_length=hop_length) #intervalos do eixo x(tempo)

    music_name = music_name.split('.')[0]

    print('\n-----------------------------------------------------------------------------------')
    print(f'Musica: {music_name}')
    print(f'Duracao: {xout[-1]}')
    print(f'Dimensões(f/t): {D.shape}')
    print('-----------------------------------------------------------------------------------\n')

    x1 = WINDOW_DISTANCE
    a = 0


    for w in window(xout, WINDOW_SIZE): # repete para cada janela na música
        lista = []

        if x1 >= WINDOW_DISTANCE: #considera apenas janelas de 50 em 50 de distancia (aproximadamente 0.5 segundos)

            w_max = w[-1] #ultimo tempo do intervalo w
            w_min = w[0] #primeiro tempo do intervalo w

            data_new = dataset.loc[dataset['title'] == music_name]
            data_new = data_new.loc[data_new['annotator'] == 'A4'] #

            data_time = data_new.loc[data_new['start'] >= w_min]
            data_time = data_time.loc[data_time['start'] < w_max]

            chords = []

            for nota in data_time["classes"]: #Percorre e salva todas as notas por janela
                #print(nota)

                chords.append(nota)

            if len(chords) == 1: #se possui apenas um acorde, nao muda
                status = 0
            elif len(set(chords)) == 1: #se todos acordes sao iguais, nao muda
                status = 0
            else: #caso nao seja nenhum dos casos, troca
                status = 1
            #if music_name != 'I Want A New Drug.wav' or a/50 != 1071.0:
            lista.append([a/WINDOW_DISTANCE,music_name, w_min, status])
            dataset_a4 = dataset_a4.append(lista) #
                
            x1 = 0
            a += WINDOW_DISTANCE
        x1+=1 

dataset_a4 = dataset_a4.rename(columns={0: 'Janela', 1: 'title', 2:'begin', 3:'status'}) #
dataset_a4.to_csv('DataBase/rotulos/status.csv', index=False) #
        
