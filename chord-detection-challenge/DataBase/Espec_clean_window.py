import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from collections import deque
import pandas as pd

WINDOW_SIZE = 50
WINDOW_DIST = 25

fail = []

flag = False

def filter(song, sr):
  D = librosa.stft(song)
  D_harmonic, D_percussive = librosa.decompose.hpss(D)
  S = D_harmonic
  s, phase = librosa.magphase(S)
  S_filter = librosa.decompose.nn_filter(s, 
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
  S_filter = np.minimum(s, S_filter)
  i_mask = librosa.util.softmask(S_filter,
                                2* (s - S_filter),
                                power=2)
  pure_instrumental = s * i_mask
  return pure_instrumental

def window(seq, n=WINDOW_SIZE):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

num_musicas = len(os.listdir('DataBase/Audio'))

for i in range(0, num_musicas):

    music_name = os.listdir('DataBase/Audio')[i]
    if music_name[num_musicas-5:] == ".wav":
        music_name = music_name[:num_musicas - 5]
    elif music_name[num_musicas-6:] == ".wav":
        music_name = music_name[:num_musicas - 6]

    if music_name == 'I Want A New Drug.wav' or flag == True:
        flag = True
        data, sr = librosa.load(f'DataBase/Audio/{music_name}', sr = None)

        D = filter(data, sr)

        n = D.shape[0]
        yout = librosa.fft_frequencies(sr=sr ,n_fft=1+(2 * (n - 1)) )

        m = D.shape[1]
        hop_length=512

        xout = librosa.frames_to_time(np.arange(m+1), sr=sr, hop_length=hop_length)

        print('\n-----------------------------------------------------------------------------------')
        print(f'Musica: {music_name}')
        print(f'Duracao: {xout[-1]}')
        print(f'Dimensões(f/t): {D.shape}')
        print('-----------------------------------------------------------------------------------\n')

        window_decibel = np.zeros((n,WINDOW_SIZE))

        x1 = WINDOW_DIST
        a = 0

        for w in window(xout, WINDOW_SIZE):

            if x1 >= WINDOW_DIST:
                try:

                    for k in range(0, n):
                        for l in range(0, len(w)):
                            window_decibel[k][l] = D[k][l + a]

                    plt.figure(figsize=(20,10))
                    librosa.display.specshow(librosa.amplitude_to_db(window_decibel), x_axis='time', y_axis='log', sr= sr, cmap = 'jet')
                    plt.xlabel("Time [s]")
                    plt.ylabel("Frequency [Hz]")
                    plt.colorbar(format = "%+2.0f db")
                    plt.savefig(f'DataBase/new_windows/{music_name}{a/WINDOW_DIST}.png', dpi = 70)
                    plt.close()
                    
                    a += WINDOW_DIST
                    x1 = 0

                except:
                    fail.append(f'{music_name}{a/WINDOW_DIST}.png')
                    print(f'Failed to save: {music_name}{a/WINDOW_DIST}.png')

            x1 += 1
            
print(fail)