import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from collections import deque
import pandas as pd


def window(seq, n=100):
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

    if 1:
      data, fs = librosa.load(f'DataBase/Audio/{music_name}', sr = None)

      D = librosa.amplitude_to_db(np.abs(librosa.stft(data)))

      n = D.shape[0]
      yout = librosa.fft_frequencies(sr=fs ,n_fft=1+(2 * (n - 1)) )

      m = D.shape[1]
      hop_length=512

      xout = librosa.frames_to_time(np.arange(m+1), sr=fs, hop_length=hop_length)

      print('\n-----------------------------------------------------------------------------------')
      print(f'Musica: {music_name}')
      print(f'Duracao: {xout[-1]}')
      print(f'Dimensões(f/t): {D.shape}')
      print('-----------------------------------------------------------------------------------\n')

      window_decibel = np.zeros((n,100))

      x1 = 50
      a = 0

      for w in window(xout, 100):
          status = []
          music = []

          if x1 >= 50:

              for k in range(0, n):
                  for l in range(0, len(w)):
                      window_decibel[k][l] = D[k][l + a]

              plt.figure(figsize=(20,10))
              librosa.display.specshow(window_decibel, x_axis='time', y_axis='log', sr= fs, cmap = 'jet')
              plt.xlabel("Time [s]")
              plt.ylabel("Frequency [Hz]")
              plt.colorbar(format = "%+2.0f db")
              plt.savefig(f'DataBase/Windows/{music_name}{a/50}.png', dpi = 70)
              plt.close()
              
              a += 50
              x1 = 0

          x1 += 1