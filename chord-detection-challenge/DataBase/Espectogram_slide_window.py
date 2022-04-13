import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

num_musicas = len(os.listdir('DataBase/Audio'))

for i in range(0, num_musicas):

    music_name = os.listdir('DataBase/Audio')[i]
    music_name = music_name[:num_musicas - 5]

    data, fs = librosa.load(f'DataBase/Audio/{music_name}', sr = 44100)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)))

    plt.figure(figsize=(20,10))
    librosa.display.specshow(D, x_axis='time', y_axis='log', sr= fs, cmap = 'jet')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(format = "%+2.0f db")

    plt.savefig(f'DataBase/Espectogramas/{music_name}.png', dpi = 300)
    plt.close()