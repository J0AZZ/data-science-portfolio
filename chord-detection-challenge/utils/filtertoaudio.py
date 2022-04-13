import librosa, librosa.display
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
import os

def filter(song, sr): 
  D = librosa.stft(song)
  D_harmonic, D_percussive = librosa.decompose.hpss(D)
  S = D_harmonic
  s, phase = librosa.magphase(D)
  S_filter = librosa.decompose.nn_filter(s, 
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
  S_filter = np.minimum(s, S_filter)
  i_mask = librosa.util.softmask(S_filter,
                                2* (s - S_filter),
                                power=2)
  pure_instrumental = s * i_mask
  T = librosa.griffinlim(np.abs(pure_instrumental)) # Model np.array to convert to audio
  return (pure_instrumental, T)



y, sr  = librosa.load("Last Nite.wav")

# y, sr = librosa.load(librosa.ex('fishin'))

pure_instrumental = filter(y, sr)
T = pure_instrumental[1]

write("Last Nite(instrumental).wav", sr, T)
