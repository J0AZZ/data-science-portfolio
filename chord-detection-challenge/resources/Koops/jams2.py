import jams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#jam_path = '/jams/159.jams'
jam_path = 'DataBase/jams/1235.jams'

jam = jams.load(jam_path)

cor = ('Blue','Red','Green','Black','Purple')

tempo = []
notas = []

i = 0;

for person in jam['annotations']:
    for chord_info in person['data']:
        tempo.append(chord_info[0])
        notas.append(chord_info[2])
    df = pd.DataFrame({"Tempo": tempo,"Notas": notas})
    df.plot(kind ='scatter', x='Tempo', y='Notas',color = cor[i], alpha = 0.4, s = 10, cmap = plt.get_cmap('jet'), figsize = (5,5))
    df.drop(columns=['Tempo'])
    df.drop(columns=['Notas'])
    tempo = []
    notas = []
    i+=1

plt.show()

print(df)