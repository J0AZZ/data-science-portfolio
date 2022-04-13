#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Particionando-bases-de-treino-e-teste-com-split-70-30%" data-toc-modified-id="Particionando-bases-de-treino-e-teste-com-split-70-30%-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Particionando bases de treino e teste com split 70-30%</a></span></li><li><span><a href="#Particionando-bases-de-treino-e-teste-com-diferentes-músicas" data-toc-modified-id="Particionando-bases-de-treino-e-teste-com-diferentes-músicas-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Particionando bases de treino e teste com diferentes músicas</a></span></li></ul></div>

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from imblearn.under_sampling import RandomUnderSampler


# In[2]:


tf.__version__


# In[3]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[4]:


df = pd.read_csv('/home/thais/CSV/status_A1.csv')
df.sort_values(by='title', inplace=True)
images_path = sorted(glob.glob('/home/thais/Windows/Train/*'))
df['Unnamed: 0'] = np.array(images_path)
df.reset_index(inplace=True, drop=True)
df['status'] = df['status'].astype(str)


# ### Particionando bases de treino e teste com split 70-30%

# In[5]:


### particiona considerando 70-30% e mantendo a frequência de amostras para treino, validação e teste de acordo com as colunas título e status (rótulo da rede)

X_train, X_test, y_train, y_test = train_test_split(df[['Unnamed: 0', 'title']], df['status'], test_size=0.30, random_state=42, stratify=df[['status', 'title']])
df_train = pd.concat([X_train, y_train], axis=1)
X_train, X_val, y_train, y_val = train_test_split(df_train[['Unnamed: 0', 'title']], df_train['status'], test_size=0.30, random_state=42, stratify=df_train[['status', 'title']])

### contatena atributos de entrada e rótulo em um único dataframe para utilizar o flow_from_dataframe do tensorflow
df_test = pd.concat([X_test, y_test], axis=1)
df_train = pd.concat([X_train, y_train], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)

print('Total de imagens de treinamento', len(df_train))
print('Total de imagens de validação', len(df_val))
print('Total de imagens de teste', len(df_test))


# In[6]:

'''
undersample_train = RandomUnderSampler(sampling_strategy='majority')
undersample_validation = RandomUnderSampler(sampling_strategy='majority')

X_undertrain, y_undertrain = undersample_train.fit_resample(df_train[['Unnamed: 0', 'title']], df_train['status'])
X_undervalidation, y_undervalidation = undersample_validation.fit_resample(df_val[['Unnamed: 0', 'title']], df_val['status'])


# In[7]:


df_train = pd.concat([X_undertrain, y_undertrain], axis=1)
df_val = pd.concat([X_undervalidation, y_undervalidation], axis=1)
'''

# ### Particionando bases de treino e teste com diferentes músicas

# In[8]:

'''
songs, n = df['title'].unique(), 5
index = np.random.choice(len(songs), 5, replace=False)  
selected_songs = songs[index] ## seleciona n músicas disponíveis para teste
df_test = df[df['title'].isin(selected_songs)] ## banco de teste contém todos os espectrogramas das n músicas selecionadas anteriormemente
df_train = df[~(df['title'].isin(selected_songs))] ## banco de treino contém os espectrogramas de todas as músicas EXCETO as selecionadas anteriormente para teste

X_train, X_val, y_train, y_val = train_test_split(df_train[['Unnamed: 0', 'title']], df_train['status'], test_size=0.30, random_state=42, stratify=df_train[['status', 'title']]) ## divide em validação considerando 30% e balanceamento de acordo com título e status

### contatena atributos de entrada e rótulo em um único dataframe para utilizar o flow_from_dataframe do tensorflow
df_train = pd.concat([X_train, y_train], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)

print('Total de imagens de treinamento', len(df_train))
print('Total de imagens de validação', len(df_val))
print('Total de imagens de teste', len(df_test))
'''

# In[8]:


datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=df_train, directory='/home/thais/Windows/Train/', x_col='Unnamed: 0', y_col="status", class_mode="binary", target_size=(224,224), batch_size=32)
valid_generator=datagen.flow_from_dataframe(dataframe=df_val, directory='/home/thais/Windows/Train/', x_col='Unnamed: 0', y_col="status", class_mode="binary", target_size=(224,224), batch_size=32)
test_generator=datagen.flow_from_dataframe(dataframe=df_test, directory='/home/thais/Windows/Train/', x_col='Unnamed: 0', y_col="status", class_mode="binary", target_size=(224,224), batch_size=1,shuffle=False)


# In[9]:


#from tensorflow.keras.models import Model
restnet = tf.keras.applications.VGG16(
    include_top=False, # não vai aproveitar a camada de saída 
    weights=None, #não pega os pesso da imagenet
    input_shape=(224,224,3)
)
output = restnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
restnet = tf.keras.Model(inputs=restnet.input, outputs=output)
for layer in restnet.layers: #treina tudo do zero
    layer.trainable = True
restnet.summary()


# In[10]:


mc = tf.keras.callbacks.ModelCheckpoint('resnet_model.h5', monitor='val_binary_accuracy', mode='max', save_best_only=True)

model = tf.keras.models.Sequential()
model.add(restnet)
model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=(224,224,3)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(
    optimizer=tf.keras.optimizers.Adamax(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(df_train['status']),
                                                 df_train['status'])
class_weights = dict(enumerate(class_weights))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    callbacks = [mc])


# In[20]:


STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

print('---------------Teste-------------')
test_generator.reset()
predictions = model.predict(test_generator,
                        steps=STEP_SIZE_TEST,
                        verbose=1)


# In[32]:


y_pred = predictions > 0.5


# In[29]:


predicted_class_indices=np.argmax(predictions,axis=1)



# In[27]:


print(accuracy_score(test_generator.labels, y_pred))


# In[28]:


print(classification_report(test_generator.labels, y_pred))


# In[ ]:




