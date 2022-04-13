#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# In[50]:


df = pd.read_csv('/home/thais/CSV/status_A1.csv')
df.sort_values(by='title', inplace=True)
images_path = sorted(glob.glob('/home/thais/Windows/Train/*'))
df['Unnamed: 0'] = np.array(images_path)
df.reset_index(inplace=True, drop=True)
df['status'] = df['status'].astype(str)


# ### Particionando bases de treino e teste com split 70-30%

# In[49]:


### particiona considerando 70-30% e mantendo a frequência de amostras para treino, validação e teste de acordo com as colunas título e status (rótulo da rede)

X_train, X_test, y_train, y_test = train_test_split(df[['Unnamed: 0', 'title']], df['status'], test_size=0.30, random_state=42, stratify=df[['status', 'title']])
df_train = pd.concat([X_train, y_train], axis = 1)
X_train, X_val, y_train, y_val = train_test_split(df_train[['Unnamed: 0', 'title']], df_train['status'], test_size=0.30, random_state=42, stratify=df_train[['status', 'title']])

### contatena atributos de entrada e rótulo em um único dataframe para utilizar o flow_from_dataframe do tensorflow
df_test = pd.concat([X_test, y_test], axis=1)
df_train = pd.concat([X_train, y_train], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)

print('Total de imagens de treinamento', len(df_train))
print('Total de imagens de validação', len(df_val))
print('Total de imagens de teste', len(df_test))


# ### Particionando bases de treino e teste com diferentes músicas

# In[52]:

# In[54]:


datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=df_train, directory='/home/thais/Windows/Train/', x_col='Unnamed: 0', y_col="status", class_mode="binary", target_size=(32,32), batch_size=32)
valid_generator=datagen.flow_from_dataframe(dataframe=df_val, directory='/home/thais/Windows/Train/', x_col='Unnamed: 0', y_col="status", class_mode="binary", target_size=(32,32), batch_size=32)
test_generator=datagen.flow_from_dataframe(dataframe=df_test, directory='/home/thais/Windows/Train/', x_col='Unnamed: 0', y_col="status", class_mode="binary", target_size=(32,32), batch_size=32)


# In[87]:

model = tf.keras.models.load_model('resnet_model.h5')

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
buceta = model.predict_generator(test_generator,
                        steps=STEP_SIZE_TEST,
                        verbose=1)


y_pred = buceta > 0.5
print(confusion_matrix(df_test['status'].values, y_pred))
#predicted_class_indices=np.argmax(buceta,axis=1)
#labels = (train_generator.class_indices)
#labels2 = dict((v,k) for k,v in labels.items())
#predictions = [labels2[k] for k in predicted_class_indices]
#print(accuracy_score(df_test['status'].values, predictions))
