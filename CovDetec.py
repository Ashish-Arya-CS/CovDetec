import streamlit as st
import IPython.display as ipd
from pydub import AudioSegment 
import numpy as np
from PIL import Image
from load_css import local_css

import pandas as pd
import os
import datetime

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder

# to play the audio files
from IPython.display import Audio

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *

import base64

st.set_page_config(layout="centered")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def load_our_model():
  model = tf.keras.models.load_model('covidtest.h5')
  return model

local_css("style.css")
 
st.markdown(" <h1 style='text-align: center; color: black;'>Corona Detection App</h1>", unsafe_allow_html=True)
st.markdown("\n")
st.markdown(" <div>To Know about the working of App and to Display Wave Plot, please click on Expand to show option Button below.</div>", unsafe_allow_html=True)


my_expander = st.beta_expander("Expand to show option", expanded=False)
with my_expander:
    choice = st.multiselect("Enter Your Choice", ('How does it work ?', 'Display Wave Plot'))

if 'How does it work ?' in choice:
    st.markdown("<div>Hello and Welcome to our AI enabled Covid Detection App.Let us describe you how it works :- </div>", unsafe_allow_html=True)
    st.markdown("<div>• Upload an audio of about three seconds in which your cough sound can be heard clearly </div>", unsafe_allow_html=True)
    st.markdown("<div> by clicking on the Browse Files button </span></div>", unsafe_allow_html=True)
    st.markdown("<div>• Once the file is uploaded the AI Model will display the result on the screen.</div>", unsafe_allow_html=True)
    st.markdown("<div>• Once your result is displayed and you want to obtain a prediction for any other audio file then</div>", unsafe_allow_html=True)
    st.markdown("<div> it is recommended to reload the page. </span></div>", unsafe_allow_html=True)
    st.markdown("<div>• At last, we wish you to stay healthy and Covid Negative. Don't forget to wear Mask and</div>", unsafe_allow_html=True)
    st.markdown("<div> maintain Social Distancing.</div>", unsafe_allow_html=True)
    st.markdown("\n")
st.markdown(" <h3 style='text-align: center; color: black;'>Upload Your Audio File Below</h3>", unsafe_allow_html=True)
st.markdown(" <div>The audio file should be of about three seconds in which the cough sound can be heard clearly.</div>", unsafe_allow_html=True)

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background: url("data:image/png;base64,%s");
    background-size: auto;    
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('abcd1.png')

from numpy import load
Y = load('Y.npy', allow_pickle = True)
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
tr = load('tr.npy')
te = load('te.npy')

scaler = StandardScaler()
tr = scaler.fit_transform(tr)
te = scaler.transform(te)

directory = datetime.datetime.now().time()

st.set_option('deprecation.showPyplotGlobalUse', False)

uploaded_file = st.file_uploader("Insert File", type="mp3")
if uploaded_file is not None:
  audio_path = "./tmp/" + directory.strftime('%Y%m%d%H%M%S') + ".wav"
  file_var = AudioSegment.from_mp3(uploaded_file) 
  file_var.export(audio_path, format='wav')
  st.audio(audio_path)

  def extract_features(data, sample_rate):
      # ZCR
      result = np.array([])
      zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
      result=np.hstack((result, zcr)) # stacking horizontally

      # Chroma_stft
      stft = np.abs(librosa.stft(data))
      chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
      result = np.hstack((result, chroma_stft)) # stacking horizontally
      # MFCC
      mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
      result = np.hstack((result, mfcc)) # stacking horizontally

      # Root Mean Square Value
      rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
      result = np.hstack((result, rms)) # stacking horizontally

      # MelSpectogram
      mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
      result = np.hstack((result, mel)) # stacking horizontally
      
      return result
  
  def create_waveplot(data, sr):
      plt.figure(figsize=(10, 3))
      plt.title('Waveplot for audio that you uploaded.', size=15)
      librosa.display.waveplot(data, sr=sr)
      plt.show()
      st.pyplot()

  def get_features(data, sample_rate):
      # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
      res1 = extract_features(data, sample_rate)
      result = np.array(res1)
      return result



  X = []
  data, sample_rate = librosa.load(audio_path, duration=2.0, offset=0.6)
  
  if 'Display Wave Plot' in choice:  
    create_waveplot(data, sample_rate)
  feature = get_features(data, sample_rate)
  for ele in feature:
    X.append(ele)

  X = np.array(X)
  X = X.reshape(1, -1)
  X = scaler.transform(X)
  X = np.expand_dims(X, axis=2)

  present_model = load_our_model() 

  prediction = present_model.predict(X)
  prediction_pred = encoder.inverse_transform(prediction)
  if str(prediction_pred[0][0]) == 'covid':
    st.markdown("<div><span class='highlight coral'>Our AI model has predicted you as Covid positive. So, we highly recommend you to visit your</span></div>", unsafe_allow_html=True)
    st.markdown("<div><span class='highlight coral'> nearest hopital</span></div>", unsafe_allow_html=True)

  else:
    st.markdown("<div><span class='highlight green'>According to our AI Model you are Covid Negative. Always wear a mask while going </span></div>", unsafe_allow_html=True)
    st.markdown("<div><span class='highlight green'> outside your home and maintain Social Distancing. Be Safe and Be Healthy.</span></div>", unsafe_allow_html=True)
  st.markdown("\n")
  st.markdown("<div>Before uploading any next audio file please reload the page.</div>", unsafe_allow_html=True)
  st.markdown("\n")
  st.markdown("<div>Thank You for using our Covid Detection App.</div>", unsafe_allow_html=True)
  os.remove(audio_path) 
  
else:
  st.write("Please upload a .mp3 file") 
my_expander = st.beta_expander("Connect with the Developers", expanded=True)
with my_expander:
    st.markdown(" [![](https://img.shields.io/badge/LinkedIn-AshishArya-blue?logo=Linkedin&logoColor=blue&labelColor=white)](https://www.linkedin.com/in/ashish-arya-65923b16b/) [![](https://img.shields.io/badge/LinkedIn-AnshulChaudhary-blue?logo=Linkedin&logoColor=blue&labelColor=white)](https://www.linkedin.com/in/anshul-chaudhary-2001/)", unsafe_allow_html=True)
my_expander = st.beta_expander("Give us your Feedback", expanded=True)
with my_expander:
    st.markdown("[Feedback Form](https://forms.gle/afWUXQyB3nvftEm17)", unsafe_allow_html=True)
