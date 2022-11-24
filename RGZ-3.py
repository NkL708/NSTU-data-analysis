import streamlit as st

import json
import pickle
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt


with open(r'./assets/dump/m1_info.json', 'r') as f:
    m1_info = json.load(f)
with open(r'./assets/dump/m2_info.json', 'r') as f:
    m2_info = json.load(f)
# Загрузка шкалёров
with open(r'./assets/dump/scalerNormForX.txt', 'rb') as f:
    scalerNormForX = pickle.load(f)
with open(r'./assets/dump/scalerNormForY.txt', 'rb') as f:
    scalerNormForY = pickle.load(f)
# Загрузка моделей
with open(r'./assets/dump/m1.txt', 'rb') as f:
    m1 = pickle.load(f)
m2 = tf.keras.models.load_model(r'./assets/dump/m2.h5')

# Левая панель (сайдбар)
weekday = st.sidebar.slider('День недели (weekday)', 0, 6, 1)
mnth = st.sidebar.slider('Месяц (mnth)', 1, 12, 1)
season = st.sidebar.slider('Время года (season)', 1, 4, 1)
hum = st.sidebar.slider('Влажность (hum)', 0.0, 1.0, step=0.05)
weathersit = st.sidebar.slider('Непогода (weathersit)', 1, 4, 1)

# Формирование данных для предсказания выходного параметра моделями
df_to_predict = pd.DataFrame(
    data=[[weekday, mnth, season, hum, weathersit]],
    columns=['weekday',  'mnth', 'season', 'hum', 'weathersit'])

# Нормирование данных
dfNorm_to_predict = pd.DataFrame(
    data=scalerNormForX.transform(df_to_predict),
    columns=df_to_predict.columns)

# Предсказание выходного параметра
yPred = m1.predict(df_to_predict[['season', 'hum', 'weathersit']])
yNormPred = m2.predict(
    df_to_predict[['weekday',  'mnth', 'season', 'hum', 'weathersit']])

# Основная часть
st.markdown('# Расчёт количества арендованных велосипедов')
st.write('Исходные данные')
st.write(df_to_predict)
st.write('Нормализованные данные')
st.write(dfNorm_to_predict)

# Колонки
col1, col2 = st.columns(2)
with col1:
    st.markdown('## Линейная регрессия')
    r2 = m1_info['R2']
    st.write(f'R^2={r2:>9,.3f}')
    rmse = m1_info['RMSE']
    st.write(f'R^2={rmse:>9,.3f}')
    st.write('Y в исходной шкале')
    st.write(yPred)
with col2:
    st.markdown('## Нейронная сеть')
    r2 = m2_info['R2']
    st.write(f'R^2={r2:>9,.3f}')
    rmse = m2_info['RMSE']
    st.write(f'R^2={rmse:>9,.3f}')
    st.write('Y нормализированный')
    st.write(yNormPred)
    st.write('Y в исходной шкале')
    st.write(scalerNormForY.inverse_transform(yNormPred))

st.markdown('# Информация об m1')
st.write(m1_info)
st.markdown('# Информация об m2')
st.write(m2_info)
