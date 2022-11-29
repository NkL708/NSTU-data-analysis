import streamlit as st

import json
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


week_dict = {
    'Воскресенье': 0,
    'Понедельник': 1,
    'Вторник': 2,
    'Среда': 3,
    'Четверг': 4,
    'Пятница': 5,
    'Суббота': 6,
}

season_dict = {
    'Весна': 1,
    'Лето': 2,
    'Осень': 3,
    'Зима': 4,
}

weathersit_dict = {
    'Ясная погода': 1,
    'Туман': 2,
    'Непогода': 3,
    'Сильная непогода': 4,
}

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

# Левая панель (sidebar)
weekday = week_dict.get(st.sidebar.select_slider(
    'День недели (weekday)',
    options=week_dict.keys()))

hr = st.sidebar.slider('Час (hr)', 0, 23, 1)

season = season_dict.get(st.sidebar.select_slider(
    'Время года (season)',
    options=season_dict.keys()))

temp = st.sidebar.slider('Температура (temp)', 0.0, 1.0, step=0.01)

weathersit = weathersit_dict.get(st.sidebar.select_slider(
    'Непогода (weathersit)',
    options=weathersit_dict.keys()))

# Формирование данных для предсказания выходного параметра моделями
df_to_predict = pd.DataFrame(
    data=[[weekday, hr, season, temp, weathersit]],
    columns=['weekday',  'hr', 'season', 'temp', 'weathersit'])

# Нормирование данных
dfNorm_to_predict = pd.DataFrame(
    data=scalerNormForX.transform(df_to_predict),
    columns=df_to_predict.columns)

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
    st.write(f"R^2={m1_info['R2']:>9,.3f}")
    st.write(f"RMSE{m1_info['RMSE']:>9,.3f}")
    st.write('Y в исходной шкале')
    yPred = m1.predict(df_to_predict[m1_info['features']])
    st.write(yPred)

with col2:
    st.markdown('## Нейронная сеть')
    st.write(f"R^2={m2_info['R2']:>9,.3f}")
    st.write(f"RMSE={m2_info['RMSE']:>9,.3f}")
    st.write('Y нормализированный')
    yNormPred = m2.predict(dfNorm_to_predict[m2_info['features']])
    st.write(yNormPred)
    st.write('Y в исходной шкале')
    st.write(scalerNormForY.inverse_transform(yNormPred))

df_to_check = pd.DataFrame(
    data=[[6, 0, 1, 0.24, 1, 16, 95, 24],
          [5, 8, 1, 0.2, 1, 210, 83, 260],
          [6, 4, 1, 0.18, 3, 1, 31, 24],
          [6, 12, 1, 0.2, 2, 98, 60, 82],
          [6, 17, 1, 0.16, 1, 69, 71, 65],],
    columns=['weekday',  'hr', 'season', 'temp',
             'weathersit', 'cnt', 'predLR', 'predNN'])
st.markdown('# Таблица для сверки точности')
st.write(df_to_check)

st.markdown('# Информация об m1')
st.write(m1_info)
st.markdown('# Информация об m2')
st.write(m2_info)
