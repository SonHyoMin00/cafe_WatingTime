import pandas as pd
import nltk
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt


def get_xy():
    cafe = pd.read_csv('data/cafe.csv', index_col=0)
    # print(cafe)  # [500 rows x 13 columns]
    # print(cafe.values.shape)  # (500, 13)

    scaler = preprocessing.MinMaxScaler()  # 최소, 최대 범위를 0~1로
    values = scaler.fit_transform(cafe.values)

    grams = nltk.ngrams(values, 7+1)
    grams = np.float32(list(grams))  # 튜플의 리스트라 원하는 연산을 못함. 넘파이로 바꿔줌

    x = np.float32([g[:-1] for g in grams])
    # print(x.shape)  # (493, 7, 13)
    y = np.float32([g[-1, -1:] for g in grams])
    # print(y.shape)  # (493, 1)

    return x, y, scaler.data_min_[-1], scaler.data_max_[-1]


def model_cafe():
    x, y, data_min, data_max = get_xy()  # 최대 최소값을 이용, 계산해 원래의 값으로 복구시킨다

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    # hidden stage의 사이즈를 늘리거나 simplernn layer를 늘릴수록 정확도가 높아짐
    # return_sequences:true->입력한 값만큼,false->1개  # rnn은 x는 3차원 데이터가 들어옴. y는 2차원임

    model.add(keras.layers.Dense(1))
    # model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.mse,
                  metrics='mae')    # mae는 정답과의 오차

    model.fit(x_train, y_train, epochs=1000, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test)

    plt.subplot(1, 2, 1)
    plt.plot(y_test, 'r', label='target')  # 데이터를 섞어서 시각화가 제대로 되지 않음 # 셔플옵션  false로 주고오기
    plt.plot(p, 'g', label='prediction')
    plt.legend()  # label 값을 표에 표시할수있다

    p = (data_max - data_min) * p + data_min
    # print((data_max-data_min)*p+data_min)
    y_test = (data_max - data_min) * y_test + data_min

    plt.subplot(1, 2, 2)
    plt.plot(y_test, 'r')  # 데이터를 섞어서 시각화가 제대로 되지 않음 # 셔플옵션  false로 주고오기
    plt.plot(p, 'g')
    # plt.ylim(2650, 3000)
    plt.show()


model_cafe()



