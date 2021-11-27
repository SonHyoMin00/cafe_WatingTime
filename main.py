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

    values = [cafe['아메리카노 수'], cafe['핫커피 수'], cafe['아이스커피 수'], cafe['핫음료 수'],
              cafe['아이스음료 수'], cafe['블랜딩음료 수'], cafe['티 수'], cafe['펄 수'], cafe['총 잔 수'],
              cafe['근무자 수'], cafe['밀린 주문 수'], cafe['대기시간 (+밀린 주문 수)']]
    values = np.transpose(values)
    # print(values.shape)  # (500, 12)

    scaler = preprocessing.MinMaxScaler()  # 최소, 최대 범위를 0~1로
    values = scaler.fit_transform(cafe.values)

    grams = nltk.ngrams(values, 5+1)
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
    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.LSTM(128, return_sequences=True))
    model.add(keras.layers.LSTM(108, return_sequences=True))
    model.add(keras.layers.SimpleRNN(56, return_sequences=True))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    # hidden stage의 사이즈를 늘리거나 simplernn layer를 늘릴수록 정확도가 높아짐
    # return_sequences:true->입력한 값만큼,false->1개  # rnn은 x는 3차원 데이터가 들어옴. y는 2차원임

    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse,
                  metrics='acc')    # mae는 정답과의 오차

    model.fit(x_train, y_train, epochs=1000, verbose=2,
              validation_split=0.2)
    print(model.evaluate(x_test, y_test, verbose=0))

    t = [[[1, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 6.1],
         [2, 0, 0, 0, 2, 1, 2, 0, 0, 5, 2, 1, 11.6],
         [3, 0, 1, 0, 0, 1, 0, 0, 0, 2, 2, 2, 14.5],
         [4, 0, 0, 0, 0, 0, 0, 3, 0, 3, 2, 2, 15.1],
         [5, 2, 0, 1, 0, 0, 0, 0, 0, 3, 2, 3, 19.3]]]

    # line3-7
    p = model.predict(t)
    p = (data_max - data_min) * p + data_min
    print(p)

    exit()

    plt.subplot(1, 2, 1)
    plt.plot(y_test, 'r', label='target')
    plt.plot(p, 'g', label='prediction')
    plt.legend()

    p = (data_max - data_min) * p + data_min
    y_test = (data_max - data_min) * y_test + data_min

    plt.subplot(1, 2, 2)
    plt.plot(y_test, 'r')
    plt.plot(p, 'g')
    plt.show()


model_cafe()



