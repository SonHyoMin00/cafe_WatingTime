# Day_07_02_stock.py

import pandas as pd
import nltk
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt


# 퀴즈 3
# 80%의 데이터로 학습하고 20%의 데이터에 대해 결과를 예측하세요
def get_xy():
    # 퀴즈 1
    # stock_daily.csv 파일로부터 x, y를 반환하는 함수를 만드세요
    # multipleRegression, 7일치 데이터를 이용해 학습
    # serial 데이터니까 rnn 으로 처리해야함 -> x가 3차원 데이터, batch_size, seq_length, n_features = 32,7,5
    stock = pd.read_csv('data/stock_daily.csv', skiprows=2, header=None)
    # print(stock)      # [732 rows x 5 columns]
    # print(stock.values)

    # values = preprocessing.scale(stock.values)
    # values = preprocessing.minmax_scale(stock.values)
    scaler = preprocessing.MinMaxScaler()  # 원래 주식가격을 시각화에 사용하기 위해서는 minmax를 쓰지 않고 MinMaxScaler를 사용함
    values = scaler.fit_transform(stock.values)  # 공부하고 변환까지 한번에 하는 함수
    values = values[::-1]   # 슬라이싱 문법을 이용해서 뒤집는다=>자주사용함! # 오래된 데이터를 가지고 최신의 데이터 도출하기 위해서 뒤집음

    # print(scaler.scale_)    # [2.91411157e-03 2.89040914e-03 2.93437760e-03 8.96298288e-08 2.91445143e-03]
    # print(scaler.data_max_)  # [8.37809998e+02 8.41950012e+02 8.28349976e+02 1.11649000e+07 8.35669983e+02]
    # print(scaler.data_min_)  # [ 494.652237  495.97823   487.562205 7900.        492.552239]
    # =>컬럼이 5개니까 5개가 나옴

    # scaler.inverse_transform()  # 원래 데이터의 상태로 돌아갈 수 있다. (스케일링 이전으로)
    grams = nltk.ngrams(values, 7+1)
    # print(list(grams)[0])
    grams = np.float32(list(grams))  # 튜플의 리스트라 원하는 연산을 못함. 넘파이로 바꿔줌
    # print(grams.shape)  # (725, 8, 5)

    x = np.float32([g[:-1] for g in grams])  # g는 8행 5열, 7행이어야해서 슬라이싱
    # print(x.shape)  # (725, 7, 5)
    y = np.float32([g[-1, -1:] for g in grams])
    # print(y.shape)  # (725, 1)

    """내가 푼 퀴즈 1
        x = stock.values[:, :-1]
        y = stock.values[:, -1:]
        # print(x.shape, y.shape)     # (732, 4) (732, 1)

        x = nltk.ngrams(x, 7)
        y = nltk.ngrams(y, 7)
        print(np.array(list(x)).shape, np.array(list(y)).shape)  # (726, 7, 4) (726, 7, 1)
        """

    return x, y, scaler.data_min_[-1], scaler.data_max_[-1]


def model_stock():
    # 퀴즈 2
    # 앞에서 만든 데이터에 대해 모델을 구축하세요
    x, y, data_min, data_max = get_xy()     # 최대 최소값을 이용, 계산해 원래의 값으로 복구시킨다

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    # 7행 5열의 순서는 미리 정해져 있기 때문에 셔플을 해도 영향을 주지 않는다. 셔플을 하면 오히려 더 좋다.
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    # hidden stage의 사이즈를 늘리거나 simplernn layer를 늘릴수록 정확도가 높아짐
    # return_sequences:true->입력한 값만큼,false->1개  # rnn은 x는 3차원 데이터가 들어옴. y는 2차원임

    model.add(keras.layers.Dense(1))  # 850 딱 하나만 나온다
    # model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.mse,
                  metrics='mae')    # mae는 정답과의 오차

    model.fit(x_train, y_train, epochs=10, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    # 1104 퀴즈 1
    # 정답과 예측 결과를 시각화하세요
    p = model.predict(x_test)

    plt.subplot(1, 2, 1)
    plt.plot(y_test, 'r', label='target')    # 데이터를 섞어서 시각화가 제대로 되지 않음 # 셔플옵션  false로 주고오기
    plt.plot(p, 'g', label='prediction')
    plt.legend()    # label 값을 표에 표시할수있다

    # 1104 퀴즈 2
    # 예측 결과를 원래 값으로 복구하세요
    # 공식 : skyil.tistory.com/50 ???

    p = (data_max - data_min) * p + data_min
    # print((data_max-data_min)*p+data_min)
    y_test = (data_max - data_min) * y_test + data_min

    plt.subplot(1, 2, 2)
    plt.plot(y_test, 'r')  # 데이터를 섞어서 시각화가 제대로 되지 않음 # 셔플옵션  false로 주고오기
    plt.plot(p, 'g')
    plt.show()


model_stock()






