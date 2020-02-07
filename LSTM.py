import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv()   #读取文件

training_set = dataset_train.iloc[:,1:2].values   #取文件的所有行和第二第三列


#将数据标准化处理
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
fig, ax = plt.subplots(figsize=(14,4))
ax.plot(range(len(training_set)), training_set_scaled)
ax.set_xlabel('time (from 2018-09-28 to 2010-07-10)'); ax.set_ylabel('open （rescaled）'); 

ndat = len(training_set_scaled)

#用60天的数据模拟一天
X_train = []
y_train = []
for i in range(60, ndat):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print ('Shape of X_train and y_train is ', X_train.shape, y_train.shape)

#搭建LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer='adam', loss='mean_squared_error') 

regressor.summary()


#用新数据测试
test_data = test.iloc[:,1:2].values

inputs = test_data[ len(test_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

regressor.fit(X_train, y_train, epochs = 10, batch_size=32)

predicted = regressor.predict(X_test)
predicted = sc.inverse_transform(predicted)

#剩下作图就可以了