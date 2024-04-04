# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:29:41 2024

@author: Administrator
"""
import tensorflow as tf

def analyze_lottery_with_lstm(input_data, output_data, rnn_size=128, num_layers=2, batch_size=64, learning_rate=0.01):
    end_points = {}

    cell = tf.keras.layers.LSTM(rnn_size, return_state=True)
    states = [cell.get_initial_state(batch_size=batch_size, dtype=tf.float32) for _ in range(num_layers)]

    inputs = tf.keras.layers.Input(shape=(input_data.shape[1], input_data.shape[2]))
    outputs = inputs
    for state in states:
        outputs, state = cell(outputs, states=state, training=False)
        states = [state]

    outputs = tf.keras.layers.Dense(output_data.shape[1], activation='softmax')(outputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(input_data, output_data, epochs=10, batch_size=batch_size)

def analyze_lottery_with_gru(input_data, output_data, rnn_size=128, num_layers=2, batch_size=64, learning_rate=0.01):
    end_points = {}

    cell = tf.keras.layers.GRU(rnn_size, return_state=True)
    states = [cell.get_initial_state(batch_size=batch_size, dtype=tf.float32) for _ in range(num_layers)]

    inputs = tf.keras.layers.Input(shape=(input_data.shape[1], input_data.shape[2]))
    outputs = inputs
    for state in states:
        outputs, state = cell(outputs, states=state, training=False)
        states = [state]

    outputs = tf.keras.layers.Dense(output_data.shape[1], activation='softmax')(outputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(input_data, output_data, epochs=10, batch_size=batch_size)
    
    def analyze_and_predict_lottery_with_lstm(input_data, output_data, prediction_input):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(input_data.shape[1], input_data.shape[2])),
        tf.keras.layers.Dense(output_data.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_data, output_data, epochs=10, batch_size=64)

    predictions = model.predict(prediction_input)
    return predictions

def analyze_and_predict_lottery_with_gru(input_data, output_data, prediction_input):
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(128, input_shape=(input_data.shape[1], input_data.shape[2])),
        tf.keras.layers.Dense(output_data.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_data, output_data, epochs=10, batch_size=64)

    predictions = model.predict(prediction_input)
    return predictions
# 这段代码使用了TensorFlow 2.x的Keras API构建了带有预测功能的LSTM和GRU模型。您可以调用analyze_and_predict_lottery_with_lstm函数来使用LSTM模型分析和预测彩票数据，或者调用analyze_and_predict_lottery_with_gru函数来使用GRU模型分析和预测彩票数据。

# 在调用这些函数之前，您需要准备好训练数据input_data和output_data，以及用于预测的数据prediction_input。训练模型后，您可以使用模型对新的彩票数据进行预测。

# 希望这能帮助您进行彩票数据的分析和预测。如果您有任何其他问题或需要进一步帮助，请随时告诉我！

# 以下是一个简单的示例代码，演示如何在 analyze_lottery_with_lstm 方法中添加预测功能：

def analyze_lottery_with_lstm(input_data, output_data, prediction_input):
    # 构建训练模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(input_data.shape[1], input_data.shape[2])),
        tf.keras.layers.Dense(output_data.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_data, output_data, epochs=10, batch_size=64)

    # 使用训练好的模型进行预测
    predictions = model.predict(prediction_input)

    # 返回预测结果
    return predictions