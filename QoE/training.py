import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

sphour = 12
hpweek = 168 * 6
services = 4
batch = 64
ratio = 1


def QoEmetric(y_true, y_pred, batch_size, graph=False):
    """
    Metric used by the model
    Inputs:
        - y_test: True values
        - y_pred: Predicted values
        - batch_size: batch_size
        - graph: Results will be used for graphs (splited loss)
    Outputs:
        - cost: The cost of the model
    """
    alpha = 1
    alphacost = alpha * y_pred
    epsilon = 1e-3
    alphacost = np.clip(alphacost, 0 + epsilon, max(alphacost))
    cong = np.divide(y_true, y_pred + epsilon)
    QoE = -0.891 + 5.082 / (np.sqrt(np.abs(cong)))
    stepQoE = np.round(QoE / 2, decimals=1) * 2
    stepQoE2 = np.clip(stepQoE, 0, 5)
    QoEcost = []
    for elem in stepQoE2:
        if elem >= 4:
            QoEcost.append(0)
        elif elem > 3:
            QoEcost.append(0.2)
        elif elem > 2:
            QoEcost.append(0.4)
        elif elem > 1:
            QoEcost.append(0.7)
        else:
            QoEcost.append(1)
    QoEcost = np.array(QoEcost)

    a = ratio / (ratio + 1)
    if graph == True:
        return QoEcost, alphacost
    return 0.05 * (a * QoEcost + (1 - a) * alphacost)


def loss_wrapper(x, y_true, opt, noise, nnloss, nn, batch_size, used_metric, batch_size_loss=64, clr=None):
    """
    Define the loss function update to use in a wrapper
    Inputs:
        -x: Inputs of the neural network
        -y_true: True values from the main model
        -opt: Optimizer of the main neural network
        -noise: The noise to add to the output predicted values
        -nnloss/nn: The loss neural network/ The main neural network
        -batch_size: batch size used by the main neural network
        -used_metric: The metric to use for the loss neural network
        -batch_size_loss: batch size used by the loss neural network (<batch_size)
        -clr : CLR object (if used)
    Outputs:
        -loss_value: The
        -error: y_pred-y_true
        -met_value: Metric value of the batch
        (y_true,y_pred): The cost function of the model
    """
    # @tf.function
    def train_main(x, y_true):
        with tf.GradientTape() as tape:
            y_pred = nn(x, training=True)
            y_pred = tf.dtypes.cast(tf.reshape(y_pred, [-1]), tf.float64)
            y_pred = y_pred + noise
            xloss = tf.stack((y_true, y_pred), axis=1)
            # +np.sum(np.sum(np.sum(np.absolute(nn.trainable_weights))))
            loss_value = K.mean(nnloss(xloss))
        # Update the weights of the model.
        gradients = tape.gradient(loss_value, nn.trainable_weights)
        opt.apply_gradients(zip(gradients, nn.trainable_weights))
        error = y_pred - y_true
        return y_pred, xloss, loss_value, error

    y_pred, xloss, loss_value, error = train_main(x, y_true)
    met = QoEmetric(y_true.numpy(), y_pred.numpy(), batch_size)
    # met=metric(y_true,y_pred,used_metric)
    nnloss.fit(xloss, met, epochs=1,
               batch_size=batch_size_loss, verbose=0)
    if clr:
        clr.on_batch_end()
    met_value = np.mean(met)

    return loss_value, error, met_value


def dual_nn_train(dataset, nn, nnloss, used_metric, batch_size=64, epochs=50, val_data=None, clr=None, clr2=None):
    """
    Train both neural networks model at the same time
    Inputs:
        -dataset: Tensorflow dataset object composed with (X,y)
        -nnloss/nn: The loss neural network/ The main neural network
        -used_metric: The metric to use for the loss neural network
        -batch_size: batch size used by the main neural network
        -epoch: number of epochs to complete during training
    Outputs:
        -nn,nnloss: both updated neural networks
        -full_loss: history of all loss of the main neural network
    """
    dataset = dataset.batch(batch_size, drop_remainder=True)
    learning_rate = 0.001
    sigma = 0.01
    initial_learning_rate = 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    full_loss = []
    full_error = []
    train_met = []
    val_met = []
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataset):
            noise = np.random.normal(0, sigma, batch_size)
            noiseadd = np.reshape(noise, (batch_size, 1, 1))
            x = tf.concat([x, noiseadd], 1)

            # Open a GradientTape
            loss_value, error_value, met_value = loss_wrapper(
                x, y, optimizer, noise, nnloss, nn, batch_size, used_metric, clr=clr)

            full_loss.append(loss_value)
            full_error.append(error_value)
            train_met.append(met_value)
            if clr2:
                clr2.on_batch_end2(optimizer)
            if step % 100 == 1:
                print(
                    "Training loss (for one batch) at step %d: %.7f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
                print(optimizer.lr)

        if val_data != None:
            xval, yval = val_data
            yvalpred = nn.predict(xval)
            yvalpred = tf.reshape(yvalpred, [-1])
            yval = tf.cast(yval, tf.float32)
            yvalpred = tf.cast(yvalpred, tf.float32)
            # val_met.append(np.mean(metric(yval,yvalpred.numpy(), used_metric)))
            QoE = QoEmetric(yval, yvalpred, np.size(yval))
            print(QoE)
            val_met.append(np.mean(QoE))
            print(np.mean(QoE))

            # Display metrics at the end of each epoch.
        # train_acc = train_acc_metric.result()
        # print("Training acc over epoch: %.4f" % (float(train_acc),))

    return nn, nnloss, full_loss, full_error, train_met, val_met
