

import time

import numpy as np
import tensorflow as tf
from keras import backend as K


class LossLeaP:
    def __init__(self, nn_main, nn_loss, metric_fc, isrnn=False, lr_main=0.001, batch_size_main=64, batch_size_loss=64, epochs=50, clr=None):
        """
        Create The LossLearningPredictor object
        Inputs:
            -nn_main: Main neural network predictor (must be uncompiled)
            -nn_loss: Loss neural network
            -metric_fc: Metric used to train the nn_loss
            -isrnn: Boolean to know if nn_main is a Recurent Neural Network (RNN)
            -lr_main: Learning Rate for the nn_main
            -batch_size_main: batch size used by nn_main (>= batch_size_loss)
            -batch_size_loss: batch size used by nn_loss
            -epochs: Epochs for the training phase
            -clr : Cyclic Learning Rate object (if used)
        """
        super(LossLeaP, self).__init__()
        # Variables definition
        self.nn_main = nn_main
        self.nn_loss = nn_loss
        self.metric_fc = metric_fc
        self.batch_size_main = batch_size_main
        self.batch_size_loss = batch_size_loss
        self.epochs = epochs
        self.clr = clr
        self.lr_main = lr_main
        self.isrnn = isrnn
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=self.lr_main)
        # Keep results during training
        self.history = {}

    def dual_train(self, X, Y, v_inputs=None, validation_data=None, noise=0):
        """
        Train neural networks models
        Inputs:
            -X: Input dataset
            -Y: Target values
            -v_inputs: External KPI
            -validation_data: Validation data (X_val, y_val)
            -noise: Noise added to the model
        """
        # 2 different cases if additionnal inputs are used
        if v_inputs is None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (X, Y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (X, Y, v_inputs))
        dataset = dataset.batch(self.batch_size_main, drop_remainder=False)
        for epoch in range(self.epochs):
            start_time = time.time()
            for step, elems in enumerate(dataset):
                if v_inputs is None:
                    x, y = elems
                    y = tf.dtypes.cast(y, tf.float64)
                    yvec = y
                else:
                    x, y, v = elems
                    v = tf.reshape(v, [np.shape(x)[0], -1])
                    y = tf.dtypes.cast(y, tf.float64)
                    yvec = tf.concat((y, v), axis=1)
                # noise shaped differently if nn is an RNN or a standard MLP
                if self.isrnn == False:
                    noiseadd = tf.random.normal(
                        [np.shape(x)[0], 1], 0, noise, tf.float64)
                else:
                    noiseadd = tf.random.normal(
                        [np.shape(x)[0], 1, 1], 0, noise, tf.float64)
                if noise != 0:
                    x = tf.concat([x, noiseadd], 1)
                noiseadd = tf.reshape(noiseadd, [-1, 1])
                loss_value, y_pred, X_loss = self.train_step(
                    x, yvec, noiseadd)
                met = self.metric_fc(y, y_pred)
                met_value = np.mean(met)
                self.nn_loss.fit(X_loss, met, epochs=1,
                                 batch_size=self.batch_size_loss, verbose=0)
                if self.clr:
                    self.clr.on_batch_end2(self.opt)
                self.history.setdefault('loss', []).append(loss_value)
                self.history.setdefault('metric', []).append(met_value)
                # print training loss every step iterations
                if step % 200 == 1:
                    print(
                        "Training loss (for one batch) at step %d from epochs %d: %.4f  and   %.4f"
                        % (step, epoch, float(met_value), float(loss_value))
                    )
                    print("Seen so far: %s samples" %
                          ((step + 1) * self.batch_size_main))
                    print(self.opt.lr)
            # Run a validation loop at the end of each epoch.
            if validation_data != None:
                xval, yval = validation_data
                if self.isrnn == False:
                    if noise != 0:
                        noiseadd = tf.zeros([np.shape(xval)[0], 1], tf.float64)
                        xval = tf.concat([xval, noiseadd], 1)
                else:
                    if noise != 0:
                        noiseadd = tf.zeros(
                            [np.shape(xval)[0], 1, 1], tf.float64)
                        xval = tf.concat([xval, noiseadd], 1)
                yvalpred = self.nn_main(xval)
                yvalpred = tf.dtypes.cast(yvalpred, tf.float64)
                val_met = np.mean(self.metric_fc(yval, yvalpred))
                self.history.setdefault('val_metric', []).append(val_met)
            print("Validation metric: %.4f" % (float(val_met),))
            print("Time taken: %.2fs" % (time.time() - start_time))

    # Simple train of the main nn using tf function
    @tf.function
    def train_step(self, x, y, noiseadd):
        with tf.GradientTape() as tape:
            tape.watch(self.nn_main.trainable_weights)
            y_pred = self.nn_main(x, training=True)
            y_pred = tf.dtypes.cast(y_pred, tf.float64)
            y_pred = y_pred + noiseadd
            X_loss = tf.concat((y, y_pred), axis=1)
            loss_value = K.mean(self.nn_loss(X_loss))
        gradients = tape.gradient(
            loss_value, self.nn_main.trainable_weights)
        self.opt.apply_gradients(
            zip(gradients, self.nn_main.trainable_weights))

        return loss_value, y_pred, X_loss
