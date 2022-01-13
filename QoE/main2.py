import clr
import LLP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
# read_data.py not present for confidentiality reasons
from getdata import read_data
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential


def QoEmetric(y_true, y_pred, graph=False):
    """
    Metric used by the model
    Inputs:
        - y_test: True values
        - y_pred: Predicted values
        - graph: Results will be used for graphs (splited loss)
    Outputs:
        - cost: The cost of the model
    """
    alpha = 1
    ratio = 2
    alphacost = alpha * y_pred
    epsilon = 1e-2
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
    return 0.05 * (a * QoEcost + (1 - a) * np.abs(alphacost))


def nnloss_model(activ, errorf, lr):
    """
    Create the Neural network model as Loss Function
    Inputs:
        - lookback: Number of past samples used
        - activ: Activation function to use
        - errorf: loss function to use
        - lr : Learning rate used by the model
    Outputs:
        - model: The loss neural network model
    """
    model = Sequential()
    model.add(Dense(25, activation=activ, input_shape=(2,)))
    model.add(Dense(25, activation=activ,
              kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    # Adding the output layer
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), loss=errorf)
    # model = Sequential()
    # model.add(Dense(15, activation=activ,
    #           input_shape=(2,)))
    # model.add(Dense(15, activation=activ,))
    # model.add(Dense(15, activation=activ,)    kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    # model.add(Dense(1),)
    # model.compile(optimizer=tf.keras.optimizers.Adam(
    #     learning_rate=lr), loss=errorf)
    return model


def main_nn_model(input):
    """
    Create the Main Neural network
    Inputs:
        -lookback: Number of past samples used
    Outputs:
        -model: The neural network model
    """
    model = Sequential()
    # # Adding LSTM layers and some Dropout regularisation
    model.add(LSTM(units=30, return_sequences=True,
                   input_shape=(input, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=30))
    # Adding the output layer
    model.add(Dense(units=1))
    # Must nor be compile to dual train the model
    # model = Sequential()
    # model.add(LSTM(50, activation='relu', input_shape=(
    #     input, 1), return_sequences=True))
    # model.add(LSTM(units=30))
    return model

#
# def compare_shifted(y_test, y_pred):
#     """
#     Compare results with the same model shifted
#     Inputs:
#         - y_test: True values
#         - y_pred: Predicted values
#     """
#     full_loss = []
#     shifts = np.linspace(-0.2, 0.2, 200)
#     for shift in shifts:
#         y_shift = y_pred + shift
#         sumQoE = np.sum(training.QoEmetric(y_test, y_shift, np.size(y_test)))
#         full_loss.append(sumQoE)
#
#     figshif = plt.figure()
#     plt.scatter(shifts, full_loss)
#     plt.xlabel('Shift', fontsize=16)
#     plt.ylabel('Error', fontsize=16)
#     plt.grid()
#     figshif.savefig('Shiftfed')


def main():
    # Definding variables

    lookback = 6  # looking at past 30 minutes data as 1 week of data=2016; 1 day of data= 288; 1 hour = 12
    # Load and format the data
    X_train, X_test, Y_train, Y_test = read_data("partial_data.csv")
    # Split test and train
    print(np.shape(X_train))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    Y_train = tf.reshape(Y_train, [-1, 1])
    Y_test = tf.reshape(Y_test, [-1, 1])
    for i in range(1):
        epo = 40
        batch_used = 16
       # Create neural networks
        nnloss = nnloss_model(
            'relu', 'mean_squared_error', 0.0001)
        model = main_nn_model(X_train.shape[1] + 1)
        # Create Cyclic learning rate object
        clr2 = clr.CyclicLR(base_lr=0.00001, max_lr=0.001, nb_cycles=3,
                            full_size=np.shape(X_train)[0], epochs=epo, batch_size=batch_used, mode='triangular')
        LLPitem = LLP.LossLeaP(model, nnloss, QoEmetric, isrnn=True,
                               batch_size_main=batch_used, batch_size_loss=batch_used, clr=clr2, epochs=epo)
        # Train the model
        LLPitem.dual_train(X_train, Y_train,  validation_data=(
            X_test, Y_test), noise=0.01)
        LLPmet = LLPitem.history['val_metric']
        # Save nn models
        path = "Networks/test2_nnloss.h5"
        path2 = "Networks/test2_nn.h5"
        nnloss.save(path)
        model.save(path2)
        noisenul = np.zeros((X_test.shape[0], 1, 1))
        X_test = np.concatenate((X_test, noisenul), axis=1)
        print(np.shape(X_test))
    #     # Estimate models
        Y_pred = model.predict(X_test)
        Y_pred = np.abs(Y_pred)
    #     y_predtrain = nn.predict(X_trainbis)
    #     compare_shifted(np.array(y_test).flatten(), np.array(y_pred).flatten())
    #     y_pred = np.flatten((y_pred))
    #     y_predtrain = np.flatten((y_predtrain))
    #
        Qoetest, alphatest = QoEmetric(
            Y_test, Y_pred, graph=True)
        print("QoE=", np.mean(Qoetest))
        print("alpha=", np.mean(alphatest))
        print(np.shape(Y_test))
        print(np.shape(Y_pred))
    #
    #     # Forecast
        figpred = plt.figure()
        plt.xlabel('Sample')
        plt.ylabel('Traffic')
        plt.plot(Y_pred)
        plt.plot(Y_test)
        plt.legend(["y_pred", "y_test"])
        plt.show()
        # path2 = "Networks/" + traffic + '/' + cluster + ratio + "_Traffic"
        # figpred.savefig(path2)
    #
    #     # Cost
        print(np.shape(Qoetest))
        print(np.shape(alphatest))
        alphatest = alphatest.flatten()
        figpred2 = plt.figure()
        plt.scatter(np.linspace(0, np.size(Y_test),
                    np.size(Y_test)), Qoetest, alpha=0.5, s=20, color='red')
        plt.scatter(np.linspace(0, np.size(Y_test), np.size(Y_test)),
                    alphatest + Qoetest, alpha=0.7, s=20)
        plt.scatter(np.linspace(0, np.size(Y_test),
                    np.size(Y_test)), alphatest, alpha=0.7, s=20)

        plt.xlabel('Sample', fontsize=16)
        plt.ylabel('Cost', fontsize=16)
        plt.legend(["$Cost_{capa}$", "$Cost_{QoE}$", "$M$"])
        plt.show()
    #     path2 = "Networks/" + traffic + '/' + cluster + ratio + "_Cost"
    #     figpred2.savefig(path2)
    #
        # 2D Loss
        y_test2 = tf.random.shuffle(Y_test)
        adderrortest3 = np.reshape(np.linspace(-0.07, 0.07, 2016), (2016, 1))
        Xtest2 = tf.transpose(tf.stack((y_test2, y_test2 + adderrortest3)))
        final = nnloss.predict(Xtest2)
        figloss2 = plt.figure()
        plt.ylabel('Loss perceived')
        plt.xlabel('y_pred-y_true')
        plt.scatter(adderrortest3, np.abs(final))
        plt.grid()
        plt.xlim([-0.1, 0.1])
        plt.show()
        figloss2.savefig('Loss2')
    #
        # 3D Loss
        y_3d = np.linspace(0.2, 0.8, 200)
        adderrortest3 = np.linspace(-0.5, 0.5, 300)

        x = []
        y = []
        z = []
        print("'''''''''''''''''''''''''''''")
        for elem in y_3d:
            ypred_3d = adderrortest3 + elem
            ypart_3d = np.zeros(300) + elem
            Xtest2 = tf.transpose(tf.stack((ypart_3d, ypred_3d)))
            final = nnloss.predict(Xtest2)
            x.append(ypart_3d)
            y.append(ypred_3d)
            z.append(final)

        x = np.array(x).flatten()
        y = np.array(y).flatten()
        z = np.array(z).flatten()
        # configure_plotly_browser_state()
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, mode='markers', marker=dict(color=z, colorscale='viridis'))])
        fig.update_layout(scene=dict(
            xaxis_title="d<sub>t</sub>",
            yaxis_title="s<sub>t+1</sub>",
            zaxis_title='M<sub>t+1</sub>'))
        fig.show()
        fig.write_html("3Dshape3.html")

        # figmet = plt.figure()
        # plt.ylabel('Metric')
        # plt.xlabel('Batch nb')
        # plt.yscale('log')
        # axis2 = np.linspace(epo, np.size(train_met), num=epo)
        # plt.plot(train_met)
        # plt.plot(axis2, val_met, color='red')
        # plt.legend(["Training", "Val"])
        # plt.grid()
    #
    #     alltrainmet.append(train_met)
    #     allvalmet.append(val_met)
    #
    #     loss_test = training.QoEmetric(y_test, y_pred, np.size(y_test))
    #     loss_train = training.QoEmetric(y_train, y_predtrain, np.size(y_train))
    #
    #     print("Training metric:")
    #     print("\t", (np.mean(loss_train)))
    #     ltrain.append((np.mean(loss_train)))
    #     print("Testing metric:")
    #     print("\t", (np.mean(loss_test)))
    #     ltest.append((np.mean(loss_test)))
    #
    #     # Plot the loss distribution
    #     fighist2 = plt.figure()
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.xlabel('Loss')
    #     plt.ylabel('Frequency')
    #     plt.hist(x=full_loss, bins='auto', color='#0504aa',
    #              density=True, alpha=0.5, rwidth=1, label='Full Train')
    #     plt.hist(x=loss_test, bins='auto', color='red',
    #              density=True, alpha=0.5, rwidth=1, label='Test')
    #     plt.legend()
    #     plt.show()
    #     fighist2.savefig('Histon')
    #
    # print("\t-Train Set:")
    # print("\t\t", np.mean(ltrain))
    # print("\t\t", np.std(ltrain))
    # print("\t-Test Set:")
    # print("\t\t", np.mean(ltest))
    # print("\t\t", np.std(ltest))
    #
    # figmetavr = plt.figure()
    # plt.ylabel('Metric')
    # plt.xlabel('Batch nb')
    # plt.yscale('log')
    # axis2 = np.linspace(epo, np.size(train_met), num=epo)
    # plt.plot(np.mean(alltrainmet, axis=0))
    # plt.plot(axis2, np.mean(allvalmet, axis=0), color='red')
    # plt.legend(["Training", "Val"])
    # plt.grid()
    # figmetavr.savefig('Metric')


if __name__ == "__main__":
    main()
