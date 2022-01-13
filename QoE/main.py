import clr
import LLP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from getdata import read_data
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

sphour = 12
hpweek = 168 * 6
services = 4


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
    model.add(Dense(40, activation=activ, input_shape=(2,)))
    model.add(Dense(40, activation=activ))
    model.add(Dense(40, activation=activ))
    model.add(Dense(25, activation=activ))
    # Adding the output layer
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), loss=errorf)
    # model = Sequential()
    # model.add(Dense(15, activation=activ, input_shape=(2,)))
    # model.add(Dense(15, activation=activ))
    # model.add(Dense(1))
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
    # Adding LSTM layers and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(input, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=30))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))
    # Must nor be compile to dual train the model
    # model = Sequential()
    # model.add(LSTM(50, activation='relu', input_shape=(input, 1)))
    # model.add(Dense(1))
    return model


def compare_shifted(y_test, y_pred):
    """
    Compare results with the same model shifted
    Inputs:
        - y_test: True values
        - y_pred: Predicted values
    """
    full_loss = []
    shifts = np.linspace(-0.2, 0.2, 200)
    for shift in shifts:
        y_shift = y_pred + shift
        sumQoE = np.sum(training.QoEmetric(y_test, y_shift, np.size(y_test)))
        full_loss.append(sumQoE)

    figshif = plt.figure()
    plt.scatter(shifts, full_loss)
    plt.xlabel('Shift', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.grid()
    figshif.savefig('Shiftfed')


def main():
    # Definding variables
    metricstr = "omc"
    traffic = "facebook"
    cluster = "10"
    ratio = "1"
    serv = 0
    lookback = 6  # looking at past 30 minutes data as 1 week of data=2016; 1 day of data= 288; 1 hour = 12
    # Load and format the data
    Xs_nn, full_ox, full_cx, maxi = read_data("partial_data.csv", cluster)
    # Split test and train
    X_test_l = []
    y_test_l = []
    nb_train = 9000
    nb_test = 2016
    X_train = np.array(Xs_nn[serv][:nb_train]).astype('float64')
    X_test = np.array(
        Xs_nn[serv][nb_train + 1:nb_test + nb_train + 1]).astype('float64')
    y_train = np.array(full_ox[serv][:nb_train]).astype('float64')
    y_test = np.array(
        full_ox[serv][nb_train + 1:nb_test + nb_train + 1]).astype('float64')
    y_train = tf.dtypes.cast(tf.reshape(y_train, [-1]), tf.float64)
    y_test = tf.dtypes.cast(tf.reshape(y_test, [-1]), tf.float32)
    train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    noiseadd = np.reshape(
        np.zeros(np.shape(X_train)[0]), (np.shape(X_train)[0], 1, 1))
    X_trainbis = tf.concat([X_train, noiseadd], 1)
    noise = np.zeros(np.shape(X_test)[0])
    noiseadd = np.reshape(noise, (np.shape(X_test)[0], 1, 1))
    X_testbis = tf.concat([X_test, noiseadd], 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    ltest = []
    ltrain = []
    alltrainmet = []
    allvalmet = []
    ovrl = []
    slal = []
    for i in range(1):
        epo = 20
        # Create neural network models
        nnloss = networks.make_nnloss_model(
            lookback, 'relu', 'mean_squared_error', 0.001)
        nn = networks.make_nn_model(lookback)

        # Create CLR tool
        clr2 = clr.CyclicLR(base_lr=0.00001, max_lr=0.001, nb_cycles=epo / 10,
                            full_size=9000, epochs=epo, batch_size=64, mode='triangular')
        nn, nnloss, full_loss, full_error, train_met, val_met = training.dual_nn_train(
            train_dataset, nn, nnloss, metricstr, epochs=epo, val_data=(X_testbis, y_test), clr2=clr2)

        # Save nn models
        path = "Networks/" + traffic + '/' + cluster + ratio + "_nnloss.h5"
        path2 = "Networks/" + traffic + '/' + cluster + ratio + "_nn.h5"
        nnloss.save(path)
        nn.save(path2)

        # Estimate models
        y_pred = nn.predict(X_testbis)
        y_predtrain = nn.predict(X_trainbis)
        compare_shifted(np.array(y_test).flatten(), np.array(y_pred).flatten())
        y_pred = np.reshape(y_pred, (2016,))
        y_predtrain = np.reshape(y_predtrain, (9000,))

        Qoetest, alphatest = training.QoEmetric(
            y_test, y_pred, np.size(y_test), graph=True)
        print("QoE=", np.mean(Qoetest))
        print("alpha=", np.mean(alphatest))
        print(np.shape(y_test))
        print(np.shape(y_pred))
        print(y_test)
        print(y_pred)

        # Forecast
        figpred = plt.figure()
        plt.xlabel('Sample')
        plt.ylabel('Traffic')
        plt.plot(y_pred)
        plt.plot(y_test)
        plt.legend(["y_pred", "y_test"])
        plt.show()
        path2 = "Networks/" + traffic + '/' + cluster + ratio + "_Traffic"
        figpred.savefig(path2)

        # Cost
        figpred2 = plt.figure()
        plt.scatter(np.linspace(0, np.size(y_test),
                    np.size(y_test)), Qoetest, alpha=0.5, s=20, color='red')
        plt.scatter(np.linspace(0, np.size(y_test), np.size(y_test)),
                    alphatest + Qoetest, alpha=0.7, s=20)
        plt.scatter(np.linspace(0, np.size(y_test),
                    np.size(y_test)), alphatest, alpha=0.7, s=20)

        plt.xlabel('Sample', fontsize=16)
        plt.ylabel('Cost', fontsize=16)
        plt.legend(["$Cost_{capa}$", "$Cost_{QoE}$", "$M$"])
        plt.show()
        path2 = "Networks/" + traffic + '/' + cluster + ratio + "_Cost"
        figpred2.savefig(path2)

        # 2D Loss
        y_test2 = tf.random.shuffle(y_test)
        adderrortest3 = np.linspace(-0.07, 0.07, 2016)
        Xtest2 = tf.transpose(tf.stack((y_test2, y_test2 + adderrortest3)))
        final = nnloss.predict(Xtest2)
        figloss2 = plt.figure()
        plt.ylabel('Loss perceived')
        plt.xlabel('y_pred-y_true')
        plt.scatter(adderrortest3, final)
        plt.grid()
        plt.xlim([-0.1, 0.1])
        plt.show()
        figloss2.savefig('Loss2')

        # 3D Loss
        y_3d = np.linspace(0.2, 0.8, 200)
        adderrortest3 = np.linspace(-0.3, 0.3, 300)

        x = []
        y = []
        z = []
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
        fig.write_html("3Dshape.html")

        figmet = plt.figure()
        plt.ylabel('Metric')
        plt.xlabel('Batch nb')
        plt.yscale('log')
        axis2 = np.linspace(epo, np.size(train_met), num=epo)
        plt.plot(train_met)
        plt.plot(axis2, val_met, color='red')
        plt.legend(["Training", "Val"])
        plt.grid()

        alltrainmet.append(train_met)
        allvalmet.append(val_met)

        loss_test = training.QoEmetric(y_test, y_pred, np.size(y_test))
        loss_train = training.QoEmetric(y_train, y_predtrain, np.size(y_train))

        print("Training metric:")
        print("\t", (np.mean(loss_train)))
        ltrain.append((np.mean(loss_train)))
        print("Testing metric:")
        print("\t", (np.mean(loss_test)))
        ltest.append((np.mean(loss_test)))

        # Plot the loss distribution
        fighist2 = plt.figure()
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.hist(x=full_loss, bins='auto', color='#0504aa',
                 density=True, alpha=0.5, rwidth=1, label='Full Train')
        plt.hist(x=loss_test, bins='auto', color='red',
                 density=True, alpha=0.5, rwidth=1, label='Test')
        plt.legend()
        plt.show()
        fighist2.savefig('Histon')

    print("\t-Train Set:")
    print("\t\t", np.mean(ltrain))
    print("\t\t", np.std(ltrain))
    print("\t-Test Set:")
    print("\t\t", np.mean(ltest))
    print("\t\t", np.std(ltest))

    figmetavr = plt.figure()
    plt.ylabel('Metric')
    plt.xlabel('Batch nb')
    plt.yscale('log')
    axis2 = np.linspace(epo, np.size(train_met), num=epo)
    plt.plot(np.mean(alltrainmet, axis=0))
    plt.plot(axis2, np.mean(allvalmet, axis=0), color='red')
    plt.legend(["Training", "Val"])
    plt.grid()
    figmetavr.savefig('Metric')


if __name__ == "__main__":
    main()
