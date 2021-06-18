from keras.models import Sequential
from keras.layers import Dense,CuDNNLSTM , CuDNNGRU , Dropout , Activation, Embedding
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

def new_net(X,embeddings,Y):
    limit=80
    new_net_out=[]
    for i in Y:
        new_net_out.append(embeddings[i])


    new_net_out=np.array(new_net_out)

    Y=new_net_out.reshape([-1,128])

    
##    sfl = np.arange(X.shape[0])
##    np.random.shuffle(sfl)
##    X=X[sfl]
##    Y=Y[sfl]
    
    print(X[0].shape)

    model = Sequential()
##    model.add(CuDNNLSTM(100, return_sequences=True))
##    model.add(Activation('relu'))
##    model.add(Dropout(0.8))
##    model.add(CuDNNLSTM(100, return_sequences=True))
##    model.add(Activation('relu'))
##    model.add(Dropout(0.8))
    model.add(CuDNNLSTM(1250))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(128, kernel_initializer='uniform'))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mae', 'acc'])

    
    history=model.fit(X[:limit], Y[:limit], epochs=200
        , batch_size=10, validation_data=(X[limit:], Y[limit:]), verbose=1)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.show()
    return model
