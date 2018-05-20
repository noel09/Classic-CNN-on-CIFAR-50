from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.layers import add
from keras.backend import int_shape
from normalization import LRN2D


def m_alexnet(training_dataset, testing_dataset, learning_rate, epochs, load_weights, save_weights):
    
    print("\Training AlexNet Model\n")
    
    # This is the modified alexnet model to work with CIFAR-50 dataset

    x_train, y_train = training_dataset
    x_test, y_test = testing_dataset
    
    #tb = TensorBoard(log_dir="../CIFAR-100 Project/graph", histogram_freq=0,write_graph=True, write_images=True)

    model = Sequential()

    # First Convolutional Layer
    model.add(Conv2D(96, (3,3), strides= (1,1), padding='same', activation='relu',
                            input_shape=x_train.shape[1:]))
    model.add(LRN2D())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))


    # Second Convolutional Layer
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(LRN2D())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))


    # Third Convolutional Layer
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))

    # Fourth Convolutional Layer
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))

    # Fifth Convolutional Layer
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    #First Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    #Second Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    #Softmax
    model.add(Dense(50, activation='softmax'))
    
    if load_weights:
        model.load_weights("../CIFAR-100 Project/model/m_alexnet.h5")
    
    model.summary()

    batch_size = 128
    sgd = SGD(lr=learning_rate, decay=5e-4, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy','top_k_categorical_accuracy'])
    model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(x_test, y_test))
                     #callbacks=[tb])

    if save_weights:
        model.save_weights("../CIFAR-100 Project/model/m_alexnet.h5")


def m_googlenet(training_dataset, testing_dataset, learning_rate, epochs, load_weights, save_weights):
    print("\nTraining GoogLeNet Model\n")
    
    #This is the modified googlenet model to work with CIFAR-50 dataset

    x_train, y_train = training_dataset
    x_test, y_test = testing_dataset
    
    #tb = TensorBoard(log_dir="../CIFAR-100 Project/graph", histogram_freq=0,write_graph=True, write_images=True)
    
    input = Input(shape=x_train.shape[1:])
    
    # --------- First Convolution ---------
    conv_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input)
    
    # --------- Max Pool ---------
    pool_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(conv_1)
    
    # --------- Second Convolution with Reduction ---------
    conv_2_r = Conv2D(64,(1, 1), strides=(1, 1), activation='relu')(pool_1)
    conv_2 = Conv2D(192,(3, 3), strides=(1, 1), activation='relu')(conv_2_r)
    
    # --------- Max Pool ---------
    pool_2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='valid')(conv_2)

    
    # --------- Inception (3a) ----------
    # 1 x 1 Convolution
    inception_3a_1x1 = Conv2D(64,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_2)
    
    # 3 x 3 Convolution with Redcution
    inception_3a_3x3_r = Conv2D(96,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_2)
    inception_3a_3x3 = Conv2D(128,(3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_3a_3x3_r)
    
    # 5 x 5 Convolution with Reduction
    inception_3a_5x5_r = Conv2D(16,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_2)
    inception_3a_5x5 = Conv2D(32,(5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_3a_5x5_r)
    
    # Max Pooling Projection
    inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(pool_2)
    inception_3a_pool_p = Conv2D(32,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_3a_pool)
    
    # Inception 3a output
    inception_3a_out = concatenate([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_p],axis=3)
    
    
    # --------- Inception (3b) ----------
    # 1 x 1 Convolution
    inception_3b_1x1 = Conv2D(128,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_3a_out)
    
    # 3 x 3 Convolution with Redcution
    inception_3b_3x3_r = Conv2D(128,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_3a_out)
    inception_3b_3x3 = Conv2D(192,(3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_3b_3x3_r)
    
    # 5 x 5 Convolution with Reduction
    inception_3b_5x5_r = Conv2D(32,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_3a_out)
    inception_3b_5x5 = Conv2D(96,(5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_3b_5x5_r)
    
    # Max Pooling Projection
    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inception_3a_out)
    inception_3b_pool_p = Conv2D(64,(1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_3b_pool)
    
    # Inception 3b output
    inception_3b_out = concatenate([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_p],axis=3)
    
    
    # --------- Max Pool ---------
    pool_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(inception_3b_out)


    # --------- Inception (4a) ----------
    # 1 x 1 Convolution
    inception_4a_1x1 = Conv2D(192, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_3)

    # 3 x 3 Convolution with Redcution
    inception_4a_3x3_r = Conv2D(96, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_3)
    inception_4a_3x3 = Conv2D(208, (3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_4a_3x3_r)

    # 5 x 5 Convolution with Reduction
    inception_4a_5x5_r = Conv2D(16, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_3)
    inception_4a_5x5 = Conv2D(48, (5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_4a_5x5_r)

    # Max Pooling Projection
    inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(pool_3)
    inception_4a_pool_p = Conv2D(64, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4a_pool)

    # Inception 4a output
    inception_4a_out = concatenate([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_p], axis=3)


    # --------- Auxiliary 1 - 4a ----------
    # Average Pooling
    pool_aux1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), border_mode='valid')(inception_4a_out)

    # Dimension Reduction
    conv_aux1 = Conv2D(128, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_aux1)

    # Fully Connected
    flat_aux1 = Flatten()(conv_aux1)
    linear_aux1 = Dense(1024, activation='relu')(flat_aux1)

    # Dropout
    dropout_aux1 = Dropout(0.7)(linear_aux1)

    # Softmax
    softmax_aux1 = Dense(50, activation='softmax')(dropout_aux1)

    # --------- Inception (4b) ----------
    # 1 x 1 Convolution
    inception_4b_1x1 = Conv2D(160, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4a_out)

    # 3 x 3 Convolution with Redcution
    inception_4b_3x3_r = Conv2D(112, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4a_out)
    inception_4b_3x3 = Conv2D(224, (3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_4b_3x3_r)

    # 5 x 5 Convolution with Reduction
    inception_4b_5x5_r = Conv2D(24, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4a_out)
    inception_4b_5x5 = Conv2D(64, (5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_4b_5x5_r)

    # Max Pooling Projection
    inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inception_4a_out)
    inception_4b_pool_p = Conv2D(64, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4b_pool)

    # Inception 4b output
    inception_4b_out = concatenate([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_p], axis=3)


    # --------- Inception (4c) ----------
    # 1 x 1 Convolution
    inception_4c_1x1 = Conv2D(128, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4b_out)

    # 3 x 3 Convolution with Redcution
    inception_4c_3x3_r = Conv2D(128, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4b_out)
    inception_4c_3x3 = Conv2D(256, (3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_4c_3x3_r)

    # 5 x 5 Convolution with Reduction
    inception_4c_5x5_r = Conv2D(24, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4b_out)
    inception_4c_5x5 = Conv2D(64, (5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_4c_5x5_r)

    # Max Pooling Projection
    inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inception_4b_out)
    inception_4c_pool_p = Conv2D(64, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4c_pool)

    # Inception 4c output
    inception_4c_out = concatenate([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_p], axis=3)


    # --------- Inception (4d) ----------
    # 1 x 1 Convolution
    inception_4d_1x1 = Conv2D(112, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4c_out)

    # 3 x 3 Convolution with Redcution
    inception_4d_3x3_r = Conv2D(144, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4c_out)
    inception_4d_3x3 = Conv2D(288, (3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_4d_3x3_r)

    # 5 x 5 Convolution with Reduction
    inception_4d_5x5_r = Conv2D(32, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4c_out)
    inception_4d_5x5 = Conv2D(64, (5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_4d_5x5_r)

    # Max Pooling Projection
    inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inception_4c_out)
    inception_4d_pool_p = Conv2D(64, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4d_pool)

    # Inception 4d output
    inception_4d_out = concatenate([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_p], axis=3)


    # --------- Auxiliary 2 - 4d ----------
    # Average Pooling
    pool_aux2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), border_mode='valid')(inception_4d_out)

    # Dimension Reduction
    conv_aux2 = Conv2D(128, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_aux2)

    # Fully Connected
    flat_aux2 = Flatten()(conv_aux2)
    linear_aux2 = Dense(1024, activation='relu')(flat_aux2)

    # Dropout
    dropout_aux2 = Dropout(0.7)(linear_aux2)

    # Softmax
    softmax_aux2 = Dense(50, activation='softmax')(dropout_aux2)


    # --------- Inception (4e) ----------
    # 1 x 1 Convolution
    inception_4e_1x1 = Conv2D(256, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4d_out)

    # 3 x 3 Convolution with Redcution
    inception_4e_3x3_r = Conv2D(160, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4d_out)
    inception_4e_3x3 = Conv2D(320, (3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_4e_3x3_r)

    # 5 x 5 Convolution with Reduction
    inception_4e_5x5_r = Conv2D(32, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4d_out)
    inception_4e_5x5 = Conv2D(128, (5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_4e_5x5_r)

    # Max Pooling Projection
    inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inception_4d_out)
    inception_4e_pool_p = Conv2D(128, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_4e_pool)

    # Inception 4e output
    inception_4e_out = concatenate([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_p], axis=3)


    # --------- Max Pool ---------
    pool_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(inception_4e_out)


    # --------- Inception (5a) ----------
    # 1 x 1 Convolution
    inception_5a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_4)

    # 3 x 3 Convolution with Redcution
    inception_5a_3x3_r = Conv2D(160, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_4)
    inception_5a_3x3 = Conv2D(320, (3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_5a_3x3_r)

    # 5 x 5 Convolution with Reduction
    inception_5a_5x5_r = Conv2D(32, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(pool_4)
    inception_5a_5x5 = Conv2D(128, (5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_5a_5x5_r)

    # Max Pooling Projection
    inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(pool_4)
    inception_5a_pool_p = Conv2D(128, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_5a_pool)

    # Inception 5a output
    inception_5a_out = concatenate([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_p], axis=3)


    # --------- Inception (5b) ----------
    # 1 x 1 Convolution
    inception_5b_1x1 = Conv2D(384, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_5a_out)

    # 3 x 3 Convolution with Redcution
    inception_5b_3x3_r = Conv2D(192, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_5a_out)
    inception_5b_3x3 = Conv2D(384, (3, 3), strides=(1, 1), border_mode='same', activation='relu')(inception_5b_3x3_r)

    # 5 x 5 Convolution with Reduction
    inception_5b_5x5_r = Conv2D(48, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_5a_out)
    inception_5b_5x5 = Conv2D(128, (5, 5), strides=(1, 1), border_mode='same', activation='relu')(inception_5b_5x5_r)

    # Max Pooling Projection
    inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inception_5a_out)
    inception_5b_pool_p = Conv2D(128, (1, 1), strides=(1, 1), border_mode='same', activation='relu')(inception_5b_pool)

    # Inception 5b output
    inception_5b_out = concatenate([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_p], axis=3)


    # --------- Average Pool ---------
    pool_5 = AveragePooling2D(pool_size=(5, 5), strides=(1, 1), border_mode='valid')(inception_5b_out)

    # --------- Flatten ---------
    flat = Flatten()(pool_5)

    # --------- Dropout ---------
    dropout = Dropout(0.4)(flat)

    # --------- Softmax ---------
    softmax = Dense(50, activation='softmax')(dropout)
    
    model = Model(input=input, output=[softmax, softmax_aux1, softmax_aux2])
    
    if load_weights:
        model.load_weights("../CIFAR-100 Project/model/m_googlenet.h5")
    
    model.summary()

    batch_size = 128
    sgd = SGD(lr=learning_rate, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy','top_k_categorical_accuracy'])
    model.fit(x_train, [y_train,y_train,y_train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, [y_test,y_test,y_test]))
              #callbacks=[tb])

    if save_weights:
        model.save_weights("../CIFAR-100 Project/model/m_googlenet.h5")


def m_resnet18(training_dataset, testing_dataset, learning_rate, epochs, load_weights, save_weights):
    print("\nTraining ResNet-18 Model\n")
    
    #This is the modified ResNet18 to work with CIFAR-50 Dataset

    x_train, y_train = training_dataset
    x_test, y_test = testing_dataset

    input = Input(shape=x_train.shape[1:])

    #First Convolution Layer
    x = Conv2D(64, (7,7), strides=(2,2))(input)
    # Batch Normalization
    x = BatchNormalization(axis=3)(x)
    # ReLu
    x = Activation('relu')(x)

    #Max Pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #First Residual Block (3x3, 64) x2
    x = m_resnet18_block(x, 64, (3,3), True, strides=(1,1))
    x = Activation('relu')(x)

    #Second Residual Block (3x3, 64) x2
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)

    #Third Residual Block (3x3, 128) x2
    x = m_resnet18_block(x, 128, (3, 3), True)
    x = Activation('relu')(x)

    #Fourth Residual Block (3x3, 128) x2
    x = m_resnet18_block(x, 128, (3, 3), False)
    x = Activation('relu')(x)

    #Fifth Residual Block (3x3, 256) x2
    x = m_resnet18_block(x, 256, (3, 3), True)
    x = Activation('relu')(x)

    #Sixth Residual Block (3x3, 256) x2
    x = m_resnet18_block(x, 256, (3, 3), False)
    x = Activation('relu')(x)

    #Seventh Residual Block (3x3, 512) x2
    x = m_resnet18_block(x, 512, (3, 3), True)
    x = Activation('relu')(x)

    #Eighth Residual Block (3x3, 512) x2
    x = m_resnet18_block(x, 512, (3, 3), False)
    x = Activation('relu')(x)

    #Average Pooling
    x = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), border_mode='valid')(x)

    #Flatten
    flat = Flatten()(x)

    #Softmax
    softmax = Dense(50, activation='softmax')(flat)

    model = Model(input=input, output=[softmax])
    
    if load_weights:
        model.load_weights("../CIFAR-100 Project/model/m_resnet18.h5")

    model.summary()

    batch_size = 128
    sgd = SGD(lr=learning_rate, decay=5e-4, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy','top_k_categorical_accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    if save_weights:
        model.save_weights("../CIFAR-100 Project/model/m_resnet18.h5")
    
def m_resnet50(training_dataset, testing_dataset, learning_rate, epochs, load_weights, save_weights):
    print("\nTraining ResNet-50 Model\n")
    #This is the modified ResNet110 to work with CIFAR-50 Dataset

    x_train, y_train = training_dataset
    x_test, y_test = testing_dataset

    input = Input(shape=x_train.shape[1:])

    #First Convolution Layer
    x = Conv2D(64, (3,3), strides=(1,1))(input)
    # Batch Normalization
    x = BatchNormalization(axis=3)(x)
    # ReLu
    x = Activation('relu')(x)

    #Max Pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)

    ############ conv2_x
    #1st Residual Block
    x = m_resnet50_block(x, 64, 256, True, strides=(1,1))
    x = Activation('relu')(x)
    #2nd Residual Block
    x = m_resnet50_block(x, 64, 256, False)
    x = Activation('relu')(x)
    #3rd Residual Block
    x = m_resnet50_block(x, 64, 256, False)
    x = Activation('relu')(x)
    
    ############ conv3_x
    #4th Residual Block
    x = m_resnet50_block(x, 128, 512, True)
    x = Activation('relu')(x)
    #5th Residual Block
    x = m_resnet50_block(x, 128, 512, False)
    x = Activation('relu')(x)
    #6th Residual Block
    x = m_resnet50_block(x, 128, 512, False)
    x = Activation('relu')(x)
    #7th Residual Block
    x = m_resnet50_block(x, 128, 512, False)
    x = Activation('relu')(x)
    
    ############ conv4_x
    #8th Residual Block
    x = m_resnet50_block(x, 256, 1024, True)
    x = Activation('relu')(x)
    #9th Residual Block
    x = m_resnet50_block(x, 256, 1024, False)
    x = Activation('relu')(x)
    #10th Residual Block
    x = m_resnet50_block(x, 256, 1024, False)
    x = Activation('relu')(x)
    #11th Residual Block
    x = m_resnet50_block(x, 256, 1024, False)
    x = Activation('relu')(x)
    #12th Residual Block
    x = m_resnet50_block(x, 256, 1024, False)
    x = Activation('relu')(x)
    #13th Residual Block
    x = m_resnet50_block(x, 256, 1024, False)
    x = Activation('relu')(x)
    
    ############# conv5_x
    #14th Residual Block
    x = m_resnet50_block(x, 512, 2048, True)
    x = Activation('relu')(x)
    #15th Residual Block
    x = m_resnet50_block(x, 512, 2048, False)
    x = Activation('relu')(x)
    #16th Residual Block
    x = m_resnet50_block(x, 512, 2048, False)
    x = Activation('relu')(x)
    
    #Average Pooling
    x = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), border_mode='valid')(x)

    #Flatten
    flat = Flatten()(x)

    #Softmax
    softmax = Dense(50, activation='softmax')(flat)

    model = Model(input=input, output=[softmax])
    
    if load_weights:
        model.load_weights("../CIFAR-100 Project/model/m_resnet50.h5")

    model.summary()

    
    batch_size = 128
    sgd = SGD(lr=learning_rate, decay=5e-4, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy','top_k_categorical_accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    if save_weights:
        model.save_weights("../CIFAR-100 Project/model/m_resnet50.h5")


def m_resnet110(training_dataset, testing_dataset, learning_rate, epochs, load_weights, save_weights):
    print("\nTraining ResNet-110 Model\n")

    x_train, y_train = training_dataset
    x_test, y_test = testing_dataset

    input = Input(shape=x_train.shape[1:])

    # First Convolution Layer
    x = Conv2D(64, (3, 3), padding='same')(input)
    # Batch Normalization
    x = BatchNormalization(axis=3)(x)
    # ReLu
    x = Activation('relu')(x)

    # Feature map = 32 x 32, n = 18
    x = m_resnet18_block(x, 16, (3, 3), True, strides=(1,1))
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 16, (3, 3), False)
    x = Activation('relu')(x)

    # Feature map = 16 x 16, n = 18
    x = m_resnet18_block(x, 32, (3, 3), True)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 32, (3, 3), False)
    x = Activation('relu')(x)

    # Feature map = 8 x 8, n = 18
    x = m_resnet18_block(x, 64, (3, 3), True)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)
    x = m_resnet18_block(x, 64, (3, 3), False)
    x = Activation('relu')(x)

    # Average Pooling
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode='valid')(x)

    # Flatten
    flat = Flatten()(x)

    # Softmax
    softmax = Dense(50, activation='softmax')(flat)

    model = Model(input=input, output=[softmax])

    if load_weights:
        model.load_weights("../CIFAR-100 Project/model/m_resnet110.h5")

    model.summary()

    batch_size = 128
    sgd = SGD(lr=learning_rate, decay=5e-4, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy','top_k_categorical_accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    if save_weights:
        model.save_weights("../CIFAR-100 Project/model/m_resnet110.h5")


def m_resnet18_block(input, num_kernels, kernel, downsampling, strides=(2,2)):

    fx = input

    #Downsampling
    if downsampling:
        fx = Conv2D(num_kernels, (1,1), strides=strides, padding='valid')(fx)

    #First Convolution
    fx = Conv2D(num_kernels, kernel, strides=(1, 1), padding='same')(fx)
    #Batch Normalization
    fx = BatchNormalization(axis=3)(fx)
    #ReLu
    fx = Activation('relu')(fx)

    #Second Convolution
    fx = Conv2D(num_kernels, kernel, strides=(1, 1), padding='same')(fx)
    # Batch Normalization
    fx = BatchNormalization(axis=3)(fx)
    # ReLu
    fx = Activation('relu')(fx)

    x = input

    if downsampling:
        #Projection shortcut
        x = Conv2D(num_kernels,(1,1),strides=strides)(x)
        # Batch Normalization
        x = BatchNormalization(axis=3)(x)
        # ReLu
        x = Activation('relu')(x)

    return add([x,fx])
    
def m_resnet50_block(input, num_kernels1, num_kernels2 , downsampling, strides=(2,2)):
    fx = input

    #Downsampling
    if downsampling:
        fx = Conv2D(num_kernels1, (1,1), strides=strides, padding='valid')(fx)

    #First Convolution
    fx = Conv2D(num_kernels1, (1,1), strides=(1, 1), padding='same')(fx)
    #Batch Normalization
    fx = BatchNormalization(axis=3)(fx)
    #ReLu
    fx = Activation('relu')(fx)

    #Second Convolution
    fx = Conv2D(num_kernels1, (3,3), strides=(1, 1), padding='same')(fx)
    # Batch Normalization
    fx = BatchNormalization(axis=3)(fx)
    # ReLu
    fx = Activation('relu')(fx)
    
    #Third Convolution
    fx = Conv2D(num_kernels2, (1,1), strides=(1, 1), padding='same')(fx)
    # Batch Normalization
    fx = BatchNormalization(axis=3)(fx)
    # ReLu
    fx = Activation('relu')(fx)

    x = input

    if downsampling:
        #Projection shortcut
        x = Conv2D(num_kernels2,(1,1),strides=strides)(x)
        # Batch Normalization
        x = BatchNormalization(axis=3)(x)
        # ReLu
        x = Activation('relu')(x)

    return add([x,fx])