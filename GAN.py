import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose,Dense, Dropout, ReLU, Conv2D,Input,Concatenate,Reshape,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, multiply,AveragePooling2D,Flatten
from tensorflow.keras.models import load_model
 
# load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

 
num_classes = len(np.unique(y_train))

class_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
 
 
# convert class to onehot
#####################################
y_train = np.identity(10)[y_train].reshape([len(y_train),-1])
y_test = np.identity(10)[y_test].reshape([len(y_test),-1])
 
X_train = np.float32(X_train)
X_train = X_train / 255
 
X_test = np.float32(X_test)
X_test = X_test / 255


###########################################
z = Input(shape=(100,))
labels = Input(shape=(10,))

# Generator
merged_layer = Concatenate()([z, labels])

x = Dense(2 * 2 * 512)(merged_layer)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Reshape((2, 2, 512))(x)

x = Conv2DTranspose(256, kernel_size=5, strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(128, kernel_size=5, strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(64, kernel_size=5, strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)

out = Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", use_bias=False, activation="sigmoid",)(x)

generator = Model(inputs=[z, labels], outputs=out, name="generator")


#########################

img_input = Input(shape=(X_train[0].shape))

# Discriminator
x = Conv2D(64, kernel_size=5, strides=2, padding="same")(img_input)
x = ReLU()(x)

x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
x = ReLU()(x)

x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

merged_layer = Concatenate()([x, labels])
x = Dense(512, activation="relu")(merged_layer)
out = Dense(1, activation="sigmoid")(x)

discriminator = Model(inputs=[img_input, labels], outputs=out, name="discriminator")
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
                      loss="binary_crossentropy",
                      metrics=["binary_accuracy"])

discriminator.trainable = False



###########################################################

label = Input(shape=(10,), name='label')
z = Input(shape=(100,), name='z')

fake_img = generator([z, label])
validity = discriminator([fake_img, label])

GANs = Model(inputs=generator.inputs , outputs=discriminator([generator.outputs,generator.input[1]]),name = "GANs")


GANs.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)



###################################################################

epochs = 120
batch_size = 32
smooth = 0.1
latent_dim = 100

real = tf.ones(shape=(batch_size, 1))
fake = tf.zeros(shape=(batch_size, 1))

d_loss = []
d_g_loss = []

for e in range(epochs + 1):
    for i in range(len(X_train) // batch_size):

        # Train Discriminator weights
        discriminator.trainable = True

        # Real samples
        X_batch = X_train[i * batch_size: (i + 1) * batch_size]
        # real_labels = tf.keras.utils.to_categorical(
        #     y_train[i * batch_size: (i + 1) * batch_size].reshape(-1, 1),
        #     num_classes=10
        # )
        
        real_labels = y_train[i * batch_size: (i + 1) * batch_size]

        d_loss_real = discriminator.train_on_batch(
            [X_batch, real_labels], real * (1 - smooth)
        )

        # Fake Samples
        z = tf.random.normal(shape=(batch_size, latent_dim), mean=0, stddev=1)
        random_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, 10, batch_size).reshape(-1, 1),
            num_classes=10
        )
        X_fake = generator.predict_on_batch([z, random_labels])
        d_loss_fake = discriminator.train_on_batch(
            [X_fake, random_labels], fake
        )

        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        
        #Train Generator weights
        discriminator.trainable = False
        
        z = tf.random.normal(shape=(batch_size,latent_dim),mean=0,stddev=1)
        
        random_labels = tf.keras.utils.to_categorical(
            np.random.randint(0,10,batch_size).reshape(-1,1),num_classes=10
            )
        d_g_loss_batch = GANs.train_on_batch([z,random_labels],real)
        
        print(f"{e} loss = {d_g_loss_batch[0]}")
        
        
    d_loss.append(d_loss_batch)
    d_g_loss.append(d_g_loss_batch[0])

    print(
        "epoch = %d/%d, d_Loss=%.3f, g_Loss=%.3f"
        % (e + 1, epochs, d_loss[-1], d_g_loss[-1]),
        100 * " "
    )