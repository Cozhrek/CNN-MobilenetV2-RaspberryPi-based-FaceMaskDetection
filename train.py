# import paket/lib yang diperlukan
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct argument parser dan parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path untuk input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path untuk output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path untuk output model deteksi masker")
args = vars(ap.parse_args())

# masukan nilai initial learning rate, jumlah epochs untuk dilatih,
# dan batch size
INIT_LR = 1e-4
EPOCHS = 40  # ???
BS = 32

# mengambil gambar dari folder dataset,
# dan menginisialiasi data dan kelas data
print("[INFO] memuat gambar...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop untuk image paths
for imagePath in imagePaths:
    # ekstrak label class dari filename
    label = imagePath.split(os.path.sep)[-2]

    # memuat input gambar (224x224) dan melakukan preprocessing
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update data dan label list
    data.append(image)
    labels.append(label)

# konversi data dan label ke numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# parsisi data ke training dan testing, 75% training  dan 25% testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# construct training image generator untuk augmentasi data
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# memuat MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct head dari model yang akan diletakkan diatas base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# meletakkan head FC model diatas base model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop semua layer di base model dan freeze jadi layernya
# tidak akan di update selama proses training pertama
for layer in baseModel.layers:
    layer.trainable = False

# compile model
print("[INFO] meng-compile model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train head dari network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# buat prediksi untuk testing set
print("[INFO] mengevaluasi network...")
predIdxs = model.predict(testX, batch_size=BS)

# untuk setiap gambar di testing set harus ditemukan index
# dari label telebih dahulu dengan hasil prediksi paling besar
predIdxs = np.argmax(predIdxs, axis=1)

# menampilkan laporan klasifikasi
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# simpan model ke storage
print("[INFO] menyimpan model deteksi masker...")
model.save(args["model"], save_format="h5")

# plot untuk training loss dan accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss dan Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
