# import paket/lib yang dibutuhkan
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    # ambil dimensi dari frame dan construct blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # umpan blob dan ambil deteksi wajah
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # inisialiasi daftar wajah, lokasi dan daftar prediksi dari jaringan deteksi masker
    faces = []
    locs = []
    preds = []

    # loop deteksi
    for i in range(0, detections.shape[2]):
        # ekstrak nilai confidence (perkiraan/peluang) yang berhubungan dengan deteksi
        confidence = detections[0, 0, i, 2]

        # melakukan filter untuk deteksi yang dinilai lemah dan memastikan nilai confidence lebih besar dari nilai minimum
        if confidence > args["confidence"]:
            # menghitung koordinat (x,y) dari bounding box untuk objek yang dideteksi
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # memastikan bounding box tepat berada pada dimensi frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #  ekstrak ROI wajah, konversi dari BRG ke channel RGB, resize ke 244x244 px dan lakukan prepocess
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # menambahkan wajah dan bounding box ke respective list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # hanya membuat prdiksi jika minimal ada 1 wajah
    if len(faces) > 0:
        # deteksi wajah lebih dari 1 pada saat bersamaan
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct argument parser dan parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# memuat model deteksi wajah pada disk (ssd)
print("[INFO] memuat model deteksi wajah...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# memuat model deteksi masker dari disk
print("[INFO] memuat model deteksi masker...")
maskNet = load_model(args["model"])

# inisialiasasi video stream dari kamera
print("[INFO] memulai video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


# loop frame dari video stream
while True:
    # ambil frame dari stream video dan resize ukuran ke maximum 400x400 px
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # deteksi wajah pada frame menggunakan masker atau tidak
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # looping pada wajah lokasi wajah yang terdeteksi
    for (box, pred) in zip(locs, preds):
        # unpack bounding box dan prediksi
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # label kelas yang akan ditampilkan dan warna dari bounding box yang akan digunakan
        if mask > withoutMask:
            label = "Anda Telah Menggunakan Masker."
            color = (0, 255, 0)

        else:
            label = "Silahkan Pakai Masker Anda."
            color = (0, 0, 255)

      #  label = "Masker" if mask > withoutMask else "Tanpa Masker"
      #  color = (0, 255, 0) if label == "Masker" else (0, 0, 255)

      #  label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # tampilkan label dan bounding box pada output frame
        cv2.putText(frame, label, (startX-50, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # tampilkan pada frame
    cv2.imshow("Sistem Deteksi Masker", frame)
    key = cv2.waitKey(1) & 0xFF

    # tekan 'q' untuk menghentikan loop
    if key == ord("q"):
        break

fps.update()
fps.stop()
print("[INFO] waktu berjalan: {:.2f}".format(fps.elapsed()))
print("[INFO] perkiraan. FPS: {:.2f}".format(fps.fps()))

# stop
cv2.destroyAllWindows()
vs.stop()
