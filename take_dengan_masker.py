import cv2

name = 'dengan_masker'  # input ke folder dengan_masker

cam = cv2.VideoCapture(0)

cv2.namedWindow("Ambil Foto (tekan spasi)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ambil Foto (tekan spasi)", 500, 300)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Gagal mengambil gambar")
        break
    cv2.imshow("Ambil Foto (tekan spasi)", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC untuk keluar
        print("Escape ditekan, menutup...")
        break
    elif k % 256 == 32:
        # SPACE untuk ambil foto
        img_name = "dataset/" + name + "/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} input sukses!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
