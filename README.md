# CNN-MobilenetV2-RaspberryPi-based-FaceMaskDetection
This is a Face mask detection project using CNN algorithm, design to run on a raspberry pi 3 b+. But, the model was trained on my notebook.


How to use:

1. Install all requirements needed : (i recommend you to just install conda first, it's easier to minimize error on installation)
Python 3.7 or newer,
cv2,
numpy,
matplotlib,
tensorflow,
imutils
    
2. Clone the repository

3. Run the take_dengan_masker.py and take_tanpa_masker.py to take your dataset or you can add dataset manually to the folder (dataset). note: dengan=With, tanpa=Without

4. After that you can run train.py to make your model. ('python train.py -d dataset -p plot.png -m model.model')

5. Now you can run the face mask detection. ('python run sistem_deteksi_masker.py -m modelname.model')
<img src="/demo.png" style="height: 600px; width:600px;"/>
