# Face Mask Detection (CNN + Transfer Learning)

Detects whether a person is **wearing a mask** or **not** using TensorFlow/Keras with a pretrained CNN (VGG16/ResNet).

---

## Objective

Automatically classify images into:

- WithMask
- WithoutMask

---

## Workflow

1. Prepare dataset (Train / Validation / Test).
2. Preprocess images (resize, rescale, augment).
3. Load pretrained CNN (VGG16/ResNet), add custom layers.
4. Train and validate model.
5. Test on unseen images.
6. Use `load.py` to predict new images.

---

## Model Architecture

Input (224x224x3) → Pretrained CNN → Global Pooling → Dense (ReLU) → Dropout → Softmax (2 classes)

---

## Dataset Structure

Dataset/

├── Train/

├── Validation/

└── Test/

Each contains:
├── WithMask/
└── WithoutMask/

---

## Project Files

train_model.ipynb # training
load.py # prediction
requirements.txt
.gitignore
README.md

---

## How to Run

pip install -r requirements.txt

Run `train_model.ipynb` to train.  
Run `python load.py` to test.

---

## Tech Used

Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Scikit-learn

---

## Notes

- Dataset and model files are not included (large size).
- Works with any similar mask dataset.

---

## Author

Naveen Kumar, Deepak Kumar, Hans Raj
