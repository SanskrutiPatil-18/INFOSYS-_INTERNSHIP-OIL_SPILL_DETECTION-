# INFOSYS-INTERNSHIP-OIL_SPILL_DETECTION-
PROJECT SUBMISSIONS

# 🛢️ Oil Spill Detection using Deep Learning

This project focuses on **detecting oil spills in water bodies** from satellite images using **image segmentation** with deep learning. The model is trained to separate oil spill regions from background (water/land) and provides predictions in the form of binary masks.

---

## 📌 Project Overview

* **Goal:** Automatically detect oil spills from satellite imagery.
* **Dataset:** ~1000 annotated training images and masks(oil spill vs background).
* **Approach:**

  * Preprocessing and augmentation of training data.
  * U-Net–style convolutional neural network for segmentation.
  * Training, validation, and evaluation using IoU and Dice coefficient.
  
* **Outcome:**

  * Mean Dice Score: ~0.81
  * Mean IoU: ~0.75

---

## 📂 Repository Structure

```
├── oilspill_training.ipynb   # Main notebook: training & evaluation
├── data/                     # Dataset (not included, path only)
├── models/                   # Saved trained models (.keras)
├── results/                  # Example predictions & visualizations
├── README.md                 # Project documentation
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

---

## 🚀 Training Pipeline

1. **Preprocessing**

   * Resize images & masks.
   * Normalize pixel values.
   * Apply augmentations (rotation, flips, brightness).

2. **Model Architecture**

   * U-Net encoder–decoder style CNN.
   * Convolutional + BatchNorm + ReLU blocks.
   * Skip connections to preserve spatial info.

3. **Loss & Metrics**

   * Binary Crossentropy + Dice loss.
   * Evaluation metrics: Accuracy, Dice coefficient, IoU.

4. **Training**

   ```python
   history = model.fit(
       train_ds,
       validation_data=val_ds,
       epochs=50,
       callbacks=[early_stopping, model_checkpoint]
   )
   ```

5. **Evaluation**

   ```python
   dice, iou = evaluate_model(best_model, val_ds)
   print("Mean Dice:", dice, "Mean IoU:", iou)
   ```

---

## 📊 Results

* **Mean Dice:** 0.81
* **Mean IoU:** 0.75
* **Accuracy:** ~91%. 

---

## ⚠️ Limitations

* Dataset is small (~1000 images) and lacks diversity.
* Model sometimes under-segments oil spill boundaries.

---

## 🔮 Future Work

* Collect more diverse dataset (different lighting, regions).
* Use **transfer learning** with pretrained backbones (ResNet, EfficientNet).
* Experiment with advanced loss functions (Focal Loss, Tversky Loss).
* Deploy model as a **web app** (Streamlit/Gradio) for real-time usage.

---

## 👩‍💻 Author

* **Sanskruti Patil**
  *Infosys Springboard Internship Project – Oil Spill Detection*
