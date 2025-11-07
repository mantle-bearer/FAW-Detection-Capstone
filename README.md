# 🐛 Fall Armyworm Supervised AI Detection

**Capstone Project – AI Bootcamp 2025**

This project focuses on developing a **supervised machine learning model** for the detection and classification of the **Fall Armyworm (FAW)** (*Spodoptera frugiperda*) from visual data (images or videos).  
It applies core **supervised learning** techniques—classification and detection—to address a real-world agricultural problem threatening food security across Africa.

---

## 🌍 Project Overview

The **Fall Armyworm** is a destructive pest responsible for over **31% annual maize loss in Africa**.  
Early and accurate detection of FAW larvae, pupae, or moths can significantly reduce crop damage.

This project aims to build a **compact, deployable, and performant AI model** capable of detecting FAW presence from visual inputs.  
The trained model will be exported to **ONNX format**, making it ready for real-world deployment on mobile or embedded systems.

---

## 🎯 Objectives

- **Apply Supervised Learning:** Implement classification or regression using Python-based AI/ML libraries.  
- **Develop Computer Vision Skills:** Handle image data preprocessing, augmentation, and model training.  
- **Optimize Model Performance:** Select lightweight architectures suitable for limited compute environments (e.g., Google Colab).  
- **Enable Deployment Readiness:** Export trained models to **ONNX format** for cross-platform compatibility.  
- **Ensure Reproducibility:** Maintain clear documentation and well-structured code.

---

## ⚙️ Technical Setup

| Category | Requirement | Notes / Tools |
|-----------|--------------|----------------|
| **Development Environment** | Online only (no local setup) | Google Colaboratory (Colab) |
| **Libraries / Tools** | Free and open-source | Python (TensorFlow, Keras, scikit-learn, PyTorch if feasible) or No-Code (MATLAB, Teachable Machine) |
| **Model Type** | Supervised Learning (Classification / Detection) | Focus on FAW presence or stage classification |
| **Data Source** | Custom FAW image dataset (plus optional public data) | Data augmentation encouraged |
| **Output Format** | ONNX | Enables deployment on web, mobile, or embedded devices |

---

## 🧠 Methodology

1. **Data Loading & Preprocessing**
   - Load FAW dataset  
   - Clean and augment images (rotation, flipping, resizing)  

2. **Model Selection & Training**
   - Train using CNN to train models (e.g., MobileNetV2, EfficientNetB0, ResNet50)  
   - Pick the best performing model using f1-score and recall metrics to evaluate (DenseNet121)
   - Tune hyperparameters for optimal accuracy   

4. **Evaluation**
   - Measure Accuracy, Precision, Recall, and F1-Score  
   - Visualize training and validation performance  

5. **ONNX Export**
   - Convert the trained model to ONNX format for portability  

---

## 📊 Deliverables

- ✅ **Google Colab Notebook**: End-to-end implementation (Data → Model → Evaluation → Export)  
- ✅ **ONNX Model File**: Final trained model (`model.onnx`)  
- ✅ **Documentation**: Clear explanation of pipeline, dataset, results, and deployment readiness  
- ✅ **Presentation Slides**: Summary of results and key findings  

---

## 🧩 Example Tech Stack

- Python 3.x  
- TensorFlow / Keras / PyTorch  
- OpenCV, NumPy, Pandas, Matplotlib  
- Google Colab  
- ONNX Runtime  

---

## Data Sources
1. [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/armyworm-crop-challenge)
2. [Google](https://www.google.com/search?sa=X&sca_esv=87839103094c6a77&rlz=1CDGOYI_enUS1110US1111&hl=en-US&udm=2&sxsrf=AE3TifOP8ZwH2Ie9XcBCSjQxbABsqomYXw:1762340176363&q=maiz&stick=H4sIAAAAAAAAAFvEypKbmFkFAE_so2wHAAAA&source=univ&ved=2ahUKEwjTzL3h7NqQAxVa_8kDHWw3DWIQrNwCegUI0AEQAA&biw=375&bih=640&dpr=3)
3. [Roboflow](https://universe.roboflow.com/gluwxy-nqeon/fall_armyworm-detection)
 
- Total number of Datasets collected from these sources: = 2,989 Images
 
- Number of FAW images: = 1,882
 
- Number of No FAW images: = 1,107 images

---

## 🛠️ Setup Instructions (Google Colab Only)

This section guides you through setting up and running the **Fall Armyworm Supervised AI Detection** project entirely in **Google Colaboratory (Colab)** — no local installation is required.

---

### 1. Open in Google Colab

1. Visit [Google Colab](https://colab.research.google.com/).
2. Click on **File → Open Notebook → GitHub**.
3. Paste the repository link (e.g., `https://github.com/mantle-bearer/FAW-Detection-Capstone`).
4. Open the notebook file (usually named `FAW_Detection.ipynb` or similar).

Alternatively, you can open it directly by clicking the **“Open in Colab”** badge if included in the repository.

---

### 2. Mount Google Drive

Mount your Google Drive to access and save datasets, logs, and trained models.

```python
from google.colab import drive
drive.mount('/content/drive')
```
Once mounted, ensure your working directory points to your project folder:
```python
%cd /content/drive/MyDrive/FAW_Detection_Capstone/
```

---

### 3. Install Dependencies

Google Colab comes with many pre-installed libraries.
However, to ensure full compatibility, install any missing dependencies using:
```python
!pip install tensorflow keras scikit-learn matplotlib opencv-python onnx onnxruntime
```
You can also include any other libraries your notebook references (e.g., pandas, torch, albumentations).

---

### 4. Load the Dataset

A custom Fall Armyworm (FAW) dataset will be provided during the bootcamp.
To load it into Colab:

Upload it to your Google Drive (e.g., /MyDrive/FAW_Dataset/).

Extract the dataset if it’s compressed:
1. Upload it to your Google Drive (e.g., /MyDrive/FAW_Dataset/).
2. Extract the dataset if it’s compressed:
   ```python
   import zipfile
   with zipfile.ZipFile('/content/drive/MyDrive/FAW_Dataset.zip', 'r') as zip_ref:
   zip_ref.extractall('/content/drive/MyDrive/FAW_Dataset/')
   ```

3. Confirm dataset accessibility:
   ```python
   import os
   os.listdir('/content/drive/MyDrive/FAW_Dataset/')
   ```
---
   
### 5. Run the Notebook Cells Sequentially

-   Run each cell in order from top to bottom:
-   Data Loading
-   Preprocessing & Augmentation
-   Model Building & Training
-   Evaluation
-   Export to ONNX

Each step is clearly labeled within the Colab notebook.
Avoid skipping cells to maintain reproducibility.

---

### 6. Save Your Trained Model

Once training is complete, save your best-performing model to Drive:
```
model.save('/content/drive/MyDrive/FAW_Model/FAW_Classifier.h5')
```

Convert to ONNX format for deployment compatibility:
```python
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model('/content/drive/MyDrive/FAW_Model/FAW_Classifier.h5')
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "/content/drive/MyDrive/FAW_Model/FAW_Classifier.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
print("✅ Model successfully exported to ONNX!")
```
---

### 7. Verify ONNX Model

To ensure your exported ONNX model works properly:
```python
import onnx
import onnxruntime as ort

onnx_model = onnx.load('/content/drive/MyDrive/FAW_Model/FAW_Classifier.onnx')
onnx.checker.check_model(onnx_model)
print("✅ ONNX Model is valid and ready for deployment.")
```
---

### 8. Proceed to Evaluation and Documentation

After confirming successful training and export:

-   Record metrics (Accuracy, Precision, Recall, F1-Score).
-   Document data preprocessing, architecture choice, training process, and results directly within the Colab notebook (using Markdown cells).
-   Ensure all visuals (graphs, confusion matrices) are properly displayed.

---

💡 Tip: Always connect to a GPU runtime for faster training.
Go to Runtime → Change runtime type → Hardware accelerator → GPU.

✅ You’re now ready to train, evaluate, and document your Fall Armyworm Detection Model entirely in Google Colab.

---

## 👥 Team

**Team Name:** Group U 
**Members:**  
- Goodluck  
- Deborah
- ThankGod
- Daniel
- Augustine
- Opeyemi
- Faiz
- Christabel
- Aisha
- Kazeem
- Similoluwa
- Chinazor
- Jeffery
- Jesse
- Lawal
  
**Instructor/Supervisor:** Mr. Hammed OBASEKORE

---

## 📚 References

- [Final Dataset Source](https://drive.google.com/drive/folders/1yP-eU-6Itm0Vb_wbMAcyJtpqPfkkQ3E5?usp=sharing)
- [ONNX Documentation](https://onnx.ai/)  
- [TensorFlow](https://www.tensorflow.org/) / [PyTorch](https://pytorch.org/)  
- [Teachable Machine](https://teachablemachine.withgoogle.com/)

---

### 🧾 License
This project is for academic and research purposes only under the Techcrush AI Bootcamp 2025 Capstone framework.
