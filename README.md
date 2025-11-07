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

## 🛠️ Setup Instructions

Follow these steps to get the **Fall Armyworm Supervised Learning Model Classification** project running on your **Google Colab** environment.

---

### **1. Open Repository In Google Colab (Recommended)**
Open the repository in **Google Colab** and mount your **Google Drive** for storing the data.

---

### **2. Set Up Google Colab **

1. **Open Colab**:
   Go to [Google Colab](https://colab.research.google.com/) and open the notebook from **Google Drive**.

2. **Mount Google Drive**:
   Mount your Google Drive to access the dataset and save the trained models.

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Dataset**:
   Make sure the **Fall Armyworm dataset** is available in your **Google Drive**. You can use the **shared link** from the **Data Sources** section to download it.

---

### **3. Model Training and Export to ONNX**

* After setting up, **train your model** (if not already done) using the provided Google Colab notebook.
* Once the training is complete, **export the model to ONNX** format using the script provided in the notebook.

Example:

```python
import onnx
onnx.save(onnx_model, 'faw_model.onnx')
```

Upload the **`model.onnx`** to your **FastAPI app** directory for inference.

---

### **4. Deployment (Optional)**

* The **ONNX model** can be loaded into the deployed server for real-time inference.

---

### **5. Troubleshooting**

If you encounter issues while running the **Fall Armyworm Detection Model** in **Google Colab**, here are common problems and their fixes:

---

 ⚠️ 1. **"ModuleNotFoundError" or "Package Not Found"**

**Cause:**  
Some required libraries might not be pre-installed in Colab.

**Fix:**  
Re-run the installation cell to ensure all dependencies are installed correctly:

```python
!pip install tensorflow keras matplotlib scikit-learn opencv-python
```
---

This section should provide a clear setup path for running the project in **Google Colab**.

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
