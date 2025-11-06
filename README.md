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
   - Train using CNN or lightweight architectures (e.g., MobileNet, ResNet18)  
   - Tune hyperparameters for optimal accuracy  

3. **Evaluation**
   - Measure Accuracy, Precision, Recall, and F1-Score  
   - Visualize training and validation performance  

4. **ONNX Export**
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
1. [Kaggle](#)
2. [Google](#)
3. [roboflow](#)
 
 -**Total number of Datasets collected from these sources:2,992 Images**
 
 -**Number of images with FAW damage:1,882**
 
 -**Number of images with no FAW damage (Healthy Crops):1,110 images**

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

- [FAW Dataset Source](#)  
- [ONNX Documentation](https://onnx.ai/)  
- [TensorFlow](https://www.tensorflow.org/) / [PyTorch](https://pytorch.org/)  
- [Teachable Machine](https://teachablemachine.withgoogle.com/)

---

### 🧾 License
This project is for academic and research purposes only under the Techcrush AI Bootcamp 2025 Capstone framework.
