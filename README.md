# 🚘 AI Number Plate Detection System

A professional **AI-powered Number Plate Detection System** built with **YOLO** and **Streamlit**. The application allows users to upload vehicle images and automatically detects number plates using a custom-trained YOLO model.

The system provides a simple and modern web interface for running the trained computer vision model without requiring users to interact directly with Python code.

---

## 📌 Project Overview

Number plate detection is an important computer vision task used in applications such as:

* Intelligent transportation systems
* Vehicle monitoring
* Parking management
* Traffic monitoring
* Automated vehicle identification
* Security and surveillance systems

This project uses a custom-trained YOLO object detection model to locate number plates within vehicle images.

---

## ✨ Features

* **YOLO-based Object Detection**
* Custom-trained number plate detection model
* Image upload through a web interface
* Automatic number plate localization
* Bounding-box visualization
* Detection confidence scores
* Configurable confidence threshold
* Detection statistics
* Clean and responsive Streamlit interface
* Detection details displayed in a structured table
* Supports JPG, JPEG, and PNG images

---

## 🧠 Technology Stack

| Technology  | Purpose                       |
| ----------- | ----------------------------- |
| Python      | Core programming language     |
| YOLO        | Number plate object detection |
| Ultralytics | YOLO model implementation     |
| Streamlit   | Web application interface     |
| OpenCV      | Image processing              |
| Pillow      | Image handling                |
| Pandas      | Detection data handling       |
| PyTorch     | Deep learning framework       |

---

## 📂 Project Structure

```text
NumberPlateAI/
│
├── app.py
├── Best_Plate_Number_Detecting_Model.pt
├── requirements.txt
└── README.md
```

### File Description

**`app.py`**

Main Streamlit application containing the user interface and YOLO inference code.

**`Best_Plate_Number_Detecting_Model.pt`**

Custom-trained YOLO model used for number plate detection.

**`requirements.txt`**

Contains the Python dependencies required to run the application.

**`README.md`**

Project documentation and setup instructions.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd NumberPlateAI
```

---

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Streamlit will start a local web server and provide a URL that can be opened in a browser.

---

## 🖼️ How to Use

### Step 1 — Open the Application

Launch the Streamlit application.

### Step 2 — Upload an Image

Upload a vehicle image using the image uploader.

Supported formats:

```text
JPG
JPEG
PNG
```

### Step 3 — Configure Detection

Adjust the confidence threshold using the sidebar.

### Step 4 — Run Detection

Click:

```text
Detect Number Plate
```

The YOLO model analyzes the image and detects number plates.

### Step 5 — View Results

The application displays:

* Original image
* Detection result
* Bounding boxes
* Number of detected plates
* Average confidence
* Highest confidence
* Detection coordinates

---

## 🔍 Detection Output

For each detected number plate, the system provides information such as:

```text
Detection
Class
Confidence
X1
Y1
X2
Y2
```

The bounding-box coordinates represent the detected number plate location within the image.

---

## 🎯 Model

The project uses the following trained model:

```text
Best_Plate_Number_Detecting_Model.pt
```

The model is loaded using the Ultralytics YOLO framework:

```python
from ultralytics import YOLO

model = YOLO("Best_Plate_Number_Detecting_Model.pt")
```

---

## ⚠️ Important Note

This project performs **number plate detection**, meaning the model identifies and localizes the number plate using a bounding box.

Detection and character recognition are different tasks.

If the trained model only detects the number plate region, it will not automatically convert the characters on the plate into text such as:

```text
ABC-123
```

Character recognition would require an additional **OCR (Optical Character Recognition)** stage.

---

## 🚀 Future Improvements

Potential improvements include:

* Number plate character recognition using OCR
* Real-time webcam detection
* Video-based number plate detection
* Multiple vehicle tracking
* Automatic plate text extraction
* Detection history
* Database integration
* Vehicle entry and exit logging
* Cloud deployment
* Authentication and user management

---

## 📊 Application Workflow

```text
Vehicle Image
      │
      ▼
Image Upload
      │
      ▼
YOLO Model
      │
      ▼
Number Plate Detection
      │
      ▼
Bounding Box + Confidence
      │
      ▼
Streamlit Results
```

---

## 🔧 Requirements

The project requires:

```text
Python
Streamlit
Ultralytics
PyTorch
Torchvision
Pillow
Pandas
OpenCV
```

All dependencies are listed in:

```text
requirements.txt
```

---

## 🛡️ Model Usage

The model is intended for computer vision experimentation, learning, research, and application development.

Detection performance can vary depending on:

* Image quality
* Lighting conditions
* Camera angle
* Plate visibility
* Vehicle distance
* Plate design
* Dataset quality
* Model training parameters

---

## 👨‍💻 Author

**M.Aliyaan**

Computer Vision & Machine Learning Project

---

## 📄 License

This project can be used for educational and development purposes. If you redistribute the project or trained model, ensure that the underlying dataset and model components permit redistribution under their respective licenses.

---

## ⭐ Project

If you find this project useful for learning or development, consider giving the repository a star.
