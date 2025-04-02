# Pothole and Crack Detection using YOLOv8s

## Overview
This project implements an advanced computer vision system for automated pothole and crack detection in road infrastructure. The model is built using YOLOv8s (You Only Look Once) architecture, enabling efficient and accurate detection of road surface defects in real-time under various environmental conditions.

## Features
- **Real-time object detection** for potholes and cracks.
- **Custom-trained YOLOv8s model** on a curated dataset.
- **Data augmentation and preprocessing** for improved generalization.
- **Optimized hyperparameters** to enhance precision and recall.
- **100 training epochs** for balanced detection accuracy and processing speed.

## Dataset
The model was trained on a dataset of **195 road surface images**, containing diverse examples of potholes and cracks. Various data preprocessing and augmentation techniques were applied to improve model robustness.

## Technical Implementation
1. **Dataset Preparation:**
   - Collected and labeled images of potholes and cracks.
   - Applied preprocessing techniques like normalization and resizing.
   - Augmented images to increase dataset diversity.

2. **Model Training:**
   - Used YOLOv8s for efficient object detection.
   - Trained the model for **100 epochs** to achieve optimal accuracy.
   - Fine-tuned hyperparameters for enhanced performance.

3. **Evaluation & Optimization:**
   - Assessed model accuracy using **precision, recall, and F1-score**.
   - Adjusted parameters to balance speed and accuracy.
   - Validated performance on unseen test data.

## Results
- Developed an **efficient real-time detection system** for potholes and cracks.
- Successfully identified road defects despite a limited dataset.
- Created a scalable solution to improve infrastructure maintenance workflows.

## Installation & Usage
### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8

### Installation
```bash
pip install ultralytics opencv-python numpy torch torchvision
```

### Running the Model
```python
from ultralytics import YOLO
import cv2

# Load the trained YOLOv8s model
model = YOLO("path_to_trained_model.pt")

# Load and detect objects in an image
image = cv2.imread("test_image.jpg")
results = model(image)

# Display results
results.show()
```

## Future Improvements
- Expand dataset with more diverse road surface conditions.
- Optimize inference speed for deployment on edge devices.
- Implement integration with GIS for automated mapping of detected defects.

## Contributors
- [Your Name]

## License
This project is licensed under the MIT License.

