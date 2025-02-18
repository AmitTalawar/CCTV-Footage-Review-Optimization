# CCTV Footage Review Optimization

## Overview
This project presents a system for optimizing the review of CCTV footage by automating the detection of motion and object identification. By using advanced techniques in motion detection, adaptive thresholding, and object detection (MobileNet SSD and YOLOv8), the system efficiently extracts relevant video segments, reducing manual review time and improving overall surveillance effectiveness.

## Table of Contents
- [Overview](#overview)
- [Project Details](#project-details)
- [Methodology](#methodology)
  - [Motion Detection](#motion-detection)
  - [Object Detection](#object-detection)
- [Results and Discussion](#results-and-discussion)

## Project Details
CCTV Footage Review Optimization addresses the challenge of manually reviewing extensive CCTV footage. Given that a significant portion of the footage contains no activity, this project automates the process of detecting motion and extracting segments of interest. The system leverages both traditional computer vision techniques (such as background estimation and various thresholding methods) and state-of-the-art deep learning models for object detection.

## Methodology

### Motion Detection
- **Background Estimation:**  
  The system employs the median frame technique to robustly estimate a static background, reducing the impact of transient objects.
  
- **Thresholding Techniques:**  
  Several thresholding methods are implemented to convert grayscale images into binary images for effective motion detection:
  - **Global Thresholding:** Uses a fixed threshold value to segment the image.
  - **Otsu Binarization:** Automatically calculates the optimal threshold by maximizing the between-class variance.
  - **Adaptive Thresholding:** Dynamically computes local threshold values, including:
    - Adaptive Mean Thresholding
    - Adaptive Gaussian Thresholding
  
  After thresholding, contours and bounding boxes are drawn around detected motion areas, and timestamps are recorded to mark the start and end of each motion event.

### Object Detection
- **MobileNet SSD:**  
  Initially used for object detection, this model performs well for near-field objects but may struggle with distant ones.
  
- **YOLOv8:**  
  Adopted for its improved speed and accuracy, YOLOv8 is an anchor-free detection model that excels in varied environments and real-time applications.

## Results and Discussion
Testing on various CCTV scenarios, including open areas and nighttime footage, demonstrated:
- **Motion Detection:**  
  Adaptive thresholding methods significantly outperformed global thresholding by accurately detecting motion regardless of lighting conditions and scene variability.
- **Object Detection:**  
  YOLOv8 provided more reliable object detection compared to MobileNet SSD, especially in challenging conditions.
  
Overall, the system effectively reduces the amount of footage that requires manual review, thereby enhancing the efficiency of surveillance operations.
