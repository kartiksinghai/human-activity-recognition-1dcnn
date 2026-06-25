# Human Activity Recognition from Smartphone Sensor Signals using a 1D CNN

This project classifies six human activities from smartphone sensor signals.

It compares two approaches:

1. **Random Forest** trained on 561 manually engineered features.
2. **1D Convolutional Neural Network (1D CNN)** trained directly on raw smartphone sensor signals.

The main aim is to check whether a 1D CNN can learn activity patterns directly from raw sensor data and compare its performance with a strong classical machine-learning baseline.

---

## Why This Project Is Useful

Human Activity Recognition (HAR) means identifying what a person is doing using sensor data from devices such as smartphones, smartwatches, fitness bands, and IoT devices.

This type of system can be used in:

- Fitness tracking applications
- Step and movement monitoring
- Smartwatches and health bands
- Elderly-care monitoring
- Fall detection systems
- Rehabilitation monitoring
- Patient mobility tracking
- Workplace safety monitoring
- Smart-home automation
- Sports activity tracking

For example, a smartphone can use accelerometer and gyroscope signals to identify whether a person is walking, sitting, standing, or lying down.

---

## Objective

The main question explored in this project is:

> Can a 1D CNN learn useful human activity patterns directly from raw smartphone sensor signals, and how close can it get to a Random Forest model trained on handcrafted features?

The project compares:

- A classical machine-learning model using manually engineered features.
- A deep learning model using raw time-series sensor signals.

---

## Dataset

This project uses the UCI Human Activity Recognition Using Smartphones Dataset.

The dataset contains smartphone sensor data collected while participants performed daily activities.

### Dataset Summary

- Total subjects: **30**
- Total samples: **10,299**
- Training samples: **7,352**
- Test samples: **2,947**
- Number of activities: **6**
- Sampling frequency: **50 Hz**
- Window length: **128 time steps**
- Window duration: approximately **2.56 seconds**
- Official train and test split is subject-disjoint.

Subject-disjoint means that the people in the test dataset are different from the people in the training dataset.

This is important because the model must recognize activities for new users instead of memorizing movement patterns from people it has already seen.

---

## Activity Classes

The six activities are:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

---

## Input Data

The project uses two different forms of input data.

### 1. Raw Sensor Signals for CNN

The 1D CNN receives raw sensor sequences from 9 channels:

- Body acceleration X-axis
- Body acceleration Y-axis
- Body acceleration Z-axis
- Body gyroscope X-axis
- Body gyroscope Y-axis
- Body gyroscope Z-axis
- Total acceleration X-axis
- Total acceleration Y-axis
- Total acceleration Z-axis

Each sample has the shape:

```text
(128 time steps, 9 sensor channels)
