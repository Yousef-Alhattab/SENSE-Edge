# Air Quality Classification — Time-Series to Image Encodings

This module of SENSE-Edge converts air-quality time-series data into visual representations using six image-encoding techniques.
These encoded images are used to train Vision-Language Models (VLMs) for PM2.5 air-quality classification.

---

## Dataset

We use the Beijing PM2.5 Air Quality Dataset (5-Class Version).

```
data/
 ├── beijing_pm25_5classes_paper.npz
 ├── PRSA_data_2010.1.1-2014.12.31.csv
 ├── clip_vlm_best.pt
 └── vlm_images_all_6_methods/
```

The .npz file contains:

- X_train (N, T, D)  
- y_train (labels 0–4)  
- X_test  
- y_test  

Class labels:

```
0  Very Low Risk  
1  Low Risk  
2  Medium Risk  
3  High Risk  
4  Very High Risk
```

---

## Time-Series to Image Encoding (6 Methods)

All encodings are implemented in:

```
create_all_6_image_encodings.py
```

### 1. Gramian Angular Field (GAF)
- GAF-Summation  
- GAF-Difference  
Transforms normalized signals into polar coordinates and constructs Gramian matrices.

### 2. Recurrence Plot (RP)
Binary recurrence structure describing repeated temporal states.

### 3. Continuous Wavelet Transform (CWT)
Time–frequency representation using manually implemented Ricker wavelets.

### 4. Markov Transition Field (MTF)
Encodes temporal transition probabilities between quantized states.

### 5. Greyscale Encoding
Simple normalization-based mapping of the multivariate time-series into a 2D greyscale image.

### 6. Spectrogram (STFT)
Frequency evolution using the Short-Time Fourier Transform.

---

## Output Folder Structure

After running the script:

```
vlm_images_all_6_methods/
 ├── gaf_summation/
 │     ├── train/<class>/*.png
 │     └── test/<class>/*.png
 ├── gaf_difference/
 ├── recurrence_plot/
 ├── cwt/
 ├── mtf/
 ├── greyscale/
 └── spectrogram/
```

Each method generates training and test images for all five classes.  
All images are resized to 224 × 224 × 3.

---

## How to Run

Inside the `air_quality` directory:

```
python create_all_6_image_encodings.py
```

The script performs:

1. Loading of the PM2.5 dataset  
2. Generation of all six image encodings  
3. Saving output images under `data/vlm_images_all_6_methods/`  
4. Creating a comparison figure `all_methods_comparison.png`

---

## Model Training

Below is an example of the VLM training progress:

![Training Progress](doc/images/train_progress.png)

Training scripts provided:

```
train_mlp_simple.py
train_vlm_simple.py
train_vlm_fixed.py
```

Outputs include:

```
*_results.txt
*_training_history.png
all_methods_comparison.png
```

These scripts support comparing different encoding methods to determine the best-performing representation.

---

## Role in SENSE-Edge

This module demonstrates the Sensor-to-Image pipeline in SENSE-Edge:

1. Raw environmental data is collected as time series  
2. Converted into structured visual encodings  
3. Vision-Language Models process the generated images  
4. Suitable for real-time, explainable AI on edge devices  

This supports the neuro-symbolic and agentic architecture of SENSE-Edge
