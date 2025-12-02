
Air Quality Classification — Time-Series to Image Encodings

This module of SENSE-Edge converts air-quality time-series data into visual representations using six image-encoding techniques inspired by the Time-VLM framework (ICML 2025).
These encoded images are then used to train Vision-Language Models (VLMs) for PM2.5 air-quality classification.

Dataset

We use the Beijing PM2.5 Air Quality Dataset (5-Class Version).

data/
 ├── beijing_pm25_5classes_paper.npz
 ├── PRSA_data_2010.1.1-2014.12.31.csv
 ├── clip_vlm_best.pt
 └── vlm_images_all_6_methods/


The .npz file contains:

X_train (N, T, D)

y_train (labels 0–4)

X_test

y_test

Class labels:

0  Very Low Risk
1  Low Risk
2  Medium Risk
3  High Risk
4  Very High Risk

Time-Series to Image Encoding (6 Methods)

All encodings are implemented in create_all_6_image_encodings.py.

1. Gramian Angular Field (GAF)

GAF-Summation

GAF-Difference
Transforms normalized signals into polar coordinates and constructs Gramian matrices.

2. Recurrence Plot (RP)

Binary recurrence structure showing repeated temporal states.

3. Continuous Wavelet Transform (CWT)

Time–frequency representation using manually implemented Ricker wavelets.

4. Markov Transition Field (MTF)

Encodes transition probabilities between quantized temporal states.

5. Greyscale Encoding

Direct normalization-based mapping of the time series into a 2D greyscale pattern.

6. Spectrogram (STFT)

Frequency evolution using Short-Time Fourier Transform.

Output Folder Structure

After running the script, the following structure will be created:

vlm_images_all_6_methods/
 ├── gaf_summation/train/<class>/*.png
 ├── gaf_summation/test/<class>/*.png
 ├── gaf_difference/...
 ├── recurrence_plot/...
 ├── cwt/...
 ├── mtf/...
 ├── greyscale/...
 └── spectrogram/...


Each method generates images for all five classes, split into train and test sets.
All images are resized to 224x224x3.

How to Run

From inside the air_quality directory:

python create_all_6_image_encodings.py


The script will:

Load the PM2.5 dataset

Generate images for all six encoding methods

Save outputs inside data/vlm_images_all_6_methods/

Create a comparison figure named all_methods_comparison.png

Model Training

Training scripts provided in this module:

train_mlp_simple.py
train_vlm_simple.py
train_vlm_fixed.py


Output files include:

*_results.txt
*_training_history.png
all_methods_comparison.png


These allow comparison across encoding types to identify the best-performing representation.
