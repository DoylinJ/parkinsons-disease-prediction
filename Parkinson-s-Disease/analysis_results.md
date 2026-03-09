# Parkinson's Disease Gait Detection Codebase Analysis

## Overview
This repository contains an end-to-end pipeline for detecting Parkinsonian symptoms from video sequences of patient turning tasks. It combines computer vision (via MediaPipe kinematics) with custom machine learning ensembles. Notably, due to the lack of clinical "healthy control" videos, the system robustly simulates control data and heavily employs Monte Carlo inference and custom bagging variants to build probabilistically confident results (rather than overly deterministic pseudo-results).

## Component Analysis

### 1. [pose_extract.py](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/pose_extract.py) (Stage 1: Tracking)
- **Role**: MediaPipe integration for skeletal joint tracking.
- **Mechanism**: Extracts (x, y) coordinates of the 6 core leg joints: Left/Right Hip, Knee, Ankle. 
- **Optimizations**: Downscales input videos (width > 480) and skips frames (`FRAME_SKIP = 3`) to dramatically accelerate inference without sacrificing tracking quality. 
- **Output**: Returns a NumPy array of shape [(Frames, 6, 2)](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/custom_bagging.py#15-52).

### 2. [turning_features.py](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/turning_features.py) (Stage 2: Feature Engineering)
- **Role**: Biomechanical analysis.
- **Mechanism**: Translates the raw temporal 2D coordinate streams into 8 interpretable, gait-specific spatial-temporal measurements indicative of Parkinson's:
  - `total_turn_angle` and `mean_angular_velocity`
  - `turn_duration`
  - `num_steps`, `mean_step_length`, `step_variability` (to track shuffling)
  - `freeze_frames` (freezing of gait proxy)
  - `knee_rom` (range of motion)

### 3. [build_feature_csv.py](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/build_feature_csv.py) (Data Pipeline and Augmentation)
- **Role**: Dataset instantiation and expansion.
- **Process**:
  1. Pulls real clinical patient videos from `Actual videos/` directory and extracts features.
  2. **PD Augmentation**: Synthesizes 10 additional variations for each real PD video by adding a ±5% Gaussian noise jitter, simulating variations in camera angle and measurement.
  3. **Synthetic Controls Construction**: To generate the "Healthy / Non-PD" class without actual data, it transforms the augmented PD cases using randomized scalar factors (e.g., shorter duration, faster velocity, larger stride, minimal freezing). It crucially includes an overall ±8% noise factor to create a probabilistic "fuzzy" class border to ensure the model doesn't overfit perfectly.
- **Output**: Populates [turning_pd_features.csv](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/turning_pd_features.csv) containing structurally balanced 50/50 samples.

### 4. [custom_bagging.py](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/custom_bagging.py) (Algorithm Implementation)
- **Role**: Custom Ensemble Algorithm.
- **Mechanism**: Implements an enhanced Bagging classifier from scratch. It differs from strictly standard `sklearn` Bagging through structural feature subsampling (combining row bootstrapping *with* random forest column drops) and has a tailored [predict_proba()](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/custom_bagging.py#53-73) method to gracefully sum unnormalized tree votes across estimators.

### 5. [train_model.py](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/train_model.py) (Training Orchestration)
- **Role**: Model Generation.
- **Process**: Trains the custom model using 1,000 un-pruned `DecisionTreeClassifier` objects. Extremely importantly, it limits individual tree strength (`min_samples_leaf=4` and `max_depth=8`) so no single tree is able to act deterministically. The aggregation across 1000 weak learners forms the baseline probability curve.
- **Output**: Persists the model state as [model.pkl](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/model.pkl).

### 6. [predict_folder.py](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/predict_folder.py) (Inference Engine)
- **Role**: Clinical outcome grading and reporting.
- **Mechanism**: Scans the `Videos/` directory for test cases.
- **Monte Carlo Inference**: Recognizing that MediaPipe is imperfect, this script does not just classify once. It runs 200 distinct inferences for the video, injecting ±6% Gaussian noise corresponding to anticipated sensor uncertainty per trial. 
- **Output**: Bypasses binary predictions by aggregating the Monte Carlo tests, yielding a mean score and a robust 95% Confidence Interval grading of PD manifestation.

### 7. Scripts and Assets
- **[run_pipeline.bat](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/run_pipeline.bat)**: User-friendly sequential script invoking data extraction, training, and prediction consecutively.
- **[README.md](file:///c:/Users/arvin/.gemini/antigravity/scratch/honors_project_Parkinson-s_disease/README.md)**: Provides substantial documentation explaining the pipeline, mathematical rationale, architecture diagrams, and medical disclaimers.
