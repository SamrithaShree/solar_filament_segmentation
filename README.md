# Solar Filament Segmentation — MAGFiLO 2026

Deep Learning-based instance segmentation of solar filaments using multi-head UNet++, solar image preprocessing, multi-scale inference, and instance-level post-processing.

## 1. Introduction

Solar filaments are elongated structures of relatively cool, dense plasma suspended in the Sun's corona by magnetic fields. They appear as dark, thread-like structures against the brighter solar disk in H-alpha observations.

Accurately identifying and segmenting these structures is important for studying solar activity, magnetic configurations, and potentially eruptive events such as coronal mass ejections (CMEs).

However, automatic solar filament segmentation presents several challenges:

- **Thin and elongated structures:** Filaments have irregular shapes, varying widths, and large aspect ratios.
- **Low contrast:** Some filaments are difficult to distinguish from the surrounding solar surface.
- **Complex morphology:** Filaments may be fragmented, curved, connected, or overlapping.
- **Class imbalance:** Filament pixels occupy only a very small fraction of the solar disk compared with the background.
- **Annotation disagreement:** Multiple annotators may produce different filament boundaries and instance labels for the same observation.

### Objective

The objective of this project is to develop an instance segmentation pipeline that can identify individual solar filaments from H-alpha solar images, predict their segmentation masks, and distinguish separate filament instances.

Rather than relying solely on conventional binary segmentation, the approach combines pixel-level segmentation with filament spine and distance predictions, followed by a dedicated instance decoding pipeline.

## 2. Dataset Description

This project uses the **MAGFiLO 2026 solar filament segmentation competition dataset**.

The dataset contains H-alpha solar observations with COCO-format annotations.

### Dataset Statistics

| Component | Count |
|---|---:|
| Training images | 707 |
| Unique observations | 707 |
| Total annotations | 8,199 |
| Test images | 180 |
| Training observations | 603 |
| Validation observations | 104 |

### Annotation Format

The training annotations are provided in COCO JSON format.

The annotation data includes:

- Image metadata and image identifiers.
- Filament segmentation annotations.
- Polygon based segmentation representations.
- Filament spine coordinates.
- Multiple annotations/annotator records for observations were available.

The pipeline uses these annotations to construct three training targets:

1. **Filament mask:** Pixel-level filament segmentation.
2. **Filament spine:** Centreline representation of individual filaments.
3. **Distance map:** Distance-transform representation of filament regions.

### Dataset Organization

The notebook automatically locates the competition annotation JSON and derives the corresponding image directories.

The expected Kaggle dataset structure is:

```text
MAGFiLO_1.0_Kaggle_2026/
│
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│   └── train_images/
│       ├── image_1.jpg
│       ├── image_2.jpg
│       └── ...
│
└── test/
    └── test_images/
        ├── image_1.jpg
        ├── image_2.jpg
        └── ...
```

> **Note:** The competition dataset is not included in this repository. Access it through the official MAGFiLO 2026 competition and follow its data-access and usage requirements.

## 3. Methodology and Pipeline

The complete workflow consists of the following stages.

### 3.1 Solar-Physics-Aware Preprocessing

A custom three-channel preprocessing pipeline is applied to the original H-alpha observations.

The preprocessing includes:

- Solar disk detection and separation.
- Limb flattening to reduce intensity variation across the solar disk.
- Construction of three input channels incorporating solar image information.
- Resizing images to the selected model input resolution.

The disk mask is also retained to restrict predictions to the solar disk.

This preprocessing is designed to improve the visibility of faint and elongated filament structures.

### 3.2 Dataset Preparation and Exploratory Data Analysis

COCO annotations are parsed and converted into segmentation masks, spine representations, and distance maps.

Multiple annotator annotations are retained to account for differences in segmentation.

The notebook also performs exploratory data analysis covering:

- Distribution of filament counts per observation.
- Annotator disagreement.
- Filament size and aspect-ratio distribution.
- Percentage of solar disk occupied by filaments.
- Temporal variation in filament counts.

A date-based train-validation split is used to prevent observations from the same day from appearing in both sets.

The validation ratio is 15%, with a fixed random seed of 42.

### 3.3 Multi-Scale Training Data Preparation

The training pipeline uses filament-biased random-affine crops.

Crop sizes are sampled from multiple scales to improve the model's ability to recognize filaments of different sizes and aspect ratios.

The notebook's quality preset uses:

- Input resolution: 1024 × 1024
- Crop size: 512 × 512
- Crop scales: 0.75, 1.0, 1.25

All sampled crops are resized to a consistent dimension before being passed to the model.

### 3.4 Model Architecture

The segmentation architecture consists of a **three-head UNet++ model with an EfficientNet-B5 encoder**.

The encoder uses ImageNet-pretrained Noisy Student weights.

UNet++ uses nested skip pathways to combine features from different encoder and decoder levels.

The model predicts three outputs:

| Head | Prediction | Purpose |
|---|---|---|
| Mask head | Filament probability map | Identifies filament regions |
| Spine head | Filament spine probability map | Captures filament centreline structure |
| Distance head | Inner-distance map | Provides geometric information for instance separation |

This multi-head architecture allows the network to learn both filament regions and their structural properties.

### 3.5 Loss Functions

The model combines multiple loss functions to address class imbalance and preserve thin filament structures.

The implemented losses include:

- Focal Binary Cross-Entropy
- Lovász-Hinge Loss
- Soft Dice Loss
- clDice Loss
- Spine Loss
- Distance-map Loss

The combined objective is designed to balance pixel-level accuracy, overlap quality, and structural consistency.

### 3.6 Test-Time Augmentation (TTA)

The model uses test-time augmentation to improve prediction robustness.

The inference pipeline combines:

- Four rotations (0°, 90°, 180°, 270°).
- Horizontal flipping.
- Multiple image scales.

The final enhanced configuration uses four inference scales:

```python
INF_SCALES = (0.85, 1.0, 1.25, 1.5)
```

This results in 32 augmented forward passes per image (4 rotations × 2 flip states × 4 scales).

Predictions are transformed back to their original orientation and averaged.

### 3.7 Instance Decoding and Post-Processing

The predicted mask, spine, and distance probability maps are combined to generate individual filament instances.

The pipeline includes:

- Hysteresis-based thresholding.
- Morphological processing and gap bridging.
- Spine-guided instance separation.
- Connected-component and watershed-based operations.
- Minimum-area filtering.
- Instance-level evaluation.

Decoder parameters are optimized on the validation set using Optuna's Tree-structured Estimator.

A Gradient Boosting Classifier is also implemented to assess decoded instances using descriptive instance-level features. It is enabled only when validation results demonstrate a sufficient improvement.

### 3.8 Post-Training Improvements
The notebook includes additional experimental improvements:

- Wider multi-scale TTA.
- Expanded Optuna decoder search (200 trials).
- Instance-quality filtering with a 22-feature configuration.
- Batch Normalization Test Adaptation (BNTA).
- Gaussian probability-map smoothing.

These components are evaluated on the validation set before being applied to test inference.

### 3.10 Submission

The final instance labels are resized to the native image resolution of 2048 × 2048.

Each filament instance is encoded using COCO Run-Length Encoding (RLE).

The generated submission contains filament_id and its segmentation_rle.


## 4. Experimental Results

### 4.1 Dataset Analysis

| Metric | Result |
|---|---:|
| Mean filaments per observation | 7.36 |
| Mean annotator disagreement (std. of filament counts) | 0.50 |
| Mean filament coverage of solar disk | 0.46% |
| Mean filament aspect ratio | 5.19 |
| High-disagreement observations (std. > 1.5) | 75 |

The analysis demonstrates the elongated nature of solar filaments and the presence of annotation variability.

### 4.2 Validation Performance

The initial decoder configuration achieved a validation Panoptic Quality (PQ) of 0.3573.

Following Optuna-based decoder optimization, the validation performance improved:

| Metric | Optimized Validation Result |
|---|---:|
| Panoptic Quality (PQ) | 0.4379 |
| Recognition Quality (RQ) | 0.6529 |
| Segmentation Quality (SQ) | 0.6707 |

The expanded decoder search used 200 Optuna trials.

### 4.3 Post-Training Experiments

Additional improvements were evaluated independently.

- The 22-feature instance-quality filter was disabled because it did not improve validation PQ.
- Batch Normalization Test Adaptation and Gaussian smoothing were evaluated.
- Gaussian smoothing with sigma = 0.55 achieved PQ = 0.4333 after the BN adaptation stage.


## 5. Repository Structure

```text
solar_filament_segmentation/
│
├── pipeline.ipynb
│
├── README.md

```

## 6. Outputs and Model Artifacts

The trained model checkpoints, configuration files, experimental visualizations, and final submission generated by this project are hosted separately in a shared SharePoint folder.

**[Access All Project Outputs and Model Artifacts](https://ssneduin-my.sharepoint.com/:f:/g/personal/samrithashree24110486_snuchennai_edu_in/IgCOGHjEnN_FRZs4g7HKoIQaAVc_ao-jquK6vOwxrkNotdM?e=KBDaHj)**

### Available Files

| File | Description |
|---|---|
| `best_ema.pt` | Trained model checkpoint containing Exponential Moving Average (EMA) weights. |
| `model_config.json` | Model architecture and configuration parameters. |
| `eda_overview.png` | Exploratory Data Analysis visualizations of the solar filament dataset. |
| `optuna_decoder.png` | Visualization of the Optuna-based decoder optimization experiments. |
| `tuned_decoder_optuna_*.json` | Optimized decoder configuration obtained through Optuna hyperparameter search. |
| `submission.csv` | Final test predictions containing filament instance IDs and COCO RLE-encoded segmentation masks. |
| `test_partial.pkl` | Intermediate test inference cache used to support resuming interrupted inference. |


## 7. Reproducibility Instructions

The notebook is designed to run in **Kaggle Notebooks with GPU acceleration**.

Running it in Kaggle is recommended because the dataset is already available in it.

### Prerequisites

- A Kaggle account.
- Access to the MAGFiLO 2026 competition dataset.
- A Kaggle Notebook with GPU acceleration enabled.

### Step 1: Clone the Repository

```bash
git clone https://github.com/SamrithaShree/solar_filament_segmentation.git

cd solar_filament_segmentation
```

Alternatively, download the notebook directly and upload it to Kaggle Notebooks.

Open `[pipeline].ipynb`.

### Step 2: Attach the Dataset

In the Kaggle Notebook:

1. Open the **Add Input** section.
2. Add the MAGFiLO 2026 competition dataset.
3. Confirm that the annotation JSON and training/test image directories are accessible.

The notebook automatically searches for:

```text
MAGFiLO_1.0_Annotations_kaggle2026_train.json
```

It then locates the corresponding training and test image directories.

### Step 3: Attach the Trained Model Checkpoint

Download the trained model checkpoint from the shared [Project Outputs and Model Artifacts](https://ssneduin-my.sharepoint.com/:f:/g/personal/samrithashree24110486_snuchennai_edu_in/IgCOGHjEnN_FRZs4g7HKoIQaAVc_ao-jquK6vOwxrkNotdM?e=KBDaHj) folder.

- `best_ema.pt`

Upload the downloaded checkpoint to a Kaggle Dataset or Model and attach it to the notebook using the **Add Input** option.

The notebook automatically detects the checkpoint and stages it at `/kaggle/working/best_ema.pt` before inference.


### Step 4: Configure GPU Acceleration

In Kaggle:

1. Open Notebook Settings.
2. Select GPU as the accelerator.
3. Enable Internet access if pretrained encoder weights need to be downloaded.

The executed notebook used a Tesla T4 GPU with approximately 14.6 GiB VRAM.

### Step 5: Install Dependencies

The notebook contains an installation cell for its additional packages.

Run:

```python
!pip install -q segmentation-models-pytorch pycocotools scikit-image scikit-learn tqdm optuna
```

The remaining dependencies include:

```text
Python
PyTorch
NumPy
Pandas
OpenCV
Matplotlib
Seaborn
SciPy
scikit-learn
scikit-image
segmentation-models-pytorch
timm
pycocotools
tqdm
Optuna
```

Kaggle's preconfigured PyTorch environment provides the core GPU-enabled PyTorch installation.


### Step 6: Retrieve the Output

After execution, the final submission will be available at:

```text
/kaggle/working/submission.csv
```

In the Kaggle Notebook, open the output files and download `submission.csv`.

The notebook also performs sanity checks to verify:

- Correct submission columns.
- No missing values.
- Unique filament IDs.
- Successful RLE decoding of a sample prediction.


## 8.Reproducibility Notes

- The notebook uses a fixed random seed of 42 for its main data preparation and model setup.
- The date-based validation split is generated using a separate deterministic random seed.
- Intermediate observation caches, validation maps, decoder configurations, and partial test predictions are saved under the output directory.
- The checkpoint must be attached separately; it is not embedded in the GitHub repository.
- The notebook expects the competition dataset's original annotation and image organization.
- Running the notebook from the beginning in a fresh session will recompute the necessary preprocessing and validation stages unless valid caches are available.
- The full pipeline is optimized for Kaggle GPU execution rather than CPU-only local execution.

## 9. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Segmentation Architecture | UNet++ |
| Encoder | EfficientNet-B5 |
| Preprocessing | OpenCV, NumPy |
| Image Processing | SciPy, scikit-image |
| Annotation Handling | COCO, pycocotools |
| Optimization | Optuna, AdamW |
| Evaluation | Panoptic Quality, Recognition Quality, Segmentation Quality |
| Instance Classification | Gradient Boosting Classifier |
| Visualization | Matplotlib, Seaborn |
| Execution Environment | Kaggle Notebooks |


## 10. Acknowledgements

This project uses the MAGFiLO 2026 competition dataset and builds upon deep learning-based semantic and instance segmentation techniques.

---
