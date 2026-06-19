#  Scoliosis Analysis & Cobb Angle Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end computer vision application that automatically detects spinal vertebrae on standard X-ray radiographs, maps spinal curvature, and mathematically estimates the clinical **Cobb Angle** — the standard metric used to quantify scoliosis severity.

<p align="center">
  <img src="/scoliosis_result.jpg" alt="Demo output of Cobb angle estimation overlay" width="480"/>
  <br/>
  <em>Example output: detected vertebrae, fitted spinal curve, and estimated Cobb angle overlay.</em>
</p>

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset Attribution](#dataset-attribution)
- [Project Architecture](#project-architecture)
- [Installation and Execution](#installation-and-execution)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Automated Vertebrae Detection** — A custom-trained YOLOv8 object detection model performs high-precision anatomical landmark localization directly on radiographic images.
- **Continuous Curve Fitting** — A 3rd-degree polynomial fit (`np.polyfit`) renders a smooth, continuous spinal trajectory across all detected vertebral nodes.
- **Geometric Angle Calculation** — The Cobb angle is computed via cosine-similarity operations on vectors derived from the superior, apical, and inferior vertebrae coordinates.
- **Multilingual User Interface** — A responsive, state-persistent web app built with Gradio Blocks, supporting real-time language toggling between English and Turkish.
- **Diagnostic Visualization** — Semi-transparent segmentation masks and computed metrics are overlaid directly onto the source radiograph for fast visual review.

## Technology Stack

| Layer | Tools |
|---|---|
| Deep Learning | Ultralytics YOLOv8, PyTorch |
| Computer Vision & Math | OpenCV, NumPy, SciPy |
| Application Interface | Gradio (Blocks Architecture) |
| Data Engineering | Pandas, Scikit-learn |

## Model Performance & Metrics

The model was evaluated on the validation set, demonstrating robust localization of vertebrae structures. Below are the primary convergence metrics and confidence evaluations extracted from the training cycle.

### Training & Validation Loss
<p align="center">
  <img src="assets\results.png" width="400">
  <br/>
</p>

### Inference Evaluation
| F1-Confidence Curve | Precision-Recall Curve |
| :---: | :---: |
| <img src="assets\BoxF1_curve.png" width="400"> | <img src="assets\BoxPR_curve.png" width="400"> |

### Dataset & Predictions
| Normalized Confusion Matrix | Dataset Label Distribution |
| :---: | :---: |
| <img src="assets\confusion_matrix_normalized.png" width="400"> | <img src="assets\labels.jpg" width="400"> |

## Dataset Attribution

The model was trained and validated using the publicly available **Scoliosis YOLOv5 Annotated Spine X-Ray Dataset** hosted on Kaggle.

- **Publisher:** Muhammad Salman
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/salmankey/scoliosis-yolov5-annotated-spine-x-ray-dataset)
- **Description:** Spinal X-ray radiographs pre-annotated with bounding boxes for vertebrae localization object-detection tasks.

## Project Architecture

```text
Scoliosis-Analysis/
├── Data/
│   ├── processed/           # Normalized and YOLO-formatted datasets
│   └── raw/                 # Immutable original source data
├── runs/
│   └── detect/
│       └── train/
│           └── weights/     # Serialized model weights (best.pt)
├── src/
│   ├── app.py               # Primary Gradio application and UI routing
│   ├── predict.py           # Standalone inference and algorithmic testing script
│   └── train.py             # YOLOv8 training configuration pipeline
├── dataset.yaml             # YOLO dataset definition and class mapping
├── README.md                # Repository documentation
└── LICENSE                  # Open-source license definition
```

## Installation and Execution

1. **Clone the repository**

```bash
   git clone https://github.com/GokceGokben/scoliosis-analysis.git
   cd scoliosis-analysis
```

2. **Install dependencies**

   It is recommended to use an isolated Python environment (`venv` or `conda`).

```bash
   pip install ultralytics opencv-python numpy scipy gradio pandas scikit-learn
```

3. **Launch the application**

```bash
   python src/app.py
```

## Disclaimer

This project is intended for **educational and research purposes only**. It is **not a certified medical device** and must not be used as a substitute for diagnosis, screening, or treatment decisions by a licensed radiologist or physician. All Cobb angle estimates should be independently verified by a qualified medical professional before any clinical interpretation.

## Contributing

Contributions are welcome. If you'd like to propose a change:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with clear messages
4. Open a pull request describing the motivation and changes

Bug reports and dataset/model improvement suggestions can be opened as [GitHub Issues](https://github.com/GokceGokben/scoliosis-analysis/issues).

## License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for full details.

