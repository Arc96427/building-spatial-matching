# Building Spatial Matching

A project for matching and enriching building footprints between OSM (OpenStreetMap) and official data sources using deep learning approaches.

## Project Overview

This project implements two deep learning approaches to match building geometries between different data sources:

1. **CNN Autoencoder**: Converts building polygons to images, learns a compact representation, and matches buildings based on feature similarity
2. **ResNet Feature Extractor**: Uses a pre-trained ResNet model to extract features from building images for matching

The matching process not only improves the geometric alignment between datasets but also enriches building entities with additional semantic attributes like usage type, accessibility, and metadata from different sources.

## Repository Structure

```
building-spatial-matching/
├── data/                     # Data directory (not tracked by git)
│   ├── raw/                  # Original data files
│   └── processed/            # Processed datasets and outputs
├── notebooks/                # Jupyter notebooks for exploration 
├── src/                      # Source code
│   ├── data.py               # Data processing functions
│   ├── models.py             # Model implementations
│   ├── matching.py           # Building matching algorithms
│   └── utils.py              # Utility functions
├── scripts/                  # Executable scripts
│   ├── run_autoencoder.py    # Run autoencoder approach
│   └── run_resnet.py         # Run ResNet approach
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Arc96427/building-spatial-matching.git
cd building-spatial-matching
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Place your data files in the `data/raw/` directory:
- OSM building data: `maxvorstadt_osm.geojson`
- Official building data: `maxvorstadt_official.geojson`

Both files should contain building polygon geometries in GeoJSON format.

## Running the Code

### Autoencoder Approach

```bash
python scripts/run_autoencoder.py
```

This will:
- Convert building polygons to images
- Train an autoencoder to learn building representations
- Match buildings based on feature similarity
- Merge attributes from matched buildings
- Generate visualizations and statistics

### ResNet Approach

```bash
python scripts/run_resnet.py
```

This will:
- Convert building polygons to images
- Use ResNet18 to extract building features
- Match buildings based on feature similarity
- Merge attributes from matched buildings
- Generate visualizations and statistics

## Results

The processed results will be saved in the `data/processed/` directory:
- Merged GeoJSON files containing enriched building data
- Matching statistics text files
- Visualizations of building matches
- Similarity distribution plots

## Features

- Converts building polygon geometries to image representations
- Uses deep learning to extract meaningful features from building shapes
- Matches buildings across datasets based on shape similarity
- Enriches building entities with attributes from both sources
- Provides visualizations and statistics for matching performance

## Requirements

- Python 3.7+
- PyTorch
- GeoPandas
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- torchvision


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{wang2025building,
  author = {Arc964},
  title = {Building Spatial Matching: Deep Learning Approaches for Integrating OpenStreetMap with Official Datasets},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chihchiwang/building-spatial-matching}}
}
```

## Acknowledgments

- Technical University of Munich (TUM) and the AI4EO course
- OpenStreetMap contributors for the building data
- Landesamt für Digitalisierung, Breitband und Vermessung for the official building data
