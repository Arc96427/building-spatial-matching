# Building Spatial Matching

## Project Overview

The matching process not only improves the geometric alignment between datasets but also enriches building entities with additional semantic attributes like usage type, accessibility, and metadata from different sources.

This project addresses the challenge of matching and integrating building data from different sources, specifically OpenStreetMap (OSM) and official municipal datasets. Using deep learning approaches, we achieve highly accurate spatial matching and attribute integration, creating enriched building datasets for urban planning, disaster response, and smart city applications.
This project implements two deep learning approaches to match building geometries between different data sources:
1. **CNN Autoencoder**: Converts building polygons to images, learns a compact representation, and matches buildings based on feature similarity
2. **ResNet Feature Extractor**: Uses a pre-trained ResNet model to extract features from building images for matching
   
Key features:
- Polygon-to-image conversion for neural network processing
- Custom CNN-based autoencoder for building feature extraction
- Transfer learning approach using pre-trained ResNet18
- Cosine similarity-based building matching
- Attribute fusion from OSM to official building records

Our ResNet-based approach achieves 99.95% matching accuracy on 5,749 buildings in Munich's Maxvorstadt district, significantly outperforming traditional geometric methods.

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
python scripts/run_resnet.py.py
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

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@software{building_spatial_matching,
  author = {Arc964},
  title = {Building Spatial Matching: Deep Learning Approaches for Integrating OpenStreetMap with Official Datasets},
  year = {2025},
  url = {https://github.com/Arc96427/building-spatial-matching}
}
```

## Acknowledgments

- Technical University of Munich (TUM) and the AI4EO course
- OpenStreetMap contributors for the building data
- Landesamt für Digitalisierung, Breitband und Vermessung for the official building data
