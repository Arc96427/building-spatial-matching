# Building Spatial Matching

Deep learning approaches for matching and integrating building footprints between OpenStreetMap and official datasets.

## Project Overview

This project addresses the challenge of matching and integrating building data from different sources, specifically OpenStreetMap (OSM) and official municipal datasets. Using deep learning approaches, we achieve highly accurate spatial matching and attribute integration, creating enriched building datasets for urban planning, disaster response, and smart city applications.

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
├── data/                        # Data directory (not tracked by git)
│   ├── raw/                     # Original data files
│   └── processed/               # Processed datasets
├── notebooks/                   # Jupyter notebooks for exploration and visualization
│   ├── data_exploration.ipynb   # Dataset analysis
│   └── results_visualization.ipynb  # Visualizing matching results
├── src/                         # Source code
│   ├── data/                    # Data loading and processing scripts
│   │   ├── __init__.py
│   │   ├── load_data.py         # Functions to load GeoJSON files
│   │   └── preprocess.py        # Data preprocessing functions
│   ├── models/                  # Model implementations
│   │   ├── __init__.py
│   │   ├── autoencoder.py       # CNN autoencoder implementation
│   │   └── resnet.py            # ResNet feature extractor implementation
│   ├── matching/                # Building matching algorithms
│   │   ├── __init__.py
│   │   ├── similarity.py        # Similarity calculation functions
│   │   └── threshold.py         # Threshold-based matching
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── polygon_to_image.py  # Convert polygons to images
│       ├── evaluation.py        # Evaluation metrics
│       └── visualization.py     # Visualization functions
├── scripts/                     # Executable scripts
│   ├── download_data.py         # Script to download required datasets
│   ├── run_autoencoder.py       # Run the CNN autoencoder approach
│   └── run_resnet.py            # Run the ResNet approach
├── requirements.txt             # Project dependencies
├── environment.yml              # Conda environment file
├── LICENSE                      # License information
└── README.md                    # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- GDAL for geospatial data processing
- PyTorch 1.9+ with CUDA support (recommended for GPU acceleration)

### Option 1: Using conda (recommended)

```bash
# Clone the repository
git clone https://github.com/chihchiwang/building-spatial-matching.git
cd building-spatial-matching

# Create and activate conda environment
conda env create -f environment.yml
conda activate building-matching

# Install the package in development mode
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/chihchiwang/building-spatial-matching.git
cd building-spatial-matching

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Data Preparation

### Downloading the Data

```bash
# Run the data download script
python scripts/download_data.py --region maxvorstadt
```

This script will download:
1. OpenStreetMap data for the specified region using Overpass API
2. Administrative boundaries from Geofabrik

**Note**: The Bavarian Building Footprint dataset must be manually downloaded from the Landesamt für Digitalisierung, Breitband und Vermessung website due to licensing restrictions. After downloading, place the files in the `data/raw/` directory.

### Data Preprocessing

```bash
# Preprocess the raw data
python src/data/preprocess.py --input_dir data/raw --output_dir data/processed
```

This script will:
- Reproject all datasets to EPSG:4326
- Extract buildings within the Maxvorstadt district
- Create necessary files for model training

## Running the Models

### CNN Autoencoder Approach

```bash
# Run the CNN autoencoder approach
python scripts/run_autoencoder.py \
    --data_dir data/processed \
    --output_dir results/autoencoder \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --latent_dim 64 \
    --similarity_threshold 0.85
```

### ResNet Feature Extraction Approach

```bash
# Run the ResNet approach
python scripts/run_resnet.py \
    --data_dir data/processed \
    --output_dir results/resnet \
    --output_dim 128 \
    --similarity_threshold 0.85
```

## Evaluating Results

```bash
# Evaluate matching results
python src/utils/evaluation.py \
    --ground_truth data/processed/ground_truth.geojson \
    --predictions results/resnet/matched_buildings.geojson \
    --output results/evaluation_metrics.json
```

## Visualizing Results

```bash
# Generate visualizations of matching results
python src/utils/visualization.py \
    --input_dir results/resnet \
    --output_dir visualizations
```

Alternatively, you can use the Jupyter notebook for interactive visualizations:

```bash
jupyter notebook notebooks/results_visualization.ipynb
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{wang2025building,
  author = {Wang, Chih-Chi},
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
