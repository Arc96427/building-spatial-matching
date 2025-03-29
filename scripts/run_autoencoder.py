import os
import sys
import torch
import numpy as np


# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_osm_data, load_official_data, get_desired_columns,
    prepare_image_dataset, prepare_data_for_autoencoder,
    set_random_seeds, get_device
)
from src.models import (
    BuildingAutoencoder, train_autoencoder, extract_autoencoder_features
)
from src.matching import (
    calculate_cosine_similarities, apply_threshold_matching,
    merge_matched_data, calculate_matching_statistics, save_matching_statistics
)
from src.utils import (
    visualize_matches, visualize_similarity_distribution,
    visualize_autoencoder_reconstruction
)


def main():
    # Set parameters
    IMG_SIZE = 64
    LATENT_DIM = 64
    NUM_EPOCHS = 50
    SIMILARITY_THRESHOLD = 0.85

    # Set file paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    osm_file_path = os.path.join(project_root, 'data', 'raw', 'maxvorstadt_osm.geojson')
    official_file_path = os.path.join(project_root, 'data', 'raw', 'maxvorstadt_official.geojson')

    print("Starting building matching process...")
    # 添加实际执行的代码
    print("Loading data...")
    osm_gdf = load_osm_data(osm_file_path)
    official_gdf = load_official_data(official_file_path)

    print("Processing images...")
    # 更多的处理步骤...


if __name__ == "__main__":
    main()