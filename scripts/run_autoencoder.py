import os
import sys
import torch
import numpy as np


#将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 现在可以导入src目录下的模块
from src.data import (
    load_osm_data, load_official_data, get_desired_columns,
    prepare_image_dataset, prepare_data_for_autoencoder,
    set_random_seeds, get_device
)
# Add the src directory to the Python path
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
    osm_file_path = os.path.join('data', 'raw', 'maxvorstadt_osm.geojson')
    official_file_path = os.path.join('data', 'raw', 'maxvorstadt_official.geojson')
    output_dir = os.path.join('data', 'processed')
    model_path = os.path.join(output_dir, 'building_autoencoder_model.pth')
    merged_data_path = os.path.join(output_dir, 'merged_with_autoencoder.geojson')
    stats_path = os.path.join(output_dir, 'autoencoder_matching_statistics.txt')
    matches_viz_path = os.path.join(output_dir, 'building_matches')
