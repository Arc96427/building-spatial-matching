import os
import sys
import torch
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_osm_data, load_official_data, get_desired_columns,
    prepare_image_dataset, get_resnet_preprocessing,
    set_random_seeds, get_device
)
from src.models import (
    ResNetFeatureExtractor, extract_resnet_features
)
from src.matching import (
    calculate_cosine_similarities, apply_threshold_matching,
    merge_matched_data, calculate_matching_statistics,
    save_matching_statistics, get_similarity_distribution
)
from src.utils import (
    visualize_matches, visualize_similarity_distribution
)


def main():
    # Set parameters
    IMG_SIZE = 224  # ResNet requires 224x224 input
    OUTPUT_DIM = 128
    SIMILARITY_THRESHOLD = 0.85

    # Set file paths
    osm_file_path = os.path.join('data', 'raw', 'maxvorstadt_osm.geojson')
    official_file_path = os.path.join('data', 'raw', 'maxvorstadt_official.geojson')
    output_dir = os.path.join('data', 'processed')
    vectors_dir = os.path.join(output_dir, 'vectors')
    os.makedirs(vectors_dir, exist_ok=True)

    merged_data_path = os.path.join(output_dir, 'merged_with_resnet.geojson')
    stats_path = os.path.join(output_dir, 'resnet_matching_statistics.txt')
    matches_viz_path = os.path.join(output_dir, 'building_matches_resnet.png')
    sim_dist_path = os.path.join(output_dir, 'similarity_distribution_resnet.png')

    # Set random seeds for reproducibility
    set_random_seeds(42)

    # Get device (GPU/CPU)
    device = get_device()

    # Load data
    print("Loading data...")
    osm_gdf = load_osm_data(osm_file_path)
    official_gdf = load_official_data(official_file_path)
    desired_columns = get_desired_columns()

    # Prepare image data
    print("Preparing image data...")
    osm_images = prepare_image_dataset(osm_gdf, img_size=IMG_SIZE, model_type="resnet")
    official_images = prepare_image_dataset(official_gdf, img_size=IMG_SIZE, model_type="resnet")

    # Create ResNet model
    print("Creating model...")
    model = ResNetFeatureExtractor(output_dim=OUTPUT_DIM).to(device)

    # Get preprocessing transformations
    preprocess = get_resnet_preprocessing()

    # Extract features
    print("Extracting features...")
    osm_vectors = extract_resnet_features(model, osm_images, preprocess, device)
    official_vectors = extract_resnet_features(model, official_images, preprocess, device)

    # Save vectors for future analysis
    np.save(os.path.join(vectors_dir, 'resnet_osm_vectors.npy'), osm_vectors)
    np.save(os.path.join(vectors_dir, 'resnet_official_vectors.npy'), official_vectors)

    # Match buildings
    print("Matching buildings...")
    similarities = calculate_cosine_similarities(official_vectors, osm_vectors)
    matches = apply_threshold_matching(similarities, SIMILARITY_THRESHOLD)

    # Merge data
    print("Merging data...")
    merged_gdf = merge_matched_data(matches, official_gdf, osm_gdf, desired_columns)

    # Save results
    print("Saving results...")
    merged_gdf.to_file(merged_data_path, driver="GeoJSON")

    # Calculate and save matching statistics
    stats = calculate_matching_statistics(matches, official_gdf)
    save_matching_statistics(stats, stats_path)

    # Print matching statistics
    print(f"\nMatching Statistics:")
    print(f"Total buildings in official dataset: {stats['total_buildings']}")
    print(f"Number of matched buildings: {stats['matched_buildings']}")
    print(f"Matching rate: {stats['matching_rate']:.2f}%")

    # Visualize some results
    print("Visualizing results...")
    visualize_matches(osm_images, official_images, matches, matches_viz_path)

    # Visualize similarity distribution
    similarity_distribution = get_similarity_distribution(official_vectors, osm_vectors)
    visualize_similarity_distribution(similarity_distribution, SIMILARITY_THRESHOLD, sim_dist_path)

    print(f"Results saved to: {merged_data_path}")
    print(f"Visualization saved to: {matches_viz_path}")
    print(f"Similarity distribution saved to: {sim_dist_path}")


if __name__ == "__main__":
    main()
