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
    merged_data_path = os.path.join(project_root,'data', 'processed', 'merged_with_autoencoder.geojson')
    stats_path = os.path.join(project_root, 'data','processed', 'autoencoder_matching_statistics.txt')
    matches_viz_path = os.path.join(project_root,'data', 'processed', 'building_matches_autoencoder.png')
    sim_dist_path = os.path.join(project_root, 'data','processed', 'similarity_distribution_autoencoder.png')
    model_path = os.path.join(project_root, 'data','processed', 'building_autoencoder_model.pth')
    output_dir = os.path.join(project_root, 'data', 'processed')

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
    osm_images = prepare_image_dataset(osm_gdf, img_size=IMG_SIZE, model_type="autoencoder")
    official_images = prepare_image_dataset(official_gdf, img_size=IMG_SIZE, model_type="autoencoder")

    # Prepare data for autoencoder
    train_loader, osm_images_tensor, official_images_tensor = prepare_data_for_autoencoder(
        osm_images, official_images
    )

    # Create and train model
    print("Creating and training model...")
    model = BuildingAutoencoder(img_size=IMG_SIZE, latent_dim=LATENT_DIM).to(device)
    train_autoencoder(model, train_loader, device, num_epochs=NUM_EPOCHS)

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), model_path)

    # Extract features
    print("Extracting features...")
    osm_vectors = extract_autoencoder_features(model, osm_images_tensor, device)
    official_vectors = extract_autoencoder_features(model, official_images_tensor, device)

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

    # Visualize results
    print("Visualizing results...")
    visualize_matches(osm_images, official_images, matches, matches_viz_path)

    # Visualize similarity distribution
    similarity_distribution = [m['similarity'] for m in similarities]
    visualize_similarity_distribution(similarity_distribution, SIMILARITY_THRESHOLD, sim_dist_path)

    # Visualize autoencoder reconstruction
    visualize_autoencoder_reconstruction(
        model,
        official_images_tensor[:5].to(device),
        os.path.join(output_dir, 'autoencoder_reconstruction.png')
    )

    print(f"Results saved to: {merged_data_path}")
    print(f"Visualization saved to: {matches_viz_path}")
    print(f"Similarity distribution saved to: {sim_dist_path}")


if __name__ == "__main__":
    main()