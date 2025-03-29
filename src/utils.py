import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd


def visualize_matches(osm_images, official_images, matches, save_path, n_samples=5):
    """
    Visualize building matches side by side.
    
    Args:
        osm_images (np.ndarray): OSM building images
        official_images (np.ndarray): Official building images
        matches (list): List of match dictionaries
        save_path (str): Path to save the visualization
        n_samples (int): Number of samples to visualize
    """
    plt.figure(figsize=(15, 5))
    
    # Adjust the number of samples to display
    n_samples = min(n_samples, len(matches))
    
    if n_samples > 0:
        # Display n_samples matching results
        for i, match in enumerate(matches[:n_samples]):
            official_idx = match['official_idx']
            osm_idx = match['osm_idx']
            
            plt.subplot(2, n_samples, i + 1)
            
            # Handle different image formats (1 or 3 channels)
            if len(official_images.shape) == 4 and official_images.shape[1] == 1:
                # Autoencoder format (batch, channels, height, width)
                plt.imshow(official_images[official_idx][0], cmap='gray')
            else:
                plt.imshow(official_images[official_idx])
                
            plt.title(f"Official {official_idx}")
            plt.axis('off')
            
            plt.subplot(2, n_samples, n_samples + i + 1)
            
            # Handle different image formats
            if len(osm_images.shape) == 4 and osm_images.shape[1] == 1:
                # Autoencoder format
                plt.imshow(osm_images[osm_idx][0], cmap='gray')
            else:
                plt.imshow(osm_images[osm_idx])
                
            plt.title(f"OSM {osm_idx}\nSimilarity: {match['similarity']:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        print("No matches found, cannot generate visualization.")


def visualize_similarity_distribution(similarities, threshold, save_path):
    """
    Visualize similarity distribution to help choose threshold.
    
    Args:
        similarities (np.ndarray): Array of similarity scores
        threshold (float): Current threshold value
        save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.7)
    plt.xlabel('Best Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Best Cosine Similarity for Building Matching')
    plt.axvline(x=threshold, color='r', linestyle='--', 
                label=f'Current Threshold ({threshold:.2f})')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Similarity distribution visualization saved to {save_path}")


def visualize_autoencoder_reconstruction(model, test_images, save_path, n_samples=5):
    """
    Visualize original and reconstructed images from the autoencoder.
    
    Args:
        model: Trained autoencoder model
        test_images (tensor): Test images
        save_path (str): Path to save the visualization
        n_samples (int): Number of samples to visualize
    """
    model.eval()
    with torch.no_grad():
        # Get a sample of test images
        sample_images = test_images[:n_samples]
        # Get reconstructions
        reconstructions = model(sample_images)
    
    plt.figure(figsize=(12, 6))
    for i in range(n_samples):
        # Original
        plt.subplot(2, n_samples, i+1)
        plt.imshow(sample_images[i][0].cpu().numpy(), cmap='gray')
        plt.title(f"Original {i+1}")
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(2, n_samples, n_samples+i+1)
        plt.imshow(reconstructions[i][0].cpu().numpy(), cmap='gray')
        plt.title(f"Reconstructed {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Reconstruction visualization saved to {save_path}")


def compare_models_performance(autoencoder_stats, resnet_stats):
    """
    Compare performance between autoencoder and ResNet approaches.
    
    Args:
        autoencoder_stats (dict): Autoencoder matching statistics
        resnet_stats (dict): ResNet matching statistics
        
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    comparison = pd.DataFrame({
        'Model': ['Autoencoder', 'ResNet'],
        'Total Buildings': [autoencoder_stats['total_buildings'], resnet_stats['total_buildings']],
        'Matched Buildings': [autoencoder_stats['matched_buildings'], resnet_stats['matched_buildings']],
        'Matching Rate (%)': [autoencoder_stats['matching_rate'], resnet_stats['matching_rate']]
    })
    
    return comparison
