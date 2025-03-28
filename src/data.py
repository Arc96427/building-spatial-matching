import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import cv2
import pandas as pd


def load_osm_data(file_path):
    """
    Load OSM GeoJSON data and filter columns of interest.
    
    Args:
        file_path (str): Path to the OSM GeoJSON file
        
    Returns:
        gpd.GeoDataFrame: OSM GeoDataFrame with selected columns
    """
    osm_gdf = gpd.read_file(file_path)
    
    # Define columns of interest
    desired_columns = [
        "addr_postc", "addr_stree", "addr_hou_1",
        "contact_ph", "contact_we", "amenity",
        "architect", "building", "Buildings_6",
        "wheelchair", "wikimedia"
    ]
    
    # Ensure only columns of interest are kept
    osm_gdf = osm_gdf[["geometry"] + [col for col in desired_columns if col in osm_gdf.columns]]
    
    return osm_gdf


def load_official_data(file_path):
    """
    Load official GeoJSON data.
    
    Args:
        file_path (str): Path to the official GeoJSON file
        
    Returns:
        gpd.GeoDataFrame: Official GeoDataFrame
    """
    official_gdf = gpd.read_file(file_path)
    return official_gdf


def get_desired_columns():
    """
    Get the list of desired attribute columns for building matching.
    
    Returns:
        list: List of column names
    """
    return [
        "addr_postc", "addr_stree", "addr_hou_1",
        "contact_ph", "contact_we", "amenity",
        "architect", "building", "Buildings_6",
        "wheelchair", "wikimedia"
    ]


def polygon_to_image_autoencoder(polygon, img_size=64):
    """
    Convert polygon to binary image for autoencoder.
    
    Args:
        polygon: Shapely polygon object
        img_size (int): Output image size
        
    Returns:
        np.ndarray: Binary image representation of the polygon
    """
    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Create an empty image
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Map polygon to image space
    def map_to_pixel(x, y):
        px = int((x - minx) * (img_size - 1) / (maxx - minx)) if maxx > minx else 0
        py = int((y - miny) * (img_size - 1) / (maxy - miny)) if maxy > miny else 0
        return px, py
    
    # Get polygon points
    if hasattr(polygon, 'exterior'):
        # For simple polygons
        coords = np.array(polygon.exterior.coords)
    else:
        # For multi-part polygons, take the first part
        try:
            coords = np.array(polygon.geoms[0].exterior.coords)
        except:
            # If first part cannot be accessed, return empty image
            return img
    
    # Map points to image space
    points = np.array([map_to_pixel(x, y) for x, y in coords], dtype=np.int32)
    
    # Draw polygon
    cv2.fillPoly(img, [points], 255)
    
    return img


def polygon_to_image_resnet(polygon, img_size=224):
    """
    Convert polygon to three-channel image for use with ResNet.
    
    Args:
        polygon: Shapely polygon object
        img_size (int): Output image size (ResNet requires 224x224)
        
    Returns:
        np.ndarray: RGB image representation of the polygon
    """
    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Create an empty image
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Map polygon to image space
    def map_to_pixel(x, y):
        px = int((x - minx) * (img_size - 1) / (maxx - minx)) if maxx > minx else 0
        py = int((y - miny) * (img_size - 1) / (maxy - miny)) if maxy > miny else 0
        return px, py
    
    # Get polygon points
    if hasattr(polygon, 'exterior'):
        # For simple polygons
        coords = np.array(polygon.exterior.coords)
    else:
        # For multi-part polygons, take the first part
        try:
            coords = np.array(polygon.geoms[0].exterior.coords)
        except:
            # If first part cannot be accessed, return empty image
            return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Map points to image space
    points = np.array([map_to_pixel(x, y) for x, y in coords], dtype=np.int32)
    
    # Draw polygon
    cv2.fillPoly(img, [points], 255)
    
    # Convert to three-channel image
    img_rgb = np.stack([img, img, img], axis=-1)
    
    return img_rgb


def prepare_image_dataset(gdf, img_size=64, model_type="autoencoder"):
    """
    Prepare image dataset from GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame containing building polygons
        img_size (int): Output image size
        model_type (str): Type of model ("autoencoder" or "resnet")
        
    Returns:
        np.ndarray: Array of images
    """
    images = []
    
    if model_type == "autoencoder":
        polygon_to_image_func = lambda poly: polygon_to_image_autoencoder(poly, img_size)
    else:  # resnet
        polygon_to_image_func = lambda poly: polygon_to_image_resnet(poly, img_size)
    
    for _, row in gdf.iterrows():
        try:
            img = polygon_to_image_func(row.geometry)
            images.append(img)
        except Exception as e:
            print(f"Error processing geometry: {e}")
            # If error occurs, add a blank image
            if model_type == "autoencoder":
                blank = np.zeros((img_size, img_size), dtype=np.uint8)
            else:  # resnet
                blank = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            images.append(blank)
    
    # For autoencoder, reshape and normalize
    if model_type == "autoencoder":
        images = np.array(images).reshape(-1, 1, img_size, img_size) / 255.0
    
    return np.array(images)


def prepare_data_for_autoencoder(osm_images, official_images):
    """
    Prepare image data for autoencoder training.
    
    Args:
        osm_images (np.ndarray): OSM building images
        official_images (np.ndarray): Official building images
        
    Returns:
        tuple: (train_loader, osm_tensor, official_tensor)
    """
    # Combine datasets for training
    all_images = np.concatenate([osm_images, official_images], axis=0)
    
    # Convert to PyTorch tensors
    all_images_tensor = torch.FloatTensor(all_images)
    osm_images_tensor = torch.FloatTensor(osm_images)
    official_images_tensor = torch.FloatTensor(official_images)
    
    # Create data loader
    train_dataset = TensorDataset(all_images_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    return train_loader, osm_images_tensor, official_images_tensor


def get_resnet_preprocessing():
    """
    Get preprocessing transformations for ResNet.
    
    Returns:
        transforms.Compose: Preprocessing pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_device():
    """
    Determine the best available device (MPS, CUDA, or CPU).
    
    Returns:
        torch.device: Device to use for computations
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    return device
