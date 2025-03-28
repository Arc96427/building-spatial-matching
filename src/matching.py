import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarities(official_vectors, osm_vectors):
    """
    Calculate cosine similarity between official and OSM building vectors.
    
    Args:
        official_vectors (np.ndarray): Official building feature vectors
        osm_vectors (np.ndarray): OSM building feature vectors
        
    Returns:
        list: List of similarity dictionaries with indices and similarity scores
    """
    all_similarities = []
    
    for i, official_vector in enumerate(official_vectors):
        # Calculate cosine similarity with all OSM vectors
        similarities = cosine_similarity([official_vector], osm_vectors)[0]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        all_similarities.append({
            'official_idx': i,
            'osm_idx': best_match_idx,
            'similarity': best_similarity
        })
    
    return all_similarities


def apply_threshold_matching(similarities, threshold=0.85):
    """
    Apply threshold-based matching to similarity scores.
    
    Args:
        similarities (list): List of similarity dictionaries
        threshold (float): Similarity threshold for matching
        
    Returns:
        list: Filtered list of matches above the threshold
    """
    return [match for match in similarities if match['similarity'] > threshold]


def merge_matched_data(matches, official_gdf, osm_gdf, desired_columns):
    """
    Merge data from official and OSM sources based on matches.
    
    Args:
        matches (list): List of match dictionaries
        official_gdf (gpd.GeoDataFrame): Official GeoDataFrame
        osm_gdf (gpd.GeoDataFrame): OSM GeoDataFrame
        desired_columns (list): List of columns to merge
        
    Returns:
        gpd.GeoDataFrame: Merged GeoDataFrame
    """
    merged_data = []
    
    for official_idx, official_row in official_gdf.iterrows():
        # Check if there's a match
        match = next((m for m in matches if m['official_idx'] == official_idx), None)
        
        if match:
            osm_idx = match['osm_idx']
            osm_row = osm_gdf.iloc[osm_idx]
            
            # Merge data
            merged_entry = official_row.to_dict()
            for col in desired_columns:
                if col in osm_row and not pd.isna(osm_row[col]):
                    merged_entry[col] = osm_row[col]
            
            merged_data.append(merged_entry)
        else:
            # If no match, add official data directly
            merged_data.append(official_row.to_dict())
    
    # Convert to GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(merged_data, geometry="geometry", crs=official_gdf.crs)
    
    return merged_gdf


def get_similarity_distribution(official_vectors, osm_vectors):
    """
    Get the distribution of best similarity scores for all buildings.
    Useful for threshold selection.
    
    Args:
        official_vectors (np.ndarray): Official building feature vectors
        osm_vectors (np.ndarray): OSM building feature vectors
        
    Returns:
        np.ndarray: Array of best similarity scores for each official building
    """
    all_similarities = []
    
    for i, official_vector in enumerate(official_vectors):
        # Calculate cosine similarity with all OSM vectors
        similarities = cosine_similarity([official_vector], osm_vectors)[0]
        best_similarity = np.max(similarities)
        all_similarities.append(best_similarity)
    
    return np.array(all_similarities)


def calculate_matching_statistics(matches, official_gdf):
    """
    Calculate matching statistics.
    
    Args:
        matches (list): List of matches
        official_gdf (gpd.GeoDataFrame): Official GeoDataFrame
        
    Returns:
        dict: Dictionary with matching statistics
    """
    total_buildings = len(official_gdf)
    matched_buildings = len(matches)
    matching_rate = (matched_buildings / total_buildings) * 100 if total_buildings > 0 else 0
    
    return {
        'total_buildings': total_buildings,
        'matched_buildings': matched_buildings,
        'matching_rate': matching_rate
    }


def save_matching_statistics(stats, file_path):
    """
    Save matching statistics to a text file.
    
    Args:
        stats (dict): Dictionary with matching statistics
        file_path (str): Path to save the statistics
    """
    with open(file_path, "w") as f:
        f.write("Matching Statistics:\n")
        f.write(f"Total buildings in official dataset: {stats['total_buildings']}\n")
        f.write(f"Number of matched buildings: {stats['matched_buildings']}\n")
        f.write(f"Matching rate: {stats['matching_rate']:.2f}%\n")
    
    print(f"Matching statistics saved to {file_path}")
