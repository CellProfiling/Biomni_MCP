#!/usr/bin/env python3
"""
SubCell MCP Server for Biomni integration.
Provides low-level subcellular localization prediction functions via MCP protocol.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from mcp.server.fastmcp import FastMCP
from mcp import InitializeResult
from mcp import types
from pydantic import AnyUrl

from subcell_wrapper import SubCellWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("subcell-mcp-server")

# Initialize MCP server
mcp = FastMCP("SubCell")

# Global model instance (initialized on first use)
_subcell_model: Optional[SubCellWrapper] = None


def get_model() -> SubCellWrapper:
    """Get or initialize SubCell model instance."""
    global _subcell_model
    if _subcell_model is None:
        logger.info("Initializing SubCell model...")
        _subcell_model = SubCellWrapper(model_channels="rybg", model_type="mae_contrast_supcon_model")
        logger.info("SubCell model initialized successfully")
    return _subcell_model


@mcp.tool()
def predict_subcellular_localization(
    r_image_path: Optional[str] = None,
    y_image_path: Optional[str] = None,
    b_image_path: Optional[str] = None, 
    g_image_path: Optional[str] = None,
    return_attention: bool = False
) -> Dict[str, Any]:
    """
    Predict subcellular localization from multi-channel fluorescence microscopy images.
    
    This is the main prediction function that returns comprehensive results including
    top predictions, full probability distribution, and feature embeddings.
    
    Args:
        r_image_path: Absolute path to red channel image (microtubules marker)
        y_image_path: Absolute path to yellow channel image (ER marker)  
        b_image_path: Absolute path to blue channel image (nuclei marker)
        g_image_path: Absolute path to green channel image (protein of interest)
        return_attention: Whether to include attention maps (not implemented yet)
        
    Returns:
        Dictionary containing:
        - predictions: Top 3 predicted localizations with probabilities
        - probabilities: Full 31-class probability distribution
        - embedding: 1536-dimensional feature vector
        - model_info: Model metadata
    """
    try:
        model = get_model()
        result = model.predict_subcellular_localization(
            r_image_path=r_image_path,
            y_image_path=y_image_path, 
            b_image_path=b_image_path,
            g_image_path=g_image_path,
            return_attention=return_attention
        )
        logger.info(f"Prediction completed for images: r={r_image_path}, y={y_image_path}, b={b_image_path}, g={g_image_path}")
        return result
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def extract_subcellular_features(
    r_image_path: Optional[str] = None,
    y_image_path: Optional[str] = None,
    b_image_path: Optional[str] = None,
    g_image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract feature embeddings from fluorescence microscopy images.
    
    This function returns only the 1536-dimensional feature vector that represents
    the subcellular localization pattern, without classification probabilities.
    Useful for similarity searches and clustering analyses.
    
    Args:
        r_image_path: Absolute path to red channel image (microtubules marker)
        y_image_path: Absolute path to yellow channel image (ER marker)
        b_image_path: Absolute path to blue channel image (nuclei marker) 
        g_image_path: Absolute path to green channel image (protein of interest)
        
    Returns:
        Dictionary containing:
        - embedding: 1536-dimensional feature vector
        - embedding_dim: Dimensionality of the embedding
        - model_info: Model metadata
    """
    try:
        model = get_model()
        result = model.extract_subcellular_features(
            r_image_path=r_image_path,
            y_image_path=y_image_path,
            b_image_path=b_image_path, 
            g_image_path=g_image_path
        )
        logger.info(f"Feature extraction completed for images: r={r_image_path}, y={y_image_path}, b={b_image_path}, g={g_image_path}")
        return result
    except Exception as e:
        error_msg = f"Feature extraction failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def get_localization_probabilities(
    r_image_path: Optional[str] = None,
    y_image_path: Optional[str] = None,
    b_image_path: Optional[str] = None,
    g_image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get subcellular localization probability predictions only.
    
    This function returns classification results without the feature embeddings,
    focusing on the localization predictions and class probabilities.
    
    Args:
        r_image_path: Absolute path to red channel image (microtubules marker)
        y_image_path: Absolute path to yellow channel image (ER marker)
        b_image_path: Absolute path to blue channel image (nuclei marker)
        g_image_path: Absolute path to green channel image (protein of interest)
        
    Returns:
        Dictionary containing:
        - predictions: Top 3 predicted subcellular localizations
        - probabilities: Full 31-class probability distribution  
        - class_names: Mapping of class IDs to localization names
        - model_info: Model metadata
    """
    try:
        model = get_model()
        result = model.get_localization_probabilities(
            r_image_path=r_image_path,
            y_image_path=y_image_path,
            b_image_path=b_image_path,
            g_image_path=g_image_path
        )
        logger.info(f"Probability extraction completed for images: r={r_image_path}, y={y_image_path}, b={b_image_path}, g={g_image_path}")
        return result
    except Exception as e:
        error_msg = f"Probability extraction failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def find_similar_proteins(
    query_embedding: List[float],
    reference_model: str = "MAE",
    num_neighbors: int = 5
) -> Dict[str, Any]:
    """
    Find proteins with similar subcellular localization patterns.
    
    Given a feature embedding from extract_subcellular_features, this function
    searches a reference database to find proteins with similar localization patterns
    based on cosine similarity in the embedding space.
    
    Args:
        query_embedding: 1536-dimensional feature vector from extract_subcellular_features
        reference_model: Reference database to use ("MAE" or "ViT")
        num_neighbors: Number of most similar proteins to return
        
    Returns:
        Dictionary containing:
        - similar_proteins: List of dicts with gene_name and cell_line for similar proteins
        - similarities: Cosine similarity scores for each protein
        - reference_model: Which reference database was used
        - query_embedding_dim: Dimensionality check for input embedding
    """
    try:
        model = get_model()
        result = model.find_similar_proteins(
            query_embedding=query_embedding,
            reference_model=reference_model,
            num_neighbors=num_neighbors
        )
        logger.info(f"Similar protein search completed with {len(result.get('similar_proteins', []))} results")
        return result
    except FileNotFoundError as e:
        error_msg = f"Reference database file not found: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except ValueError as e:
        error_msg = f"Invalid input data: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Similar protein search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


@mcp.tool()
def get_model_info() -> Dict[str, Any]:
    """
    Get comprehensive model information and metadata.
    
    Returns information about the loaded SubCell model including supported
    channels, available localization classes, and technical specifications.
    
    Returns:
        Dictionary containing:
        - model_channels: Which image channels are supported (rybg)
        - model_type: Architecture type (mae_contrast_supcon_model)
        - device: Compute device being used
        - num_classes: Number of subcellular localization classes (31)
        - class_names: Mapping of class IDs to localization names
        - class_colors: Color codes for visualization
        - embedding_dimension: Size of feature vectors (1536)
        - supported_channels: Description of each image channel
    """
    try:
        model = get_model() 
        result = model.get_model_info()
        logger.info("Model info retrieved successfully")
        return result
    except Exception as e:
        error_msg = f"Failed to get model info: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


if __name__ == "__main__":
    # Run the server directly with FastMCP
    mcp.run()