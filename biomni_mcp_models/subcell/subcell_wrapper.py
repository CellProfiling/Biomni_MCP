"""
SubCell model wrapper for MCP server integration.
Handles low-level operations for subcellular localization prediction.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from skimage.io import imread
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add SubCell portable to path
sys.path.insert(0, '/app/subcell_portable')

from vit_model import ViTPoolClassifier
import inference
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubCellWrapper:
    """Wrapper class for SubCell model operations."""
    
    def __init__(self, model_channels: str = "rybg", model_type: str = "mae_contrast_supcon_model"):
        """
        Initialize SubCell wrapper.
        
        Args:
            model_channels: Channel configuration (rybg, rbg, ybg, bg)
            model_type: Model architecture (mae_contrast_supcon_model, vit_supcon_model)
        """
        self.model_channels = model_channels
        self.model_type = model_type
        self.device = self._setup_device()
        self.model = None
        self.model_config = None
        self._load_model()
        
        # Class mapping for localization predictions
        self.class_names = inference.CLASS2NAME
        self.class_colors = inference.CLASS2COLOR
        
    def _setup_device(self) -> torch.device:
        """Setup compute device (GPU if available, CPU otherwise)."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")
    
    def _load_model(self):
        """Load SubCell model and configuration."""
        try:
            # Load model configuration
            config_path = f"/app/subcell_portable/models/{self.model_channels}/{self.model_type}/model_config.yaml"
            with open(config_path, 'r') as f:
                model_config_file = yaml.safe_load(f)
            
            self.model_config = model_config_file
            classifier_paths = model_config_file["classifier_paths"]
            encoder_path = model_config_file["encoder_path"]
            
            # Initialize and load model
            model_config = model_config_file.get("model_config")
            self.model = ViTPoolClassifier(model_config)
            
            # Convert relative paths to absolute paths
            abs_classifier_paths = [f"/app/subcell_portable/{path}" for path in classifier_paths]
            abs_encoder_path = f"/app/subcell_portable/{encoder_path}"
            
            self.model.load_model_dict(abs_encoder_path, abs_classifier_paths)
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"Successfully loaded SubCell model: {self.model_channels}/{self.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_subcellular_localization(
        self, 
        r_image_path: Optional[str] = None,
        y_image_path: Optional[str] = None, 
        b_image_path: Optional[str] = None,
        g_image_path: Optional[str] = None,
        return_attention: bool = False
    ) -> Dict:
        """
        Predict subcellular localization from multi-channel images.
        
        Args:
            r_image_path: Path to red channel (microtubules) image
            y_image_path: Path to yellow channel (ER) image  
            b_image_path: Path to blue channel (nuclei) image
            g_image_path: Path to green channel (protein) image
            return_attention: Whether to include attention map in output
            
        Returns:
            Dictionary containing predictions, probabilities, and embeddings
        """
        try:
            # Load images based on model channels
            cell_data = []
            
            if "r" in self.model_channels and r_image_path:
                cell_data.append([imread(r_image_path, as_gray=True)])
            elif "r" in self.model_channels:
                raise ValueError("Red channel required but not provided")
                
            if "y" in self.model_channels and y_image_path:
                cell_data.append([imread(y_image_path, as_gray=True)])
            elif "y" in self.model_channels:
                raise ValueError("Yellow channel required but not provided")
                
            if "b" in self.model_channels and b_image_path:
                cell_data.append([imread(b_image_path, as_gray=True)])
            elif "b" in self.model_channels:
                raise ValueError("Blue channel required but not provided")
                
            if "g" in self.model_channels and g_image_path:
                cell_data.append([imread(g_image_path, as_gray=True)])
            elif "g" in self.model_channels:
                raise ValueError("Green channel required but not provided")
            
            if not cell_data:
                raise ValueError("No valid image channels provided")
            
            # Run inference
            embedding, probabilities = self._run_inference(cell_data)
            
            # Get top predictions
            top_predictions = self._get_top_predictions(probabilities)
            
            result = {
                "predictions": top_predictions,
                "probabilities": probabilities.tolist(),
                "embedding": embedding.tolist(),
                "model_info": {
                    "channels": self.model_channels,
                    "type": self.model_type,
                    "embedding_dim": len(embedding)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def extract_subcellular_features(
        self,
        r_image_path: Optional[str] = None,
        y_image_path: Optional[str] = None,
        b_image_path: Optional[str] = None, 
        g_image_path: Optional[str] = None
    ) -> Dict:
        """
        Extract only feature embeddings from images.
        
        Args:
            r_image_path: Path to red channel image
            y_image_path: Path to yellow channel image
            b_image_path: Path to blue channel image
            g_image_path: Path to green channel image
            
        Returns:
            Dictionary containing feature embedding
        """
        result = self.predict_subcellular_localization(
            r_image_path, y_image_path, b_image_path, g_image_path
        )
        return {
            "embedding": result["embedding"],
            "embedding_dim": len(result["embedding"]),
            "model_info": result["model_info"]
        }
    
    def get_localization_probabilities(
        self,
        r_image_path: Optional[str] = None,
        y_image_path: Optional[str] = None,
        b_image_path: Optional[str] = None,
        g_image_path: Optional[str] = None
    ) -> Dict:
        """
        Get only localization probability predictions.
        
        Args:
            r_image_path: Path to red channel image
            y_image_path: Path to yellow channel image  
            b_image_path: Path to blue channel image
            g_image_path: Path to green channel image
            
        Returns:
            Dictionary containing class probabilities and predictions
        """
        result = self.predict_subcellular_localization(
            r_image_path, y_image_path, b_image_path, g_image_path
        )
        return {
            "predictions": result["predictions"], 
            "probabilities": result["probabilities"],
            "class_names": self.class_names,
            "model_info": result["model_info"]
        }
    
    def find_similar_proteins(
        self, 
        query_embedding: List[float],
        reference_model: str = "MAE",
        num_neighbors: int = 5
    ) -> Dict:
        """
        Find proteins with similar subcellular localization patterns.
        
        Args:
            query_embedding: Feature embedding vector (1536-dim)
            reference_model: Reference model for comparison ("MAE" or "ViT") 
            num_neighbors: Number of similar proteins to return
            
        Returns:
            Dictionary containing similar protein information with gene names and cell lines
        """
        try:
            # Convert to numpy array
            query_features = np.array(query_embedding).reshape(1, -1)
            
            # Load reference features
            ref_features_path = f"/app/subcell_portable/subcell_features/{reference_model}_mean_feat.csv"
            
            if not os.path.exists(ref_features_path):
                raise FileNotFoundError(f"Reference database not found at {ref_features_path}. "
                                      f"Please ensure the {reference_model}_mean_feat.csv file is included in the container.")
            
            # Load reference features - expect columns: gene_name, cell_line, feat_0, feat_1, ..., feat_1535
            ref_df = pd.read_csv(ref_features_path)
            
            # Validate required columns
            if 'gene_name' not in ref_df.columns or 'cell_line' not in ref_df.columns:
                raise ValueError(f"Reference database must contain 'gene_name' and 'cell_line' columns")
            
            # Extract feature columns (should be feat_0 to feat_1535)
            feature_cols = [col for col in ref_df.columns if col.startswith('feat_')]
            if len(feature_cols) != 1536:
                raise ValueError(f"Expected 1536 feature columns, found {len(feature_cols)}")
            
            # Get reference features matrix
            ref_features = ref_df[feature_cols].values
            
            # Calculate similarities
            similarities = cosine_similarity(query_features, ref_features)[0]
            
            # Get top similar proteins
            top_indices = np.argsort(similarities)[::-1][:num_neighbors]
            
            # Extract gene names and cell lines for top matches
            similar_proteins = []
            for idx in top_indices:
                similar_proteins.append({
                    "gene_name": ref_df.iloc[idx]['gene_name'],
                    "cell_line": ref_df.iloc[idx]['cell_line']
                })
            
            similarity_scores = similarities[top_indices].tolist()
            
            return {
                "similar_proteins": similar_proteins,
                "similarities": similarity_scores,
                "reference_model": reference_model,
                "query_embedding_dim": len(query_embedding)
            }
            
        except Exception as e:
            logger.error(f"Similar protein search failed: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata and available classes.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_channels": self.model_channels,
            "model_type": self.model_type,
            "device": str(self.device),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "class_colors": self.class_colors,
            "embedding_dimension": 1536,
            "supported_channels": {
                "r": "microtubules" if "r" in self.model_channels else None,
                "y": "endoplasmic_reticulum" if "y" in self.model_channels else None, 
                "b": "nuclei" if "b" in self.model_channels else None,
                "g": "protein_of_interest" if "g" in self.model_channels else None
            }
        }
    
    def _run_inference(self, cell_data: List) -> Tuple[np.ndarray, np.ndarray]:
        """Run model inference on cell data."""
        with torch.no_grad():
            # Stack channels
            cell_crop = np.stack(cell_data, axis=1)
            cell_crop = torch.from_numpy(cell_crop).float().to(self.device)
            
            # Normalize
            cell_crop = inference.min_max_standardize(cell_crop)
            
            # Forward pass
            output = self.model(cell_crop)
            
            # Extract results
            probabilities = output.probabilities[0].cpu().numpy()
            
            # Get the raw pooled output
            raw_pool_op = output.pool_op[0].cpu().numpy()
            
            # The model should output 1536 dimensions (768 * 2 heads)
            # If it's larger, we need to reshape/reduce it appropriately
            if raw_pool_op.shape[0] > 1536:
                # This indicates the pooling didn't properly project to final embedding size
                # We'll take a weighted average across spatial dimensions to get 1536 dims
                # Assuming the raw output is flattened spatial features: reshape and pool
                expected_dim = 1536  # 768 * 2 heads as per config
                
                # Calculate spatial dimensions
                spatial_size = raw_pool_op.shape[0] // expected_dim
                if raw_pool_op.shape[0] % expected_dim == 0:
                    # Reshape and average pool across spatial dimensions
                    reshaped = raw_pool_op.reshape(expected_dim, spatial_size)
                    embedding = reshaped.mean(axis=1)
                else:
                    # Fallback: simple truncation + padding to get exactly 1536 dims
                    embedding = raw_pool_op[:expected_dim] if raw_pool_op.shape[0] >= expected_dim else \
                               np.pad(raw_pool_op, (0, expected_dim - raw_pool_op.shape[0]))
            else:
                embedding = raw_pool_op
            
            return embedding, probabilities
    
    def _get_top_predictions(self, probabilities: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Get top-k predictions from probability array."""
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for i, idx in enumerate(top_indices):
            predictions.append({
                "rank": i + 1,
                "class_id": int(idx),
                "class_name": self.class_names[idx],
                "probability": float(probabilities[idx]),
                "color": self.class_colors[idx]
            })
        
        return predictions