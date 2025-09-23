# Biomni MCP Models Integration

This directory contains MCP (Model Context Protocol) integrations for external models that can be used with Biomni. Each model is containerized using Singularity for HPC compatibility and isolated dependency management.

## Overview

The MCP Models framework provides a scalable approach to integrate complex machine learning models with Biomni while addressing the following challenges:

1. **Dependency Conflicts**: Models with conflicting package requirements are isolated in containers
2. **HPC Compatibility**: Uses Singularity containers compatible with HPC clusters  
3. **Resource Management**: Pre-packages models and weights to avoid download issues
4. **Standardized Interface**: Consistent MCP protocol for all model integrations

## Architecture

```
biomni_mcp_models/
├── subcell/                    # SubCell integration
│   ├── singularity.def         # Container definition
│   ├── mcp_server.py          # MCP server implementation
│   ├── subcell_wrapper.py     # Model wrapper
│   ├── subcell_portable/      # SubCell source code
│   ├── build_container.sh     # Build script
│   └── tests/                 # Unit tests
├── templates/                  # Templates for new models
│   ├── singularity_template.def
│   └── mcp_server_template.py
└── subcell_mcp_config.yaml    # Biomni MCP configuration
```

## SubCell Integration

SubCell is a protein subcellular localization prediction model from the Lundberg lab. This integration provides:

### **Low-level MCP Tools:**
- `predict_subcellular_localization`: Full prediction with probabilities and embeddings
- `extract_subcellular_features`: Feature embeddings only (1536-dim vectors)
- `get_localization_probabilities`: Classification probabilities only  
- `find_similar_proteins`: Find proteins with similar localization patterns
- `get_model_info`: Model metadata and class information

### **Use Cases:**

#### 1. Protein Similarity Analysis
Find proteins with similar subcellular localization patterns:

```python
from biomni.agent import A1

agent = A1()
agent.add_mcp(config_path="./biomni_mcp_models/subcell_mcp_config.yaml")

# Extract features from your protein images
features = agent.go("""
    Use extract_subcellular_features to get embeddings from these 4-channel images:
    r_image_path='/path/to/microtubules.tif'
    y_image_path='/path/to/er.tif' 
    b_image_path='/path/to/nuclei.tif'
    g_image_path='/path/to/protein.tif'
""")

# Find similar proteins  
similar = agent.go(f"""
    Use find_similar_proteins with the embedding vector from the previous result
    to find the 5 most similar proteins in the reference database.
""")
```

#### 2. Localization Annotation
Get subcellular localization predictions:

```python
# Get localization predictions
annotation = agent.go("""
    Use get_localization_probabilities to predict subcellular localization from:
    r_image_path='/path/to/microtubules.tif'
    y_image_path='/path/to/er.tif'
    b_image_path='/path/to/nuclei.tif' 
    g_image_path='/path/to/protein.tif'
""")

# Get model info for context
model_info = agent.go("Use get_model_info to get the class names and model details")
```

## Installation and Setup

### Prerequisites
- Singularity 3.0+ (for HPC environments)
- Docker (alternative for development)
- Python 3.11+

### Build SubCell Container

1. **Navigate to SubCell directory:**
   ```bash
   cd biomni_mcp_models/subcell
   ```

2. **Build the container:**
   ```bash
   ./build_container.sh
   ```
   
   This will:
   - Create a Singularity container with all dependencies
   - Download the RYBG MAE model (~2GB)
   - Set up the MCP server inside the container

3. **Test the container:**
   ```bash
   singularity run build/subcell-mcp.sif
   ```

### Integration with Biomni

1. **Add to your Biomni configuration:**
   ```python
   from biomni.agent import A1
   
   agent = A1()
   agent.add_mcp(config_path="./biomni_mcp_models/subcell_mcp_config.yaml")
   ```

2. **Verify tools are available:**
   ```python
   # List available tools
   tools = agent.list_custom_tools()
   print([t for t in tools if 'subcellular' in t.lower()])
   ```

## Technical Specifications

### SubCell Model Details
- **Channels**: RYBG (Red: microtubules, Yellow: ER, Blue: nuclei, Green: protein)
- **Architecture**: MAE-contrast-supcon with Vision Transformer backbone
- **Output Classes**: 31 subcellular localizations
- **Embedding Dimension**: 1536 features
- **Input Size**: 448x448 pixels
- **Model Size**: ~2GB

### Container Specifications
- **Base**: python:3.11-slim
- **PyTorch**: 2.4.1 (CPU/GPU compatible)
- **Dependencies**: Isolated environment with exact version requirements
- **Size**: ~4GB with models included

### HPC Compatibility
- **Singularity**: Designed for HPC clusters without root access
- **GPU Support**: Automatic GPU detection with CPU fallback
- **Memory**: ~8GB RAM recommended for large images
- **Storage**: Models pre-packaged to avoid download issues

## Testing

Run the comprehensive test suite:

```bash
cd biomni_mcp_models/subcell
python test_subcell_integration.py
```

Tests cover:
- SubCell wrapper functionality
- MCP server tools
- Biomni integration compatibility  
- End-to-end workflows for both use cases

## Adding New Models

Use the provided templates to add new models:

1. **Copy templates:**
   ```bash
   cp -r templates/ new_model/
   cd new_model/
   ```

2. **Customize the template:**
   - Edit `singularity_template.def` for container definition
   - Modify `mcp_server_template.py` for MCP tools
   - Add model-specific wrapper class

3. **Follow the pattern:**
   - Isolated environment in Singularity container
   - Consistent MCP tool interface
   - Comprehensive testing
   - Clear documentation

## Troubleshooting

### Common Issues

1. **Singularity build fails:**
   - Ensure you have sudo access or fakeroot support
   - Check available disk space (need ~10GB for build)

2. **Model download fails:**
   - Check internet connectivity during build
   - S3 bucket may be temporarily unavailable

3. **GPU not detected:**
   - Container will automatically fall back to CPU
   - Verify CUDA_VISIBLE_DEVICES environment variable

4. **Tool not found in Biomni:**
   - Verify MCP configuration path is correct
   - Check that container is executable
   - Ensure Biomni can find the configuration file

### Performance Tips

1. **GPU Usage:**
   - Set `CUDA_VISIBLE_DEVICES` to specify GPU
   - Container automatically detects and uses available GPUs

2. **Memory Management:**
   - Large images may require significant RAM
   - Consider resizing images if memory is limited

3. **Storage:**
   - Models are cached inside container
   - No additional downloads needed after build

## Contributing

To contribute new model integrations:

1. Follow the template structure
2. Ensure comprehensive testing
3. Document use cases and API
4. Test on HPC environments
5. Submit pull request with integration

## License

This integration framework is licensed under Apache 2.0. Individual models may have different licenses - check each model's documentation.