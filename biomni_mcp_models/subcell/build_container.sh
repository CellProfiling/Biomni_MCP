#!/bin/bash

# Build script for SubCell Singularity container
# This script builds the SubCell MCP server container with all dependencies and models

set -e

echo "Building SubCell Singularity container..."

# Check if singularity is available
if ! command -v singularity &> /dev/null; then
    echo "Error: Singularity is not installed or not in PATH"
    echo "Please install Singularity before building the container"
    exit 1
fi

# Create build directory
BUILD_DIR="./build"
mkdir -p $BUILD_DIR

# Set container name and path
CONTAINER_NAME="subcell-mcp.sif"
CONTAINER_PATH="$BUILD_DIR/$CONTAINER_NAME"

echo "Building container: $CONTAINER_PATH"
echo "This may take several minutes as models need to be downloaded..."

# Build the container (requires sudo/fakeroot)
if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
    echo "Using sudo for container build..."
    singularity build --force $CONTAINER_PATH singularity.def
elif singularity build --help | grep -q "\--fakeroot"; then
    echo "Using fakeroot for container build..."
    singularity build --fakeroot --force $CONTAINER_PATH singularity.def
else
    echo "Error: Need either sudo access or fakeroot support in Singularity"
    echo "Container build requires elevated privileges"
    exit 1
fi

echo "Build completed successfully!"
echo "Container saved to: $CONTAINER_PATH"
echo ""
echo "To test the container:"
echo "  singularity run $CONTAINER_PATH"
echo ""
echo "To use with Biomni, add this to your MCP configuration:"
echo "  mcp_servers:"
echo "    subcell:"
echo "      command: [\"singularity\", \"run\", \"$PWD/$CONTAINER_PATH\"]"
echo "      enabled: true"
echo ""