#!/bin/bash

# Build script for IWR6843 driver Docker environment
set -e

cd "$(dirname "$0")/.."

echo "Building Docker image..."
docker-compose -f docker/docker-compose.yml build

echo "Docker image built successfully!"
echo "Run './scripts/run.sh' to start the development environment."