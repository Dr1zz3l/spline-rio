#!/bin/bash

# Run script for IWR6843 driver development environment
set -e

cd "$(dirname "$0")/.."

echo "Starting IWR6843 development container..."
echo "Note: Run 'catkin_make' inside the container to build the workspace."
echo ""

docker-compose -f docker/docker-compose.yml up -d
docker exec -it iwr6843-dev /bin/bash