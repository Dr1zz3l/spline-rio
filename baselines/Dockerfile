# Baselines image: repo's ROS1 image + build/run deps for the Doer/Trommer rio
# toolbox (ODRPACK in reve is Fortran). Build from the repo root:
#   docker build -t rio-baselines -f baselines/Dockerfile .
# (requires the base image: docker compose -f docker/docker-compose.yml build)
FROM docker-iwr6843:latest

RUN apt-get update && apt-get install -y \
    gfortran \
    && rm -rf /var/lib/apt/lists/*
