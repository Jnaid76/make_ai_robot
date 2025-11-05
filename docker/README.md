# Docker Setup for CRAIP

## Overview
This Docker environment sets up the environment for CRAIP with ROS2 Jazzy + Gazebo Harmonic.

## Key Features
- Local `make_ai_robot` folder is mounted to `/home/ros/craip_ws` in the container
- Local code changes are immediately reflected in the container
- GPU support (NVIDIA)

## Usage

### 1. Build Docker Image
```bash
cd <path to repository>
bash docker/compose_build.bash
```

### 2. Run Container
```bash
docker compose up -d
```

### 3. Access Container
```bash
docker exec -it craip25 /bin/bash
```

After this, proceed to the [🚀 Usage section](../README.md#-usage) in the main README.


## Container Management Commands
### To access from another terminal, enter the following command in each terminal:
```bash
docker exec -it craip25 /bin/bash
```

### Stop Container
```bash
docker compose down
# or docker stop craip25
```

### Restart Container
```bash
docker compose restart
```

### View Logs
```bash
docker compose logs -f
```

## Notes
- When local files are changed, rebuild is required in the container (`colcon build`)
- X11 forwarding is configured to run GUI applications
- NVIDIA GPU is required