# Alternative Setup Methods for Iris

This document describes alternative ways to set up and run Iris, including manual Docker setup and Apptainer. Use these methods if Docker Compose is not suitable for your workflow.

---

## Manual Docker Setup

If you prefer to build and run Docker containers manually:

```shell
# Build the Docker image
./docker/build.sh <image-name>

# Run the container
./docker/run.sh <image-name>

# Install Iris in development mode
cd iris && pip install -e .
```

---

## Apptainer

If you prefer to use Apptainer:

```shell
# Build the Apptainer image
./apptainer/build.sh

# Run the container
./apptainer/run.sh

# Activate the environment
source activate.sh
``` 
