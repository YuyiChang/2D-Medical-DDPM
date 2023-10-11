#!/bin/bash

MOUNT_OPS="-v $(pwd):/workspace"
OPTS=""
IMAGE="ddpm"


if [ $# -eq 0 ]; then
    docker run -it --rm --gpus all $MOUNT_OPS $OPTS $IMAGE
elif [ "$1" = "cpu" ]; then
    docker run -it --rm $MOUNT_OPS $OPTS $IMAGE
elif [ "$1" = "build" ]; then
    docker build -t $IMAGE -f docker/Dockerfile .
else
    echo "Unknown argument: $1"
fi