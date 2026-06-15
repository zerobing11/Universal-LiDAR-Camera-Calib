#!/bin/bash

IMAGE_NAME="zerobing12/cali_gdb"
IMAGE_TAG="latest"
CONTAINER_NAME='uni_cali'

xhost +

if [ $# -eq 0 ]; then
    # Check whether Docker is running.
    if ! docker info >/dev/null 2>&1; then
        echo "Docker is not running. Please start the Docker service."
        exit 1
    fi

    # Start the container, or enter it if it already exists.
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
            echo "Container already exists and is running. Entering it..."
            docker exec -u 0 -it $CONTAINER_NAME bash -c "source /opt/ros/kinetic/setup.bash && cd /home/CamLidar_cail && exec bash"
        else
            echo "Container exists but is stopped. Starting it..."
            docker start $CONTAINER_NAME
            docker exec -u 0 -it $CONTAINER_NAME bash -c "source /opt/ros/kinetic/setup.bash && cd /home/CamLidar_cail && exec bash"
        fi
    else
        echo "Creating and starting container $CONTAINER_NAME..."
        
        # Use the current host user identity inside the container.
        HOST_UID=$(id -u)
        HOST_GID=$(id -g)
        HOST_USER=$(whoami)
        HOST_HOME=$HOME
        
        # Ensure the host ROS config directory exists.
        mkdir -p "$HOST_HOME/.ros"

        # Create and start the container with basic X11 visualization support.
        docker run -itd \
            --name "$CONTAINER_NAME" \
            --user "$HOST_UID:$HOST_GID" \
            --network host \
            -e DISPLAY=$DISPLAY \
            -e QT_X11_NO_MITSHM=1 \
            -e USER=$HOST_USER \
            -e HOME=/home/$HOST_USER \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "$(pwd)/":/home/CamLidar_cail \
            -v "$HOST_HOME/.ros":/home/$HOST_USER/.ros \
            -v /etc/passwd:/etc/passwd:ro \
            -v /etc/group:/etc/group:ro \
            --ipc=host \
            "${IMAGE_NAME}:${IMAGE_TAG}" \
            bash -c " bash"

        # Enter the container if it was created successfully.
        if [ $? -eq 0 ]; then
            echo "Container created successfully. Entering it..."
            docker exec -u 0 -it $CONTAINER_NAME bash -c "source /opt/ros/kinetic/setup.bash && cd /home/CamLidar_cail && exec bash"
        else
            echo "Failed to create the container. Please check the error message above."
            exit 1
        fi
    fi
    
elif [ "$1" = "commit" ]; then
    # Require a target image name.
    if [ -z "$2" ]; then
        echo "Error: please specify a new image name in the format image_name:tag."
        echo "Example: $0 commit my_new_image:latest \"custom commit message\""
        exit 1
    fi
    
    NEW_IMAGE="$2"
    COMMIT_MESSAGE="${3:-Commit from container $CONTAINER_NAME}"
    
    echo "Committing container '$CONTAINER_NAME' as image '$NEW_IMAGE'..."
    echo "Commit message: $COMMIT_MESSAGE"
    
    # Check whether the container exists.
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        # Stop the container first so the committed image has a consistent state.
        if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
            echo "Container is running. Stopping it first for a consistent commit..."
            docker stop $CONTAINER_NAME >/dev/null
        fi
        
        # Commit the container as a new image.
        docker commit \
            --message "$COMMIT_MESSAGE" \
            $CONTAINER_NAME $NEW_IMAGE
        
        if [ $? -eq 0 ]; then
            echo "Commit succeeded. New image information:"
            docker images | grep $(echo $NEW_IMAGE | cut -d':' -f1)
        else
            echo "Commit failed. Please check the error message above."
            exit 1
        fi
    else
        echo "Error: container '$CONTAINER_NAME' does not exist."
        exit 1
    fi

else
    echo "Invalid argument. Available options:"
    echo "no argument              - Start or enter the container"
    echo "commit <image_name:tag>  - Commit the container as a new image"
    exit 1
fi
