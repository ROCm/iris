#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

cd $parent_dir

size=1024
while getopts "s:" opt; do
    case $opt in
        s)
            size=$OPTARG
            ;;
        *)
            echo "Usage: $0 [-s size]"
            exit 1
            ;;
    esac
done

# Create filesystem image overlay, if it doesn't exist
timestamp=$(date +%s)
overlay="/tmp/iris_overlay_$(whoami)_${timestamp}.img"
if [ ! -f $overlay ]; then
    echo "[Log] Overlay image ${overlay} does not exist. Creating overlay of ${size} MiB..."
    apptainer overlay create --size ${size} --create-dir /var/cache/iris ${overlay}
else
    echo "[Log] Overlay image ${overlay} already exists. Using this one."
fi
echo "[Log] Utilize the directory /var/cache/iris as a sandbox to store data you'd like to persist between container runs."

# Run the container
image="apptainer/images/iris.sif"
apptainer exec --overlay ${overlay} --cleanenv $image bash --rcfile /etc/bash.bashrc