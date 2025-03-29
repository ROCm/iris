#!/bin/bash

alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME:$HOME -w $HOME --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864'
drun sk-pyrocshmem
