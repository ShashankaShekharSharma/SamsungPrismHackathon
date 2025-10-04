#!/bin/bash

# Start X virtual framebuffer
Xvfb :99 -screen 0 1920x1080x24 &
sleep 2

# Start window manager
fluxbox &
sleep 1

# Start VNC server (optional, for remote viewing)
x11vnc -display :99 -forever -nopw -quiet &

# Run the application
python final.py "$@"