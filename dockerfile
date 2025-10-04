FROM python:3.10-slim

# Install system dependencies for pygame and display
RUN apt-get update && apt-get install -y \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libfreetype6 \
    libportmidi0 \
    libpng16-16 \
    libjpeg62-turbo \
    xvfb \
    x11vnc \
    fluxbox \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY final.py .
COPY logo.jpg ./
COPY models/ ./models/

# Create directory for game data
RUN mkdir -p /app/data

# Set display for X virtual framebuffer
ENV DISPLAY=:99
ENV SDL_VIDEODRIVER=x11

# Expose VNC port
EXPOSE 5900

# Create startup script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]