# 1. Base Image: Lightweight Python 3.10
FROM python:3.10-slim

# 2. Prevent Python from buffering stdout (Logs appear instantly)
ENV PYTHONUNBUFFERED=1

# 3. Install System Dependencies (Critical for OpenCV)
# libgl1 and libglib2.0 are mandatory for cv2 to work on Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Work Directory
WORKDIR /app

# 5. Copy Dependencies First (Better Caching)
# We copy from the 'Environment details' folder relative to build context
COPY ["Environment details/requirements.txt", "./"]

# 6. Install Python Libraries
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the Pipeline Code and Model
COPY ["Pipeline code/main.py", "./"]
COPY ["Pipeline code/EDSR_x4.pb", "./"] 
# (Note: We will copy the model file dynamically or map it)

# 8. Command to run the pipeline
# We assume the user mounts their input file to /app/inputs
CMD ["python", "main.py"]