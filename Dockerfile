# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR "C:/Users/HP/ask-multiple-pdfs-main"

# Copy the requirements.txt file into the container
COPY requirements.txt requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Specify the command to run on container start
CMD ["streamlit", "run", "app.py"]
