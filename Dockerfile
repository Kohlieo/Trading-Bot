# Use an official Python runtime as a base image
FROM python:3.13-slim

FROM python:3.9-slim

# Install tzdata for time zone info
RUN apt-get update && apt-get install -y tzdata

# Set the time zone to Pacific Standard Time
ENV TZ=America/Los_Angeles

# Optional: reconfigure tzdata non-interactively
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set the working directory
WORKDIR /app

# Install pandas_ta and its dependencies
RUN pip install --upgrade pip
RUN pip install pandas_ta

# Copy only requirements first for caching
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of your code
COPY . /app

# Expose the port your app runs on
EXPOSE 5000

# Command to run your bot
CMD python main.py