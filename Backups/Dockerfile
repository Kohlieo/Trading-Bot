# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose any ports that your app may use (if applicable)
# EXPOSE 8080  # You're using port 5000 in your app, so expose that instead
EXPOSE 5000

# Command to run your bot
CMD ["python", "ibkr_bot.py"]