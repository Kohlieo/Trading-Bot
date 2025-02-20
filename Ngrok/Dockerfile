# Use the official ngrok image
FROM wernight/ngrok

# Expose ngrok's web interface port (optional, for monitoring)
EXPOSE 4040

# Set environment variable for ngrok authtoken (replace with your actual authtoken)
ENV NGROK_AUTHTOKEN=2rStottMXL4yaqIRKt7lBbaoLi9_6d1RkJaLZMJSJvd2wXeXf

# Command to start ngrok and forward traffic to port 8080 (or any other port your bot is using)
CMD ngrok http 5000
