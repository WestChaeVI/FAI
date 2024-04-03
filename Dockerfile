# Use a slim Python base image
FROM tiangolo/uvicorn-gunicorn:python3.11

MAINTAINER eric@eoncare.org

# Expose port 8000 for the server
EXPOSE 8001

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the rest of the application code
COPY ./ ./

# Run the server with Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8001"]