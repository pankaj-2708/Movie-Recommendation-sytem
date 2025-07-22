# base image
FROM python:3.12-slim

# setting a working directory
WORKDIR /app

#copying requirements and installing dependeces
COPY requirements.txt .
RUN pip install -r requirements.txt

#copying rest of the code
COPY . .

#exposing port
EXPOSE 8000

#command to start fast api application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

#docker command
#docker build -t pankajmaulekhi/movie-recomm-frontend .
#docker push docker push pankajmaulekhi/movie-recomm-frontend 