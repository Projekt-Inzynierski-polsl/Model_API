FROM python:3.12

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "./main.py"]