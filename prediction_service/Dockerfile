FROM python:3.11.9-slim

RUN pip install -U pip

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

# RUN pip3 install evidently

COPY app.py . 
COPY preprocess.py . 
COPY predict.py . 
COPY ./model ./model

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5000"]
