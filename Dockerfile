FROM python:3.8

WORKDIR /usr/src/app

COPY flask-app/requirements.txt .

RUN pip install -r requirements.txt

COPY flask-app .

CMD ["python", "app.py"]