FROM python:3.11

WORKDIR /assembly-agent

ADD . /assembly-agent

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]




