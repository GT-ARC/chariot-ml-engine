FROM python:3.6
COPY ./app-docker /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD ["bash", "-c", "python3 /app/ChariotCloudAPI.py  -port 27017 -u mldb -p chariot -db admin -H mongo"]
