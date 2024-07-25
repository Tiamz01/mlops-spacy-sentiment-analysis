#!/bin/bash
echo
echo '1. COPYING LATEST TRAINED MODEL AND SCRIPTS'
echo
cp -r ../train_model/model .
cp ../train_model/preprocess.py .
cp ../train_model/predict.py .

echo
echo '2. BUILDING DOCKER IMAGE...'
echo
docker build -t prediction-service .

sleep 5

echo
echo '3. RUNNING DOCKER IMAGE... API available on port 5555'
echo
docker run -p 5555:5000 prediction-service:latest &
# --name prediction-service 

sleep 5

echo
echo '4. CALLING API via curl to TEST API...'
echo
bash test-calls.sh

sleep 5

echo
echo '5. STOPPING DOCKER IMAGE...'
echo
# docker stop prediction-service
docker rm $(docker stop $(docker ps -a -q --filter ancestor=prediction-service --format="{{.ID}}"))