#!/bin/bash
echo
echo '1. COPYING LATEST TRAINED MODEL AND SCRIPTS'
echo
cp -r ../model .
cp ../preprocess.py .
cp ../predict.py .

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


