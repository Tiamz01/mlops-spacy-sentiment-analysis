#!/bin/bash

TEXT1="I did not like the book. Don't recommend to buy it!"
TEXT2="I really liked reading the book, awesome story!"

echo 'TESTING API (2 calls)'
echo
echo $TEXT1
curl -X POST -H "Content-Type: application/json" -d '{"text":"'"$TEXT1"'"}'  http://127.0.0.1:5555/predict
echo
echo $TEXT2
curl -X POST -H "Content-Type: application/json" -d '{"text":"'"$TEXT2"'"}'  http://127.0.0.1:5555/predict
