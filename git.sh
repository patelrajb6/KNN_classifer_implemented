#!/bin/zsh
git add README.md iris.csv git.sh rpate375-KNN.ipynb

read message

git commit -m $message

git push origin master

