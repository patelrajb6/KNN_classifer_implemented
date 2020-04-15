#!/bin/zsh
git add .
read message

git commit -m $message

git push origin master

