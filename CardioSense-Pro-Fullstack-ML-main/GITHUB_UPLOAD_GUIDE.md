# GitHub Upload Guide (One Shot)

## Recommended Repository Name

CardioSense-Pro-Fullstack-ML

## Recommended Description

End-to-end heart disease prediction platform with interactive frontend, FastAPI backend, JWT multi-user authentication, prediction history, and Kaggle-trained machine learning pipeline.

## Topics (optional)

fastapi, machine-learning, heart-disease-prediction, kaggle, jwt-auth, fullstack, python, healthcare-ai

## 1) Open terminal at project root

cd "/Users/aadilmansuri/Desktop/untitled folder"

## 2) Initialize git

git init
git branch -M main

## 3) Add all files

git add .

## 4) First commit

git commit -m "Complete CardioSense Pro fullstack ML platform"

## 5) Add GitHub remote

Replace USERNAME and REPO with your values:

git remote add origin https://github.com/USERNAME/CardioSense-Pro-Fullstack-ML.git

If remote already exists:

git remote set-url origin https://github.com/USERNAME/CardioSense-Pro-Fullstack-ML.git

## 6) Push

git push -u origin main

## 7) Verify on GitHub

Check these files are visible:
- README.md
- CardioSense_Pro_v2.html
- backend/README.md
- backend/app/main.py
- backend/ml/train.py

## Notes

- .env is not committed (use backend/.env.example)
- Local virtual environment is ignored (.venv)
- Model artifact files and raw dataset files inside backend are ignored by backend/.gitignore
