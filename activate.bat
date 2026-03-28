@echo off
cd /d C:\Users\shahar\mlops\ml-01_knowledge-trainer
powershell -NoExit -Command "& { Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force; .\venv\Scripts\Activate.ps1 }"