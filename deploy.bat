@echo off
REM Quick deployment script for Galaxy Classifier (Windows)
REM Usage: deploy.bat [local|docker|cloud]

echo ==========================================
echo Galaxy Classifier Deployment Script
echo ==========================================
echo.

if "%1"=="" goto menu
if "%1"=="local" goto deploy_local
if "%1"=="docker" goto deploy_docker
if "%1"=="cloud" goto deploy_cloud
echo Invalid argument. Use: deploy.bat [local^|docker^|cloud]
exit /b 1

:menu
echo Select deployment method:
echo 1) Local (Python virtual environment)
echo 2) Docker (Docker Compose)
echo 3) Cloud (AWS/GCP/Azure)
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" goto deploy_local
if "%choice%"=="2" goto deploy_docker
if "%choice%"=="3" goto deploy_cloud
echo Invalid choice
exit /b 1

:deploy_local
echo.
echo Deploying locally...
echo ====================

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start services
echo.
echo Starting services...
echo ====================
echo.
echo Choose which service to start:
echo 1) REST API (port 8000)
echo 2) Web App (port 5001)
echo 3) Both
set /p svc_choice="Enter choice (1-3): "

if "%svc_choice%"=="1" (
    echo Starting REST API...
    python src\api_server.py
) else if "%svc_choice%"=="2" (
    echo Starting Web App...
    python src\inference_server.py
) else if "%svc_choice%"=="3" (
    echo Starting both services...
    echo API will run on port 8000
    echo Web App will run on port 5001
    echo.
    start "Galaxy API" python src\api_server.py
    start "Galaxy Web App" python src\inference_server.py
    echo.
    echo Services started!
    echo API: http://localhost:8000
    echo Web App: http://localhost:5001
    echo.
    echo Close the terminal windows to stop services
) else (
    echo Invalid choice
    exit /b 1
)
goto end

:deploy_docker
echo.
echo Deploying with Docker...
echo ========================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed. Please install Docker first.
    echo Visit: https://docs.docker.com/get-docker/
    exit /b 1
)
echo Docker is installed

REM Check if models exist
if not exist "models\*.pth" (
    echo Warning: No trained models found in models\ directory
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Build image
echo.
echo Building Docker image...
docker build -t galaxy-classifier:latest .

REM Start services
echo.
echo Starting services with Docker Compose...
docker-compose up -d

echo.
echo Deployment complete!
echo.
echo Services are running:
echo   - API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo   - Web App: http://localhost:5001
echo.
echo View logs: docker-compose logs -f
echo Stop services: docker-compose down
goto end

:deploy_cloud
echo.
echo Cloud Deployment Helper
echo =======================
echo.
echo Select cloud provider:
echo 1) AWS (EC2 + ECR)
echo 2) Google Cloud (Cloud Run)
echo 3) Azure (Container Instances)
echo 4) Heroku
set /p cloud_choice="Enter choice (1-4): "

if "%cloud_choice%"=="1" (
    echo.
    echo AWS Deployment Instructions:
    echo ============================
    echo.
    echo 1. Build and tag image:
    echo    docker build -t galaxy-classifier:latest .
    echo.
    echo 2. Tag for ECR:
    echo    docker tag galaxy-classifier:latest ^<account-id^>.dkr.ecr.^<region^>.amazonaws.com/galaxy-classifier:latest
    echo.
    echo 3. Push to ECR:
    echo    aws ecr get-login-password --region ^<region^> ^| docker login --username AWS --password-stdin ^<account-id^>.dkr.ecr.^<region^>.amazonaws.com
    echo    docker push ^<account-id^>.dkr.ecr.^<region^>.amazonaws.com/galaxy-classifier:latest
    echo.
    echo 4. Deploy on EC2:
    echo    SSH into EC2 and run:
    echo    docker pull ^<account-id^>.dkr.ecr.^<region^>.amazonaws.com/galaxy-classifier:latest
    echo    docker run -d -p 8000:8000 -v /path/to/models:/app/models:ro ^<account-id^>.dkr.ecr.^<region^>.amazonaws.com/galaxy-classifier:latest
) else if "%cloud_choice%"=="2" (
    echo.
    echo Google Cloud Run Deployment:
    echo ============================
    echo.
    echo 1. Build and push to GCR:
    echo    gcloud builds submit --tag gcr.io/^<project-id^>/galaxy-classifier
    echo.
    echo 2. Deploy to Cloud Run:
    echo    gcloud run deploy galaxy-classifier --image gcr.io/^<project-id^>/galaxy-classifier --platform managed --port 8000 --memory 4Gi --allow-unauthenticated
) else if "%cloud_choice%"=="3" (
    echo.
    echo Azure Container Instances:
    echo ==========================
    echo.
    echo 1. Build and push to ACR:
    echo    az acr build --registry ^<registry-name^> --image galaxy-classifier:latest .
    echo.
    echo 2. Deploy to ACI:
    echo    az container create --resource-group ^<rg-name^> --name galaxy-classifier --image ^<registry-name^>.azurecr.io/galaxy-classifier:latest --dns-name-label galaxy-classifier --ports 8000
) else if "%cloud_choice%"=="4" (
    echo.
    echo Heroku Deployment:
    echo ==================
    echo.
    echo 1. Login to Heroku:
    echo    heroku login
    echo    heroku container:login
    echo.
    echo 2. Create app:
    echo    heroku create ^<app-name^>
    echo.
    echo 3. Push container:
    echo    heroku container:push web -a ^<app-name^>
    echo.
    echo 4. Release:
    echo    heroku container:release web -a ^<app-name^>
) else (
    echo Invalid choice
    exit /b 1
)

echo.
echo For detailed instructions, see DEPLOYMENT.md

:end
echo.
pause
