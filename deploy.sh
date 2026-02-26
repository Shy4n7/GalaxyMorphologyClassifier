#!/bin/bash
# Quick deployment script for Galaxy Classifier
# Usage: ./deploy.sh [local|docker|cloud]

set -e

echo "=========================================="
echo "Galaxy Classifier Deployment Script"
echo "=========================================="
echo ""

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    echo "✓ Docker is installed"
}

# Function to check if models exist
check_models() {
    if [ ! -d "models" ] || [ -z "$(ls -A models/*.pth 2>/dev/null)" ]; then
        echo "⚠️  Warning: No trained models found in models/ directory"
        echo "Please train models first or download pre-trained models"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ Models found"
    fi
}

# Local deployment
deploy_local() {
    echo ""
    echo "Deploying locally..."
    echo "===================="
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python -m venv .venv
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source .venv/bin/activate || source .venv/Scripts/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Start services
    echo ""
    echo "Starting services..."
    echo "===================="
    echo ""
    echo "Choose which service to start:"
    echo "1) REST API (port 8000)"
    echo "2) Web App (port 5001)"
    echo "3) Both"
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo "Starting REST API..."
            python src/api_server.py
            ;;
        2)
            echo "Starting Web App..."
            python src/inference_server.py
            ;;
        3)
            echo "Starting both services..."
            echo "API will run on port 8000"
            echo "Web App will run on port 5001"
            echo ""
            echo "Starting API in background..."
            python src/api_server.py &
            API_PID=$!
            echo "API PID: $API_PID"
            
            echo "Starting Web App..."
            python src/inference_server.py &
            WEBAPP_PID=$!
            echo "Web App PID: $WEBAPP_PID"
            
            echo ""
            echo "Services started!"
            echo "API: http://localhost:8000"
            echo "Web App: http://localhost:5001"
            echo ""
            echo "Press Ctrl+C to stop both services"
            
            # Wait for interrupt
            trap "kill $API_PID $WEBAPP_PID" INT
            wait
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
}

# Docker deployment
deploy_docker() {
    echo ""
    echo "Deploying with Docker..."
    echo "========================"
    
    check_docker
    check_models
    
    # Build image
    echo ""
    echo "Building Docker image..."
    docker build -t galaxy-classifier:latest .
    
    # Start services
    echo ""
    echo "Starting services with Docker Compose..."
    docker-compose up -d
    
    echo ""
    echo "✓ Deployment complete!"
    echo ""
    echo "Services are running:"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Web App: http://localhost:5001"
    echo ""
    echo "View logs: docker-compose logs -f"
    echo "Stop services: docker-compose down"
}

# Cloud deployment helper
deploy_cloud() {
    echo ""
    echo "Cloud Deployment Helper"
    echo "======================="
    echo ""
    echo "Select cloud provider:"
    echo "1) AWS (EC2 + ECR)"
    echo "2) Google Cloud (Cloud Run)"
    echo "3) Azure (Container Instances)"
    echo "4) Heroku"
    read -p "Enter choice (1-4): " cloud_choice
    
    case $cloud_choice in
        1)
            echo ""
            echo "AWS Deployment Instructions:"
            echo "============================"
            echo ""
            echo "1. Build and tag image:"
            echo "   docker build -t galaxy-classifier:latest ."
            echo ""
            echo "2. Tag for ECR:"
            echo "   docker tag galaxy-classifier:latest <account-id>.dkr.ecr.<region>.amazonaws.com/galaxy-classifier:latest"
            echo ""
            echo "3. Push to ECR:"
            echo "   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com"
            echo "   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/galaxy-classifier:latest"
            echo ""
            echo "4. Deploy on EC2:"
            echo "   SSH into EC2 and run:"
            echo "   docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/galaxy-classifier:latest"
            echo "   docker run -d -p 8000:8000 -v /path/to/models:/app/models:ro <account-id>.dkr.ecr.<region>.amazonaws.com/galaxy-classifier:latest"
            ;;
        2)
            echo ""
            echo "Google Cloud Run Deployment:"
            echo "============================"
            echo ""
            echo "1. Build and push to GCR:"
            echo "   gcloud builds submit --tag gcr.io/<project-id>/galaxy-classifier"
            echo ""
            echo "2. Deploy to Cloud Run:"
            echo "   gcloud run deploy galaxy-classifier \\"
            echo "     --image gcr.io/<project-id>/galaxy-classifier \\"
            echo "     --platform managed \\"
            echo "     --port 8000 \\"
            echo "     --memory 4Gi \\"
            echo "     --allow-unauthenticated"
            ;;
        3)
            echo ""
            echo "Azure Container Instances:"
            echo "=========================="
            echo ""
            echo "1. Build and push to ACR:"
            echo "   az acr build --registry <registry-name> --image galaxy-classifier:latest ."
            echo ""
            echo "2. Deploy to ACI:"
            echo "   az container create \\"
            echo "     --resource-group <rg-name> \\"
            echo "     --name galaxy-classifier \\"
            echo "     --image <registry-name>.azurecr.io/galaxy-classifier:latest \\"
            echo "     --dns-name-label galaxy-classifier \\"
            echo "     --ports 8000"
            ;;
        4)
            echo ""
            echo "Heroku Deployment:"
            echo "=================="
            echo ""
            echo "1. Login to Heroku:"
            echo "   heroku login"
            echo "   heroku container:login"
            echo ""
            echo "2. Create app:"
            echo "   heroku create <app-name>"
            echo ""
            echo "3. Push container:"
            echo "   heroku container:push web -a <app-name>"
            echo ""
            echo "4. Release:"
            echo "   heroku container:release web -a <app-name>"
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
    
    echo ""
    echo "For detailed instructions, see DEPLOYMENT.md"
}

# Main menu
if [ $# -eq 0 ]; then
    echo "Select deployment method:"
    echo "1) Local (Python virtual environment)"
    echo "2) Docker (Docker Compose)"
    echo "3) Cloud (AWS/GCP/Azure)"
    read -p "Enter choice (1-3): " method
    
    case $method in
        1) deploy_local ;;
        2) deploy_docker ;;
        3) deploy_cloud ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
else
    case $1 in
        local) deploy_local ;;
        docker) deploy_docker ;;
        cloud) deploy_cloud ;;
        *)
            echo "Usage: $0 [local|docker|cloud]"
            exit 1
            ;;
    esac
fi
