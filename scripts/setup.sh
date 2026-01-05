#!/bin/bash
# =============================================================================
# ML Observability Platform - Setup Script
# =============================================================================
# Run this script to set up the project on a fresh machine
# Usage: ./scripts/setup.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo ""
echo "=============================================="
echo "   ML Observability Platform Setup Script"
echo "=============================================="
echo ""

# Check for required tools
print_info "Checking required tools..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version 2>&1 | cut -d' ' -f3 | tr -d ',')
    print_success "Docker found: $DOCKER_VERSION"
else
    print_warning "Docker is not installed. You'll need it to run the full stack."
fi

# Check Docker Compose
if command -v docker compose &> /dev/null || command -v docker-compose &> /dev/null; then
    print_success "Docker Compose found"
else
    print_warning "Docker Compose is not installed. You'll need it to run the full stack."
fi

# Check Git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version 2>&1 | cut -d' ' -f3)
    print_success "Git found: $GIT_VERSION"
else
    print_error "Git is not installed. Please install Git."
    exit 1
fi

echo ""

# Create virtual environment
print_info "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
print_info "Installing Python dependencies (this may take a few minutes)..."
pip install -e ".[dev,notebooks]" > /dev/null 2>&1
print_success "Dependencies installed"

# Install pre-commit hooks
print_info "Installing pre-commit hooks..."
pre-commit install > /dev/null 2>&1
print_success "Pre-commit hooks installed"

# Create .env file if it doesn't exist
print_info "Setting up environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success ".env file created from .env.example"
    print_warning "Please review and update .env with your settings"
else
    print_warning ".env file already exists"
fi

# Create necessary directories
print_info "Creating data directories..."
mkdir -p data/raw data/processed data/reference models logs
touch data/raw/.gitkeep data/processed/.gitkeep data/reference/.gitkeep models/.gitkeep
print_success "Data directories created"

# Summary
echo ""
echo "=============================================="
echo "   Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Review and update .env file with your settings"
echo "     ${YELLOW}nano .env${NC}"
echo ""
echo "  2. Start all services with Docker Compose:"
echo "     ${YELLOW}make up${NC}"
echo ""
echo "  3. Or run locally without Docker:"
echo "     ${YELLOW}source venv/bin/activate${NC}"
echo "     ${YELLOW}make api${NC}        # In one terminal"
echo "     ${YELLOW}make dashboard${NC}  # In another terminal"
echo ""
echo "  4. Access the services:"
echo "     - API Docs:    http://localhost:8000/docs"
echo "     - Dashboard:   http://localhost:8501"
echo "     - Grafana:     http://localhost:3000"
echo "     - Prometheus:  http://localhost:9090"
echo ""
echo "  5. Train models and generate data:"
echo "     ${YELLOW}make train${NC}"
echo "     ${YELLOW}make generate-data${NC}"
echo ""
echo "Happy monitoring! 🚀"
echo ""
