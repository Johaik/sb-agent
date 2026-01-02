#!/bin/bash
# Test runner script for the Research Agent System
# Usage: ./scripts/run_tests.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load .env file if it exists (for API keys like TAVILY_API_KEY)
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo -e "${GREEN}Loaded environment from .env${NC}"
fi

echo -e "${GREEN}=== Research Agent Test Suite ===${NC}"
echo ""

# Check if test infrastructure is running
check_test_infra() {
    echo -e "${YELLOW}Checking test infrastructure...${NC}"
    
    # Check PostgreSQL
    if ! nc -z localhost 5433 2>/dev/null; then
        echo -e "${RED}PostgreSQL test container not running on port 5433${NC}"
        echo "Start it with: docker-compose -f tests/docker-compose.test.yml up -d"
        exit 1
    fi
    
    # Check Redis
    if ! nc -z localhost 6380 2>/dev/null; then
        echo -e "${RED}Redis test container not running on port 6380${NC}"
        echo "Start it with: docker-compose -f tests/docker-compose.test.yml up -d"
        exit 1
    fi
    
    echo -e "${GREEN}Test infrastructure is running${NC}"
}

# Install test dependencies
install_deps() {
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -r tests/requirements-test.txt
}

# Run tests based on argument
case "${1:-all}" in
    "unit")
        echo -e "${YELLOW}Running unit tests...${NC}"
        pytest tests/unit/ -v
        ;;
    "integration")
        echo -e "${YELLOW}Running integration tests...${NC}"
        check_test_infra
        pytest tests/integration/ -v -m integration
        ;;
    "e2e")
        echo -e "${YELLOW}Running end-to-end tests...${NC}"
        check_test_infra
        pytest tests/e2e/ -v -m e2e
        ;;
    "real")
        echo -e "${YELLOW}Running real API tests...${NC}"
        echo -e "${RED}Warning: This will use real API credits!${NC}"
        pytest tests/ -v -m real_api
        ;;
    "coverage")
        echo -e "${YELLOW}Running all tests with coverage...${NC}"
        check_test_infra
        pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing -m "not real_api"
        echo -e "${GREEN}Coverage report generated in htmlcov/${NC}"
        ;;
    "all")
        echo -e "${YELLOW}Running all tests (excluding real API tests)...${NC}"
        check_test_infra
        pytest tests/ -v -m "not real_api"
        ;;
    "install")
        install_deps
        ;;
    "infra")
        echo -e "${YELLOW}Starting test infrastructure...${NC}"
        docker-compose -f tests/docker-compose.test.yml up -d
        echo -e "${GREEN}Test infrastructure started${NC}"
        echo "PostgreSQL: localhost:5433"
        echo "Redis: localhost:6380"
        ;;
    "stop")
        echo -e "${YELLOW}Stopping test infrastructure...${NC}"
        docker-compose -f tests/docker-compose.test.yml down
        echo -e "${GREEN}Test infrastructure stopped${NC}"
        ;;
    *)
        echo "Usage: $0 {unit|integration|e2e|real|coverage|all|install|infra|stop}"
        echo ""
        echo "Commands:"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests (requires test infra)"
        echo "  e2e         - Run end-to-end tests (requires test infra)"
        echo "  real        - Run real API tests (uses real API credits)"
        echo "  coverage    - Run all tests with coverage report"
        echo "  all         - Run all tests except real API tests"
        echo "  install     - Install test dependencies"
        echo "  infra       - Start test infrastructure (PostgreSQL, Redis)"
        echo "  stop        - Stop test infrastructure"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"

