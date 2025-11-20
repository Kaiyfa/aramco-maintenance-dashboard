#!/bin/bash

# Tailwind CSS Setup Script for Django
# This script sets up Tailwind CSS in your Django project

echo "🚀 Setting up Tailwind CSS for Django..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create directory structure
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p dashboard/static/dashboard/css
mkdir -p dashboard/static/dashboard/js
mkdir -p dashboard/static/dashboard/img
mkdir -p dashboard/templates/dashboard/components
mkdir -p dashboard/templates/dashboard/pages

# Copy Tailwind config and input CSS
echo -e "${BLUE}📋 Setting up Tailwind configuration...${NC}"
cp tailwind.config.js .
cp input.css dashboard/static/dashboard/css/

# Check if Tailwind CLI is installed
if ! command -v tailwindcss &> /dev/null; then
    echo -e "${RED}❌ Tailwind CSS CLI not found!${NC}"
    echo -e "${BLUE}📥 Installing Tailwind CSS CLI...${NC}"
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if [[ $(uname -m) == 'arm64' ]]; then
            curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-arm64
            chmod +x tailwindcss-macos-arm64
            sudo mv tailwindcss-macos-arm64 /usr/local/bin/tailwindcss
        else
            curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-x64
            chmod +x tailwindcss-macos-x64
            sudo mv tailwindcss-macos-x64 /usr/local/bin/tailwindcss
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-linux-x64
        chmod +x tailwindcss-linux-x64
        sudo mv tailwindcss-linux-x64 /usr/local/bin/tailwindcss
    else
        echo -e "${RED}❌ Unsupported OS. Please install Tailwind CSS manually.${NC}"
        exit 1
    fi
fi

# Compile Tailwind CSS
echo -e "${BLUE}🎨 Compiling Tailwind CSS...${NC}"
tailwindcss -i dashboard/static/dashboard/css/input.css -o dashboard/static/dashboard/css/output.css --minify

# Create watch script for development
echo -e "${BLUE}📝 Creating watch script for development...${NC}"
cat > watch-tailwind.sh << 'EOF'
#!/bin/bash
echo "👀 Watching Tailwind CSS for changes..."
tailwindcss -i dashboard/static/dashboard/css/input.css -o dashboard/static/dashboard/css/output.css --watch
EOF

chmod +x watch-tailwind.sh

echo -e "${GREEN}✅ Tailwind CSS setup complete!${NC}"
echo ""
echo -e "${BLUE}📚 Next steps:${NC}"
echo "1. Update your settings.py STATICFILES_DIRS"
echo "2. Run './watch-tailwind.sh' during development"
echo "3. Include the compiled CSS in your base template:"
echo "   <link rel=\"stylesheet\" href=\"{% static 'dashboard/css/output.css' %}\">"
echo ""
echo -e "${GREEN}🎉 Ready to build with Tailwind CSS!${NC}"
