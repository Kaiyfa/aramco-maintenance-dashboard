#!/bin/bash

echo "🔄 Switching to Tailwind CSS v3.4.17..."

# Remove Tailwind v4
echo "📦 Removing Tailwind v4..."
sudo rm /usr/local/bin/tailwindcss

# Download Tailwind v3.4.17 standalone CLI
echo "📥 Downloading Tailwind CSS v3.4.17..."

# Detect OS and architecture
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == 'arm64' ]]; then
        # macOS ARM64
        curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.17/tailwindcss-macos-arm64
        chmod +x tailwindcss-macos-arm64
        sudo mv tailwindcss-macos-arm64 /usr/local/bin/tailwindcss
    else
        # macOS x64
        curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.17/tailwindcss-macos-x64
        chmod +x tailwindcss-macos-x64
        sudo mv tailwindcss-macos-x64 /usr/local/bin/tailwindcss
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.17/tailwindcss-linux-x64
    chmod +x tailwindcss-linux-x64
    sudo mv tailwindcss-linux-x64 /usr/local/bin/tailwindcss
else
    echo "❌ Unsupported OS"
    exit 1
fi

# Verify installation
echo "✅ Verifying installation..."
tailwindcss --help | head -n 1

echo "🎉 Tailwind CSS v3.4.17 installed successfully!"
