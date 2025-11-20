#!/bin/bash
echo "👀 Watching Tailwind CSS for changes..."
tailwindcss -i dashboard/static/dashboard/css/input.css -o dashboard/static/dashboard/css/output.css --watch
