@echo off
title AI Agents - Learn & Build

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║                   🤖 AI Agents — Learn & Build               ║
echo ║                                                              ║
echo ║   Comprehensive educational resource for designing,          ║
echo ║   building, and deploying AI agents using LLMs, RAG,        ║
echo ║   and cutting-edge fine-tuning techniques.                  ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🚀 Starting AI Agents Documentation Website...
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Install/upgrade dependencies
echo Installing/updating dependencies...
pip install -q mkdocs mkdocs-material

REM Start the documentation server
echo.
echo ✅ Starting documentation server at http://localhost:8000
echo.
echo 📚 Navigate to the URL above to explore the documentation
echo 🔍 Press Ctrl+C to stop the server
echo.

mkdocs serve --dev-addr=localhost:8000

pause