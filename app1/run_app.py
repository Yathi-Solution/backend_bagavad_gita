#!/usr/bin/env python3
"""
Startup script for Bhagavad Gita Chat Application
This script starts the FastAPI backend server
"""

import uvicorn
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting Bhagavad Gita Chat Application...")
    print("=" * 50)
    print("Backend API: http://localhost:8000")
    print("Frontend App: http://localhost:8000/app")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        uvicorn.run(
            "endpoints.chat:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped. Thank you for using Bhagavad Gita Chat!")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
