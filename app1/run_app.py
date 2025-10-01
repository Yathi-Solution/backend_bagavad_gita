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
    # Get port from environment variable (for Render deployment) or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Check if running in production (Render sets RENDER environment variable)
    is_production = os.getenv("RENDER") is not None
    
    print("Starting Bhagavad Gita Chat Application...")
    print("=" * 50)
    if is_production:
        print(f"Running in PRODUCTION mode on port {port}")
    else:
        print(f"Backend API: http://localhost:{port}")
        print(f"Frontend App: http://localhost:{port}")
        print(f"API Docs: http://localhost:{port}/docs")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        uvicorn.run(
            "endpoints.chat:app",
            host="0.0.0.0",
            port=port,
            reload=not is_production,  # Disable reload in production
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped. Thank you for using Bhagavad Gita Chat!")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
