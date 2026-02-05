#!/usr/bin/env python3
"""
AgenticML Launcher Script
Simple script to launch the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Launch the AgenticML Streamlit application"""
    print("🤖 Starting AgenticML...")
    print("📊 AI-Powered AutoML Platform")
    print("-" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the AgenticML directory")
        sys.exit(1)
    
    # Check if api_key.txt exists
    if not os.path.exists("api_key.txt") and "GROQ_API_KEY" not in os.environ:
        print("⚠️  Warning: api_key.txt not found and GROQ_API_KEY not set")
        print("Please create api_key.txt with your Groq API key or set GROQ_API_KEY environment variable")
        print("Get your API key at: https://console.groq.com/")
        
        # Ask if they want to continue anyway
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    try:
        # Launch Streamlit
        print("🚀 Launching Streamlit application...")
        print("🌐 Your browser should open automatically at http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 40)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "light"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 AgenticML stopped. Thanks for using our platform!")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
