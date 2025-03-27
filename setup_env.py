import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Get environment variables with defaults
MODEL_FILE = os.environ.get('MODEL_FILE', 'best_model.pkl')
PORT = int(os.environ.get('PORT', 10000))

# Set up model directory
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path():
    """Return the full path to the model file"""
    return os.path.join(MODEL_DIR, MODEL_FILE)

def setup_environment():
    """Set up any additional environment configuration"""
    logging.info(f"Setting up environment for {MODEL_FILE}")
    logging.info(f"Model directory: {MODEL_DIR}")
    logging.info(f"Server will run on port: {PORT}")
    
    # Create any necessary directories
    os.makedirs('logs', exist_ok=True)
    
    return {
        'model_path': get_model_path(),
        'port': PORT
    }

if __name__ == "__main__":
    # Test the setup
    config = setup_environment()
    logging.info(f"Environment setup complete: {config}")
