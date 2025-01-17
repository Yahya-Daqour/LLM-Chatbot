import os
import subprocess
from retrieval import VectorDBBuilder

def main():
    # Backend initialization tasks (e.g., building vector DBs)
    config_file = '../config/config.yaml'
    builder = VectorDBBuilder(config_file)
    builder.build_all_vector_dbs()
    print("Vector DBs have been built successfully.")

    # Define the path to the Streamlit app
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/app.py'))
    
    # Launch the Streamlit app
    try:
        subprocess.run(["streamlit", "run", frontend_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch Streamlit app: {e}")

if __name__ == '__main__':
    main()
