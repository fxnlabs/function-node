# -*- mode: python -*-
# Function Node Tiltfile

# Parse configuration options
options = {}

if config.main_path == __file__:
    config.define_string(
        "config-file",
        args = False,
        usage = "Path to the node configuration file (default: config.yaml)",
    )
    
    config.define_string(
        "model-backend-file",
        args = False,
        usage = "Path to the model backend configuration file (default: model_backend.yaml)",
    )
    
    config.define_string(
        "gpu-backend",
        args = False,
        usage = "GPU backend to use: cuda, metal, cpu, or auto (default: auto)",
    )

    options = config.parse()

# Set default values
config_file = options.get("config-file", "config.yaml")
model_backend_file = options.get("model-backend-file", "model_backend.yaml")
gpu_backend = options.get("gpu-backend", "auto")

# Pass configuration to the .devops Tiltfile via environment variables
os.putenv("FXN_NODE_CONFIG_FILE", config_file)
os.putenv("FXN_NODE_MODEL_BACKEND_FILE", model_backend_file)
os.putenv("FXN_NODE_GPU_BACKEND", gpu_backend)

# Include the .devops Tiltfile
include(".devops/tilt/Tiltfile")