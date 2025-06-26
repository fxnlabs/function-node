# -*- mode: python -*-

def go_files(*paths):
    """Returns a list of all go files in the given paths"""
    files = []
    for path in paths:
        files += str(local(
            "find %s -name '*.go' " % path,
            echo_off = True,
            quiet = True,
        )).strip().splitlines()
    return files

def detect_gpu_backend(requested_backend="auto"):
    """Detects the best available GPU backend"""
    gpu_backend = str(local(
        "./detect_gpu.sh %s" % requested_backend,
        dir = os.path.dirname(__file__),
        echo_off = True,
        quiet = True,
    )).strip()
    
    if not gpu_backend:
        gpu_backend = "cpu"
    
    return gpu_backend

def validate_config_files(main_dir, config_file, model_backend_file):
    """Validates that required configuration files exist"""
    missing_files = []
    
    # Check config.yaml
    config_path = os.path.join(main_dir, config_file)
    if not os.path.exists(config_path):
        template = "config.yaml.template" if os.path.exists(os.path.join(main_dir, "config.yaml.template")) else None
        missing_files.append({
            "file": config_file,
            "template": template,
            "description": "Node configuration file"
        })
    
    # Check model_backend.yaml
    model_backend_path = os.path.join(main_dir, model_backend_file)
    if not os.path.exists(model_backend_path):
        template = "model_backend.yaml.template" if os.path.exists(os.path.join(main_dir, "model_backend.yaml.template")) else None
        missing_files.append({
            "file": model_backend_file,
            "template": template,
            "description": "Model backend configuration file"
        })
    
    # Check nodekey.json
    nodekey_path = os.path.join(main_dir, "nodekey.json")
    if not os.path.exists(nodekey_path):
        missing_files.append({
            "file": "nodekey.json",
            "template": None,
            "description": "Node identity key file"
        })
    
    return missing_files

def format_setup_instructions(missing_files, main_dir):
    """Formats setup instructions for missing files"""
    if not missing_files:
        return None
    
    instructions = ["⚠️  Missing required configuration files:"]
    instructions.append("")
    
    for f in missing_files:
        instructions.append("• %s - %s" % (f["file"], f["description"]))
        if f["template"]:
            instructions.append("  Copy from template: cp %s %s" % (
                f["template"],
                f["file"]
            ))
        elif f["file"] == "nodekey.json":
            instructions.append("  Generate with: go run cmd/fxn/main.go keygen")
    
    instructions.append("")
    instructions.append("Run these commands in %s:" % main_dir)
    
    return "\n".join(instructions)