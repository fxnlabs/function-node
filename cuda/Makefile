# CUDA compiler
NVCC = nvcc

# Prefix for direnv load_prefix
PREFIX = .

# CUDA flags
CUDA_FLAGS = -arch=sm_50 -O3 -Xcompiler -fPIC
CUDA_FLAGS_DEBUG = -arch=sm_50 -g -G -Xcompiler -fPIC

# Include directories
INCLUDES = -I.

# Source files
CUDA_SOURCES = matmul.cu cuda_utils.cu
TEST_SOURCES = test_matmul.cu

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cu=.o)

# Output directories
LIB_DIR = $(PREFIX)/lib
BIN_DIR = $(PREFIX)/bin
INCLUDE_DIR = $(PREFIX)/include

# Shared library
SHARED_LIB = $(LIB_DIR)/libmatmul_cuda.so

# Test executable
TEST_EXEC = $(BIN_DIR)/test_matmul

# Default target
all: $(SHARED_LIB) $(TEST_EXEC)

# Create directories
$(LIB_DIR) $(BIN_DIR) $(INCLUDE_DIR):
	mkdir -p $@

# Build shared library
$(SHARED_LIB): $(CUDA_OBJECTS) | $(LIB_DIR)
	$(NVCC) -shared -o $@ $^

# Build test executable
$(TEST_EXEC): $(TEST_OBJECTS) $(CUDA_OBJECTS) | $(BIN_DIR)
	$(NVCC) -o $@ $^

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: CUDA_FLAGS = $(CUDA_FLAGS_DEBUG)
debug: clean all

# Clean build files
clean:
	rm -f $(CUDA_OBJECTS) $(TEST_OBJECTS)
	rm -f libmatmul_cuda.so test_matmul  # Remove old build artifacts
	rm -rf $(LIB_DIR) $(BIN_DIR) $(INCLUDE_DIR)

# Run tests
test: $(TEST_EXEC)
	LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $(TEST_EXEC)

# Install headers to local include directory
install-headers: | $(INCLUDE_DIR)
	@echo "Installing headers to $(INCLUDE_DIR)..."
	@cp matmul.h cuda_utils.h $(INCLUDE_DIR)/

# Install everything locally (for direnv load_prefix)
install-local: $(SHARED_LIB) $(TEST_EXEC) install-headers
	@echo "Installed to local prefix: $(PREFIX)"
	@echo "Library: $(SHARED_LIB)"
	@echo "Binary: $(TEST_EXEC)"
	@echo "Headers: $(INCLUDE_DIR)/"

# Install library (requires sudo for system directories)
install-system: $(SHARED_LIB)
	@echo "Installing to /usr/local/lib..."
	@mkdir -p /usr/local/lib
	@cp $(SHARED_LIB) /usr/local/lib/libmatmul_cuda.so
	@mkdir -p /usr/local/include/cuda
	@cp matmul.h cuda_utils.h /usr/local/include/cuda/
	@ldconfig

# Uninstall from system
uninstall-system:
	@echo "Uninstalling from system..."
	@rm -f /usr/local/lib/libmatmul_cuda.so
	@rm -f /usr/local/include/cuda/matmul.h
	@rm -f /usr/local/include/cuda/cuda_utils.h

.PHONY: all clean debug test install-headers install-local install-system uninstall-system