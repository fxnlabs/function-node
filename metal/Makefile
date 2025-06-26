# Metal Backend Makefile

# Output directories
LIB_DIR = lib
OBJ_DIR = obj
METALLIB_DIR = $(LIB_DIR)

# Source files
OBJC_SRC = src/metal_backend.m
METAL_SRC = src/matmul.metal

# Object files
OBJC_OBJ = $(OBJ_DIR)/metal_backend.o

# Library names
STATIC_LIB = $(LIB_DIR)/libmetal_backend.a
DYNAMIC_LIB = $(LIB_DIR)/libmetal_backend.dylib
METALLIB = $(METALLIB_DIR)/matmul.metallib

# Compiler and flags
CC = clang
METAL = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
AR = ar

CFLAGS = -I./include -x objective-c -fobjc-arc -fPIC -O3
LDFLAGS = -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

# Targets
.PHONY: all clean directories static dynamic metallib install

all: directories static dynamic metallib

directories:
	@mkdir -p $(LIB_DIR) $(OBJ_DIR)

# Compile Objective-C source
$(OBJC_OBJ): $(OBJC_SRC) | directories
	$(CC) $(CFLAGS) -c $< -o $@

# Build static library
static: $(STATIC_LIB)

$(STATIC_LIB): $(OBJC_OBJ)
	$(AR) rcs $@ $^
	@echo "Built static library: $@"

# Build dynamic library
dynamic: $(DYNAMIC_LIB)

$(DYNAMIC_LIB): $(OBJC_OBJ)
	$(CC) -dynamiclib -o $@ $^ $(LDFLAGS)
	@echo "Built dynamic library: $@"

# Compile Metal shaders
metallib: $(METALLIB)

$(METALLIB): $(METAL_SRC) | directories
	$(METAL) -c $< -o $(OBJ_DIR)/matmul.air
	$(METALLIB_TOOL) $(OBJ_DIR)/matmul.air -o $@
	@rm -f $(OBJ_DIR)/matmul.air
	@echo "Built Metal shader library: $@"

# Install libraries to parent directory
install: all
	@cp $(STATIC_LIB) ../
	@cp $(DYNAMIC_LIB) ../
	@cp $(METALLIB) ../
	@echo "Installed libraries to parent directory"

# Install to local lib directory (for testing)
install-local: all
	@echo "Libraries built in $(LIB_DIR)/"

# Clean build artifacts
clean:
	@rm -rf $(LIB_DIR) $(OBJ_DIR)
	@echo "Cleaned Metal build artifacts"