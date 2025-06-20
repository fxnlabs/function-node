#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[256];
    unsigned long total_memory;
    unsigned long available_memory;
    int max_threads_per_threadgroup;
    char gpu_family[64];
    int is_unified_memory;
    int is_removable;
    int supports_mps;
} MetalDeviceInfo;

// Initialize Metal device and resources
int metal_init(void);

// Check if Metal is available
int metal_check_device(void);

// Get Metal device information
int metal_get_device_info(MetalDeviceInfo* info);

// Perform matrix multiplication using Metal Performance Shaders
int metal_matmul_mps(const float* h_A, const float* h_B, float* h_C, int M, int N, int K);

// Perform matrix multiplication using custom Metal kernel
int metal_matmul_kernel(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, int use_tiled);

// Cleanup Metal resources
int metal_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // METAL_BACKEND_H