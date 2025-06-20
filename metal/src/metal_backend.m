#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include "../include/metal_backend.h"

// Metal context for managing resources
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> matmulSimplePipeline = nil;
static id<MTLComputePipelineState> matmulTiledPipeline = nil;

// Buffer pool for memory reuse
#define BUFFER_POOL_SIZE 16
typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
    int in_use;
} BufferPoolEntry;

static BufferPoolEntry bufferPool[BUFFER_POOL_SIZE];
static dispatch_queue_t bufferPoolQueue = nil;

// Get or allocate buffer from pool
id<MTLBuffer> get_pooled_buffer(size_t size) {
    __block id<MTLBuffer> buffer = nil;

    dispatch_sync(bufferPoolQueue, ^{
        // Look for existing buffer of sufficient size
        for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
            if (!bufferPool[i].in_use && bufferPool[i].buffer != nil && bufferPool[i].size >= size) {
                bufferPool[i].in_use = 1;
                buffer = bufferPool[i].buffer;
                return;
            }
        }

        // Allocate new buffer
        buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];

        // Try to add to pool
        for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
            if (bufferPool[i].buffer == nil) {
                bufferPool[i].buffer = buffer;
                bufferPool[i].size = size;
                bufferPool[i].in_use = 1;
                break;
            }
        }
    });

    return buffer;
}

// Return buffer to pool
void return_pooled_buffer(id<MTLBuffer> buffer) {
    dispatch_sync(bufferPoolQueue, ^{
        for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
            if (bufferPool[i].buffer == buffer) {
                bufferPool[i].in_use = 0;
                break;
            }
        }
    });
}

// Helper function to initialize pipelines from loaded library
static int metal_init_pipelines() {
    NSError *error = nil;
    
    // Create simple pipeline
    id<MTLFunction> matmulSimpleFunction = [library newFunctionWithName:@"matmul_simple"];
    if (matmulSimpleFunction == nil) {
        return -4;
    }

    matmulSimplePipeline = [device newComputePipelineStateWithFunction:matmulSimpleFunction error:&error];
    if (matmulSimplePipeline == nil) {
        NSLog(@"Failed to create simple pipeline: %@", error);
        return -5;
    }

    // Create tiled pipeline if available
    id<MTLFunction> matmulTiledFunction = [library newFunctionWithName:@"matmul_tiled"];
    if (matmulTiledFunction != nil) {
        matmulTiledPipeline = [device newComputePipelineStateWithFunction:matmulTiledFunction error:&error];
    }

    return 0;
}

// Initialize Metal device with embedded library data
int metal_init_with_embedded_lib(const void* lib_data, size_t lib_size) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return -1;
        }

        commandQueue = [device newCommandQueue];
        if (commandQueue == nil) {
            return -2;
        }

        // Initialize buffer pool queue
        bufferPoolQueue = dispatch_queue_create("com.fxnlabs.metal.bufferpool", DISPATCH_QUEUE_SERIAL);

        // Create the compute pipeline for matrix multiplication
        NSError *error = nil;

        // Load library from embedded data
        if (lib_data != NULL && lib_size > 0) {
            // Create dispatch_data_t from the embedded data
            dispatch_data_t libraryData = dispatch_data_create(lib_data, lib_size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            library = [device newLibraryWithData:libraryData error:&error];
            if (library != nil) {
                NSLog(@"Loaded Metal library from embedded data (%zu bytes)", lib_size);
                return metal_init_pipelines();
            } else {
                NSLog(@"Failed to load Metal library from embedded data: %@", error);
            }
        }

        NSLog(@"Error: No embedded Metal library data provided");
        return -3;
    }
}

// Initialize Metal device and resources
int metal_init() {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return -1;
        }

        commandQueue = [device newCommandQueue];
        if (commandQueue == nil) {
            return -2;
        }

        // Initialize buffer pool queue
        bufferPoolQueue = dispatch_queue_create("com.fxnlabs.metal.bufferpool", DISPATCH_QUEUE_SERIAL);

        // Create the compute pipeline for matrix multiplication
        NSError *error = nil;

        // Load compiled shader from metallib file
        // Get executable path for relative paths
        NSString *execPath = [[NSProcessInfo processInfo].arguments[0] stringByDeletingLastPathComponent];

        // Try all possible paths in a single loop
        NSArray *libraryPaths = @[
            // Primary path
            @"metal/matmul.metallib",
            // Alternative paths
            @"matmul.metallib",
            @"./metal/matmul.metallib",
            @"metal/lib/matmul.metallib",
            @"../metal/matmul.metallib",
            @"../metal/lib/matmul.metallib",
            @"../../metal/matmul.metallib",
            @"../../metal/lib/matmul.metallib",
            // Paths relative to executable
            [execPath stringByAppendingPathComponent:@"metal/matmul.metallib"],
            [execPath stringByAppendingPathComponent:@"metal/lib/matmul.metallib"],
            [execPath stringByAppendingPathComponent:@"matmul.metallib"]
        ];

        library = nil;
        for (NSString *path in libraryPaths) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                NSURL *url = [NSURL fileURLWithPath:path];
                library = [device newLibraryWithURL:url error:&error];
                if (library != nil) {
                    NSLog(@"Loaded metallib from: %@", path);
                    break;
                }
            }
        }

        if (library == nil) {
            NSLog(@"Error: Could not load Metal shader library. Please run 'make metal' to compile the Metal shaders.");
            return -3;
        }

        return metal_init_pipelines();
    }
}

// Check if Metal is available
int metal_check_device() {
    @autoreleasepool {
        id<MTLDevice> tempDevice = MTLCreateSystemDefaultDevice();
        return (tempDevice != nil) ? 0 : -1;
    }
}

// Get Metal device information
int metal_get_device_info(MetalDeviceInfo* info) {
    @autoreleasepool {
        if (device == nil || info == NULL) {
            return -1;
        }

        // Copy device name
        const char* deviceName = [[device name] UTF8String];
        strncpy(info->name, deviceName, sizeof(info->name) - 1);
        info->name[sizeof(info->name) - 1] = '\0';

        // Get memory info
        info->total_memory = [device recommendedMaxWorkingSetSize];

        // Estimate available memory (macOS doesn't provide exact available GPU memory)
        vm_size_t page_size;
        vm_statistics64_data_t vm_stat;
        mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);

        if (host_page_size(mach_host_self(), &page_size) == KERN_SUCCESS &&
            host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) == KERN_SUCCESS) {
            info->available_memory = (vm_stat.free_count + vm_stat.inactive_count) * page_size;
        } else {
            info->available_memory = info->total_memory / 2; // Fallback estimate
        }

        if (matmulSimplePipeline) {
            info->max_threads_per_threadgroup = [matmulSimplePipeline maxTotalThreadsPerThreadgroup];
        } else {
            info->max_threads_per_threadgroup = 1024; // Default for most Apple GPUs
        }

        // Check GPU family
        if ([device supportsFamily:MTLGPUFamilyApple7]) {
            strcpy(info->gpu_family, "Apple7");
        } else if ([device supportsFamily:MTLGPUFamilyApple6]) {
            strcpy(info->gpu_family, "Apple6");
        } else if ([device supportsFamily:MTLGPUFamilyApple5]) {
            strcpy(info->gpu_family, "Apple5");
        } else if ([device supportsFamily:MTLGPUFamilyApple4]) {
            strcpy(info->gpu_family, "Apple4");
        } else if ([device supportsFamily:MTLGPUFamilyApple3]) {
            strcpy(info->gpu_family, "Apple3");
        } else if ([device supportsFamily:MTLGPUFamilyApple2]) {
            strcpy(info->gpu_family, "Apple2");
        } else if ([device supportsFamily:MTLGPUFamilyApple1]) {
            strcpy(info->gpu_family, "Apple1");
        } else {
            strcpy(info->gpu_family, "Unknown");
        }

        // Check if unified memory
        info->is_unified_memory = [device hasUnifiedMemory] ? 1 : 0;
        info->is_removable = [device isRemovable] ? 1 : 0;
        info->supports_mps = 1; // MPS is supported on all modern macOS devices

        return 0;
    }
}

// Perform matrix multiplication using Metal Performance Shaders
int metal_matmul_mps(const float* h_A, const float* h_B, float* h_C, int M, int N, int K) {
    @autoreleasepool {
        if (device == nil || commandQueue == nil) {
            return -1;
        }

        // Create matrix descriptors
        MPSMatrixDescriptor *descriptorA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                                columns:K
                                                                               rowBytes:K * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descriptorB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                                columns:N
                                                                               rowBytes:N * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descriptorC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                                columns:N
                                                                               rowBytes:N * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];

        // Calculate buffer sizes
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);

        // Get buffers from pool
        id<MTLBuffer> bufferA = get_pooled_buffer(size_A);
        id<MTLBuffer> bufferB = get_pooled_buffer(size_B);
        id<MTLBuffer> bufferC = get_pooled_buffer(size_C);

        if (bufferA == nil || bufferB == nil || bufferC == nil) {
            return -2;
        }

        // Copy data to buffers
        memcpy([bufferA contents], h_A, size_A);
        memcpy([bufferB contents], h_B, size_B);

        // Create MPS matrices
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descriptorA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descriptorB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descriptorC];

        // Create matrix multiplication kernel
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                          transposeLeft:NO
                                                                          transposeRight:NO
                                                                          resultRows:M
                                                                          resultColumns:N
                                                                          interiorColumns:K
                                                                          alpha:1.0
                                                                          beta:0.0];

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        // Encode the operation
        [matmul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrixA
                           rightMatrix:matrixB
                           resultMatrix:matrixC];

        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back
        memcpy(h_C, [bufferC contents], size_C);

        // Return buffers to pool
        return_pooled_buffer(bufferA);
        return_pooled_buffer(bufferB);
        return_pooled_buffer(bufferC);

        return 0;
    }
}

// Perform matrix multiplication using custom Metal kernel
int metal_matmul_kernel(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, int use_tiled) {
    @autoreleasepool {
        if (device == nil || commandQueue == nil || matmulSimplePipeline == nil) {
            return -1;
        }

        // Choose appropriate pipeline
        id<MTLComputePipelineState> pipeline = nil;
        if (use_tiled && matmulTiledPipeline != nil) {
            pipeline = matmulTiledPipeline;
        } else {
            pipeline = matmulSimplePipeline;
        }

        // Calculate buffer sizes
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);

        // Get buffers from pool
        id<MTLBuffer> bufferA = get_pooled_buffer(size_A);
        id<MTLBuffer> bufferB = get_pooled_buffer(size_B);
        id<MTLBuffer> bufferC = get_pooled_buffer(size_C);

        if (bufferA == nil || bufferB == nil || bufferC == nil) {
            return -2;
        }

        // Copy data to buffers
        memcpy([bufferA contents], h_A, size_A);
        memcpy([bufferB contents], h_B, size_B);

        // Create buffers for dimensions
        uint32_t m = M, n = N, k = K;
        id<MTLBuffer> bufferM = [device newBufferWithBytes:&m length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferN = [device newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferK = [device newBufferWithBytes:&k length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set the pipeline and buffers
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        [encoder setBuffer:bufferM offset:0 atIndex:3];
        [encoder setBuffer:bufferN offset:0 atIndex:4];
        [encoder setBuffer:bufferK offset:0 atIndex:5];

        // Calculate thread groups
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;

        if (use_tiled) {
            // For tiled kernel, use fixed tile size
            const NSUInteger TILE_SIZE = 16;
            MTLSize threadgroupSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
            MTLSize threadgroupCount = MTLSizeMake((N + TILE_SIZE - 1) / TILE_SIZE,
                                                   (M + TILE_SIZE - 1) / TILE_SIZE,
                                                   1);
            [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        } else {
            // For simple kernel, use automatic sizing
            NSUInteger threadGroupWidth = 16;
            NSUInteger threadGroupHeight = threadGroupSize / threadGroupWidth;
            MTLSize threadgroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }

        [encoder endEncoding];

        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back to host
        memcpy(h_C, [bufferC contents], size_C);

        // Return buffers to pool
        return_pooled_buffer(bufferA);
        return_pooled_buffer(bufferB);
        return_pooled_buffer(bufferC);

        return 0;
    }
}

// Cleanup Metal resources
int metal_cleanup() {
    @autoreleasepool {
        // Clean up buffer pool
        dispatch_sync(bufferPoolQueue, ^{
            for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
                bufferPool[i].buffer = nil;
                bufferPool[i].size = 0;
                bufferPool[i].in_use = 0;
            }
        });

        matmulTiledPipeline = nil;
        matmulSimplePipeline = nil;
        library = nil;
        commandQueue = nil;
        device = nil;
        return 0;
    }
}
