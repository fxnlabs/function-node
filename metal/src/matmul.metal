#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// Constants for tiling
constant uint TILE_SIZE = 16;
constant uint VECTOR_SIZE = 4; // Process 4 elements at a time

// Simple matrix multiplication kernel (for small matrices)
kernel void matmul_simple(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}

// Tiled matrix multiplication kernel with shared memory
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint2 gid [[thread_position_in_grid]])
{
    // Shared memory for tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    // Calculate the row and column index of the element
    uint row = bid.y * TILE_SIZE + tid.y;
    uint col = bid.x * TILE_SIZE + tid.x;
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Loop over tiles
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from matrix A
        uint aRow = row;
        uint aCol = t * TILE_SIZE + tid.x;
        
        if (aRow < M && aCol < K) {
            tileA[tid.y][tid.x] = A[aRow * K + aCol];
        } else {
            tileA[tid.y][tid.x] = 0.0f;
        }
        
        // Load tile from matrix B
        uint bRow = t * TILE_SIZE + tid.y;
        uint bCol = col;
        
        if (bRow < K && bCol < N) {
            tileB[tid.y][tid.x] = B[bRow * N + bCol];
        } else {
            tileB[tid.y][tid.x] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized tiled kernel with vectorization for Apple GPUs
kernel void matmul_tiled_optimized(
    device const float4* A [[buffer(0)]],  // Vectorized input
    device const float4* B [[buffer(1)]],  // Vectorized input
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint2 gid [[thread_position_in_grid]])
{
    // Shared memory for tiles (using float4 for vectorization)
    threadgroup float4 tileA[TILE_SIZE][TILE_SIZE/4];
    threadgroup float4 tileB[TILE_SIZE][TILE_SIZE/4];
    
    uint row = bid.y * TILE_SIZE + tid.y;
    uint col = bid.x * TILE_SIZE + tid.x * 4; // Process 4 columns at once
    
    // Initialize accumulator
    float4 sum = float4(0.0f);
    
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    uint K_vec = K / 4; // Vectorized K dimension
    
    for (uint t = 0; t < numTiles; t++) {
        // Load vectorized tile from matrix A
        uint aRow = row;
        uint aCol_vec = (t * TILE_SIZE + tid.x * 4) / 4;
        
        if (aRow < M && aCol_vec < K_vec) {
            tileA[tid.y][tid.x] = A[aRow * K_vec + aCol_vec];
        } else {
            tileA[tid.y][tid.x] = float4(0.0f);
        }
        
        // Load tile from matrix B (transposed access pattern)
        for (uint i = 0; i < 4; i++) {
            uint bRow = t * TILE_SIZE + tid.y * 4 + i;
            uint bCol = col;
            
            if (bRow < K && bCol < N) {
                for (uint j = 0; j < 4; j++) {
                    if (bCol + j < N) {
                        tileB[tid.y * 4 + i][tid.x].x = B[(bRow * N + bCol + j) / 4][j];
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute using fused multiply-add
        for (uint k = 0; k < TILE_SIZE; k++) {
            float4 a_vec = tileA[tid.y][k/4];
            float a_scalar = a_vec[k % 4];
            sum = fma(a_scalar, tileB[k][tid.x], sum);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write vectorized result
    if (row < M && col < N) {
        for (uint i = 0; i < 4 && col + i < N; i++) {
            C[row * N + col + i] = sum[i];
        }
    }
}

// Kernel for matrix transpose (used for optimizing memory access patterns)
kernel void matrix_transpose(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cols || gid.y >= rows) return;
    
    // Transpose: output[col][row] = input[row][col]
    output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
}

// Kernel for matrix multiplication with pre-transposed B matrix
kernel void matmul_transposed_b(
    device const float* A [[buffer(0)]],
    device const float* B_T [[buffer(1)]], // B is pre-transposed
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    
    float sum = 0.0f;
    
    // Both A[row] and B_T[col] are contiguous in memory
    uint a_offset = gid.y * K;
    uint b_offset = gid.x * K;
    
    // Vectorized dot product
    for (uint k = 0; k < K; k += 4) {
        if (k + 3 < K) {
            float4 a_vec = float4(A[a_offset + k], A[a_offset + k + 1], 
                                  A[a_offset + k + 2], A[a_offset + k + 3]);
            float4 b_vec = float4(B_T[b_offset + k], B_T[b_offset + k + 1], 
                                  B_T[b_offset + k + 2], B_T[b_offset + k + 3]);
            sum += dot(a_vec, b_vec);
        } else {
            // Handle remainder
            for (uint i = k; i < K; i++) {
                sum += A[a_offset + i] * B_T[b_offset + i];
            }
        }
    }
    
    C[gid.y * N + gid.x] = sum;
}