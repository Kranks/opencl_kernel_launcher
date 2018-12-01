#define MIN(a, b) ((a<b)?(a):(b))

__kernel void matdouble(__global int *A, __global int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    A[i * n + j] *= 2;
}


__kernel void compute_floyd(__global int *A, __global int *B, __global int n, __global int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    B[i * n + j] = MIN(A[i * n + j], A[i * n + k] + A[k * n + j]); 
    
}