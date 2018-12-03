#define MIN(a, b) ((a<b)?(a):(b))

__kernel void matdouble(__global int *A, int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    A[i * n + j] *= 2;
}

__kernel void transmat(__global int *A, int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (A[i * n +j] == 0) {
        A[i * n + j] = 1;
    } else { 
        A[i * n + j] = 42;
    }
}

__kernel void compute_floyd(__global int *A, int n, int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    A[i * n + j] = MIN(A[i * n + j], A[i * n + k] + A[k * n + j]); 
    
}
