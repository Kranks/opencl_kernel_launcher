#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <errno.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef LOG
#define LOG 1
#endif
#ifndef OPENCL1
#define OPENCL1 0
#endif 

//
// Lecture d'un fichier source
//

char* load_program_source(const char *filename) {

    FILE *fp;
    char *source;
    int sz=0;
    struct stat status;

    fp = fopen(filename, "r");
    if (fp == 0){
        perror ("Error : opennig the program source");
        return 0;
    }

    if (stat(filename, &status) == 0)
        sz = (int) status.st_size;

    source = (char *) malloc(sz + 1);
    
    fread(source, sz, 1, fp);
    source[sz] = '\0';

    return source;
}

// print square matrix
void print_matrix(int *a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            printf("%d ", a[i*n+j]);
        }
        printf("\n");
    }
}

void displayInfoPlatforms(cl_uint numPlatforms, cl_platform_id *platforms) {
    
    size_t infoSize;
    char *infoData;
    for (int i = 0; i < numPlatforms; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &infoSize);
        infoData = malloc(infoSize);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, infoSize, infoData, NULL);
        printf("Name of platform : %s\n", infoData);
        free(infoData);
    }

}

void displayInfoDevices(cl_uint numDevices, cl_device_id *devices) {
    
    size_t infoSize;
    char *infoData;
    for (int i=0; i < numDevices; i++){
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &infoSize);
        infoData = malloc(infoSize);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, infoSize, infoData, NULL);
        printf("Name of device %d: %s\n", i, infoData);
        free(infoData);
    }
    
}

int main (int argc, char* argv[]) {

    //----------------------------------------------------
    // STEP 0: Read the kernel file
    //----------------------------------------------------

    char * programSource = load_program_source(argv[1]);

    // This code executes on the OpenCL host

    // STEP 0.5: Initialize the matrix

    // Host data
    int *A = NULL;  // Input array
    int *B = NULL;  // Output array
    int *C = NULL;
   
    // Elements in each array
    int elements = 0;

    scanf("%d", &elements);

    // Compute the size of the data
    size_t datasize = sizeof(int)*elements*elements;

    // Allocate space for input/output data
    A = (int*)malloc(datasize);
    B = (int*)malloc(datasize);
    C = (int*)malloc(datasize);
   
    // Initialize the input data
    for(int i = 0; i < elements * elements; i++) {
        scanf("%d", &A[i]);
    }
    
    print_matrix(A, elements);
    
    // Use this to check the output of each API call
    cl_int status;

    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------

    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;

    // Use clGetPlatformIDs() to retrieve the number of platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    if (!numPlatforms) {
        printf("Error : no available platform.\n");
        return 0;
    }
    
    // Allocate enough space for each platform
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    if (LOG) {
        displayInfoPlatforms(numPlatforms, platforms);
    }

    //-----------------------------------------------------
    // STEP 2: Discover and initialize the devices
    //-----------------------------------------------------

    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;

    // Use clGetDeviceIDs() to retrieve the number of
    // devices present
    status = clGetDeviceIDs(
                            platforms[0],
                            CL_DEVICE_TYPE_ALL,
                            0,
                            NULL,
                            &numDevices);

    if (LOG) {
        printf("Number of devices = %d\n", (int)numDevices);
    }

    if (!numDevices) {
        printf("Error : no devices found.\n");
        return 0;
    }
    // Allocate enough space for each device

    devices = (cl_device_id*)malloc(
                                    numDevices*sizeof(cl_device_id));

    // Fill in devices with clGetDeviceIDs()
    status = clGetDeviceIDs(
                            platforms[0],
                            CL_DEVICE_TYPE_ALL,
                            numDevices,
                            devices,
                            NULL);

    if (LOG) {
        displayInfoDevices(numDevices, devices);
    }
    
    //-----------------------------------------------------
    // STEP 3: Create a context
    //-----------------------------------------------------

    cl_context context = NULL;

    // Create a context using clCreateContext() and
    // associate it with the devices
    context = clCreateContext(
                              NULL,
                              numDevices,
                              devices,
                              NULL,
                              NULL,
                              &status);
    
    //-----------------------------------------------------
    // STEP 4: Create a command queue
    //-----------------------------------------------------

    cl_command_queue cmdQueue;

    // Create a command queue using clCreateCommandQueue(),
    // and associate it with the device you want to execute
    // on 
    if (OPENCL1) {
        cmdQueue = clCreateCommandQueue(
                                        context,
                                        devices[0],
                                        0,
                                        &status);
    } else {
        cmdQueue = clCreateCommandQueueWithProperties(
                                        context,
                                        devices[0],
                                        0,
                                        &status);    
    }

    //-----------------------------------------------------
    // STEP 5: Create device buffers
    //-----------------------------------------------------

    cl_mem bufferA;  // Input array on the device
    cl_mem bufferB;  // Input array on the device
    cl_mem bufferC;  // Output array on the device
    cl_int n; // Input for the size of the matrix
    
    // Use clCreateBuffer() to create a buffer object (d_A)
    // that will contain the data from the host array A

    bufferA = clCreateBuffer(
                             context,
                             CL_MEM_READ_WRITE,
                             datasize,
                             NULL,
                             &status);

    // Use clCreateBuffer() to create a buffer object (d_B)
    // that will contain the data from the host array B

    bufferB = clCreateBuffer(
                             context,
                             CL_MEM_READ_ONLY,
                             datasize,
                             NULL,
                             &status);

    // Use clCreateBuffer() to create a buffer object (d_C)
    // with enough space to hold the output data

    bufferC = clCreateBuffer(
                             context,
                             CL_MEM_WRITE_ONLY,
                             datasize,
                             NULL,
                             &status);
    
    n = elements;
                       
    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------

    // Use clEnqueueWriteBuffer() to write input array A to
    // the device buffer bufferA

    status = clEnqueueWriteBuffer(
                                  cmdQueue,
                                  bufferA,
                                  CL_FALSE,
                                  0,
                                  datasize,
                                  A,
                                  0,
                                  NULL,
                                  NULL);
    
    
    // Use clEnqueueWriteBuffer() to write input array B to
    // the device buffer bufferB
    /*
    status = clEnqueueWriteBuffer(
                                  cmdQueue,
                                  bufferB,
                                  CL_FALSE,
                                  0,
                                  datasize,
                                  B,
                                  0,
                                  NULL,
                                  NULL);
    */

    //-----------------------------------------------------
    // STEP 7: Create and compile the program
    //-----------------------------------------------------

    // Create a program using clCreateProgramWithSource()
    cl_program program = clCreateProgramWithSource(
                                                   context,
                                                   1,
                                                   (const char**)&programSource,
                                                   NULL,
                                                   &status);

    // Build (compile) the program for the devices with
    // clBuildProgram()
    status = clBuildProgram(
                            program,
                            numDevices,
                            devices,
                            NULL,
                            NULL,
                            NULL);

    if (status) printf("ERREUR A LA COMPILATION: %d\n", status);
    
    if (status == CL_BUILD_PROGRAM_FAILURE) {
    // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }

    //-----------------------------------------------------
    // STEP 8: Create the kernel
    //-----------------------------------------------------

    cl_kernel kernel = NULL;

    // Use clCreateKernel() to create a kernel from the
    // given kernel function

    kernel = clCreateKernel(program, "compute_floyd", &status);

    //-----------------------------------------------------
    // STEP 9: Set the kernel arguments
    //-----------------------------------------------------

    // Associate the input and output buffers with the
    // kernel using clSetKernelArg()
    status  = clSetKernelArg(
                             kernel,
                             0,
                             sizeof(cl_mem),
                             &bufferA);
    status |= clSetKernelArg(
                             kernel,
                             1,
                             sizeof(cl_int),
                             &n);
    /*
    status |= clSetKernelArg(
                             kernel,
                             2,
                             sizeof(cl_mem),
                             &bufferC);
    */
    
    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //-----------------------------------------------------

    // Define an index space (global work size) of work
    // items for execution. 
    // A workgroup size (local work size) is not
    // required, but can be used.

    size_t globalWorkSize[2];   

    // There are 'elements' work-items
    globalWorkSize[0] = elements;
    globalWorkSize[1] = elements;
   
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //-----------------------------------------------------

    // Execute the kernel by using clEnqueueNDRangeKernel().
    // 'globalWorkSize' is the 1D dimension of the
    // work-items
    for (int i = 0; i < elements; i++) {
        clSetKernelArg(kernel, 2, sizeof(cl_int), &i);
        status = clEnqueueNDRangeKernel(
                                        cmdQueue,
                                        kernel,
                                        2,
                                        NULL,
                                        globalWorkSize,
                                        NULL,
                                        0,
                                        NULL,
                                        NULL);
    }
   
    //-----------------------------------------------------
    // STEP 12: Read the output buffer back to the host
    //-----------------------------------------------------
    
    // Use clEnqueueReadBuffer() to read the OpenCL output 
    // buffer (bufferC) to the host output array (C)
    clEnqueueReadBuffer(
                        cmdQueue,
                        bufferA,
                        CL_TRUE,
                        0,
                        datasize,
                        C,
                        0,
                        NULL,
                        NULL);

    print_matrix(C, elements); 

    //-----------------------------------------------------
    // STEP 13: Release OpenCL resources
    //-----------------------------------------------------

    // Free OpenCL resources
    clReleaseKernel(kernel);
    
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(context);

    // Free host resources
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(devices);
    return 0;
}
