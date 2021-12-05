//
// Created by Cristian & Roberto on 15/11/2021.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAXLENGHT_NAME 20
#define MAXTHREADS 8
#define TOTAL_THREADS 100000000
#define MAX_THREADS_PER_BLOCK 1024
#define TEST_PER_METHOD 5


clock_t start_t, end_t, total_t;

// Declaration of variables for CUDA
int N = TOTAL_THREADS;
int numThreads = min(TOTAL_THREADS, MAX_THREADS_PER_BLOCK);
double num = (double)TOTAL_THREADS / numThreads;
int numBlocks = ceil(num);


// Function to read values from file
void read_ints(const char* fileName, double* matrix, int ID);

// Function to get number of data in file
int fsize(const char* fileName);

// Function to get the transpose of a matrix
void transpose(double* matrix, int rows, int columns);

// Serial matrix multiplication
void matMulSerial(double* matrixA, double* matrixB, double* matrixC, int rowsA, int colsA, int rowsB, int colsB);
void serial(double* matrixA, double* matrixB, double* matrixC, int rowsA, int colsA, int rowsB, int colsB, double* Runtime_Sx);

// OpenMP matrix multiplication
void matMulOMP(double* matrixA, double* matrixB, double* matrixC, int rowsA, int colsA, int rowsB, int colsB);
void openMPmatMul(double* matrixA, double* matrixB, double* matrixC, double* matrixS, int rowsA, int colsA, int rowsB, int colsB, double* Runtime_Sx);

// CUDA matrix multiplication
void cudaMatMul(double* matrixA, double* matrixB, double* matrixC, double* matrixS, int rowsA, int colsA, int rowsB, int colsB, double* Runtime_Sx);

// Print data
void printData(double* Runtime_Sx);

// Compare values with serial
int comp_vals(double* matrixC, double* matrixS, int rows, int columns);


// CUDA kernel to perform matrix multiplication
__global__ void matMulCuda(double* matrixA, double* matrixB, double* matrixC, int rowsA, int colsA, int rowsB, int colsB)
{
    // blockIdx.x
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long i, j, k;
    double sum;

    //double j_aux1, j_aux2;
    long stride = blockDim.x * gridDim.x;
    sum = 0;

    //printf("CUDA Thread \n");
   // printf("Rows A = %d, colsB = %d, C dim  = %d\n", rowsA, colsB, rowsA * colsB);


    for (i = idx; i < rowsA; i += stride)
    {
        //printf("START thread %d: i = %d, j = 0 to %d, k = 0 to %d\n", idx, i, j, k);

        for (j = 0; j < colsB; j++) {
            for (k = 0; k < rowsB; k++)
                //sum = sum + *(matrixA + i * colsA + k) * *(matrixB + j * rowsB + k);
                sum = sum + matrixA[i * colsA + k] * matrixB[j * rowsB + k];

            matrixC[i * colsB + j] = sum;
            // *(matrixC + i * colsB + j) = sum;
            sum = 0;
            //printf("i = %d, j=%d \n", i, j);
           //rintf("thread %d: matrixC[%d, %d] -> k = 0 to %d\n", idx, i, j, k);

        }
        //printf("END thread %d: i = %d, j = 0 to %d, k = 0 to %d\n", idx, i, j, k);

    }

}

int main() {

    // Declaration and initialization of variables for serial and OpenMP
    double* matA, * matB, * matC, * matC_serial;
    int rowsA, colsA, rowsB, colsB, sizeFA, sizeFB;
    double *Runtime_Sx;
    const char name_MA[MAXLENGHT_NAME] = "matrizA.txt";
    const char name_MB[MAXLENGHT_NAME] = "matrizB.txt";

    // Variables size for every matrix
    size_t datasizeA, datasizeB, datasizeC, datasizeTime;

    bool serialParallelSuccess = 0;

    // Asking for dimensions of matrices
    printf("\tIngrese el numero de filas y columnas de la matriz A: ");
    scanf("%d%d", &rowsA, &colsA);
    printf("\tIngrese el numero de filas y columnas de la matriz B: ");
    scanf("%d%d", &rowsB, &colsB);

    // Getting size of the file
    sizeFA = fsize(name_MA);
    sizeFB = fsize(name_MB);

   

    // Validating if the matrixes can be multiplied
    if (colsA != rowsB)
        printf("\tLa multiplicacion de matrices no se puede realizar con esas dimensiones.\n");
    // Validating dimensions with input data
    else if ((rowsA * colsA) != sizeFA || (rowsB * colsB) != sizeFB)
        printf("\tLas dimensiones con la cantidad de datos en el archivo no coinciden\n\tA: r=%d c=%d data=%d\n\tB: r=%d c=%d data=%d\n",
            rowsA, colsA, sizeFA, rowsB, colsB, sizeFB);
    else {
        // Showing input data info to the user
        printf("\n\tName 1: %s with dimensions Rows: %d   Cols: %d   and %d data read on file\n", name_MA, rowsA, colsA, sizeFA);
        printf("\tName 2: %s with dimensions Rows: %d   Cols: %d   and %d data read on file\n", name_MB, rowsB, colsB, sizeFB);

        // Setting size for every matrix
        datasizeA = rowsA * colsA * sizeof(double);
        datasizeB = rowsB * colsB * sizeof(double);
        datasizeC = rowsA * colsB * sizeof(double);
        datasizeTime = 3 * TEST_PER_METHOD * sizeof(double);

        // Time memory allocation
        Runtime_Sx = (double*)malloc(datasizeTime);

        // Allocating memory for matrix B
        matB = (double*)malloc(datasizeB);

        // Validating memory allocation for memory B
        if (matB == NULL)
            printf("\tMemorry allocation unsuccessful\n");
        else {
            // Reading input file for Matrix B and getting transpose
            read_ints(name_MB, matB, 2);
            transpose(matB, rowsB, colsB);
        }

        // Allocating memory for matrix A and C
        matA = (double*)malloc(datasizeA);
        matC = (double*)malloc(datasizeC);
        matC_serial = (double*)malloc(datasizeC);

        // Validating memory allocation for Matrix A, B and C
        if (matA == NULL || matB == NULL || matC == NULL || matC_serial == NULL || datasizeC == 0) {
            printf("\tMemorry allocation unsuccessful\n");
            exit(-1);
        }
        else {
            /// Validation approved
            read_ints(name_MA, matA, 1);

            //\\\\\\\\\\\\\\\\\\\\\\\\\\METODO SERIAL///////////////////////////////////
            printf("\n\t============== Serial ==============\n");
            printf("\n\tCalculating matrix in serial mode...\n");
            serial(matA, matB, matC_serial, rowsA, colsA, rowsB, colsB, Runtime_Sx);
            printf("\tDone Serial\n");

            //\\\\\\\\\\\\\\\\\\\\\\\\\\OpenMP///////////////////////////////////
            printf("\n\t============== Open MP ==============\n");
            printf("\n\tCalculating matrix in OpenMP...\n");
            openMPmatMul(matA, matB, matC, matC_serial, rowsA, colsA, rowsB, colsB, Runtime_Sx);
            

            //CUDA can be execued
            serialParallelSuccess = 1;

        }
    }

    //\\\\\\\\\\\\\\\\\\\\\\\\\\CUDA///////////////////////////////////
    if (serialParallelSuccess)
    {
        printf("\n\t============== CUDA ==============\n");
        printf("\n\tCalculating matrix in CUDA...\n");
        cudaMatMul(matA, matB, matC, matC_serial, rowsA, colsA, rowsB, colsB, Runtime_Sx);
    
        free(matA); free(matB); free(matC); free(matC_serial);

        printData(Runtime_Sx);
        free(Runtime_Sx);
        
    }


    // DEVICE info
    /*
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Maximum number of 32-bit registers: %d\n", prop.regsPerBlock);
        printf("  Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Maximum block dimension: [%d,%d,%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);
        printf("  Maximum grid size: [%d,%d,%d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) /
            1.0e6);

    }
    */
    return 0;
}


void read_ints(const char* fileName, double* matrix, int ID) {
    FILE* myfile;
    double myvariable;

    int i = 0;

    myfile = fopen(fileName, "r");

    while (fscanf(myfile, "%lf", &myvariable) == 1) // expect 1 successful conversion
    {
        //"%[^\n]%*c"
        fscanf(myfile, "%[^\n]%*c", &myvariable);
        *(matrix + i) = myvariable;
        i++;
        //printf("%.15f \n", myvariable);
    }

    fclose(myfile);
}

int fsize(const char* fileName) {
    FILE* myfile;
    double myvariable;

    int i = 0;

    myfile = fopen(fileName, "r");

    if (myfile == NULL)
    {
        printf("Error! Could not open file\n");
        exit(-1);
    }

    while (fscanf(myfile, "%lf", &myvariable) == 1) // expect 1 successful conversion
    {
        i++;
    }

    fclose(myfile);

    return i;
}

void transpose(double* matrix, int rows, int columns) {
    double* matrixAux;
    long i, j;
    size_t dimension = rows * columns * sizeof(double);
    matrixAux = (double*)malloc(dimension);

    if (matrixAux == NULL)
        printf("Memorry allocation unsuccessful\n");
    else {
        for (i = 0; i < columns; i++)
            for (j = 0; j < rows; j++)
                *(matrixAux + i * rows + j) = *(matrix + j * columns + i);

        for (i = 0; i < columns * rows; i++)
            *(matrix + i) = *(matrixAux + i);

        free(matrixAux);
    }
}

void matMulSerial(double* matrixA, double* matrixB, double* matrixC, int rowsA, int colsA, int rowsB, int colsB) {
    long i, j, k;
    double sum;

    sum = 0;

    for (i = 0; i < rowsA; i++)
        for (j = 0; j < colsB; j++) {
            for (k = 0; k < rowsB; k++)
                sum = sum + *(matrixA + i * colsA + k) * *(matrixB + j * rowsB + k);
            *(matrixC + i * colsB + j) = sum;
            sum = 0;
        }

}

void serial(double* matrixA, double* matrixB, double* matrixC, int rowsA, int colsA, int rowsB, int colsB, double* Runtime_Sx) {

    FILE* myfile;

    myfile = fopen("matrizC.txt", "w");

    for (int i = 0; i < TEST_PER_METHOD; i++) {

        start_t = clock();

        matMulSerial(matrixA, matrixB, matrixC, rowsA, colsA, rowsB, colsB);

        end_t = clock();

        total_t = end_t - start_t;

        *(Runtime_Sx + i) = (((double)total_t) / CLOCKS_PER_SEC) * 1000.0;

    }

    for (int i = 0; i < rowsA * colsB; i++)
        fprintf(myfile, "%.10lf\n", *(matrixC + i));

    fclose(myfile);
}

void matMulOMP(double* matrixA, double* matrixB, double* matrixC, int rowsA, int colsA, int rowsB, int colsB) {
    int THREADS = MAXTHREADS;
    omp_set_num_threads(THREADS);

#pragma omp parallel shared(matrixC)
    {
        int numThread = omp_get_thread_num();
        long i, j, k;
        double sum = 0;

        for (i = numThread; i < rowsA; i += THREADS)
            for (j = 0; j < colsB; j++) {
                for (k = 0; k < rowsB; k++)
                    sum = sum + *(matrixA + i * colsA + k) * *(matrixB + j * rowsB + k);
                *(matrixC + i * colsB + j) = sum;
                sum = 0;
            }


    }
}

void openMPmatMul(double* matrixA, double* matrixB, double* matrixC, double* matrixS, int rowsA, int colsA, int rowsB, int colsB, double* Runtime_Sx) {

    for (int i = 0; i < TEST_PER_METHOD; i++) {

        start_t = clock();

        matMulOMP(matrixA, matrixB, matrixC, rowsA, colsA, rowsB, colsB);

        end_t = clock();

        total_t = end_t - start_t;

        *(Runtime_Sx + i + TEST_PER_METHOD) = (((double)total_t) / CLOCKS_PER_SEC) * 1000.0;

    }

    printf("\tDone OpenMP\n");
    if (comp_vals(matrixC, matrixS, rowsA, colsB))
        printf("\tMatrix generated with OMP it IS equal than the generated in Serial\n");
    else
        printf("\tMatrix generated with OMP it IS NOT equal than the generated in Serial\n");

}

void cudaMatMul(double* matrixA, double* matrixB, double* matrixC, double* matrixS, int rowsA, int colsA, int rowsB, int colsB, double* Runtime_Sx) {

    cudaError_t  error_memCuda, error_sync;

    double* d_matA, * d_matB, * d_matC;

    size_t datasizeA = rowsA * colsA * sizeof(double);
    size_t datasizeB = rowsB * colsB * sizeof(double);
    size_t datasizeC = rowsA * colsB * sizeof(double);

    if (cudaMalloc((void**)&d_matA, datasizeA) | cudaMalloc((void**)&d_matB, datasizeB) | cudaMalloc((void**)&d_matC, datasizeC))
    {
        error_memCuda = cudaGetLastError();
        printf("Error Cuda %s \n", cudaGetErrorName(error_memCuda));
    }

    //Copy memory from readed matrixes
    cudaMemcpy(d_matA, matrixA, datasizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matrixB, datasizeB, cudaMemcpyHostToDevice);


    for (int i = 0; i < TEST_PER_METHOD; i++) {

        start_t = clock();

        matMulCuda << <numBlocks, numThreads >> > (d_matA, d_matB, d_matC, rowsA, colsA, rowsB, colsB);

        if (cudaDeviceSynchronize())
        {
            error_sync = cudaGetLastError();
            printf("Error Cuda %s \n", cudaGetErrorName(error_sync));
        }

        end_t = clock();

        total_t = end_t - start_t;

        *(Runtime_Sx + i + 2*TEST_PER_METHOD) = (((double)total_t) / CLOCKS_PER_SEC) * 1000.0;

        //Copy Calculated matrix to Host pointer
        cudaMemcpy(matrixC, d_matC, datasizeC, cudaMemcpyDeviceToHost);

    }

    printf("\tDone CUDA\n");
    if (comp_vals(matrixC, matrixS, rowsA, colsB) == 1)
        printf("\tMatrix generated with CUDA it IS equal than the generated in Serial\n\n\n");
    else
        printf("\tMatrix generated with CUDA it IS NOT equal than the generated in Serial\n\n\n");

    cudaFree(d_matA); cudaFree(d_matB); cudaFree(d_matC);
}

void printData(double* Runtime_Sx) {
    printf("\t%6s %15s %15s %15s\n", "Corrida", "Serial", "Paralelo OMP", "Paralelo CUDA");

    double promedio[3] = { 0.0, 0.0, 0.0 };

    for (int i = 0; i < TEST_PER_METHOD; i++) {
        printf("\t%6d %18.2f %15.2f %15.2f\n", i + 1, *(Runtime_Sx + i), *(Runtime_Sx + i + TEST_PER_METHOD), *(Runtime_Sx + i + 2*TEST_PER_METHOD));
        promedio[0] += *(Runtime_Sx + i);
        promedio[1] += *(Runtime_Sx + i + TEST_PER_METHOD);
        promedio[2] += *(Runtime_Sx + i + 2 * TEST_PER_METHOD);
    }

    promedio[0] /= TEST_PER_METHOD;
    promedio[1] /= TEST_PER_METHOD;
    promedio[2] /= TEST_PER_METHOD;

    printf("\t%s %16.2f %15.2f %15.2f\n", "Promedio", promedio[0], promedio[1], promedio[2]);
    printf("\t%s %15s %15.2f %15.2f\n", "%vsSerial", " - ", promedio[1] / promedio[0], promedio[2] / promedio[0]);
}

int comp_vals(double* matrixC, double* matrixS, int rows, int columns) {
    
    int i = 0, valid = 1;


    while (valid && i < rows*columns)
    {
        double a = *(matrixC + i);
        double b = *(matrixS + i);
        if ((a - b) > 0.000001)
            valid = 0;
        i++;
    }

    return valid;
}