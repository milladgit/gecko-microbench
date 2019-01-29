#include "geckoRuntime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <math.h>

#include <vector>
#include <algorithm>

using namespace std;

/* Includes, cuda */
#include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <helper_cuda.h>

/* Matrix size */
//#define N  (275)
//#define N (512)
int N = 128;

/* Host implementation of a simple version of sgemm */
static void simple_sgemm_no_acc(int n, float alpha, gecko_float A, gecko_float B,
                         float beta, gecko_float C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

/* Host implementation of a simple version of sgemm */
static void simple_sgemm_acc(int n, float alpha, gecko_float A, gecko_float B,
                         float beta, gecko_float C)
{
    int i;
    int j;
    int k;

{
int *beginLoopIndex=NULL, *endLoopIndex=NULL, jobCount, devCount, devIndex;
GeckoLocation **dev = NULL;
int ranges_count = 0;
float *ranges = NULL;
int var_count = 0;
void **var_list = NULL;
GeckoError err = geckoRegion("static", "LocB", 0, n, 1, 0, &devCount, &beginLoopIndex, &endLoopIndex, &dev, ranges_count, ranges, var_count, var_list);
jobCount = devCount;
if(err != GECKO_ERR_TOTAL_ITERATIONS_ZERO) {
#pragma omp parallel num_threads(jobCount)
{
int devIndex = omp_get_thread_num();
if(dev[devIndex] != NULL) {
int beginLI = beginLoopIndex[devIndex], endLI = endLoopIndex[devIndex];
int asyncID = dev[devIndex]->getAsyncID();
#pragma acc parallel loop private(i,j,k) collapse(2) gang deviceptr() async(asyncID) copyin(A[0:1],B[0:1],C[0:1])
for(int i = beginLI;i < endLI;++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float prod = 0;

		#pragma acc loop vector independent reduction(+:prod)
            for (int k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
#pragma acc wait(asyncID)
} // end of if(dev[devIndex]!=NULL)
} // end of OpenMP pragma 
} // end of checking: err != GECKO_ERR_TOTAL_ITERATIONS_ZERO 
geckoFreeRegionTemp(beginLoopIndex, endLoopIndex, devCount, dev, var_list);
}

geckoWaitOnLocation("LocB");

}


/* Main */
int main(int argc, char **argv)
{
    gecko_float h_A;
    gecko_float h_B;
    gecko_float h_C;
    gecko_float h_C_ref;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;

    if(argc < 3) {
        printf("%s <iter_count> <dim_size>\n", argv[0]);
        exit(1);
    }

    N = atoi(argv[2]);
    n2 = N * N;

    printf("simpleGEMM test running..\n");

geckoLoadConfigWithEnv();
    /* Allocate host memory for the matrices */
geckoMemoryInternalTypeDeclare(h_A, sizeof(float), n2, "LocB", GECKO_DISTANCE_NOT_SET);

    // h_B = (float *)malloc(n2 * sizeof(h_B[0]));
geckoMemoryInternalTypeDeclare(h_B, sizeof(float), n2, "LocB", GECKO_DISTANCE_NOT_SET);

    // h_C = (float *)malloc(n2 * sizeof(h_C[0]));
geckoMemoryInternalTypeDeclare(h_C, sizeof(float), n2, "LocB", GECKO_DISTANCE_NOT_SET);

    /* Allocate host memory for reading back the result from device memory */
//    #pragma gecko memory allocate(h_C_ref[0:n2]) type(gecko_float) location("LocA") 


    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
//        h_C_ref[i] = h_C[i];
    }


    /* Performs operation using plain C code */
//    simple_sgemm_no_acc(N, alpha, h_A, h_B, beta, h_C_ref);

    double t;
    int iter_count = atoi(argv[1]);
    vector<double> time_list;

    for(int i=0;i<iter_count;i++) {
      t = omp_get_wtime();

      /* Performs operation using plain C code */
      simple_sgemm_acc(N, alpha, h_A, h_B, beta, h_C);

geckoWaitOnLocation("LocB");

      t = omp_get_wtime() - t;
      time_list.push_back(t);
    }

    double sum = std::accumulate(time_list.begin(), time_list.end(), 0.0);
    printf("Sum: %.2fus\n", sum);
    printf("Avg: %.2fus\n", sum/time_list.size());


#if 0
    fprintf(stderr, "Checking results...\n");

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;

    for (i = 0; i < n2; ++i)
    {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Checking results...Done.\n");
#endif

    /* Memory clean up */
h_A.freeMem();

#if 0
    if (error_norm / ref_norm < 1e-6f)
    {
        printf("simpleGEMM test passed.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("simpleGEMM test failed.\n");
        exit(EXIT_FAILURE);
    }
#else
	return 0;
#endif
}

