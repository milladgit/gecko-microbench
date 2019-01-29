
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <omp.h>

#include <chrono>
using namespace std;


#ifdef USE_DOUBLE
#define real double
#else
#define real float
#endif


/* Matrix size */
int N = 128;

/* Host implementation of a simple version of sgemm */
static void simple_gemm_no_acc(int n, real alpha, real *A, real *B,
						 real beta, real *C)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			real prod = 0;

			for (int k = 0; k < n; ++k)
			{
				prod += A[k * n + i] * B[j * n + k];
			}

			C[j * n + i] = alpha * prod + beta * C[j * n + i];
		}
	}
}

/* Host implementation of a simple version of sgemm */
static void simple_gemm_acc(int n, real alpha, real *A, real *B,
						 real beta, real *C)
{
	#pragma omp parallel for 
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			real prod = 0;

			for (int k = 0; k < n; ++k)
			{
				prod += A[k * n + i] * B[j * n + k];
			}

			C[j * n + i] = alpha * prod + beta * C[j * n + i];
		}
	}
}


/* Main */
int main(int argc, char **argv) 
{
	omp_set_num_threads(omp_get_num_procs());


	real *h_A, *d_A;
	real *h_B, *d_B;
	real *h_C, *d_C;
	real *h_C_ref;

	real alpha = 2.1;
	real beta = 2.1;
	int n2 = N * N;
	int i;
	real error_norm;
	real ref_norm;
	real diff;

	int num_times;

	if(argc < 3) {
		printf("%s <dim_size> <num_times>\n", argv[0]);
		exit(1);
	}

	N = atoi(argv[1]);
	n2 = N * N;
	num_times = atoi(argv[2]);

	printf("%s test running..\n", argv[0]);

	/* Allocate device memory for the matrices */
	d_A = (real*) malloc(sizeof(real) * n2);
	d_B = (real*) malloc(sizeof(real) * n2);
	d_C = (real*) malloc(sizeof(real) * n2);


	/* Allocate host memory for reading back the result from device memory */
	h_A = (real*) malloc(sizeof(real) * n2);
	h_B = (real*) malloc(sizeof(real) * n2);
	h_C = (real*) malloc(sizeof(real) * n2);
	h_C_ref = (real*) malloc(sizeof(real) * n2);


	/* Fill the matrices with test data */
	for (i = 0; i < n2; i++)
	{
		h_A[i] = rand() / (real)RAND_MAX;
		h_B[i] = rand() / (real)RAND_MAX;
	}

	memcpy(d_A, h_A, sizeof(real) * n2);
	memcpy(d_B, h_B, sizeof(real) * n2);


	/* Performs operation using plain C code */
	simple_gemm_no_acc(N, alpha, h_A, h_B, beta, h_C_ref);

	auto start = std::chrono::steady_clock::now();

	for(int i=0;i<num_times;i++)
		/* Performs operation using plain C code */
		simple_gemm_acc(N, alpha, d_A, d_B, beta, d_C);

	auto finish = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(finish - start).count();
	printf("Elapsed time per GEMM operation: %.2fs\n", elapsed_seconds / num_times);

	memcpy(h_C, d_C, sizeof(real) * n2);

	if(num_times == 1) {
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

		error_norm = (real)sqrt((double)error_norm);
		ref_norm = (real)sqrt((double)ref_norm);

		if (fabs(ref_norm) < 1e-7)
		{
			fprintf(stderr, "!!!! reference norm is 0\n");
			return EXIT_FAILURE;
		}

		fprintf(stderr, "Checking results...Done.\n");
	}

	/* Memory clean up */
	free(h_A);
	free(h_B);
	free(h_C);
    free(h_C_ref);
	free(d_A);
	free(d_B);
	free(d_C);

	if(num_times > 1) {
		printf("No checking since num_times variable is bigger than one.\n");
		exit(EXIT_SUCCESS);
	} else {
		if (error_norm / ref_norm < 1e-6f) {
			printf("%s test passed.\n", argv[0]);
			exit(EXIT_SUCCESS);
		} else {
			printf("%s test failed. error_norm/ref_norm: %.3f, error_norm: %.3f, ref_norm: %.3f\n", argv[0], error_norm/ref_norm, error_norm, ref_norm);
			exit(EXIT_FAILURE);
		}
	}

	return 0;
}
