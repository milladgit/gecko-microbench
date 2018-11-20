#include "geckoRuntime.h"

#include <stdio.h>

#include <vector>
#include <algorithm>

using namespace std;

static char *exec_loc = "LocB";
static char *exec_pol = "static";

int main(int argc, char const *argv[]) {

	if(argc < 3) {
		printf("Usage: %s <iter_count> <elements_count>\n", argv[0]);
		return -1;
	}

geckoLoadConfigWithEnv();
	int iter_count = atoi(argv[1]);

	int N = atoi(argv[2]);

	double *X, *Y, *Z;

	vector<double> v_time;

	double **ptr_list;
	ptr_list = (double**) malloc(sizeof(double*) * 3 * iter_count);

	for(int iter=0;iter<3*iter_count;iter++) {
		double *a;
geckoMemoryDeclare((void**)&a, sizeof(double), N, exec_loc, GECKO_DISTANCE_NOT_SET);
		ptr_list[iter] = a;
	}	


	for(int iter=0;iter<iter_count;iter++) {

		X = ptr_list[iter];
		Y = ptr_list[iter+1];
		Z = ptr_list[iter+2];

		int a = 0;
		int b = N;
		double coeff = 2.3;

		// #pragma gecko region at(exec_loc) exec_pol(exec_pol) variable_list(X,Y,Z)
		// #pragma acc parallel loop 
		// for (int i = a; i<b; i++) {
		// 	X[i] = i * coeff;
		// 	Y[i] = -i * coeff;
		// }
		// #pragma gecko region end


		double time;
		time = omp_get_wtime();
{
int *beginLoopIndex=NULL, *endLoopIndex=NULL, jobCount, devCount, devIndex;
GeckoLocation **dev = NULL;
int ranges_count = 0;
float *ranges = NULL;
int var_count = 3;
void **var_list = (void **) malloc(sizeof(void*) * var_count);
for(int __v_id=0;__v_id<var_count;__v_id++) {
var_list[__v_id] = X;
var_list[__v_id] = Y;
var_list[__v_id] = Z;
}
GeckoError err = geckoRegion(exec_pol, exec_loc, a, b, 1, 0, &devCount, &beginLoopIndex, &endLoopIndex, &dev, ranges_count, ranges, var_count, var_list);
jobCount = devCount;
if(err != GECKO_ERR_TOTAL_ITERATIONS_ZERO) {
#pragma omp parallel num_threads(jobCount)
{
int devIndex = omp_get_thread_num();
if(dev[devIndex] != NULL) {
int beginLI = beginLoopIndex[devIndex], endLI = endLoopIndex[devIndex];
int asyncID = dev[devIndex]->getAsyncID();
#pragma acc parallel loop deviceptr(X,Y,Z) async(asyncID) copyin()
for(int i = beginLI;i < endLI;i++) {
			Z[i] = X[i] + Y[i];
		}
#pragma acc wait(asyncID)
} // end of if(dev[devIndex]!=NULL)
} // end of OpenMP pragma 
} // end of checking: err != GECKO_ERR_TOTAL_ITERATIONS_ZERO 
geckoFreeRegionTemp(beginLoopIndex, endLoopIndex, devCount, dev, var_list);
}

geckoWaitOnLocation(exec_loc);
		
		time = omp_get_wtime() - time;
		time *= 1E6;

		v_time.push_back(time);

	}

	for(int iter=0;iter<3*iter_count;iter++) {
		double *a;
		a = ptr_list[iter];
geckoFree(a);
	}

	free(ptr_list);

	double sum = std::accumulate(v_time.begin(), v_time.end(), 0.0);
	printf("Sum: %.2fus\n", sum);
	printf("Avg: %.2fus\n", sum/v_time.size());

	return 0;
}

