
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

	#pragma gecko config env

	int iter_count = atoi(argv[1]);

	int N = atoi(argv[2]);

	double *X, *Y, *Z;

	vector<double> v_time;

	double **ptr_list;
	ptr_list = (double**) malloc(sizeof(double*) * 3 * iter_count);

	for(int iter=0;iter<3*iter_count;iter++) {
		double *a = (double*) malloc(sizeof(double) * N);
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
		#pragma acc parallel loop copyin(X[0:N], Y[0:N]) copy(Z[0:N])
		for (int i = a; i<b; i++) {
			Z[i] = X[i] + Y[i];
		}
		
		time = omp_get_wtime() - time;
		time *= 1E6;

		v_time.push_back(time);

	}

	for(int iter=0;iter<3*iter_count;iter++) {
		double *a;
		a = ptr_list[iter];
		free(a);
	}

	free(ptr_list);

	double sum = std::accumulate(v_time.begin(), v_time.end(), 0.0);
	printf("Sum: %.2fus\n", sum);
	printf("Avg: %.2fus\n", sum/v_time.size());

	return 0;
}

