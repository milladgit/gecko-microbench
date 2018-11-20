
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

	gecko_double X, Y, Z;

	vector<double> v_time;

	gecko_double *ptr_list;
	ptr_list = (gecko_double*) malloc(sizeof(gecko_double) * 3 * iter_count);

	for(int iter=0;iter<3*iter_count;iter++) {
		gecko_double a;
		#pragma gecko memory allocate(a[0:N]) type(gecko_double) location(exec_loc) 
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
		#pragma gecko region at(exec_loc) exec_pol(exec_pol)
		#pragma acc parallel loop 
		for (int i = a; i<b; i++) {
			Z[i] = X[i] + Y[i];
		}
		#pragma gecko region end

		#pragma gecko region pause at(exec_loc) 
		
		time = omp_get_wtime() - time;
		time *= 1E6;

		v_time.push_back(time);

	}

	for(int iter=0;iter<3*iter_count;iter++) {
		gecko_double a;
		a = ptr_list[iter];
		#pragma gecko memory freeobj(a)
	}

	free(ptr_list);

	double sum = std::accumulate(v_time.begin(), v_time.end(), 0.0);
	printf("Sum: %.2fus\n", sum);
	printf("Avg: %.2fus\n", sum/v_time.size());

	return 0;
}

