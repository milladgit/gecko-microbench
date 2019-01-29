
#include <stdio.h>

#include <vector>
#include <algorithm>

using namespace std;

static char *exec_loc = "LocB";
static char *exec_pol = "static";


#define dval(a, nCols, i, j)    a[(i)*(nCols) + (j)]


void
ApplyDoubleStencil( void *vdata,
                    unsigned int nRows,
                    unsigned int nCols,
                    unsigned int nPaddedCols,
                    unsigned int nIters,
                    unsigned int nItersPerExchange,
                    void* vwCenter,
                    void* vwCardinal,
                    void* vwDiagonal,
                    void (*preIterBlockCB)(void* cbData),
                    void* cbData )
{
    double* __restrict__ data = (double*)vdata;
//    gecko_double data = *(gecko_double*) vdata;
    double wCenter = *(double*)vwCenter;
    double wCardinal = *(double*)vwCardinal;
    double wDiagonal = *(double*)vwDiagonal;


    /*
     * Our algorithm is double buffering.  We need to allocate a buffer
     * on the device of the same size as the input data.  We use
     * OpenACC's create clause on a data region to accomplish this.
     */
    double* __restrict__ other = (double*)calloc( nRows * nPaddedCols, sizeof(double) );

    #pragma acc data copyin(other[0:nRows*nPaddedCols]) 
    {

    /* Perform the stencil operation for the desired number of iterations.
     * To support the necessary halo exchanges in the truly parallel version,
     * we need to ensure that the data in the "data" matrix is valid in
     * the host memory every "nItersPerExchange" iterations.  Since OpenACC
     * doesn't give us an explicit operation to "read data from device now"
     * as we have with OpenCL and CUDA, we have to break up the iterations
     * and put each block of iterations between exchanges into its own
     * data region for the "data" array.
     *
     * For the sequential version, this logic should degenerate to a
     * single data region for "data," as long as nIters == nItersPerExchange.
     */
    unsigned int nIterBlocks = (nIters / nItersPerExchange) +
        ((nIters % nItersPerExchange) ? 1 : 0);

    for( unsigned int iterBlockIdx = 0; iterBlockIdx < nIterBlocks; iterBlockIdx++ )
    {
        unsigned int iterLow = iterBlockIdx * nItersPerExchange;
        unsigned int iterHighBound = (iterBlockIdx + 1) * nItersPerExchange;
        if( iterHighBound > nIters )
        {
            iterHighBound = nIters;
        }

        /* do any per-iteration-block work (e.g., do a halo exchange!) */
        if( preIterBlockCB != NULL )
        {
            (*preIterBlockCB)( cbData );
        }

        for( unsigned int iter = iterLow; iter < iterHighBound; iter++ )
        {
            /* apply the stencil operator */

            unsigned int nRows_1 = nRows - 1;

            // Launching following kernel at the selected location
            #pragma acc parallel loop collapse(2) independent present(data[0:nRows*nPaddedCols])
            for(unsigned int i=1; i<nRows_1; i++)
            {
                for( unsigned int j = 1; j < (nPaddedCols-1); j++ )
                {
                    double oldCenterValue = dval(data, nPaddedCols, i, j);
                    double oldNSEWValues = dval(data, nPaddedCols, i - 1, j ) +
                                            dval(data, nPaddedCols, i + 1, j ) +
                                            dval(data, nPaddedCols, i, j - 1 ) +
                                            dval(data, nPaddedCols, i, j + 1 );
                    double oldDiagonalValues = dval(data, nPaddedCols, i - 1, j - 1) +
                                                dval(data, nPaddedCols, i - 1, j + 1) +
                                                dval(data, nPaddedCols, i + 1, j - 1) +
                                                dval(data, nPaddedCols, i + 1, j + 1);

                    double newVal = wCenter * oldCenterValue +
                                    wCardinal * oldNSEWValues +
                                    wDiagonal * oldDiagonalValues;
                    dval(other, nPaddedCols, i, j ) = newVal;
                }
            }

            /* Copy the new values into the "real" array (data)
             * Note: we would like to just swap pointers between a "current" 
             * and "new" array, but have not figured out how to do this successfully
             * within OpenACC parallel region.
             */

            // Launching following kernel at the selected location
            #pragma acc parallel loop collapse(2) independent present(data[0:nRows*nPaddedCols])
            for( unsigned int i = 1; i < nRows_1; i++ )
            {
                for( unsigned int j = 1; j < (nCols - 1); j++ )
                {
                    dval(data, nPaddedCols, i, j) = dval(other, nPaddedCols, i, j);
                }
            }
        }

    }
    } /* end of OpenACC data region for "other" array */

    free(other);
}



int main(int argc, char const *argv[]) {

	if(argc < 3) {
		printf("Usage: %s <iter_count> <elements_count>\n", argv[0]);
		return -1;
	}

	int iter_count = atoi(argv[1]);

	int N = atoi(argv[2]);

	double *X;
	double w = 0.5;

	vector<double> v_time;

	X = (double*) malloc(sizeof(double) * N * N);

	#pragma acc enter data copyin(X[0:N*N])

	double time;
	time = omp_get_wtime();

	ApplyDoubleStencil(X,
	                N,
	                N,
	                N,
	                iter_count,
	                1,
	                (void*)(&w),
	                (void*)(&w),
	                (void*)(&w),
	                NULL,
	                NULL);
	
	time = omp_get_wtime() - time;
	time *= 1E6;

	v_time.push_back(time);

	#pragma acc exit data copyout(X[0:N*N])

	free(X);

	double sum = std::accumulate(v_time.begin(), v_time.end(), 0.0);
	printf("Time: %.2fus\n", sum);

	return 0;
}

