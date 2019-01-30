
#include <stdio.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;


#ifdef USE_DOUBLE
#define real double
#else
#define real float
#endif



static char *exec_loc = "LocH";


#define dval(a, nCols, i, j)    a[(i)*(nCols) + (j)]


void
ApplyDoubleStencil( void* vdata,
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
    real* __restrict__ data = (real*)vdata;
    real wCenter = *(real*)vwCenter;
    real wCardinal = *(real*)vwCardinal;
    real wDiagonal = *(real*)vwDiagonal;


    /*
     * Our algorithm is real buffering.  We need to allocate a buffer
     * on the device of the same size as the input data.  We use
     * OpenACC's create clause on a data region to accomplish this.
     */
    // real* __restrict__ other = (real*)calloc( nRows * nPaddedCols, sizeof(real) );
    real* __restrict__ other;
	#pragma gecko memory allocate(other[0:nRows*nPaddedCols]) type(real) location(exec_loc)

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
			#pragma gecko region exec_pol("runtime") variable_list(other,data) independent gang 
            for(unsigned int i=1; i<nRows_1; i++)
            {
		#pragma acc loop vector independent
                for( unsigned int j = 1; j < (nPaddedCols-1); j++ )
                {
                    real oldCenterValue = dval(data, nPaddedCols, i, j);
                    real oldNSEWValues = dval(data, nPaddedCols, i - 1, j ) +
                                            dval(data, nPaddedCols, i + 1, j ) +
                                            dval(data, nPaddedCols, i, j - 1 ) +
                                            dval(data, nPaddedCols, i, j + 1 );
                    real oldDiagonalValues = dval(data, nPaddedCols, i - 1, j - 1) +
                                                dval(data, nPaddedCols, i - 1, j + 1) +
                                                dval(data, nPaddedCols, i + 1, j - 1) +
                                                dval(data, nPaddedCols, i + 1, j + 1);

                    real newVal = wCenter * oldCenterValue +
                                    wCardinal * oldNSEWValues +
                                    wDiagonal * oldDiagonalValues;
                    dval(other, nPaddedCols, i, j ) = newVal;
                }
            }
			#pragma gecko region end

            /* Copy the new values into the "real" array (data)
             * Note: we would like to just swap pointers between a "current" 
             * and "new" array, but have not figured out how to do this successfully
             * within OpenACC parallel region.
             */

            // Launching following kernel at the selected location
			#pragma gecko region exec_pol("runtime") variable_list(other,data) independent gang 
            for( unsigned int i = 1; i < nRows_1; i++ )
            {
                #pragma acc loop vector independent
                for( unsigned int j = 1; j < (nCols - 1); j++ )
                {
                    dval(data, nPaddedCols, i, j) = dval(other, nPaddedCols, i, j);
                }
            }
			#pragma gecko region end
			#pragma gecko region pause at("LocA") 
        }

    }

	#pragma gecko memory free(other)
}



int main(int argc, char const *argv[]) {

	if(argc < 3) {
		printf("Usage: %s <elements_count> <iter_count>  \n", argv[0]);
		return -1;
	}

	#pragma gecko config file("gecko.conf")

	int iter_count = atoi(argv[2]);

	int N = atoi(argv[1]);

	real *X;
	real w = 0.5;

	vector<double> v_time;

	#pragma gecko memory allocate(X[0:N*N]) type(real) location(exec_loc) 

    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();

	ApplyDoubleStencil( X,
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
	
    t2 = std::chrono::high_resolution_clock::now();

    double runtime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    v_time.push_back(runtime);

	#pragma gecko memory free(X)

	double sum = std::accumulate(v_time.begin(), v_time.end(), 0.0);
	printf("Time: %.2fs\n", sum/v_time.size());

	return 0;
}

