#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>


#define Max(a,b) ((a)>(b)?(a):(b))

//#define N (2*2*2*2*2*2+2)
#define N 500

float maxeps = 0.1e-7;
int itmax = 1000;
int i, j, k;
float eps;
float A[N][N], B[N][N];

void relax();
float resid();
void init();
void verify(); 

int main(int an, char **as)
{
    struct timeval start_time, stop_time, elapsed_time;

    gettimeofday(&start_time, NULL);

    init();
    eps = 100;
    int iteration = 1;
    while (eps > maxeps && iteration <= itmax) {
        relax();
        eps = resid();
        printf("it=%4i  eps=%f\n", iteration, eps);
        iteration++;
    }

    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);

    verify();

    printf("\nMax error at iteration %d was %f\n", iteration - 1, eps);
    printf("Total time was %f seconds.\n", elapsed_time.tv_sec + elapsed_time.tv_usec/1000000.0);
    return 0;
}

void init()
{
    #pragma omp parallel for default(none) shared(A, B) private(i, j) 
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                A[i][j] = 0.0;
            }
            else {
                A[i][j] = (1.0 + i + j);
            }
        }
    }
} 

void relax()
{
    #pragma omp parallel for default(none) shared(A, B) private(i, j)
    for (i = 1; i < N - 1; i++) {
        for (j = 1; j < N - 1; j++) {
		    B[i][j]=(A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
	    }
    }
}

float resid()
{ 
    float eps = 0.0;
    #pragma omp parallel for default(none) shared(A, B) private(i, j) reduction(max:eps)
    for (i = 1; i < N - 1; i++) {
        for (j = 1; j < N - 1; j++) {
            float e = fabs(A[i][j] - B[i][j]);         
            A[i][j] = B[i][j];
            eps = Max(eps,e);    
        }
    }
    return eps;
}

void verify()
{
    float s = 0.0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            s += A[i][j] * (i + 1) * (j + 1) / (N * N);
        }
    }
    printf("S = %f\n", s);
}
