#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"

void GVqrrq_g(double* T0, double* T1, double* T2, double* T3, double* c, double* s, double shift, int Nstep);
void given_rotation(double* c, double* s, double* r, double a, double b);

int Lanczos_LockPurge_gpu(
    LANCZOS_BUFFER* lBuffer, 
    cublasHandle_t  cublas_handle, 
    int Asize, int Nwant, int Nstep)
{
    int j, jj;
    double  cublas_scale;
    cuDoubleComplex* U = lBuffer->dU;
    double* T0 = lBuffer->T0;
    double* T1 = lBuffer->T1;
    double* T2 = lBuffer->T2;
    double* T3 = lBuffer->T3;
    double* c  = lBuffer->c;
    double* s  = lBuffer->s;
    double* LT0 = lBuffer->LT0;

    for(j = 0; j < Nstep - Nwant; j++)
    {
        // Shifted QR for T 
        GVqrrq_g(T0, T1, T2, T3, c, s, LT0[Nwant + j], Nstep - j);

        for(jj = 0; jj < Nstep - 1 - j; jj++)
            cublasZdrot_v2(cublas_handle, Asize, U+Asize*(jj+1), 1, U+Asize*jj, 1, c+jj, s+jj);
        s[jj -1] *= T1[jj];
        cublasZdrot_v2(cublas_handle, Asize, U+Asize*(jj+1), 1, U+Asize*jj, 1, T1+jj-1, s+jj-1);
       // Normalize U(:,Nstep-1-j) 
        cublasDznrm2_v2(cublas_handle, Asize, U+Asize*jj, 1, T1+jj-1);

        cublas_scale = 1.0 / T1[jj-1];
        cublasZdscal_v2(cublas_handle, Asize, &cublas_scale, U+Asize*jj, 1);

        // Purge Talpha(Nstep-1-j) and Tbeta(Nstep-1-j) 
        T0[jj] = 0.0;
        T1[jj] = 0.0;

        jj++;
        // Purge U(:,Nstep-j) 
        cudaMemset(U+Asize*jj, 0, Asize * sizeof(cuDoubleComplex));
    } // end of j

    return 0;
}

/*
int Lanczos_LockPurge_gpu(
	LANCZOS_BUFFER* lBuffer, 
	cublasHandle_t  cublas_handle, 
	int Asize, int Nwant, int Nstep)
{
	int j, jj;
	double  cublas_scale;
    cuDoubleComplex cublas_zscale;
	cuDoubleComplex* U = lBuffer->dU;
	double* T0 = lBuffer->T0;
	double* T1 = lBuffer->T1;
	double* T2 = lBuffer->T2;
	double* T3 = lBuffer->T3;
	double* c  = lBuffer->c;
	double* s  = lBuffer->s;
	double* LT0 = lBuffer->LT0;
    double bk = 1.0;

	for(j = 0; j < Nstep - Nwant; j++)
	{
		// Shifted QR for T 
		GVqrrq_g(T0, T1, T2, T3, c, s, LT0[Nwant + j], Nstep - j);

		for(jj = 0; jj < Nstep - 1 - j; jj++)
			cublasZdrot_v2(cublas_handle, Asize, U+Asize*(jj+1), 1, U+Asize*jj, 1, c+jj, s+jj);

        bk = -bk * s[Nstep - 2 - j];
	}

    cublas_scale = T1[Nwant - 1];
    cublasZdscal_v2(cublas_handle, Asize, &cublas_scale, U+Asize*Nwant, 1);

    bk = bk * T1[Nstep - 1];
    cublas_zscale = make_cuDoubleComplex(bk, 0.0);
    cublasZaxpy_v2(cublas_handle, Asize, &cublas_zscale, U+Asize*Nstep, 1, U+Asize*Nwant, 1);

    cublasDznrm2_v2(cublas_handle, Asize, U+Asize*Nwant, 1, &T1[Nwant - 1]);
    cublas_scale = 1.0 / T1[Nwant - 1];
    cublasZdscal_v2(cublas_handle, Asize, &cublas_scale, U+Asize*Nwant, 1);

	return 0;
}
*/

void GVqrrq_g(double* T0, double* T1, double* T2, double* T3, double* c, double* s, double shift, int n)
{
	
	//T is symmetric
	//T0;  diag(T, 0)
    //T1;  diag(T, -1) 
    //T2;  diag(R, 2), where R = Q' * T
    //T3;  temp T1
	
    int i;
    double a, b, r, R[6];
    
    memcpy(T3, T1, (n-1) * sizeof(double));

    // shift
    for(i = 0; i < n; i++)
        T0[i] -= shift;

    // Q' * T
    for(i = 0; i < n - 1; i++)
    {
        a     = T0[i]; 
        b     = T1[i];

        given_rotation(&c[i], &s[i], &r, a, b);

        T0[i] = r;
        T1[i] = 0.0;
    
        a     = T3[i];
        b     = T0[i + 1];

        T3[i]      = c[i] * a - s[i] * b;
        T0[i + 1]  = s[i] * a + c[i] * b;

        if(i < n - 2)
        {
        	T2[i]   = -s[i] * T3[i+1];
            T3[i+1] =  c[i] * T3[i+1];
        }
    }

    // T * Q
    
    //R is upper triangular.
    //R = x x x 0 0 0 0 0
    //    0 x x x 0 0 0 0
    //    0 0 x x x 0 0 0
    //    0 0 0 x x x 0 0
    //    0 0 0 0 x x x 0
    //    0 0 0 0 0 x x x
    //    0 0 0 0 0 0 x x
    //    0 0 0 0 0 0 0 x

    //| R0 R1 |  [  c s]
    //|  0 R3 |  [ -s c]
    

    R[0] = T0[0];
    R[1] = T3[0];
    R[2] = T1[0];
    R[3] = T0[1];

    T0[0] = c[0] * R[0] - s[0] * R[1];
    T3[0] = s[0] * R[0] + c[0] * R[1];
    T1[0] = c[0] * R[2] - s[0] * R[3];
    T0[1] = s[0] * R[2] + c[0] * R[3];
    
    //| R0 R1 |  [ c s]
    //| R2 R3 |  [-s c]
    //|  0 R5 |
    

    for(i = 1; i < n - 1; i++)
    {
        R[0]    = T3[i - 1];
        R[1]    = T2[i - 1];
        R[2]    = T0[i];
        R[3]    = T3[i];
        R[4]    = T1[i];
        R[5]    = T0[i + 1];

        T3[i - 1] =  c[i] * R[0] - s[i] * R[1];
        T2[i - 1] =  s[i] * R[0] + c[i] * R[1];
        T0[i]     =  c[i] * R[2] - s[i] * R[3];
        T3[i]     =  s[i] * R[2] + c[i] * R[3];
        T1[i]     =  c[i] * R[4] - s[i] * R[5];
        T0[i + 1] =  s[i] * R[4] + c[i] * R[5];
    }

    for(i = 0; i < n - 1; i++)
        T1[i] = (T1[i] + T3[i]) / 2.0;

    for(i = 0; i < n; i++)
        T0[i] += shift;
}

/*
void GVqrrq_g(double* T0, double* T1, double* T2, double* T3, double* c, double* s, double shift, int n)
{
    int i;
    double a, b, r, R[6];
    
    memcpy(T3, T1, (n-1) * sizeof(double));

    // shift
    for(i = 0; i < n; i++)
        T0[i] -= shift;

    // Q' * T
    for(i = 0; i < n - 1; i++)
    {
        a     = T0[i]; 
        b     = T1[i];

        given_rotation(&c[i], &s[i], &r, a, b);

        T0[i] = r;
        T1[i] = 0.0;
    
        a     = T3[i];
        b     = T0[i + 1];

        T3[i]      = c[i] * a - s[i] * b;
        T0[i + 1]  = s[i] * a + c[i] * b;

        if(i < n - 2)
        {
            T2[i]   = -s[i] * T3[i+1];
            T3[i+1] =  c[i] * T3[i+1];
        }
    }

    for(i = 0; i < n - 1; i++)
    {
        R[0]    = T0[i];
        R[1]    = T3[i];
        R[2]    = T1[i];
        R[3]    = T0[i + 1];

        T0[i]     =  c[i] * R[0] - s[i] * R[1];
        //T3[i]     =  s[i] * R[0] + c[i] * R[1];
        T1[i]     =  c[i] * R[2] - s[i] * R[3];
        T0[i + 1] =  s[i] * R[2] + c[i] * R[3];
    }

    for(i = 0; i < n; i++)
        T0[i] += shift;
}
*/
void given_rotation(double* c, double* s, double* r, double a, double b)
{
    double t, u;
    if(fabs(a) > fabs(b))
    {
        t    = b / a;
        u    = fabs(a) / a * sqrt(1 + t * t);
        c[0] = 1.0 / u;
        s[0] = c[0] * t;
        r[0] = a * u;
        s[0] = -s[0];
    }
    else
    {
        t    = a / b;
        u    = fabs(b) / b * sqrt(1 + t * t);
        s[0] = 1.0 / u;
        c[0] = s[0] * t;
        r[0] = b * u;
        s[0] = -s[0];
    }
}