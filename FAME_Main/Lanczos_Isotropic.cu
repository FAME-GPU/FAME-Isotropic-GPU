#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "Lanczos_decomp_gpu.cuh"
#include "Lanczos_LockPurge_gpu.cuh"
#include <lapacke.h>

int Lanczos_Isotropic( 
    double*          Freq_array, 
    cuDoubleComplex* ev,
    CULIB_HANDLES    cuHandles,
    LANCZOS_BUFFER   lBuffer,
    FFT_BUFFER       fft_buffer,
    LAMBDAS_CUDA     Lambdas_cuda,
    MTX_B            mtx_B,
    ES               es,
    LS               ls,
    int Nx, int Ny, int Nz, int Nd,
    string flag_CompType, PROFILE* Profile)
{
    int i, iter, conv;
    int Nwant = es.nwant;
    int Nstep = es.nstep;
    int Asize = 2 * Nd;
    double res;

    size_t z_size = Nstep * Nstep * sizeof(cuDoubleComplex);

    /* Variables for lapack */
    lapack_int  n, lapack_info, ldz;
    n           = (lapack_int) Nstep;
    ldz         = n;

    cuDoubleComplex* U = lBuffer.dU;
    cuDoubleComplex* dz = lBuffer.dz;
    double* T0  = lBuffer.T0;
    double* T1  = lBuffer.T1;
    double* LT0 = lBuffer.LT0;
    double* LT1 = lBuffer.LT1;
    cmpx *z = lBuffer.z;

    /* Initial Decomposition */
    Lanczos_decomp_gpu(U, T0, T1, 0, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
                             ls, Nx, Ny, Nz, Nd, Nwant, Nstep, flag_CompType, Profile);

    /* Begin Lanczos iteration */
    for(iter = 1; iter <= es.maxit; iter++)
    {
        memcpy(LT0, T0, Nstep * sizeof(double));
        memcpy(LT1, T1, (Nstep-1) * sizeof(double));

        /* Get the Ritz values T_d and Ritz vectors z*/
        /* Note that T_d will stored in descending order */
        lapack_info = LAPACKE_zpteqr(LAPACK_COL_MAJOR, 'I', n, LT0, LT1, z, ldz);
        assert(lapack_info == 0);

        cudaMemcpy(dz, z, z_size, cudaMemcpyHostToDevice);

        /* Check convergence, T_e will store the residules */
        conv = 0;
        for(i = 0; i < Nwant; i++)
        {
            res = T1[Nstep - 1] * cabs(z[(i + 1) * Nstep - 1]);

            if(res < es.tol)
                    conv++;
            else
                break;
        }

        /* Converged!! */
        if(conv == Nwant)
            break;


        /* Implicit Restart: Lock and Purge */
        Lanczos_LockPurge_gpu(&lBuffer, cuHandles.cublas_handle, Asize, Nwant, Nstep);

        printf("\033[40;33m= = = = = = = = = = = = = = = Lanczos Restart : %2d = = = = = = = = = = = = = = =\033[0m\n", iter);
        /* Restart */
        Lanczos_decomp_gpu(U, T0, T1, 1, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
                                 ls, Nx, Ny, Nz, Nd, Nwant, Nstep, flag_CompType, Profile);

    }
    if(iter == es.maxit + 1)
        printf("\033[40;31m      Lanczos did not converge when restart numbers reached ES_MAXIT (%3d).\033[0m\n", es.maxit);
    
    for(i = 0; i < Nwant; i++)
        Freq_array[i] = sqrt(1.0 / LT0[i]);

    cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    cublasZgemm(cuHandles.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, Asize, Nwant, Nstep, &one, U, Asize, dz, Nstep, &zero, ev, Asize);

    return iter;
}
