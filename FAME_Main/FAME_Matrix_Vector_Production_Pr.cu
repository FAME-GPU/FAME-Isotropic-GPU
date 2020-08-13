#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"

//static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* vec_1, cuDoubleComplex* vec_out);
static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* L_2, cuDoubleComplex* vec_1, cuDoubleComplex* vec_2,cuDoubleComplex* vec_out);
static __global__ void initialize(int size, cuDoubleComplex* vec, double alpha, double beta);


////////////=========================== Create Pr function for Biiso (cuda)===========================//////////////////
int FAME_Matrix_Vector_Production_Pr(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cuDoubleComplex* vec_x, int Nx, int Ny, int Nz, int Nd, cuDoubleComplex* D_k, cuDoubleComplex* Pi_Pr, cuDoubleComplex* vec_y)
{
    int N = Nx*Ny*Nz;
	cuDoubleComplex* temp;
    cudaMalloc((void**)&temp, 3*N*sizeof(cuDoubleComplex));
    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((N-1)/BLOCK_SIZE +1,1,1);

	// Initial
    cudaMemset(temp, 0, 3*N * sizeof(cuDoubleComplex));
/*  
    //temp = Pi_Pr * vec_x
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr, vec_x, temp+N-Nd);
	cudaDeviceSynchronize();
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+Nd, vec_x, temp+N-Nd+N);
	cudaDeviceSynchronize();
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+2*Nd, vec_x, temp+N-Nd+2*N);
	cudaDeviceSynchronize();
*/
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr,         Pi_Pr+3*Nd, vec_x, vec_x+Nd, temp+N-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+Nd,      Pi_Pr+4*Nd, vec_x, vec_x+Nd, temp+N-Nd+N);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+2*Nd,    Pi_Pr+5*Nd, vec_x, vec_x+Nd, temp+N-Nd+2*N);

    IFFT_CUDA(vec_y, temp, D_k, fft_buffer, cuHandles, Nx, Ny, Nz);

	cudaFree(temp);

    return 0;
}

int FAME_Matrix_Vector_Production_Pr(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cuDoubleComplex* vec_x, int Nx, int Ny, int Nz, int Nd, cuDoubleComplex* D_kx, cuDoubleComplex* D_ky, cuDoubleComplex* D_kz, cuDoubleComplex* Pi_Pr, cuDoubleComplex* vec_y)
{
    int N = Nx*Ny*Nz;
    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((N-1)/BLOCK_SIZE +1,1,1);
    cuDoubleComplex* temp;
    cudaMalloc((void**)&temp, 3*N*sizeof(cuDoubleComplex));

    initialize<<<DimGrid, DimBlock>>>(N, temp, 0.0, 0.0);
/*	//Pi_Pr * vec_x
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr,         vec_x, temp+N-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+Nd,      vec_x, temp+N-Nd+N);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+2*Nd,    vec_x, temp+N-Nd+2*N);
*/
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr,         Pi_Pr+3*Nd, vec_x, vec_x+Nd, temp+N-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+Nd,      Pi_Pr+4*Nd, vec_x, vec_x+Nd, temp+N-Nd+N);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+2*Nd,    Pi_Pr+5*Nd, vec_x, vec_x+Nd, temp+N-Nd+2*N);


	for(int i=0; i<3; i++)
        spMV_fastT_gpu( vec_y+i*N, temp+i*N, cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, 1);

    cudaFree(temp);
    return 0;
}

static __global__ void initialize(int size, cuDoubleComplex* vec, double alpha, double beta)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size)
    {
        vec[idx*3].x = alpha; vec[idx*3].y = beta;
		vec[idx*3+1].x = alpha; vec[idx*3+1].y = beta;
		vec[idx*3+2].x = alpha; vec[idx*3+2].y = beta;
    }

}


// temp = Pi_Qr_1 dot vec_1 + Pi_Qr_2 dot vec_2
/*
static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* vec_1, cuDoubleComplex* vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size)
    {
        //vec_out[idx] = L_1[idx]*vec_1[idx] + L_2[idx]*vec_2[idx];
      vec_out[idx].x = L_1[idx].x*vec_1[idx].x + L_1[idx+3*size].x*vec_1[idx+size].x - L_1[idx].y*vec_1[idx].y - L_1[idx+3*size].y*vec_1[idx+size].y;
      vec_out[idx].y = L_1[idx].x*vec_1[idx].y + L_1[idx+3*size].y*vec_1[idx+size].x + L_1[idx].y*vec_1[idx].x + L_1[idx+3*size].x*vec_1[idx+size].y;

    }
}
*/
static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* L_2, cuDoubleComplex* vec_1, cuDoubleComplex* vec_2,cuDoubleComplex* vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size)
    {
        //vec_out[idx] = L_1[idx]*vec_1[idx] + L_2[idx]*vec_2[idx];
        vec_out[idx].x = L_1[idx].x*vec_1[idx].x + L_2[idx].x*vec_2[idx].x - L_1[idx].y*vec_1[idx].y - L_2[idx].y*vec_2[idx].y;
        vec_out[idx].y = L_1[idx].x*vec_1[idx].y + L_2[idx].y*vec_2[idx].x + L_1[idx].y*vec_1[idx].x + L_2[idx].x*vec_2[idx].y;

    }

}