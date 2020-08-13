#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"
#include "FAME_Create_cublas.cuh"
#include "FAME_Create_Buffer.cuh"
#include "FAME_Matrix_B_Isotropic.cuh"
#include "FAME_Malloc_mtx_C.h"
#include "FAME_Matrix_Lambdas.cuh"
#include "FAME_Matrix_Curl.h"
#include "FAME_Create_Frequency_txt.h"
#include "FAME_Save_Eigenvector.h"
#include "FAME_Profile.h"
#include "FAME_Destroy_Main.cuh"
#include "FAME_Fast_Algorithms_Isotropic.cuh"
#include "FAME_Matrix_Vector_Production_Qrs.cuh"
#include "FAME_Matrix_Vector_Production_Pr.cuh"
#include <complex.h>
#include "FAME_Create_C_txt.h"
void FAME_Fast_Algorithms_Driver(
	double*        Freq_array,
	cmpx*          Ele_field_mtx,
	CULIB_HANDLES  cuHandles,
	LANCZOS_BUFFER lBuffer,
	FFT_BUFFER     fft_buffer,
	LAMBDAS_CUDA   Lambdas_cuda,
	MTX_B          mtx_B,
	ES             es,
	LS             ls,
	int Nx, int Ny, int Nz, int Nd, int N, 
	char* material_type, char* lattice_type, PROFILE* Profile);

void Destroy_Lambdas(LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, char* lattice_type);
void Check_Eigendecomp(MTX_C mtx_C, LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, FFT_BUFFER fft_buffer, CULIB_HANDLES cuHandles,
	int Nx, int Ny, int Nz, int Nd, int N, char* lattice_type, PROFILE* Profile);
void Check_Residual(double* Freq_array, cmpx* Ele_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant);

int FAME_Main_Code(PAR Par, PROFILE* Profile)
{
	int Nx = Par.mesh.grid_nums[0];
    int Ny = Par.mesh.grid_nums[1];
	int Nz = Par.mesh.grid_nums[2];
	int Nd;
	int N  = Nx * Ny * Nz;
	int N3 = N * 3;
	int Nwant = Par.es.nwant;
	int Nstep = Par.es.nstep;
	int N_wave_vec = Par.recip_lattice.Wave_vec_num;
	double wave_vec_array[3];

	double accum;
	struct timespec start, end;

	double* Freq_array    = (double*) calloc(N_wave_vec * Nwant, sizeof(double));
	cmpx*   Ele_field_mtx = (cmpx*)   calloc(        N3 * Nwant, sizeof(cmpx));

	cudaSetDevice(Par.flag.device);
	
    CULIB_HANDLES  cuHandles;
	FFT_BUFFER     fft_buffer;
	LANCZOS_BUFFER lBuffer;
	MTX_B          mtx_B;
	MTX_C          mtx_C;
	LAMBDAS        Lambdas;
    LAMBDAS_CUDA   Lambdas_cuda;

	FAME_Create_cublas(&cuHandles, Nx, Ny, Nz);
	FAME_Create_Buffer(&cuHandles, &fft_buffer, &lBuffer, N, Nstep);
	
	printf("= = = = FAME_Matrix_B_Isotropic = = = = = = = = = = = = = = = = = = = = = = = = =\n");
	if(strcmp(Par.material.material_type, "isotropic") == 0)
	{
		checkCudaErrors(cudaMalloc((void**) &mtx_B.B_eps,    N3 * sizeof(double)));
		checkCudaErrors(cudaMalloc((void**) &mtx_B.invB_eps, N3 * sizeof(double)));
		FAME_Matrix_B_Isotropic(mtx_B.B_eps, mtx_B.invB_eps, Par.material, N);
	}

    FAME_Malloc_mtx_C(&mtx_C, N);

	for(int i = 0; i < N_wave_vec; i++)
    //for(int i = 0; i < 2; i++)
	{
		Profile->idx = i;

		wave_vec_array[0] = Par.recip_lattice.WaveVector[3 * i];
    	wave_vec_array[1] = Par.recip_lattice.WaveVector[3 * i + 1];
    	wave_vec_array[2] = Par.recip_lattice.WaveVector[3 * i + 2];

    	printf("\033[40;33m= = Start to compute (%3d/%3d) WaveVector = [ % .6f % .6f % .6f ] = =\033[0m\n", i + 1, Par.recip_lattice.Wave_vec_num, wave_vec_array[0], wave_vec_array[1], wave_vec_array[2]);

		printf("= = = = FAME_Matrix_Curl  = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		FAME_Matrix_Curl(&mtx_C, wave_vec_array, Par.mesh.grid_nums, Par.mesh.edge_len, Par.mesh.mesh_len, Par.lattice);
/*		
		FAME_Create_C_txt(mtx_C.C1_r, mtx_C.C1_c, mtx_C.C1_v,
			mtx_C.C2_r, mtx_C.C2_c, mtx_C.C2_v,
			mtx_C.C3_r, mtx_C.C3_c, mtx_C.C3_v, Par.mesh.grid_nums);
*/
		printf("= = = = FAME_Matrix_Lambdas = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		Nd = FAME_Matrix_Lambdas(&Lambdas_cuda, wave_vec_array, Par.mesh.grid_nums, Par.mesh.mesh_len, Par.lattice.lattice_vec_a, &Par, &Lambdas);

		printf("= = = = Check_Eigendecomp = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		clock_gettime(CLOCK_REALTIME, &start);
		Check_Eigendecomp(mtx_C, Lambdas, Lambdas_cuda, fft_buffer, cuHandles, Nx, Ny, Nz, Nd, N, Par.lattice.lattice_type, Profile);
		clock_gettime(CLOCK_REALTIME, &end);
		accum = ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / BILLION;
		printf("%*s%8.2f sec.\n", 68, "", accum);

		printf("= = = = FAME_Fast_Algorithms_Isotropic  = = = = = = = = = = = = = = = = = = = = =\n");
		clock_gettime (CLOCK_REALTIME, &start);

		FAME_Fast_Algorithms_Driver(Freq_array+i*Nwant, Ele_field_mtx, 
			cuHandles, lBuffer, fft_buffer, Lambdas_cuda, mtx_B, Par.es, Par.ls,
			Nx, Ny, Nz, Nd, N, Par.material.material_type, Par.lattice.lattice_type, Profile);

		clock_gettime (CLOCK_REALTIME, &end);
		Profile->es_time[Profile->idx] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
		
		printf("= = = = Check_Residual  = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		clock_gettime (CLOCK_REALTIME, &start);
		Check_Residual(Freq_array+i*Nwant, Ele_field_mtx, mtx_B, mtx_C, N, Nwant);
		clock_gettime (CLOCK_REALTIME, &end);
		accum = ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / BILLION;
		printf("%*s%8.2f sec.\n", 68, "", accum);

		if(Par.flag.save_eigen_vector)
		{
			printf("= = = = Save Eigen Vector = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
			FAME_Save_Eigenvector(Ele_field_mtx, Nwant, N3, i);
		}

		Destroy_Lambdas(Lambdas, Lambdas_cuda, Par.lattice.lattice_type);

		FAME_Print_Profile(*Profile);
	}

	FAME_Create_Frequency_txt(Freq_array, Nwant, Profile->idx);

	FAME_Destroy_Main(cuHandles, fft_buffer, lBuffer, mtx_B, mtx_C, Freq_array, Ele_field_mtx);

	return 0;
}

void FAME_Fast_Algorithms_Driver(
	double*        Freq_array,
	cmpx*          Ele_field_mtx,
	CULIB_HANDLES  cuHandles,
	LANCZOS_BUFFER lBuffer,
	FFT_BUFFER     fft_buffer,
	LAMBDAS_CUDA   Lambdas_cuda,
	MTX_B          mtx_B,
	ES             es,
	LS             ls,
	int Nx, int Ny, int Nz, int Nd, int N, 
	char* material_type, char* lattice_type, PROFILE* Profile)
{

	if(strcmp(material_type, "isotropic") == 0)
	{
		if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
		   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
		   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
		{

			FAME_Fast_Algorithms_Isotropic(Freq_array, Ele_field_mtx, cuHandles, lBuffer, fft_buffer,
								  Lambdas_cuda, mtx_B, es, ls, Nx, Ny, Nz, Nd, N, "Simple", Profile);
		}
		else
		{
			FAME_Fast_Algorithms_Isotropic(Freq_array, Ele_field_mtx, cuHandles, lBuffer, fft_buffer,
								 Lambdas_cuda, mtx_B, es, ls, Nx, Ny, Nz, Nd, N, "General", Profile);
		}
	}
}

void Destroy_Lambdas(LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, char* lattice_type)
{
	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
        free(Lambdas.D_k);
        free(Lambdas.D_ks);
        free(Lambdas.Lambda_x);
        free(Lambdas.Lambda_y);
        free(Lambdas.Lambda_z);

		cudaFree(Lambdas_cuda.dD_k);
		cudaFree(Lambdas_cuda.dD_ks);
	}

	else
	{
        free(Lambdas.D_kx);
        free(Lambdas.D_ky);
        free(Lambdas.D_kz);
        free(Lambdas.Lambda_x);
        free(Lambdas.Lambda_y);
        free(Lambdas.Lambda_z);

		cudaFree(Lambdas_cuda.dD_kx);
    	cudaFree(Lambdas_cuda.dD_ky);
    	cudaFree(Lambdas_cuda.dD_kz);
	}

    free(Lambdas.Lambda_q_sqrt);
    free(Lambdas.Pi_Qr);
    free(Lambdas.Pi_Pr);
    free(Lambdas.Pi_Qrs);
    free(Lambdas.Pi_Prs);

    cudaFree(Lambdas_cuda.Lambda_q_sqrt);
	cudaFree(Lambdas_cuda.dPi_Qr);
	cudaFree(Lambdas_cuda.dPi_Pr);
	cudaFree(Lambdas_cuda.dPi_Qrs);
	cudaFree(Lambdas_cuda.dPi_Prs);
}

void Check_Eigendecomp(MTX_C mtx_C, LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, FFT_BUFFER fft_buffer, CULIB_HANDLES cuHandles,
	int Nx, int Ny, int Nz, int Nd, int N, char* lattice_type, PROFILE* Profile)
{
	int i;
	int N2 = N * 2;
	int N3 = N * 3;
	int N12 = N * 12;
	size_t size, dsizeN3, dsizeNd2;

	size = N3 * sizeof(cmpx);

	cmpx* vec_x    = (cmpx*) malloc(size);
	cmpx* vec_y    = (cmpx*) malloc(size);
	cmpx* vec_temp = (cmpx*) malloc(size);

	cuDoubleComplex* N3_temp1 = cuHandles.N3_temp1;
	cuDoubleComplex* N3_temp2 = cuHandles.N3_temp2;

	cuDoubleComplex* Nd2_temp;
	dsizeN3 = N3 * sizeof(cuDoubleComplex);
	dsizeNd2 = Nd * 2 * sizeof(cuDoubleComplex);

	checkCudaErrors(cudaMalloc((void**)&Nd2_temp, dsizeNd2));

	srand(time(NULL));

	for(i = 0; i < N3; i++)
		vec_x[i] = ((double) rand()/(RAND_MAX + 1.0)) + I*((double) rand()/(RAND_MAX + 1.0));

	cmpx *vec_y_1, *vec_y_2, *vec_y_3;

	cudaMemcpy(N3_temp1, vec_x, dsizeN3, cudaMemcpyHostToDevice);

	if( (strcmp(lattice_type, "simple_cubic"          ) == 0) || \
		(strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
		(strcmp(lattice_type, "primitive_tetragonal"  ) == 0) )
	{
		FFT_CUDA(N3_temp2, N3_temp1, Lambdas_cuda.dD_ks, fft_buffer, cuHandles, Nx, Ny, Nz);
	}
	else
	{
		for(i = 0; i < 3; i++)
        	spMV_fastT_gpu(N3_temp2+i*N, N3_temp1+i*N, cuHandles, &fft_buffer, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Nx, Ny, Nz, -1);
	}

	cudaMemcpy(vec_y, N3_temp2, dsizeN3, cudaMemcpyDeviceToHost);
	vec_y_1 = &vec_y[0];  vec_y_2 = &vec_y[N];  vec_y_3 = &vec_y[N2];

	if(Nd == N - 1)
	{
		vec_y_1[0] = 0; vec_y_2[0] = 0; vec_y_3[0] = 0;
		for(i = 0; i < N - 1; i++)
		{
			vec_y_1[i + 1] = Lambdas.Lambda_x[i] * vec_y_1[i + 1];
			vec_y_2[i + 1] = Lambdas.Lambda_y[i] * vec_y_2[i + 1];
			vec_y_3[i + 1] = Lambdas.Lambda_z[i] * vec_y_3[i + 1];
		}
	}
	else
	{
		for(i = 0; i < N; i++)
		{
			vec_y_1[i] = Lambdas.Lambda_x[i] * vec_y_1[i];
			vec_y_2[i] = Lambdas.Lambda_y[i] * vec_y_2[i];
			vec_y_3[i] = Lambdas.Lambda_z[i] * vec_y_3[i];
		}
	}

	cudaMemcpy(N3_temp1, vec_y, dsizeN3, cudaMemcpyHostToDevice);

	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
		IFFT_CUDA(N3_temp2, N3_temp1, Lambdas_cuda.dD_k, fft_buffer, cuHandles, Nx, Ny, Nz);
	}
	else
	{
		for(i = 0; i < 3; i++)
			spMV_fastT_gpu(N3_temp2+i*N, N3_temp1+i*N, cuHandles, &fft_buffer, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Nx, Ny, Nz, 1);
	}

	cudaMemcpy(vec_y, N3_temp2, dsizeN3, cudaMemcpyDeviceToHost);

	mtx_prod(&vec_temp[0] , mtx_C.C1_r, mtx_C.C1_c, mtx_C.C1_v, &vec_x[0] , N2, N);
	mtx_prod(&vec_temp[N] , mtx_C.C2_r, mtx_C.C2_c, mtx_C.C2_v, &vec_x[N] , N2, N);
	mtx_prod(&vec_temp[N2], mtx_C.C3_r, mtx_C.C3_c, mtx_C.C3_v, &vec_x[N2], N2, N);

	size = N * sizeof(cmpx);
	cmpx* test_x = (cmpx*) malloc(size);
	cmpx* test_y = (cmpx*) malloc(size);
	cmpx* test_z = (cmpx*) malloc(size);

	vec_plus(test_x, 1.0, &vec_temp[0] , -1.0, &vec_y[0] , N);
	vec_plus(test_y, 1.0, &vec_temp[N] , -1.0, &vec_y[N] , N);
	vec_plus(test_z, 1.0, &vec_temp[N2], -1.0, &vec_y[N2], N);

	double C1_error = vec_norm(test_x, N);
    double C2_error = vec_norm(test_y, N);
    double C3_error = vec_norm(test_z, N);

	free(test_x); free(test_y); free(test_z);

	cmpx* Qrs_x = (cmpx*) malloc(2*Nd*sizeof(cmpx));

	cudaMemcpy(N3_temp1, vec_x, dsizeN3, cudaMemcpyHostToDevice);

	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
		FAME_Matrix_Vector_Production_Qrs(Nd2_temp, N3_temp1, cuHandles, fft_buffer, Lambdas_cuda.dD_ks, Lambdas_cuda.dPi_Qrs, Nx, Ny, Nz, Nd, Profile);
	}
	else
	{
		FAME_Matrix_Vector_Production_Qrs(Nd2_temp, N3_temp1, cuHandles, fft_buffer, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qrs, Nx, Ny, Nz, Nd, Profile);
	}

	cudaMemcpy(Qrs_x, Nd2_temp, dsizeNd2, cudaMemcpyDeviceToHost);

	for(i = 0; i < Nd; i++ )
	{
		Qrs_x[i]      = Qrs_x[i]      * Lambdas.Lambda_q_sqrt[i];
		Qrs_x[i + Nd] = Qrs_x[i + Nd] * Lambdas.Lambda_q_sqrt[i];
	}

	cudaMemcpy(Nd2_temp, Qrs_x, dsizeNd2, cudaMemcpyHostToDevice);

	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
		FAME_Matrix_Vector_Production_Pr(cuHandles, fft_buffer, Nd2_temp, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr, N3_temp1);
	}
	else
	{
		FAME_Matrix_Vector_Production_Pr(cuHandles, fft_buffer, Nd2_temp, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr, N3_temp1);
	}

	cudaMemcpy(vec_y, N3_temp1, dsizeN3, cudaMemcpyDeviceToHost);

	mtx_prod(vec_temp, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, vec_x, N12, N3);

	cmpx* test = (cmpx*) malloc(N3 * sizeof(cmpx));
	vec_plus(test, 1.0, vec_temp, -1.0, vec_y, N3);
	double SVD_test_C = vec_norm(test, N3);

	printf("          EigDecomp_test_C1 = %e\n", C1_error);
    printf("          EigDecomp_test_C2 = %e\n", C2_error);
    printf("          EigDecomp_test_C3 = %e\n", C3_error);
	printf("          SVD_test_C        = %e\n", SVD_test_C);

	if(C1_error > 1e-6 || C2_error > 1e-6 || C3_error > 1e-6 || SVD_test_C > 1e-6)
	{
		printf("\033[40;31mFAME_Main_Code(366):\033[0m\n");
        printf("\033[40;31mThe eigen decomposition is not correct.\033[0m\n");
        printf("\033[40;31mIf N = Nx * Ny * Nz > 256^3, may be caused by numerical errors, please loosen 1e-6.\n");
        printf("\033[40;31mIf not, please contact us.\033[0m\n");
        assert(0);
	}
	
	cudaFree(Nd2_temp);
	free(test); free(vec_temp); free(Qrs_x);
	free(vec_x); free(vec_y);
}

void Check_Residual(double* Freq_array, cmpx* Ele_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant)
{
	int N3 = N * 3;
	int N12 = N * 12;
	size_t size;

	size = N3 * Nwant * sizeof(cmpx);

	cmpx* vec_temp = (cmpx*)malloc(size);
	cmpx* vec_left = (cmpx*)malloc(size);
	cmpx* residual = (cmpx*)malloc(size);

	double res, omega2;
	double* B_eps = (double*)calloc(N3, sizeof(double));
	checkCudaErrors(cudaMemcpy(B_eps, mtx_B.B_eps, N3*sizeof(double), cudaMemcpyDeviceToHost));

	for(int i = 0; i < Nwant; i++)
	{
		omega2 = -pow(Freq_array[i], 2);
		mtx_prod(vec_temp, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, Ele_field_mtx + i*N3, N12, N3);
		mtx_prod(vec_left, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, vec_temp, N12, N3, "Conjugate Transpose");
		mtx_dot_prod(B_eps, Ele_field_mtx + i*N3, vec_temp, N3, 1);
		vec_plus(residual, 1.0, vec_left, omega2, vec_temp, N3);

		res = vec_norm(residual, N3);

		printf("                 ");
		if(res > 1e-10)
		{
			printf("\033[40;31mFreq(%2d) = %10.8f, residual = %e.\033[0m\n", i, Freq_array[i], res);
			Freq_array[i] = -Freq_array[i];
		}
		else
			printf("Freq(%2d) = %10.8f, residual = %e.\n", i, Freq_array[i], res);
	}

	free(vec_temp); free(vec_left); free(residual);
	free(B_eps);
}
