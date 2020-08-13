#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include "printDeviceArray.cuh"
//#include "FAME_Matrix_Vector_Production_FFT_Single_Simple.h"
/*
 update : Jia-wei
 date : 2018/08/14
*/

int Construct_Lambdas_Simple( LAMBDAS* Lambdas,
							  LAMBDAS_CUDA* Lambdas_cuda,
	                            double* wave_vec,
	                               int* grid_nums,
	                            double* mesh_lens,
	                            double  theta_x,
	                            double  theta_y,
	                            double  theta_z  );
int Construct_Lambdas_General(           LAMBDAS* Lambdas,
										LAMBDAS_CUDA* Lambdas_cuda,
	                                      double* wave_vec,
	                                         int* grid_nums,
	                                      double* mesh_lens,
	                                      double  theta_x,
	                                      double  theta_xy,
	                                      double  theta_xyz,
	                            LATTICE_CONSTANT* lattice_constant );

void ones(int* vec, int dim)
{
	for(int i = 0; i < dim; i++)
		vec[i] = 1;
}
void AP(int* vec, int dim)
{
	for(int i = 0; i < dim; i++)
		vec[i] = i;
}

int FAME_Matrix_Lambdas(LAMBDAS_CUDA* Lambdas_cuda, double* wave_vec, int* grid_nums, double* mesh_lens, double* lattice_vec_a, PAR* Par, LAMBDAS* Lambdas)
{
	int Nd;
	double theta_x = vec_inner_prod( wave_vec, &(lattice_vec_a[0]), 3 );
	double theta_y = vec_inner_prod( wave_vec, &(lattice_vec_a[3]), 3 );
	double theta_z = vec_inner_prod( wave_vec, &(lattice_vec_a[6]), 3 );

	if( (strcmp(Par->lattice.lattice_type, "simple_cubic"          ) == 0) || \
		(strcmp(Par->lattice.lattice_type, "primitive_orthorhombic") == 0) || \
		(strcmp(Par->lattice.lattice_type, "primitive_tetragonal"  ) == 0) )
		Nd = Construct_Lambdas_Simple( Lambdas, Lambdas_cuda, wave_vec, grid_nums, mesh_lens, theta_x, theta_y, theta_z);
	else
	{
		double c1;
		if(lattice_vec_a[3] > 0) // if lattice_vec_a(1,2) > 0
			c1 =     - (double)Par->lattice.lattice_constant.m1/(double)grid_nums[0];
		else if(lattice_vec_a[3] < 0)//elseif lattice_vec_a(1,2) < 0
            c1 =  1. - (double)Par->lattice.lattice_constant.m1/(double)grid_nums[0] ;
        double c2 = (double)Par->lattice.lattice_constant.m3/(double)grid_nums[1];
        double c3 = (double)Par->lattice.lattice_constant.m2/(double)grid_nums[0];
        //a2_hat = c1*lattice_vec_a(:,1) + lattice_vec_a(:,2);
        double a2_hat[3] = {c1*lattice_vec_a[0]+lattice_vec_a[3],\
                            c1*lattice_vec_a[1]+lattice_vec_a[4],\
                            c1*lattice_vec_a[2]+lattice_vec_a[5]};
        //a3_hat = lattice_vec_a(:,3) - (lattice_constant.m3/grid_nums(2))*a2_hat - (lattice_constant.m2/grid_nums(1))*lattice_vec_a(:,1) + lattice_constant.t1;
        double a3_hat[3] = {lattice_vec_a[6]-c2*a2_hat[0]-c3*lattice_vec_a[0]+Par->lattice.lattice_constant.t1[0],\
                            lattice_vec_a[7]-c2*a2_hat[1]-c3*lattice_vec_a[1]+Par->lattice.lattice_constant.t1[1],\
                            lattice_vec_a[8]-c2*a2_hat[2]-c3*lattice_vec_a[2]+Par->lattice.lattice_constant.t1[2]};

        double theta_xy  = vec_inner_prod( wave_vec, &(a2_hat[0]), 3 );
        double theta_xyz = vec_inner_prod( wave_vec, &(a3_hat[0]), 3 );

		Nd = Construct_Lambdas_General( Lambdas, Lambdas_cuda, wave_vec, grid_nums, mesh_lens, theta_x, theta_xy, theta_xyz, &(Par->lattice.lattice_constant));
	}

	return Nd;
}
int Construct_Lambdas_Simple( LAMBDAS* Lambdas, LAMBDAS_CUDA* Lambdas_cuda, double* wave_vec, int* grid_nums, double* mesh_lens, double theta_x, double theta_y, double theta_z)
{
	int N = grid_nums[0]*grid_nums[1]*grid_nums[2];
	int i;

	// Construct Lambda_x = kron(  e_z,kron(e_y,Lambda1));
	int* e_zy = (int*) malloc(grid_nums[1] * grid_nums[2] * sizeof(int));
	int* mx   = (int*) malloc(grid_nums[0] * sizeof(int));
	ones(e_zy, grid_nums[1] * grid_nums[2]);
	AP(mx, grid_nums[0]);

	cmpx* Lambda1     = (cmpx*) malloc(grid_nums[0]*sizeof(cmpx));
	for( i = 0; i<grid_nums[0]; i++)
	{
		Lambda1[i] = cexp(idouble_pi*(mx[i]+theta_x)/grid_nums[0]);
		Lambda1[i] = (Lambda1[i] - 1)/mesh_lens[0];
	}

	cmpx* Ld_x = (cmpx*) malloc(grid_nums[0] * grid_nums[1] * grid_nums[2] * sizeof(cmpx));
	kron_vec( Ld_x, e_zy, grid_nums[2]*grid_nums[1], Lambda1, grid_nums[0] );

	// Constuct Lambda_y = kron(  e_z,kron(Lambda2,e_x));
	int* e_x = (int*) malloc(grid_nums[0] * sizeof(int));
	int* e_z = (int*) malloc(grid_nums[2] * sizeof(int));
	int* my  = (int*) malloc(grid_nums[1] * sizeof(int));
	ones(e_x, grid_nums[0]);
	ones(e_z, grid_nums[2]);
	AP(my, grid_nums[1]);

	cmpx* Lambda2     = (cmpx*) malloc(grid_nums[1]*sizeof(cmpx));
	for( i = 0; i<grid_nums[1]; i++)
	{
		Lambda2[i] = cexp(idouble_pi*(my[i]+theta_y)/grid_nums[1]);
		Lambda2[i] = (Lambda2[i] - 1)/mesh_lens[1];
	}

	cmpx* tmp  = (cmpx*) malloc(grid_nums[0] * grid_nums[1] * sizeof(cmpx));
	cmpx* Ld_y = (cmpx*) malloc(grid_nums[0] * grid_nums[1] * grid_nums[2] * sizeof(cmpx));
	kron_vec( tmp, Lambda2, grid_nums[1],               e_x,              grid_nums[0] );
	kron_vec( Ld_y,    e_z, grid_nums[2], tmp, grid_nums[1]*grid_nums[0] );

	// Constuct Lambda_z = kron(  Lambda3,kron(e_y,e_x));
	int* e_yx = (int*) malloc(grid_nums[0] * grid_nums[1] * sizeof(int));
	int* mz   = (int*) malloc(grid_nums[2] * sizeof(int));
	ones(e_yx, grid_nums[0] * grid_nums[1]);
	AP(mz, grid_nums[2]);

	cmpx* Lambda3     = (cmpx*) malloc(grid_nums[2]*sizeof(cmpx));
	for( i = 0; i<grid_nums[2]; i++)
	{
		Lambda3[i] = cexp(idouble_pi*(mz[i]+theta_z)/grid_nums[2]);
		Lambda3[i] = (Lambda3[i] - 1)/mesh_lens[2];
	}

	cmpx* Ld_z = (cmpx*) malloc(N*sizeof(cmpx));
	kron_vec( Ld_z, Lambda3, grid_nums[2], e_yx, grid_nums[1]*grid_nums[0] );

	// Construct D_k and D_ks
	cmpx* D_kx = (cmpx*) malloc(grid_nums[0]*sizeof(cmpx));
	cmpx* D_ky = (cmpx*) malloc(grid_nums[1]*sizeof(cmpx));
	cmpx* D_kz = (cmpx*) malloc(grid_nums[2]*sizeof(cmpx));
	for( i = 0; i < grid_nums[0]; i++ )
		D_kx[i] = cexp( theta_x*idouble_pi*mx[i]/grid_nums[0] );
	for( i = 0; i < grid_nums[1]; i++ )
		D_ky[i] = cexp( theta_y*idouble_pi*my[i]/grid_nums[1] );
	for( i = 0; i < grid_nums[2]; i++ )
		D_kz[i] = cexp( theta_z*idouble_pi*mz[i]/grid_nums[2] );

	cmpx* tmp1 = (cmpx*) malloc(grid_nums[0] * grid_nums[1] * sizeof(cmpx));
	Lambdas->D_k  = (cmpx*) malloc(grid_nums[0] * grid_nums[1] * grid_nums[2] * sizeof(cmpx));
	kron_vec( tmp1, D_ky, grid_nums[1],         D_kx,              grid_nums[0] );
	kron_vec( Lambdas->D_k, D_kz, grid_nums[2], tmp1, grid_nums[1]*grid_nums[0] );

	Lambdas->D_ks = (cmpx*) malloc(N*sizeof(cmpx));
	for( i = 0; i < N; i++ )
		(Lambdas->D_ks)[i] = conj(Lambdas->D_k[i]);

	// Special case: wave_vec = [0,0,0]
 	int    Nd = N;
 	double wave_vec_norm = vec_norm(wave_vec, 3);
 	if(wave_vec_norm < 1e-4)
	{
		Nd = Nd-1;
		Lambdas->Lambda_x = (cmpx*) malloc(Nd*sizeof(cmpx));
		Lambdas->Lambda_y = (cmpx*) malloc(Nd*sizeof(cmpx));
		Lambdas->Lambda_z = (cmpx*) malloc(Nd*sizeof(cmpx));
		for( i = 0; i < Nd; i++ )
		{
			Lambdas->Lambda_x[i] = Ld_x[i+1];
			Lambdas->Lambda_y[i] = Ld_y[i+1];
			Lambdas->Lambda_z[i] = Ld_z[i+1];
		}
	}
	else
	{
		Lambdas->Lambda_x = (cmpx*) malloc(N*sizeof(cmpx));
		Lambdas->Lambda_y = (cmpx*) malloc(N*sizeof(cmpx));
		Lambdas->Lambda_z = (cmpx*) malloc(N*sizeof(cmpx));
		for( i = 0; i < N; i++)
		{
			Lambdas->Lambda_x[i] = Ld_x[i];
			Lambdas->Lambda_y[i] = Ld_y[i];
			Lambdas->Lambda_z[i] = Ld_z[i];
		}
	}

	// Construct Lambda_q_sqrt, temp(normalize_term_Pi_2) and tempPi(normalize_term_Pi_1) in matlab form
	double alpha = 3.,beta = 5.;
	Lambdas->Lambda_q_sqrt = (double*) malloc(  Nd*sizeof(double));
	Lambdas->Pi_Qr         = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	Lambdas->Pi_Pr         = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	Lambdas->Pi_Qrs        = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	Lambdas->Pi_Prs        = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	for( i = 0; i < Nd; i++ )
	{
		// Determine Lambda_q entrywisely
		double Lambda_q_i = creal( conj(Lambdas->Lambda_x[i])*Lambdas->Lambda_x[i] + \
	                  	           conj(Lambdas->Lambda_y[i])*Lambdas->Lambda_y[i] + \
	                  	           conj(Lambdas->Lambda_z[i])*Lambdas->Lambda_z[i] );

		cmpx temp_1 =  beta*Lambdas->Lambda_z[i] -       Lambdas->Lambda_y[i];
		cmpx temp_2 =       Lambdas->Lambda_x[i] - alpha*Lambdas->Lambda_z[i];
		cmpx temp_3 = alpha*Lambdas->Lambda_y[i] -  beta*Lambdas->Lambda_x[i];

		cmpx normalize_term_Pi_2_i = conj(temp_1)*temp_1 + conj(temp_2)*temp_2+ conj(temp_3)*temp_3;
	    cmpx normalize_term_Pi_1_i = normalize_term_Pi_2_i*Lambda_q_i;

	    // Construct Lambda_q_sqrt entrywisely (Notice that Sigma_r = [Lambda_q_sqrt;Lambda_q_sqrt])
	    Lambdas->Lambda_q_sqrt[i] = sqrt( Lambda_q_i );
	    // Determine Pi_1 and Pi_2 entrywisely
	    cmpx temp_Pi_1_i     = ( (alpha*Lambdas->Lambda_y[i] -  beta*Lambdas->Lambda_x[i])*conj(Lambdas->Lambda_y[i])- \
                                 (      Lambdas->Lambda_x[i] - alpha*Lambdas->Lambda_z[i])*conj(Lambdas->Lambda_z[i]) )/csqrt(normalize_term_Pi_1_i);
		cmpx temp_Pi_1_ipNd  = ( ( beta*Lambdas->Lambda_z[i] -       Lambdas->Lambda_y[i])*conj(Lambdas->Lambda_z[i])- \
                                 (alpha*Lambdas->Lambda_y[i] -  beta*Lambdas->Lambda_x[i])*conj(Lambdas->Lambda_x[i]) )/csqrt(normalize_term_Pi_1_i);
		cmpx temp_Pi_1_ip2Nd = ( (      Lambdas->Lambda_x[i] - alpha*Lambdas->Lambda_z[i])*conj(Lambdas->Lambda_x[i])- \
                                 ( beta*Lambdas->Lambda_z[i] -       Lambdas->Lambda_y[i])*conj(Lambdas->Lambda_y[i]) )/csqrt(normalize_term_Pi_1_i);
		cmpx temp_Pi_2_i     = (  beta*conj(Lambdas->Lambda_z[i]) -       conj(Lambdas->Lambda_y[i]) )/csqrt(normalize_term_Pi_2_i);
        cmpx temp_Pi_2_ipNd  = (       conj(Lambdas->Lambda_x[i]) - alpha*conj(Lambdas->Lambda_z[i]) )/csqrt(normalize_term_Pi_2_i);
        cmpx temp_Pi_2_ip2Nd = ( alpha*conj(Lambdas->Lambda_y[i]) -  beta*conj(Lambdas->Lambda_x[i]) )/csqrt(normalize_term_Pi_2_i);
        // Construct Pi_Qr and Pi_Pr entrywisely
        Lambdas->Pi_Qr[i     ] = temp_Pi_1_i;
		Lambdas->Pi_Qr[i+Nd  ] = temp_Pi_1_ipNd;
		Lambdas->Pi_Qr[i+2*Nd] = temp_Pi_1_ip2Nd;
		Lambdas->Pi_Qr[i+3*Nd] = temp_Pi_2_i;
		Lambdas->Pi_Qr[i+4*Nd] = temp_Pi_2_ipNd;
		Lambdas->Pi_Qr[i+5*Nd] = temp_Pi_2_ip2Nd;

		Lambdas->Pi_Pr[i     ] = -1*conj(temp_Pi_2_i);
		Lambdas->Pi_Pr[i+Nd  ] = -1*conj(temp_Pi_2_ipNd);
		Lambdas->Pi_Pr[i+2*Nd] = -1*conj(temp_Pi_2_ip2Nd);
		Lambdas->Pi_Pr[i+3*Nd] =    conj(temp_Pi_1_i);
		Lambdas->Pi_Pr[i+4*Nd] =    conj(temp_Pi_1_ipNd);
		Lambdas->Pi_Pr[i+5*Nd] =    conj(temp_Pi_1_ip2Nd);

		Lambdas->Pi_Qrs[i     ] = conj(Lambdas->Pi_Qr[i     ]);
		Lambdas->Pi_Qrs[i+Nd  ] = conj(Lambdas->Pi_Qr[i+3*Nd]);
		Lambdas->Pi_Qrs[i+2*Nd] = conj(Lambdas->Pi_Qr[i+1*Nd]);
		Lambdas->Pi_Qrs[i+3*Nd] = conj(Lambdas->Pi_Qr[i+4*Nd]);
		Lambdas->Pi_Qrs[i+4*Nd] = conj(Lambdas->Pi_Qr[i+2*Nd]);
		Lambdas->Pi_Qrs[i+5*Nd] = conj(Lambdas->Pi_Qr[i+5*Nd]);

		Lambdas->Pi_Prs[i     ] = conj(Lambdas->Pi_Pr[i     ]);
		Lambdas->Pi_Prs[i+Nd  ] = conj(Lambdas->Pi_Pr[i+3*Nd]);
		Lambdas->Pi_Prs[i+2*Nd] = conj(Lambdas->Pi_Pr[i+1*Nd]);
		Lambdas->Pi_Prs[i+3*Nd] = conj(Lambdas->Pi_Pr[i+4*Nd]);
		Lambdas->Pi_Prs[i+4*Nd] = conj(Lambdas->Pi_Pr[i+2*Nd]);
		Lambdas->Pi_Prs[i+5*Nd] = conj(Lambdas->Pi_Pr[i+5*Nd]);
	}

	///////////////////////////////////////////////////////////////////////////
	/////// PASS Simple Lambda from local to device
	///////////////////////////////////////////////////////////////////////////
	int memsize = sizeof(cuDoubleComplex)*N;
	cudaMalloc((void**)&(Lambdas_cuda->dD_k), 	memsize);
	cudaMalloc((void**)&(Lambdas_cuda->dD_ks), 	memsize);
	cudaMemcpy(Lambdas_cuda->dD_k,  Lambdas->D_k,  memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(Lambdas_cuda->dD_ks, Lambdas->D_ks, memsize, cudaMemcpyHostToDevice);

	memsize = sizeof(cuDoubleComplex)*Nd*6;

	cudaMalloc((void**)&(Lambdas_cuda->dPi_Qr), 	memsize);
	cudaMalloc((void**)&(Lambdas_cuda->dPi_Pr), 	memsize);
	cudaMalloc((void**)&(Lambdas_cuda->dPi_Qrs), 	memsize);
	cudaMalloc((void**)&(Lambdas_cuda->dPi_Prs), 	memsize);

	cudaMemcpy(Lambdas_cuda->dPi_Qr,  Lambdas->Pi_Qr, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(Lambdas_cuda->dPi_Qrs, Lambdas->Pi_Qrs, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(Lambdas_cuda->dPi_Pr,  Lambdas->Pi_Pr, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(Lambdas_cuda->dPi_Prs, Lambdas->Pi_Prs, memsize, cudaMemcpyHostToDevice);

	memsize = sizeof(double)*Nd;
	double* temp_Lambda_q_sqrt = (double*) malloc(2 * memsize);
	memcpy(temp_Lambda_q_sqrt, Lambdas->Lambda_q_sqrt, memsize);
	memcpy(temp_Lambda_q_sqrt + Nd, Lambdas->Lambda_q_sqrt, memsize);

	memsize = 2*sizeof(double)*Nd;
	cudaMalloc((void**)&(Lambdas_cuda->Lambda_q_sqrt), memsize);
	cudaMemcpy(Lambdas_cuda->Lambda_q_sqrt, temp_Lambda_q_sqrt, memsize, cudaMemcpyHostToDevice);

	free(e_x); free(e_z);
	free(e_yx); free(e_zy);
	free(mx); free(my); free(mz);
	free(Lambda1); free(Lambda2); free(Lambda3);
	free(Ld_x); free(Ld_y); free(Ld_z);
	free(D_kx); free(D_ky); free(D_kz);
	free(tmp); free(tmp1); free(temp_Lambda_q_sqrt);

	///////////////////////////////////////////////////////////////////////////
	return Nd;
}

int Construct_Lambdas_General(           LAMBDAS* Lambdas,
										 LAMBDAS_CUDA* Lambdas_cuda,
	                                      double* wave_vec,
	                                         int* grid_nums,
	                                      double* mesh_lens,
	                                      double  theta_x,
	                                      double  theta_xy,
	                                      double  theta_xyz,
	                            LATTICE_CONSTANT* lattice_constant )
{
	int N = grid_nums[0]*grid_nums[1]*grid_nums[2];

	int i, j, k; // only use in for-loop

	// Construct Lambda_x = kron(  Lambda1 ,kron(e_y, e_z));
	int* e_zy = (int*) malloc(grid_nums[1] * grid_nums[2] * sizeof(int));
	int* mx   = (int*) malloc(grid_nums[0] * sizeof(int));
	ones(e_zy, grid_nums[1] * grid_nums[2]);
	AP(mx, grid_nums[0]);


	cmpx* Lambda1 = (cmpx*) malloc(grid_nums[0] * sizeof(cmpx));
	for( i = 0; i<grid_nums[0]; i++)
	{
		Lambda1[i] = cexp(idouble_pi*(mx[i]+theta_x)/grid_nums[0]);
		Lambda1[i] = (Lambda1[i] - 1)/mesh_lens[0];
	}

	cmpx* Ld_x = (cmpx*) malloc(grid_nums[0] * grid_nums[1] * grid_nums[2] * sizeof(cmpx));
	kron_vec(Ld_x, Lambda1, grid_nums[0], e_zy, grid_nums[1] * grid_nums[2]);


	// Constuct Lambda_y = kron( Lambda2, e_z );
	int* e_x = (int*) malloc(grid_nums[0] * sizeof(int));
	int* e_y = (int*) malloc(grid_nums[1] * sizeof(int));
	int* e_z = (int*) malloc(grid_nums[2] * sizeof(int));
	int* my  = (int*) malloc(grid_nums[1] * sizeof(int));
	ones(e_x, grid_nums[0]);
	ones(e_y, grid_nums[1]);
	ones(e_z, grid_nums[2]);
	AP(my, grid_nums[1]);

	double* tmp  = (double*) malloc(grid_nums[0] * grid_nums[1] * sizeof(double));
	double* tmp1 = (double*) malloc(grid_nums[0] * grid_nums[1] * sizeof(double));
	double* mxy  = (double*) malloc(grid_nums[0] * grid_nums[1] * sizeof(double));
	kron_vec(tmp, -1.*(double)lattice_constant->m1/(double)grid_nums[0], mx, grid_nums[0], 1., e_y, grid_nums[1]);
	kron_vec(tmp1, 1., e_x, grid_nums[0], 1., my, grid_nums[1]);
	vec_plus(mxy, 1., tmp, 1.,  tmp1, grid_nums[0]*  grid_nums[1]);

	cmpx* Lambda2     = (cmpx*) malloc(grid_nums[1]*grid_nums[0]*sizeof(cmpx));
	for( i = 0; i<grid_nums[1]*grid_nums[0]; i++)
	{
		Lambda2[i] = cexp(idouble_pi*(mxy[i]+theta_xy)/grid_nums[1]);
		Lambda2[i] = (Lambda2[i] - 1)/mesh_lens[1];
	}

	cmpx* Ld_y = (cmpx*) malloc(grid_nums[0] * grid_nums[1] * grid_nums[2] * sizeof(cmpx));
	kron_vec(Ld_y, Lambda2, grid_nums[1]*grid_nums[0], e_z, grid_nums[2]);

	// Constuct Lambda_z;
	int* e_yx = (int*) malloc(grid_nums[0] * grid_nums[1] * sizeof(int));
	int* mz   = (int*) malloc(grid_nums[2] * sizeof(int));
	ones(e_yx, grid_nums[0] * grid_nums[1]);
	AP(mz, grid_nums[2]);

	double  c1   =  ((double)lattice_constant->m1*(double)lattice_constant->m3)/((double)grid_nums[1]*grid_nums[0]) - ((double)lattice_constant->m2/(double)grid_nums[0]);
	double  c2   = -((double)lattice_constant->m3/(double)grid_nums[1]);

	int* tmp3    = (int*)    malloc(N * sizeof(int));
	int* tmp4    = (int*)    malloc(grid_nums[1] * grid_nums[2] * sizeof(int));
	int* tmp5    = (int*)    malloc(N * sizeof(int));
	double* tmp6 = (double*) malloc(N * sizeof(double));
	int* tmp7    = (int*)    malloc(N * sizeof(int));
	double* mxyz = (double*) malloc(N * sizeof(double));
	kron_vec(tmp3, mx, grid_nums[0], e_zy, grid_nums[2]*grid_nums[1]);
	kron_vec(tmp4, my, grid_nums[1], e_z, grid_nums[2]);
	kron_vec(tmp5, e_x, grid_nums[0], tmp4, grid_nums[2]*grid_nums[1]);
	vec_plus(tmp6, c1, tmp3, c2, tmp5, N);
	kron_vec(tmp7, e_yx, grid_nums[1]*grid_nums[0], mz, grid_nums[2]);
	vec_plus(mxyz, 1., tmp6, 1., tmp7, N );

	cmpx* Ld_z = (cmpx*) malloc(N*sizeof(cmpx));

	for( i = 0; i < N; i++)
	{
		Ld_z[i] = cexp(idouble_pi*(mxyz[i]+theta_xyz)/grid_nums[2]);
		Ld_z[i] = (Ld_z[i] - 1)/mesh_lens[2];
	}

	// Construct D_kx
	Lambdas->D_kx = (cmpx*) malloc(grid_nums[0]*sizeof(cmpx));
	for( i = 0; i < grid_nums[0]; i++ )
		Lambdas->D_kx[i] = cexp( theta_x*idouble_pi*mx[i]/grid_nums[0] );
	// Construct D_ky
	Lambdas->D_ky = (cmpx*) malloc(grid_nums[0]*grid_nums[1]*sizeof(cmpx));
	for( i = 0; i < grid_nums[0]; i++)
	{
		cmpx phi_jx = idouble_pi * (theta_xy + i*((double)-lattice_constant->m1/(double)grid_nums[0]) ) / (double)grid_nums[1];
		for( j = 0; j < grid_nums[1]; j++ )
			Lambdas->D_ky[j + i*grid_nums[1]] = cexp( phi_jx*my[j] );
	}
	// Construct D_kz
	Lambdas->D_kz = (cmpx*) malloc(grid_nums[0]*grid_nums[1]*grid_nums[2]*sizeof(cmpx));
	for( i = 0; i < grid_nums[0]; i++)
		for( j = 0; j < grid_nums[1]; j++)
		{
			cmpx phi_k = idouble_pi * (theta_xyz + j*(  (double)-lattice_constant->m3/(double)grid_nums[1]) + \
				                                   i*(  (double) (lattice_constant->m1*lattice_constant->m3)/(double)(grid_nums[0]*grid_nums[1]) \
				                                       -(double)lattice_constant->m2/(double)grid_nums[0] )  \
				                      )/(double)grid_nums[2];
			for( k = 0; k < grid_nums[2]; k++ )
				Lambdas->D_kz[k+j*grid_nums[2] + i*grid_nums[1]*grid_nums[2]] = cexp( phi_k*mz[k] );
	}

	// Special case: wave_vec = [0,0,0]
 	int    Nd = N;
 	double wave_vec_norm = vec_norm(wave_vec, 3);

 	if(wave_vec_norm < 1e-3)
	{
		Nd = Nd-1;
		Lambdas->Lambda_x = (cmpx*) malloc(Nd*sizeof(cmpx));
		Lambdas->Lambda_y = (cmpx*) malloc(Nd*sizeof(cmpx));
		Lambdas->Lambda_z = (cmpx*) malloc(Nd*sizeof(cmpx));
		for( i = 0; i < Nd; i++ )
		{
			Lambdas->Lambda_x[i] = Ld_x[i+1];
			Lambdas->Lambda_y[i] = Ld_y[i+1];
			Lambdas->Lambda_z[i] = Ld_z[i+1];
		}
	}
	else
	{
		Lambdas->Lambda_x = (cmpx*) malloc(N*sizeof(cmpx));
		Lambdas->Lambda_y = (cmpx*) malloc(N*sizeof(cmpx));
		Lambdas->Lambda_z = (cmpx*) malloc(N*sizeof(cmpx));
		for( i = 0; i < N; i++)
		{
			Lambdas->Lambda_x[i] = Ld_x[i];
			Lambdas->Lambda_y[i] = Ld_y[i];
			Lambdas->Lambda_z[i] = Ld_z[i];
		}
	}

	// Construct Lambda_q_sqrt, temp(normalize_term_Pi_2) and tempPi(normalize_term_Pi_1) in matlab form
	Lambdas->Lambda_q_sqrt = (double*) malloc(  Nd*sizeof(double));
	Lambdas->Pi_Qr         = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	Lambdas->Pi_Pr         = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	Lambdas->Pi_Qrs        = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	Lambdas->Pi_Prs        = (cmpx*)   malloc(6*Nd*sizeof(cmpx)  );
	for( i = 0; i < Nd; i++ )
	{

		// Determine Lambda_q entrywisely
		double Lambda_q_i = creal( conj(Lambdas->Lambda_x[i])*Lambdas->Lambda_x[i] + \
	                    	       conj(Lambdas->Lambda_y[i])*Lambdas->Lambda_y[i] + \
	                  	           conj(Lambdas->Lambda_z[i])*Lambdas->Lambda_z[i] );
	    cmpx Lambda_s_i = Lambdas->Lambda_x[i] + \
	                  	  Lambdas->Lambda_y[i] + \
	                  	  Lambdas->Lambda_z[i];
	    cmpx Lambda_p_i = conj(Lambda_s_i)*Lambda_s_i;

		cmpx normalize_term_Pi_1_i = csqrt( 3*Lambda_q_i*Lambda_q_i - Lambda_q_i*Lambda_p_i );
		cmpx normalize_term_Pi_2_i = csqrt( 3*           Lambda_q_i -            Lambda_p_i );

	    // Construct Lambda_q_sqrt entrywisely (Notice that Sigma_r = [Lambda_q_sqrt;Lambda_q_sqrt])
	    Lambdas->Lambda_q_sqrt[i] = sqrt( Lambda_q_i );
	    // Determine Pi_1 and Pi_2 entrywisely
	    cmpx temp_Pi_1_i     = (Lambda_q_i - Lambdas->Lambda_x[i]*conj(Lambda_s_i)) / normalize_term_Pi_1_i;
	    cmpx temp_Pi_1_ipNd  = (Lambda_q_i - Lambdas->Lambda_y[i]*conj(Lambda_s_i)) / normalize_term_Pi_1_i;
	    cmpx temp_Pi_1_ip2Nd = (Lambda_q_i - Lambdas->Lambda_z[i]*conj(Lambda_s_i)) / normalize_term_Pi_1_i;
	    cmpx temp_Pi_2_i     = (conj(Lambdas->Lambda_z[i]) - conj(Lambdas->Lambda_y[i])) / normalize_term_Pi_2_i;
	    cmpx temp_Pi_2_ipNd  = (conj(Lambdas->Lambda_x[i]) - conj(Lambdas->Lambda_z[i])) / normalize_term_Pi_2_i;
	    cmpx temp_Pi_2_ip2Nd = (conj(Lambdas->Lambda_y[i]) - conj(Lambdas->Lambda_x[i])) / normalize_term_Pi_2_i;
	    // Construct Pi_Qr and Pi_Pr entrywisely
	    Lambdas->Pi_Qr[i     ] = temp_Pi_1_i;
		Lambdas->Pi_Qr[i+Nd  ] = temp_Pi_1_ipNd;
		Lambdas->Pi_Qr[i+2*Nd] = temp_Pi_1_ip2Nd;
		Lambdas->Pi_Qr[i+3*Nd] = temp_Pi_2_i;
		Lambdas->Pi_Qr[i+4*Nd] = temp_Pi_2_ipNd;
		Lambdas->Pi_Qr[i+5*Nd] = temp_Pi_2_ip2Nd;
		Lambdas->Pi_Pr[i     ] = -1*conj(temp_Pi_2_i);
		Lambdas->Pi_Pr[i+Nd  ] = -1*conj(temp_Pi_2_ipNd);
		Lambdas->Pi_Pr[i+2*Nd] = -1*conj(temp_Pi_2_ip2Nd);
		Lambdas->Pi_Pr[i+3*Nd] =    conj(temp_Pi_1_i);
		Lambdas->Pi_Pr[i+4*Nd] =    conj(temp_Pi_1_ipNd);
		Lambdas->Pi_Pr[i+5*Nd] =    conj(temp_Pi_1_ip2Nd);

		Lambdas->Pi_Qrs[i     ] = conj(Lambdas->Pi_Qr[i     ]);
		Lambdas->Pi_Qrs[i+Nd  ] = conj(Lambdas->Pi_Qr[i+3*Nd]);
		Lambdas->Pi_Qrs[i+2*Nd] = conj(Lambdas->Pi_Qr[i  +Nd]);
		Lambdas->Pi_Qrs[i+3*Nd] = conj(Lambdas->Pi_Qr[i+4*Nd]);
		Lambdas->Pi_Qrs[i+4*Nd] = conj(Lambdas->Pi_Qr[i+2*Nd]);
		Lambdas->Pi_Qrs[i+5*Nd] = conj(Lambdas->Pi_Qr[i+5*Nd]);

		Lambdas->Pi_Prs[i     ] = conj(Lambdas->Pi_Pr[i     ]);
		Lambdas->Pi_Prs[i+Nd  ] = conj(Lambdas->Pi_Pr[i+3*Nd]);
		Lambdas->Pi_Prs[i+2*Nd] = conj(Lambdas->Pi_Pr[i  +Nd]);
		Lambdas->Pi_Prs[i+3*Nd] = conj(Lambdas->Pi_Pr[i+4*Nd]);
		Lambdas->Pi_Prs[i+4*Nd] = conj(Lambdas->Pi_Pr[i+2*Nd]);
		Lambdas->Pi_Prs[i+5*Nd] = conj(Lambdas->Pi_Pr[i+5*Nd]);
	}

	cudaError_t cudaErr;
	///////////////////////////////////////////////////////////////////////////
    /////// PASS General Lambda from local to device
    ///////////////////////////////////////////////////////////////////////////
    cudaErr = cudaMalloc((void**)&(Lambdas_cuda->dD_kx),  grid_nums[0]*grid_nums[1]*grid_nums[2]*sizeof(cuDoubleComplex));
    assert( cudaErr == cudaSuccess );
    cudaErr = cudaMalloc((void**)&(Lambdas_cuda->dD_ky),  grid_nums[0]*grid_nums[1]*grid_nums[2]*sizeof(cuDoubleComplex));
	assert( cudaErr == cudaSuccess );
	cudaErr = cudaMalloc((void**)&(Lambdas_cuda->dD_kz),  grid_nums[0]*grid_nums[1]*grid_nums[2]*sizeof(cuDoubleComplex));
	assert( cudaErr == cudaSuccess );

	for(int ii = 0; ii<grid_nums[1]*grid_nums[2]; ii++)
    {
    	cudaErr = cudaMemcpy(Lambdas_cuda->dD_kx+ii*grid_nums[0], Lambdas->D_kx, grid_nums[0]*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    	assert( cudaErr == cudaSuccess );
    }


	for(int jj=0; jj<grid_nums[0]; jj++)
		for(int ii = 0; ii<grid_nums[2]; ii++)
		{
		    cudaErr = cudaMemcpy(Lambdas_cuda->dD_ky+ii*grid_nums[1]+jj*grid_nums[1]*grid_nums[2], Lambdas->D_ky+jj*grid_nums[1], grid_nums[1]*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		    assert( cudaErr == cudaSuccess );
		}

	cudaErr = cudaMemcpy(Lambdas_cuda->dD_kz, Lambdas->D_kz, grid_nums[0]*grid_nums[1]*grid_nums[2]*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	assert( cudaErr == cudaSuccess );

    size_t memsize = sizeof(cuDoubleComplex)*Nd*6;

    cudaErr = cudaMalloc((void**)&(Lambdas_cuda->dPi_Qr),     memsize);assert( cudaErr == cudaSuccess );
    cudaErr = cudaMalloc((void**)&(Lambdas_cuda->dPi_Pr),     memsize);assert( cudaErr == cudaSuccess );
    cudaErr = cudaMalloc((void**)&(Lambdas_cuda->dPi_Qrs),    memsize);assert( cudaErr == cudaSuccess );
    cudaErr = cudaMalloc((void**)&(Lambdas_cuda->dPi_Prs),    memsize);assert( cudaErr == cudaSuccess );

    cudaErr = cudaMemcpy(Lambdas_cuda->dPi_Qr, Lambdas->Pi_Qr, memsize, cudaMemcpyHostToDevice);assert( cudaErr == cudaSuccess );
    cudaErr = cudaMemcpy(Lambdas_cuda->dPi_Qrs, Lambdas->Pi_Qrs, memsize, cudaMemcpyHostToDevice);assert( cudaErr == cudaSuccess );
    cudaErr = cudaMemcpy(Lambdas_cuda->dPi_Pr, Lambdas->Pi_Pr, memsize, cudaMemcpyHostToDevice);assert( cudaErr == cudaSuccess );
    cudaErr = cudaMemcpy(Lambdas_cuda->dPi_Prs, Lambdas->Pi_Prs, memsize, cudaMemcpyHostToDevice);assert( cudaErr == cudaSuccess );

    memsize = sizeof(double)*Nd;
	double* temp_Lambda_q_sqrt = (double*) malloc(2 * memsize);
	memcpy(temp_Lambda_q_sqrt, Lambdas->Lambda_q_sqrt, memsize);
	memcpy(temp_Lambda_q_sqrt + Nd, Lambdas->Lambda_q_sqrt, memsize);

	memsize = 2*sizeof(double)*Nd;
	cudaMalloc((void**)&(Lambdas_cuda->Lambda_q_sqrt), memsize);
	cudaMemcpy(Lambdas_cuda->Lambda_q_sqrt, temp_Lambda_q_sqrt, memsize, cudaMemcpyHostToDevice);
	/*
	memsize = sizeof(double)*Nd;
    cudaErr = cudaMalloc((void**)&(Lambdas_cuda->Lambda_q_sqrt), memsize);assert( cudaErr == cudaSuccess );
    cudaErr = cudaMemcpy(Lambdas_cuda->Lambda_q_sqrt, Lambdas->Lambda_q_sqrt, memsize, cudaMemcpyHostToDevice);assert( cudaErr == cudaSuccess );
*/
	///////////////////////////////////////////////////////////////////////////

	free(e_x); free(e_y); free(e_z);
	free(mx); free(my); free(mz); free(mxy);
	free(e_yx); free(e_zy); free(mxyz);
	free(Lambda1); free(Lambda2);
	free(Ld_x); free(Ld_y); free(Ld_z);
	free(tmp); free(tmp1); free(tmp3); free(tmp4); free(tmp5); free(tmp6); free(tmp7);
	free(temp_Lambda_q_sqrt);
	return Nd;
}

