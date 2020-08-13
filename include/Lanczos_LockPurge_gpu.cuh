#ifndef _LANCZOS_LOCKPURGE_GPU_H_
#define	_LANCZOS_LOCKPURGE_GPU_H_

int Lanczos_LockPurge_gpu(
	LANCZOS_BUFFER*  lBuffer, 
	cublasHandle_t  cublas_handle, 
	int Asize, int Nwant, int Nstep);

#endif