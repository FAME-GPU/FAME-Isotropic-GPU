#include "FAME_Internal_Common.h"
#include <complex.h>

int FAME_Create_C_txt(
    int* C1_r, int* C1_c, cmpx* C1_v,
    int* C2_r, int* C2_c, cmpx* C2_v,
    int* C3_r, int* C3_c, cmpx* C3_v,
    int* grid_nums)
{
    int i;
    int N = grid_nums[0] * grid_nums[1] * grid_nums[2];

    FILE* fp;
    fp = fopen("Data_Mtx_C.txt", "w");

    fprintf(fp, "%% [row col val] \n");
    fprintf(fp, "C1 = [\n");
    for (i = 0; i < 2*N; i++)
        fprintf(fp, "%5d %5d %.14e %.14e\n", C1_r[i]+1, C1_c[i]+1, creal(C1_v[i]), cimag(C1_v[i]));
    fprintf(fp, "];\n\n");

    fprintf(fp, "C2 = [\n");
    for (i = 0; i < 2*N; i++)
        fprintf(fp, "%5d %5d %.14e %.14e\n", C2_r[i]+1, C2_c[i]+1, creal(C2_v[i]), cimag(C2_v[i]));
    fprintf(fp, "];\n\n");

    fprintf(fp, "C3 = [\n");
    for (i = 0; i < 2*N; i++)
        fprintf(fp, "%5d %5d %.14e %.14e\n", C3_r[i]+1, C3_c[i]+1, creal(C3_v[i]), cimag(C3_v[i]));
    fprintf(fp, "];\n\n");

    fclose(fp);

    return 0;
}
