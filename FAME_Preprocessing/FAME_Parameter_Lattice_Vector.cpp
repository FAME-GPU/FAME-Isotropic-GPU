#include "FAME_Internal_Common.h"
// 2020-02-19

void Householder(double* w, double* c, double* b, int size);
void QRp(double* vec_a, int* permutation);

int FAME_Parameter_Lattice_Vector(LATTICE* lattice)
{
    int i;

    /* content of function Lattice_vector_generate in MATLAB */
    //  Cubic system
    if(strcmp(lattice->lattice_type, "simple_cubic") == 0)
    {
        double a     = lattice->lattice_constant.a;
        double a1[3] = {a, 0, 0};
        double a2[3] = {0, a, 0};
        double a3[3] = {0, 0, a};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "face_centered_cubic") == 0)
    {
        double a     = 0.5*lattice->lattice_constant.a;
        double a1[3] = {0, a, a};
        double a2[3] = {a, 0, a};
        double a3[3] = {a, a, 0};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "body_centered_cubic") == 0)
    {
        double a     = 0.5*lattice->lattice_constant.a;
        double a1[3] = {-1*a,    a,    a};
        double a2[3] = {   a, -1*a,    a};
        double a3[3] = {   a,    a, -1*a};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    //  Tetragonal system
    else if(strcmp(lattice->lattice_type, "primitive_tetragonal") == 0)
    {
        double a     = lattice->lattice_constant.a;
        double c     = lattice->lattice_constant.c;
        double a1[3] = {a, 0, 0};
        double a2[3] = {0, a, 0};
        double a3[3] = {0, 0, c};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "body_centered_tetragonal") == 0)
    {
        double a     = 0.5*lattice->lattice_constant.a;
        double c     = 0.5*lattice->lattice_constant.c;
        double a1[3] = {-a,  a,  c};
        double a2[3] = { a, -a,  c};
        double a3[3] = { a,  a, -c};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    // Orthorhombic system
    else if(strcmp(lattice->lattice_type, "primitive_orthorhombic") == 0)
    {
        double a = lattice->lattice_constant.a;
        double b = lattice->lattice_constant.b;
        double c = lattice->lattice_constant.c;
        double a1[3] = {a, 0, 0};
        double a2[3] = {0, b, 0};
        double a3[3] = {0, 0, c};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "face_centered_orthorhombic") == 0)//no check
    {
        double a = 0.5*lattice->lattice_constant.a;
        double b = 0.5*lattice->lattice_constant.b;
        double c = 0.5*lattice->lattice_constant.c;
        double a1[3] = {0, b, c};
        double a2[3] = {a, 0, c};
        double a3[3] = {a, b, 0};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "body_centered_orthorhombic") == 0)
    {
        double a = 0.5*lattice->lattice_constant.a;
        double b = 0.5*lattice->lattice_constant.b;
        double c = 0.5*lattice->lattice_constant.c;
        double a1[3] = {-a,  b,  c};
        double a2[3] = { a, -b,  c};
        double a3[3] = { a,  b, -c};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "a_base_centered_orthorhombic") == 0)
    {
        double a = lattice->lattice_constant.a;
        double b = lattice->lattice_constant.b;
        double c = lattice->lattice_constant.c;
        double a1[3] = {a,     0,      0};
        double a2[3] = {0, 0.5*b, -0.5*c};
        double a3[3] = {0, 0.5*b,  0.5*c};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "c_base_centered_orthorhombic") == 0)
    {
        double a = lattice->lattice_constant.a;
        double b = lattice->lattice_constant.b;
        double c = lattice->lattice_constant.c;
        double a1[3] = {0.5*a, -0.5*b, 0};
        double a2[3] = {0.5*a,  0.5*b, 0};
        double a3[3] = {    0,      0, c};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    //  Hexagonal system
    else if(strcmp(lattice->lattice_type, "hexagonal") == 0)
    {
        double a = lattice->lattice_constant.a;
        double c = lattice->lattice_constant.c;
        double a1[3] = {     a,             0, 0};
        double a2[3] = {-0.5*a, sqrt(3)*0.5*a, 0};
        double a3[3] = {     0,             0, c};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    //  Rhombohedral system
    else if(strcmp(lattice->lattice_type, "rhombohedral") == 0)
    {
        double a = lattice->lattice_constant.a;
        double alpha = lattice->lattice_constant.alpha;
        double a1[3] = {            a*cos(alpha/2), -a*sin(alpha/2),                                                     0};
        double a2[3] = {            a*cos(alpha/2),  a*sin(alpha/2),                                                     0};
        double a3[3] = { a*cos(alpha)/cos(alpha/2),               0, a*sqrt(1.0-(pow(cos(alpha) / cos(alpha / 2), 2)))};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    //  Monoclinic system
    else if(strcmp(lattice->lattice_type, "primitive_monoclinic") == 0)
    {
        double a = lattice->lattice_constant.a;
        double b = lattice->lattice_constant.b;
        double c = lattice->lattice_constant.c;
        double alpha = lattice->lattice_constant.alpha;
        double a1[3] = {a, 0, 0};
        double a2[3] = {0, b, 0};
        double a3[3] = {0, c*cos(alpha), c*sin(alpha)};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else if(strcmp(lattice->lattice_type, "base_centered_monoclinic") == 0)
    {
        double a = 0.5*lattice->lattice_constant.a;
        double b = 0.5*lattice->lattice_constant.b;
        double c = lattice->lattice_constant.c;
        double alpha = lattice->lattice_constant.alpha;
        double a1[3] = {a, b, 0};
        double a2[3] = {-a, b, 0};
        double a3[3] = {0, c*cos(alpha), c*sin(alpha)};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    //  Triclinic system
    else if(strcmp(lattice->lattice_type, "triclinic") == 0)
    {
        double a = lattice->lattice_constant.a;
        double b = lattice->lattice_constant.b;
        double c = lattice->lattice_constant.c;
        double alpha = lattice->lattice_constant.alpha;
        double beta  = lattice->lattice_constant.beta;
        double gamma = lattice->lattice_constant.gamma;
        double a1[3] = {a, 0, 0};
        double a2[3] = {b*cos(gamma), b*sin(gamma), 0};
        double a3[3] = { c*cos(beta), c*(cos(alpha)-cos(beta)*cos(gamma))/sin(gamma), c*( sqrt(1-pow(cos(alpha), 2)-pow(cos(beta), 2)-pow(cos(gamma), 2)+2*cos(alpha)*cos(beta)*cos(gamma)) / sin(gamma))};
        for(i = 0; i < 3; i++)
        {
            lattice->lattice_vec_a[i    ] = a1[i];
            lattice->lattice_vec_a[i + 3] = a2[i];
            lattice->lattice_vec_a[i + 6] = a3[i];
            lattice->Permutation[i] = i + 1;
        }
    }
    else
    {
        /*
        printf("\033[40;31mFAME_Parameter_Lattice_Vector(153):\033[0m\n");
        printf("\033[40;31mLattice type %s is invalid! Please check lattice type is correct in material data.\033[0m\n", lattice->lattice_type);
        assert(0);
        */
    }
    /* the content of function FAME_Parameter_Lattice_Vector in MATLAB */

    // initialize
    lattice->lattice_constant.length_a1 = 0.0;
    lattice->lattice_constant.length_a2 = 0.0;
    lattice->lattice_constant.length_a3 = 0.0;

    for(i = 0; i < 9; i++)
        lattice->lattice_vec_a_orig[i] = lattice->lattice_vec_a[i];

    if(strcmp(lattice->lattice_type, "simple_cubic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_orthorhombic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_tetragonal") != 0 )
        QRp(lattice->lattice_vec_a, lattice->Permutation);

    // take norm of a1, a2, a3
    for(i = 0; i < 3; i++)
    {
        lattice->lattice_constant.length_a1 += pow(lattice->lattice_vec_a[i], 2);
        lattice->lattice_constant.length_a2 += pow(lattice->lattice_vec_a[i+3], 2);
        lattice->lattice_constant.length_a3 += pow(lattice->lattice_vec_a[i+6], 2);
    }

    (lattice->lattice_constant.length_a1) = sqrt(lattice->lattice_constant.length_a1);
    (lattice->lattice_constant.length_a2) = sqrt(lattice->lattice_constant.length_a2);
    (lattice->lattice_constant.length_a3) = sqrt(lattice->lattice_constant.length_a3);

    (lattice->lattice_constant.theta_1) = 0.0;
    (lattice->lattice_constant.theta_2) = 0.0;
    (lattice->lattice_constant.theta_3) = 0.0;

    for(i = 0; i < 3; i++)
    {
        lattice->lattice_constant.theta_1 += lattice->lattice_vec_a[i+3]*lattice->lattice_vec_a[i+6]/(lattice->lattice_constant.length_a2*lattice->lattice_constant.length_a3);
        lattice->lattice_constant.theta_2 += lattice->lattice_vec_a[i]*lattice->lattice_vec_a[i+6]/(lattice->lattice_constant.length_a1*lattice->lattice_constant.length_a3);
        lattice->lattice_constant.theta_3 += lattice->lattice_vec_a[i]*lattice->lattice_vec_a[i+3]/(lattice->lattice_constant.length_a1*lattice->lattice_constant.length_a2);
    }

    lattice->lattice_constant.theta_1 = acos(lattice->lattice_constant.theta_1);
    lattice->lattice_constant.theta_2 = acos(lattice->lattice_constant.theta_2);
    lattice->lattice_constant.theta_3 = acos(lattice->lattice_constant.theta_3);

    if(fabs(lattice->lattice_constant.theta_1 - pi/2) < 1e-14)
        lattice->lattice_constant.theta_1 = pi / 2;
    if(fabs(lattice->lattice_constant.theta_2 - pi/2) < 1e-14)
        lattice->lattice_constant.theta_2 = pi / 2;
    if(fabs(lattice->lattice_constant.theta_3 - pi/2) < 1e-14)
        lattice->lattice_constant.theta_3 = pi / 2;

    // Test condition 
    if(strcmp(lattice->lattice_type, "base_centered_monoclinic") == 0)
    {
        if(lattice->lattice_constant.theta_1 >= pi/2)
            lattice->lattice_constant.theta_1 = pi - lattice->lattice_constant.theta_1;

        if(lattice->lattice_constant.theta_2 >= pi/2)
            lattice->lattice_constant.theta_2 = pi - lattice->lattice_constant.theta_2;

        if(lattice->lattice_constant.theta_3 >= pi/2)
            lattice->lattice_constant.theta_3 = pi - lattice->lattice_constant.theta_3;
    }

    /* Final check */
    if(strcmp(lattice->lattice_type, "simple_cubic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_orthorhombic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_tetragonal") != 0 )
        if(lattice->lattice_constant.length_a2 > lattice->lattice_constant.length_a1|| \
                lattice->lattice_constant.length_a3 > lattice->lattice_constant.length_a1|| \
                lattice->lattice_constant.length_a2*sin(lattice->lattice_constant.theta_3) <= fabs((cos(lattice->lattice_constant.theta_1)-cos(lattice->lattice_constant.theta_2)*cos(lattice->lattice_constant.theta_3)) / sin(lattice->lattice_constant.theta_3)) )
        {
            printf("\033[40;31mFAME_Parameter_Lattice_Vector(332):\033[0m\n");
            printf("\033[40;31mThe lattice constants does not suitable for computation. Please contact us.\033[0m\n");
            //assert(0);
        }

    double l2 = (cos(lattice->lattice_constant.theta_1) - cos(lattice->lattice_constant.theta_3)*cos(lattice->lattice_constant.theta_2))/sin(lattice->lattice_constant.theta_3);
    double l3 = sqrt(pow(sin(lattice->lattice_constant.theta_2), 2) - pow(l2, 2));
    lattice->lattice_vec_a[0] = lattice->lattice_constant.length_a1;
    lattice->lattice_vec_a[1] = 0.0;
    lattice->lattice_vec_a[2] = 0.0;
    lattice->lattice_vec_a[3] = lattice->lattice_constant.length_a2*cos(lattice->lattice_constant.theta_3);
    lattice->lattice_vec_a[4] = lattice->lattice_constant.length_a2*sin(lattice->lattice_constant.theta_3);
    lattice->lattice_vec_a[5] = 0.0;
    lattice->lattice_vec_a[6] = lattice->lattice_constant.length_a3*cos(lattice->lattice_constant.theta_2);
    lattice->lattice_vec_a[7] = lattice->lattice_constant.length_a3 * l2;
    lattice->lattice_vec_a[8] = lattice->lattice_constant.length_a3 * l3;

    double lattice_vec_a_orig_P[9];
    double inv_lattice_vec_a[9];
    lattice_vec_a_orig_P[0] = lattice->lattice_vec_a_orig[(lattice->Permutation[0] - 1) * 3 + 0];
    lattice_vec_a_orig_P[1] = lattice->lattice_vec_a_orig[(lattice->Permutation[0] - 1) * 3 + 1];
    lattice_vec_a_orig_P[2] = lattice->lattice_vec_a_orig[(lattice->Permutation[0] - 1) * 3 + 2];
    lattice_vec_a_orig_P[3] = lattice->lattice_vec_a_orig[(lattice->Permutation[1] - 1) * 3 + 0];
    lattice_vec_a_orig_P[4] = lattice->lattice_vec_a_orig[(lattice->Permutation[1] - 1) * 3 + 1];
    lattice_vec_a_orig_P[5] = lattice->lattice_vec_a_orig[(lattice->Permutation[1] - 1) * 3 + 2];
    lattice_vec_a_orig_P[6] = lattice->lattice_vec_a_orig[(lattice->Permutation[2] - 1) * 3 + 0];
    lattice_vec_a_orig_P[7] = lattice->lattice_vec_a_orig[(lattice->Permutation[2] - 1) * 3 + 1];
    lattice_vec_a_orig_P[8] = lattice->lattice_vec_a_orig[(lattice->Permutation[2] - 1) * 3 + 2];
    inv3(lattice->lattice_vec_a, inv_lattice_vec_a);
    mtx_prod(lattice->Omega, lattice_vec_a_orig_P, inv_lattice_vec_a, 3, 3, 3);

    return 0;
}

void Householder(double* w, double* c, double* b, int size)
{
    double normb = vec_norm(b, size);
    double coe;
    if(fabs(b[0]) < 1e-14)
        c[0] = normb;
    else
        c[0] = -b[0] / fabs(b[0]) * normb;
    coe = sqrt(2 * normb * (normb + fabs(b[0])));
    memcpy(w, b, size * sizeof(double));

    w[0] = w[0] - c[0];
    for(int ii = 0; ii < size; ii++)
        w[ii] = w[ii] / coe;
}

void QRp(double* vec_a, int* permutation)
{
    int ii, max_idx, itemp;
    double temp, c;
    double length_a[3], w[3];
    length_a[0] = vec_a[0] * vec_a[0] + vec_a[1] * vec_a[1] + vec_a[2] * vec_a[2];
    length_a[1] = vec_a[3] * vec_a[3] + vec_a[4] * vec_a[4] + vec_a[5] * vec_a[5];
    length_a[2] = vec_a[6] * vec_a[6] + vec_a[7] * vec_a[7] + vec_a[8] * vec_a[8];

    max_idx = 0;
    if(length_a[1] > length_a[0] && length_a[1] > length_a[2])
        max_idx = 1;
    else if(length_a[2] > length_a[0] && length_a[2] > length_a[1])
        max_idx = 2;

    if(max_idx != 0)
    {
        for(ii = 0; ii < 3; ii++)
        {
            temp = vec_a[max_idx * 3 + ii];
            vec_a[max_idx * 3 + ii] = vec_a[ii];
            vec_a[ii] = temp;
        }
        temp = length_a[0];
        length_a[0] = length_a[max_idx];
        length_a[max_idx] = temp;

        permutation[0] = max_idx + 1;
        permutation[max_idx] = 1;
    }

    Householder(w, &c, &vec_a[0], 3);
    vec_a[0] = c; vec_a[1] = 0.0; vec_a[2] = 0.0;
    

    temp = vec_inner_prod(w, &vec_a[3], 3);
    for(ii = 0; ii < 3; ii++)
        vec_a[ii + 3] = vec_a[ii + 3] - 2 * temp * w[ii];

    temp = vec_inner_prod(w, &vec_a[6], 3);
    for(ii = 0; ii < 3; ii++)
        vec_a[ii + 6] = vec_a[ii + 6] - 2 * temp * w[ii];

    if(length_a[1] - vec_a[3] * vec_a[3] < length_a[2] - vec_a[6] * vec_a[6])
    {
        itemp = permutation[1];
        permutation[1] = permutation[2];
        permutation[2] = itemp;

        for(ii = 0; ii < 3; ii++)
        {
            temp = vec_a[3 + ii];
            vec_a[3 + ii] = vec_a[6 + ii];
            vec_a[6 + ii] = temp;
        }
    }

    Householder(w, &c, &vec_a[4], 2);
    vec_a[4] = c; vec_a[5] = 0.0;

    temp = vec_inner_prod(w, &vec_a[7], 2);
    for(ii = 0; ii < 2; ii++)
        vec_a[ii + 7] = vec_a[ii + 7] - 2 * temp * w[ii];
}