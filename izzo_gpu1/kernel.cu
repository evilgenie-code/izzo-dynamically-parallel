#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdio>
#include <cstdlib> 
#include <ctime>



void vers(const double* vIn, double* verOut)
{
    double vMod = 0;

    for (int i = 0; i < 3; i++)
    {
        vMod += vIn[i] * vIn[i];
    }

    double sqrtVMod = sqrt(vMod);

    for (int i = 0; i < 3; i++)
    {
        verOut[i] = vIn[i] / sqrtVMod;
    }
}

void vett(const double* vet1, const double* vet2, double* prod)
{
    prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
    prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
    prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
}

__device__ double x2tof(double x, double s, double c, int lw, int m)
{
    //printf("inn = %f, s = %f, c = %f, lw = %i, revs = %i \n", x, s, c, lw, m);
	
    double am, a, alfa, beta;

    am = s / 2;
    a = am / (1 - x * x);

    if (x < 1)//ellpise
    {
        beta = 2 * asin(sqrt((s - c) / (2 * a)));
        if (lw) beta = -beta;
        alfa = 2 * acos(x);
    }
    else
    {
        alfa = 2 * acosh(x);
        beta = 2 * asinh(sqrt((s - c) / (-2 * a)));
        if (lw) beta = -beta;
    }

    if (a > 0)
    {
        return (a * sqrt(a) * ((alfa - sin(alfa)) - (beta - sin(beta)) + 2 * acos(-1.0) * m));
    }
    else
    {
        return (-a * sqrt(-a) * ((sinh(alfa) - alfa) - (sinh(beta) - beta)));
    }
}

__global__ void xf(float *inn, float  s, float c, int lw, int revs, float t, float* xF)
{
	
    int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

    int ThreadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    int tid = blockIndex * blockDim.x * blockDim.y * blockDim.z + ThreadIndex;
	
    float xf = log(x2tof(inn[tid], s, c, lw, revs)) - logf(t);
	
    xF[tid] = xf;
}

void getXF(float *inn, double s, double c, int lw, int revs, float t, float *xF, int iterate)
{
    float *dev_inn, *dev_xf;

    cudaMalloc((void**)&dev_xf, 2 * sizeof(float));
    cudaMalloc((void**)&dev_inn, 2 * sizeof(float));
    cudaMemcpy(dev_xf, xF, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inn, inn, 2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	
    xf << <1, 2 >> > (dev_inn, s, c, lw, revs, t, dev_xf);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("3.%i iterate %i time spent executing by the GPU: %.16e milliseconds\n", iterate, iterate, gpuTime);

    cudaMemcpy(xF, dev_xf, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_xf);
}

void lambert(const float* r0, const float* rk, float t, int lw, int revs, float mu)
{
    double v1[3], v2[3], r1[3], r2[3], r2Vers[3];
    double	V, T, r2Mod = 0.0,    // R2 module
        dotProd = 0.0, // dot product
        c,		        // non-dimensional chord
        s,		        // non dimesnional semi-perimeter
        am,		        // minimum energy ellipse semi major axis
        lambda,	        //lambda parameter defined in Battin's Book
        x, x1, x2, x1F, x2F, xNew = 0, yNew, err, alfa, beta, psi, eta, eta2, sigma1, vr1, vt1, vt2, vr2, r1Mod = 0.0;
    int iterate, i, leftbranch = 0;
    const double tolerance = 1e-7;
    double ihDum[3], ih[3], dum[3];

    double a, p, theta;

    if (t <= 0)
    {
        return;
    }

    for (i = 0; i < 3; i++)
    {
        r1[i] = r0[i];
        r2[i] = rk[i];
        r1Mod += r1[i] * r1[i];
    }

    r1Mod = sqrt(r1Mod);
    V = sqrt(mu / r1Mod);
    T = r1Mod / V;

    t /= T;

    for (i = 0; i < 3; i++)
    {
        r1[i] /= r1Mod;
        r2[i] /= r1Mod;
        r2Mod += r2[i] * r2[i];
    }

    r2Mod = sqrt(r2Mod);

    for (i = 0; i < 3; i++)
        dotProd += (r1[i] * r2[i]);

    theta = acos(dotProd / r2Mod);

    if (lw)
    {
        theta = 2 * acos(-1.0) - theta;
    }

    c = sqrt(1 + r2Mod * (r2Mod - 2.0 * cos(theta)));
    s = (1 + r2Mod + c) / 2.0;
    am = s / 2.0;
    lambda = sqrt(r2Mod) * cos(theta / 2.0) / s;

    float inn1, inn2;

    xNew = 0;
    iterate = 0;
    float xF[2];

    if (revs == 0)
    {
        x1 = log(0.4767);
        x2 = log(1.5233);
        inn1 = -.5233;
        inn2 = .5233;

        
        // Newton iterations
        while (fabs(x1 - xNew) > tolerance)
        //while (iterate <= 4)
        {
            iterate++;

            float inn[2] = { inn1, inn2 };
        	
            getXF(inn, s, c, lw, revs, t, xF, iterate);

            xNew = (x1 * xF[1] - x2 * xF[0]) / (xF[1] - xF[0]);

            x1 = x2;
            x2 = xNew;

            inn1 = exp(x1) - 1;
            inn2 = exp(xNew) - 1;
        }


        x = exp(xNew) - 1;
    }
    else
    {
        if (leftbranch == 1)   // left branch
        {
            inn1 = -0.5234;
            inn2 = -0.2234;
        }
        else			   // right branch
        {
            inn1 = 0.7234;
            inn2 = 0.5234;
        }

        x1 = tan(inn1 * acos(-1.0) / 2);
        x2 = tan(inn2 * acos(-1.0) / 2);
        //x1F = x2tof(inn1, s, c, lw, revs) - t;
       // x2F = x2tof(inn2, s, c, lw, revs) - t;

        int imax = 30;
        // Newton Iteration
        while ((err > tolerance) && (x1F != x2F) && iterate < imax)
        {
             iterate++;

            float inn[2] = { inn1, inn2 };
        	
            getXF(inn, s, c, lw, revs, t, xF , iterate);

            xNew = (x1 * xF[1] - x2 * xF[0]) / (xF[1] - xF[0]);

            printf("iterate = %i, fabs = %.e \n", iterate, fabs(x1 - xNew));

            x1 = x2;
            x2 = xNew;

            inn1 = exp(x1) - 1;
            inn2 = exp(xNew) - 1;
        }

        x = atan(xNew) * 2 / acos(-1.0);

        iterate = iterate == imax ? iterate - 1 : iterate;
    }
 

    a = am / (1 - x * x);		    // solution semimajor axis
    // psi evaluation
    if (x < 1)                         // ellipse
    {
        beta = 2 * asin(sqrt((s - c) / (2 * a)));
        if (lw) beta = -beta;
        alfa = 2 * acos(x);
        psi = (alfa - beta) / 2;
        eta2 = 2 * a * pow(sin(psi), 2) / s;
        eta = sqrt(eta2);
    }
    else       // hyperbola
    {
        beta = 2 * asinh(sqrt((c - s) / (2 * a)));
        if (lw) beta = -beta;
        alfa = 2 * acosh(x);
        psi = (alfa - beta) / 2;
        eta2 = -2 * a * pow(sinh(psi), 2) / s;
        eta = sqrt(eta2);
    }

    p = (r2Mod / (am * eta2)) * pow(sin(theta / 2), 2);
    sigma1 = (1 / (eta * sqrt(am))) * (2 * lambda * am - (lambda + x * eta));
    vett(r1, r2, ihDum);
    vers(ihDum, ih);

    if (lw)
    {
        for (i = 0; i < 3; i++)
            ih[i] = -ih[i];
    }

    vr1 = sigma1;
    vt1 = sqrt(p);
    vett(ih, r1, dum);

    for (i = 0; i < 3; i++)
        v1[i] = vr1 * r1[i] + vt1 * dum[i];

    vt2 = vt1 / r2Mod;
    vr2 = -vr1 + (vt1 - vt2) / tan(theta / 2);

    vers(r2, r2Vers);
    vett(ih, r2Vers, dum);
    for (i = 0; i < 3; i++)
        v2[i] = vr2 * r2[i] / r2Mod + vt2 * dum[i];

    for (i = 0; i < 3; i++)
    {
        v1[i] *= V;
        v2[i] *= V;
    }
}

int main() {

    printf("1. start program \n");
    double AU = 1.49597870691e8;
    double fMSun = 1.32712440018e11;             // km^3/sec^2

    double UnitR = AU;
    double UnitV = sqrt(fMSun / UnitR);          // km/sec
    double UnitT = (UnitR / UnitV) / 86400;         // day

    float unitT = 100.0 / UnitT;
    float mu = 1.0;
    int lw = 0, revs = 0.0;
    float r1[3] = { -7.8941608095246896e-01, -6.2501194900473045e-01, 3.5441335698377735e-05 };
    float r2[3] = { 1.3897892184188783e+00, 1.3377137029002054e-01, -3.1287386211010106e-02 };

	printf("2. calculate program \n");
	
    lambert(r1, r2, unitT, lw, revs, mu);

    printf("4. finish program \n");
}