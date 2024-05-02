#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

   const int nIndexBits = 3;
   const int nSvSize    = (1 << nIndexBits);
   const int nTargets   = 1;
   const int nControls  = 2;
   const int adjoint    = 0;

   int targets[]  = {2};
   int controls[] = {0, 1};

   cuDoubleComplex h_sv[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1},
                                    { 0.1, 0.2}, { 0.2, 0.2}, { 0.3, 0.3},
                                    { 0.3, 0.4}, { 0.4, 0.5}};
   cuDoubleComplex h_sv_result[] = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1},
                                    { 0.4, 0.5}, { 0.2, 0.2}, { 0.3, 0.3},
                                    { 0.3, 0.4}, { 0.1, 0.2}};
   cuDoubleComplex matrix[] = {{0.0, 0.0}, {1.0, 0.0},
                               {1.0, 0.0}, {0.0, 0.0}};


   cuDoubleComplex *d_sv;
   cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex));

   cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);

   //--------------------------------------------------------------------------

   // custatevec handle initialization
   custatevecHandle_t handle;

   custatevecCreate(&handle);

   void* extraWorkspace = nullptr;
   size_t extraWorkspaceSizeInBytes = 0;

   // check the size of external workspace
   custatevecApplyMatrixGetWorkspaceSize(
       handle, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F,
       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nTargets, nControls,
       CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes);

   // allocate external workspace if necessary
   if (extraWorkspaceSizeInBytes > 0)
       cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes);

   // apply gate
   custatevecApplyMatrix(
       handle, d_sv, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F,
       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, controls,
       nullptr, nControls, CUSTATEVEC_COMPUTE_64F,
       extraWorkspace, extraWorkspaceSizeInBytes);

   // destroy handle
   custatevecDestroy(handle);

   //--------------------------------------------------------------------------

   cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
              cudaMemcpyDeviceToHost);

   bool correct = true;
   for (int i = 0; i < nSvSize; i++) {
       if ((h_sv[i].x != h_sv_result[i].x) ||
           (h_sv[i].y != h_sv_result[i].y)) {
           correct = false;
           break;
       }
   }

   if (correct)
       printf("example PASSED\n");
   else
       printf("example FAILED: wrong result\n");

   cudaFree(d_sv);
   if (extraWorkspaceSizeInBytes)
       cudaFree(extraWorkspace);

   return EXIT_SUCCESS;
}
