// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "defines.h"

// *****************************************************************************
void kGridFuncToQuad1D(const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
                       const int numElements,
                       const double* dofToQuad,
                       const int* l2gMap,
                       const double* gf,
                       double* __restrict out) {

  for (int e = 0; e < numElements; ++e) {
    double r_out[NUM_VDIM][NUM_QUAD_1D];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        r_out[v][qx] = 0;
      }
    }
    for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
      const int gid = l2gMap[(dx) + (NUM_DOFS_1D) * (e)];
      for (int v = 0; v < NUM_VDIM; ++v) {
        const double r_gf = gf[v + gid * NUM_VDIM];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          r_out[v][qx] += r_gf * dofToQuad[(qx) + (NUM_QUAD_1D) * (dx)];
        }
      }
    }
    for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
      for (int v = 0; v < NUM_VDIM; ++v) {
        out[(qx) + (NUM_QUAD_1D) * ((e) + (numElements) * (v))] = r_out[v][qx];
      }
    }
  }
}

// *****************************************************************************
void kGridFuncToQuad2D(const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
                       const int numElements,
                       const double* dofToQuad,
                       const int* l2gMap,
                       const double* gf,
                       double* __restrict out) {
  for (int e = 0; e < numElements; ++e) {
    double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          out_xy[v][qy][qx] = 0;
        }
      }
    }

    for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
      double out_x[NUM_VDIM][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          out_x[v][qy] = 0;
        }
      }

      for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
        const int gid = l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)];
        for (int v = 0; v < NUM_VDIM; ++v) {
          const double r_gf = gf[v + gid*NUM_VDIM];
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            out_x[v][qy] += r_gf * dofToQuad[ijN(qy, dx,NUM_QUAD_1D)];
          }
        }
      }

      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double d2q = dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xy[v][qy][qx] += d2q * out_x[v][qx];
          }
        }
      }
    }

    for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        for (int v = 0; v < NUM_VDIM; ++v) {
          out[_ijklNM(v, qx, qy, e,NUM_QUAD_1D,numElements)] = out_xy[v][qy][qx];
        }
      }
    }
  }
}

// *****************************************************************************
void kGridFuncToQuad3D(const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
                       const int numElements,
                       const double* dofToQuad,
                       const int* l2gMap,
                       const double* gf,
                       double* __restrict out) {
  for (int e = 0; e < numElements; ++e) {
    double out_xyz[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xyz[v][qz][qy][qx] = 0;
          }
        }
      }
    }

    for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
      double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xy[v][qy][qx] = 0;
          }
        }
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double out_x[NUM_VDIM][NUM_QUAD_1D];
        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_x[v][qx] = 0;
          }
        }

        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const int gid = l2gMap[ijklN(dx, dy, dz, e,NUM_DOFS_1D)];
          for (int v = 0; v < NUM_VDIM; ++v) {
            const double r_gf = gf[v + gid*NUM_VDIM];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_x[v][qx] += r_gf * dofToQuad[ijN(qx, dx, NUM_QUAD_1D)];
            }
          }
        }

        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = dofToQuad[ijN(qy, dy, NUM_QUAD_1D)];
          for (int v = 0; v < NUM_VDIM; ++v) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_xy[v][qy][qx] += wy * out_x[v][qx];
            }
          }
        }
      }

      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = dofToQuad[ijN(qz, dz, NUM_QUAD_1D)];
        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_xyz[v][qz][qy][qx] += wz * out_xy[v][qy][qx];
            }
          }
        }
      }
    }

    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          for (int v = 0; v < NUM_VDIM; ++v) {
            out[_ijklmNM(v, qx, qy, qz, e,NUM_QUAD_1D,numElements)] = out_xyz[v][qz][qy][qx];
          }
        }
      }
    }
  }
}