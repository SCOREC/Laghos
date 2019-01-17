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
//
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Hydrodynamics Miniapp
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
// Sample runs:
//    mpirun -np 8 laghos -p 0 -m data/square01_quad.mesh -rs 3 -tf 0.75
//    mpirun -np 8 laghos -p 0 -m data/square01_tri.mesh  -rs 1 -tf 0.75
//    mpirun -np 8 laghos -p 0 -m data/cube01_hex.mesh    -rs 1 -tf 2.0
//    mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8
//    mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 0 -tf 0.8 -ok 7 -ot 6
//    mpirun -np 8 laghos -p 1 -m data/cube01_hex.mesh    -rs 2 -tf 0.6
//    mpirun -np 8 laghos -p 2 -m data/segment01.mesh     -rs 5 -tf 0.2
//    mpirun -np 8 laghos -p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 3.0
//    mpirun -np 8 laghos -p 3 -m data/box01_hex.mesh        -rs 1 -tf 3.0
//
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.
//    p = 2  --> 1D Sod shock tube.
//    p = 3  --> Triple point.
//Possible pumi run:
//    mpirun -np 4 ./laghos -ot 0 -ok 1 -fa -p 1 


#include "laghos_solver.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include "../mfem/general/text.hpp"

#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>
#include <spr.h>

using namespace std;
using namespace mfem;
using namespace mfem::hydrodynamics;

// Choice for the problem setup.
int problem;

void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi(argc, argv);
   int myid = mpi.WorldRank();

   // Print the banner.
   if (mpi.Root()) { display_banner(cout); }

   // Parse command-line options.
   const char *mesh_file = "data/pumi/4/sedov5p.smb";
   const char *boundary_file = "data/pumi/boundary.mesh";
#ifdef MFEM_USE_SIMMETRIX
   const char *model_file = "data/pumi/geom/sedov.x_t";
#else
   const char *model_file = "data/pumi/geom/sedov5.dmg";
#endif   
   int geom_order = 1;
   double adapt_ratio = 0.05;
   
   int rs_levels = 0;
   int rp_levels = 0;
   int order_v = 2;
   int order_e = 1;
   int ode_solver_type = 4;
   double t_final = 0.5;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool p_assembly = true;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   int partition_type = 111;
   double ma_time = 0.2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
                  "--full-assembly",
                  "Activate 1D tensor-based assembly (partial assembly).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&partition_type, "-pt", "--partition",
                  "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                  "Here x,y,z are relative task ratios in each direction.\n\t"
                  "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                  "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                  "NOTE: the serially refined mesh must have the appropriate number\n\t"
                  "of zones in each direction, e.g., the number of zones in direction x\n\t"
                  "must be divisible by the number of MPI tasks in direction x.\n\t"
                  "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
   args.AddOption(&adapt_ratio, "-ar", "--adapt_ratio",
                  "adaptation factor used in MeshAdapt");
   args.AddOption(&ma_time, "-mat", "--ma_time",
                  "MeshAdapt time frequency");   
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }
   
   //3. Read the SCOREC Mesh
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
   gmi_register_mesh();

   apf::Mesh2* pumi_mesh;
   pumi_mesh = apf::loadMdsMesh(model_file, mesh_file); 
   
   int dim = pumi_mesh->getDimension();
   int nEle = pumi_mesh->count(dim);
   //rs_levels = (int)floor(log(50000./nEle)/log(2.)/dim);

   if (geom_order > 1)
   {
      crv::BezierCurver bc(pumi_mesh, geom_order, 2);
      bc.run();
   }

   // Perform Uniform refinement
   if (rs_levels > 0)
   {
      ma::Input* uniInput = ma::configureUniformRefine(pumi_mesh, rs_levels);

      if ( geom_order > 1)
      {
         crv::adapt(uniInput);
      }
      else
      {
         ma::adapt(uniInput);
      }
   }

   pumi_mesh->verify();  
   
   //Read boundary
   string bdr_tags;
   named_ifgzstream input_bdr(boundary_file);
   input_bdr >> ws;
   getline(input_bdr, bdr_tags);
   filter_dos(bdr_tags);
   Array<int> XX, YY, ZZ;
   int numOfent;
   if (bdr_tags == "X")
   {
      input_bdr >> numOfent;
      XX.SetSize(numOfent);
      for (int kk = 0; kk < numOfent; kk++) {input_bdr >> XX[kk];}
   } 
   
   skip_comment_lines(input_bdr, '#');
   input_bdr >> bdr_tags;
   filter_dos(bdr_tags);
   if (bdr_tags == "Y")
   {
      input_bdr >> numOfent;
      YY.SetSize(numOfent);
      for (int kk = 0; kk < numOfent; kk++) {input_bdr >> YY[kk];}
   }
   
   skip_comment_lines(input_bdr, '#');
   input_bdr >> bdr_tags;
   filter_dos(bdr_tags);
   if (bdr_tags == "Z")
   {
       input_bdr >> numOfent;
       ZZ.SetSize(numOfent);
       for (int kk = 0; kk < numOfent; kk++) {input_bdr >> ZZ[kk];}
   }
   
   // Create the MFEM mesh object from the PUMI mesh.
   ParMesh *pmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
   
   //Hack for the boundary condition
   apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
   apf::MeshEntity* ent ;
   int bdr_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)))
   {
      apf::ModelEntity *me = pumi_mesh->toModel(ent);
      if (pumi_mesh->getModelType(me) == (dim-1))
      {
         //Evrywhere 3 as initial
         //(pmesh->GetBdrElement(bdr_cnt))->SetAttribute(3);
         int tag = pumi_mesh->getModelTag(me);
         if (XX.Find(tag) != -1)
         {
            //XX attr -> 1
            (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(1);
         }
         else if (YY.Find(tag) != -1)
         {
            //YY attr -> 2
            (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(2);
         }
         else if (ZZ.Find(tag) != -1)
         {
            //ZZ attr -> 3
            (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(3);    
         }
         else 
         {
             cout << " !!! ERROR !!! boundary has no attribute : " << endl;
         }
         bdr_cnt++;
      }
   }
   pumi_mesh->end(itr);   
   pmesh->SetAttributes();

   // Read the serial mesh from the given mesh file on all processors.
   // Refine the mesh in serial to increase the resolution.
   //Mesh *mesh = new Mesh(mesh_file, 1, 1);
   //const int dim = mesh->Dimension();
   //for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }

   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      if (mpi.Root())
      {
         cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
      }
   }

   int nzones = pmesh->GetNE(), nzones_min, nzones_max;
   MPI_Reduce(&nzones, &nzones_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&nzones, &nzones_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << nzones_min << " " << nzones_max << endl; }


   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), tdofs1d;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
         // enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
         ess_tdofs.Append(tdofs1d);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_h1 = H1FESpace.GlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
   }

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_h1 = H1FESpace.GetVSize();

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy

   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_h1;
   true_offset[2] = true_offset[1] + Vsize_h1;
   true_offset[3] = true_offset[2] + Vsize_l2;
   BlockVector S(true_offset);

   // Define GridFunction objects for the position, velocity and specific
   // internal energy.  There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
   v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
   e_gf.MakeRef(&L2FESpace, S, true_offset[2]);
   ParGridFunction d_gf(&H1FESpace);
   d_gf = 0.0;

   // Initialize x_gf using the starting mesh coordinates. This also links the
   // mesh positions to the values in x_gf.
   pmesh->SetNodalGridFunction(&x_gf);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs.  Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   ParGridFunction rho(&L2FESpace);
   FunctionCoefficient rho_coeff(hydrodynamics::rho0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
   l2_rho.ProjectCoefficient(rho_coeff);
   rho.ProjectGridFunction(l2_rho);
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(0, 0, 0.25);
      l2_e.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on a function that stays constant on the moving
   // mesh.
   //L2_FECollection mat_fec(0, pmesh->Dimension());
   //ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   //ParGridFunction mat_gf(&mat_fes);
   //FunctionCoefficient mat_coeff(hydrodynamics::gamma);
   //mat_gf.ProjectCoefficient(mat_coeff);
   //GridFunctionCoefficient *mat_gf_coeff = new GridFunctionCoefficient(&mat_gf);
   Coefficient *material_pcf = new FunctionCoefficient(hydrodynamics::gamma);

   // Additional details, depending on the problem.
   int source = 0; bool visc;
   switch (problem)
   {
      case 0: if (pmesh->Dimension() == 2) { source = 1; }
         visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; break;
      default: MFEM_ABORT("Wrong problem specification!");
   }

   LagrangianHydroOperator oper(S.Size(), H1FESpace, L2FESpace,
                                ess_tdofs, rho, source, cfl, material_pcf,
                                visc, p_assembly, cg_tol, cg_max_iter);

   socketstream vis_rho, vis_v, vis_e, vis_u;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   if (visualization || visit) { oper.ComputeDensity(rho_gf); }

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      vis_u.precision(8);

      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets

      VisualizeField(vis_rho, vishost, visport, rho_gf,
                     "Density", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(vis_v, vishost, visport, v_gf,
                     "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(vis_e, vishost, visport, e_gf,
                     "Specific Internal Energy", Wx, Wy, Ww, Wh);
      
      Wx += offx;
      VisualizeField(vis_u, vishost, visport, d_gf,
                     "Displacement", Wx, Wy, Ww, Wh);      
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, pmesh);
   if (visit)
   {
      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(oper);
   oper.ResetTimeStepEstimate();
   double t = 0.0, dt = oper.GetTimeStepEstimate(S), t_old, ma_t = 0.0, vis_time = 0.0;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   ParGridFunction dgf_old(d_gf);
   
   //MeshAdapt fields 
   apf::Field* Vmag_field = 0;
   apf::Field* Vel_field  = 0;
   apf::Field* Dis_field  = 0;
   apf::Field* U_field    = 0;   
   apf::Field* Enr_field  = 0;
   apf::Field* ipfield    = 0;
   apf::Field* sizefield  = 0; 
   
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }

      //Update displacement d = d + X_(n+1) - X_n
      //First step : d = d - X_n
      dgf_old = d_gf;
      d_gf -= x_gf;
      
      S_old = S;
      t_old = t;
      oper.ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Init(oper);
      ode_solver->Step(S, t, dt);
      steps++;
      
      // Adaptive time step control.
      const double dt_est = oper.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         d_gf = dgf_old;
         oper.ResetQuadratureData();
         if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Make sure that the mesh corresponds to the new solution state.
      pmesh->NewNodes(x_gf, false);

      //Update displacement d = d + X_(n+1) - X_n
      //Second step : d = d + X_(n+1)
      d_gf += x_gf; 
      
      vis_time += dt;
      if (last_step || vis_time > ma_time/10.0 || ma_t == 0.0)//(ti % vis_steps) == 0
      {
         vis_time = 0.0;
         double loc_norm = e_gf * e_gf, tot_norm;
         MPI_Allreduce(&loc_norm, &tot_norm, 1, MPI_DOUBLE, MPI_SUM,
                       pmesh->GetComm());
         if (mpi.Root())
         {
            cout << fixed;
            cout << "step " << setw(5) << ti
                 << ",\tt = " << setw(5) << setprecision(4) << t
                 << ",\tdt = " << setw(5) << setprecision(6) << dt
                 << ",\t|e| = " << setprecision(10)
                 << sqrt(tot_norm) << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit || gfprint) { oper.ComputeDensity(rho_gf); }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets

            VisualizeField(vis_rho, vishost, visport, rho_gf,
                           "Density", Wx, Wy, Ww, Wh);
            Wx += offx;
            VisualizeField(vis_v, vishost, visport,
                           v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            VisualizeField(vis_e, vishost, visport, e_gf,
                           "Specific Internal Energy", Wx, Wy, Ww,Wh);
            Wx += offx;
            
            VisualizeField(vis_u, vishost, visport, d_gf,
                           "Displacement", Wx, Wy, Ww,Wh);
            Wx += offx;            
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }

         if (gfprint)
         {
            ostringstream mesh_name, rho_name, v_name, e_name;
            mesh_name << basename << "_" << ti
                      << "_mesh." << setfill('0') << setw(6) << myid;
            rho_name  << basename << "_" << ti
                      << "_rho." << setfill('0') << setw(6) << myid;
            v_name << basename << "_" << ti
                   << "_v." << setfill('0') << setw(6) << myid;
            e_name << basename << "_" << ti
                   << "_e." << setfill('0') << setw(6) << myid;

            ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->Print(mesh_ofs);
            mesh_ofs.close();

            ofstream rho_ofs(rho_name.str().c_str());
            rho_ofs.precision(8);
            rho_gf.Save(rho_ofs);
            rho_ofs.close();

            ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.Save(v_ofs);
            v_ofs.close();

            ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.Save(e_ofs);
            e_ofs.close();
         }
      }
      
      if (mpi.Root() && (ti % 100) == 0)
       {
            cout << fixed;
            cout << "step " << setw(5) << ti
                 << ",\tt = " << setw(5) << setprecision(4) << t
                 << ",\tdt = " << setw(5) << setprecision(6) << dt << endl;
         }      
      //meshAdapt 
      //Field* createStepField(Mesh* m, const char* name, int valueType)
      //  Field transfer. V,X,e solution fields and Velocity magnitude field for
      //  error estimation are created for the pumi mesh.
      ma_t += dt;
      if (ma_t > ma_time) //(steps % ma_step) == 0
      {
          ma_t = 0.0;
    if (myid == 0) { cout << "Beginning MeshAdapt : " << endl;}      
          if (order_v > geom_order)
          {
             Vmag_field = apf::createField(pumi_mesh, "field_mag",
                                           apf::SCALAR, apf::getLagrange(order_v));
             Vel_field = apf::createField(pumi_mesh, "V_field",
                                           apf::VECTOR, apf::getLagrange(order_v));
             Dis_field = apf::createField(pumi_mesh, "Crd_field",
                                           apf::VECTOR, apf::getLagrange(order_v)); 
             U_field = apf::createField(pumi_mesh, "U_field",
                                           apf::VECTOR, apf::getLagrange(order_v));          
          }
          else
          {
             Vmag_field = apf::createFieldOn(pumi_mesh, "field_mag",apf::SCALAR);
             Vel_field = apf::createFieldOn(pumi_mesh, "V_field", apf::VECTOR);
             Dis_field = apf::createFieldOn(pumi_mesh, "Crd_field", apf::VECTOR); 
             U_field = apf::createFieldOn(pumi_mesh, "U_field", apf::VECTOR);          
          }
          Enr_field = apf::createStepField(pumi_mesh, "E_field", apf::SCALAR);

          ParPumiMesh* pPPmesh = dynamic_cast<ParPumiMesh*>(pmesh);
          pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &x_gf, Dis_field, Vmag_field);
          pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &d_gf, U_field, Vmag_field);      
          pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &v_gf, Vel_field, Vmag_field);
          pPPmesh->DGFieldMFEMtoPUMI(pumi_mesh, &e_gf, Enr_field);


          ipfield= spr::getGradIPField(Vmag_field, "MFEM_gradip", 2);
          sizefield = spr::getSPRSizeField(ipfield, adapt_ratio);

          apf::destroyField(Vmag_field);
          apf::destroyField(ipfield);
          apf::destroyNumbering(pumi_mesh->findNumbering("LocalVertexNumbering"));

          //write vtk file
          apf::writeVtkFiles("before_ma", pumi_mesh);

          // 19. Perform MesAdapt
          ma::Input* erinput = ma::configure(pumi_mesh, sizefield);
          erinput->shouldFixShape = true;
          erinput->maximumIterations = 2;
          //erinput->shouldCoarsen = false;
          if ( geom_order > 1)
          {
             crv::adapt(erinput);
          }
          else
          {
             ma::adapt(erinput);
          }

          //write vtk file
          apf::writeVtkFiles("After_ma", pumi_mesh);      

          ParMesh* Adapmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
          pPPmesh->UpdateMesh(Adapmesh);
          delete Adapmesh;
          
          //Hack for the boundary condition
          apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
          apf::MeshEntity* ent ;
          int bdr_cnt = 0;
          while ((ent = pumi_mesh->iterate(itr)))
          {
              apf::ModelEntity *me = pumi_mesh->toModel(ent);
              if (pumi_mesh->getModelType(me) == (dim-1))
              {
                  //Evrywhere 3 as initial
                  //(pmesh->GetBdrElement(bdr_cnt))->SetAttribute(3);
                  int tag = pumi_mesh->getModelTag(me);
                  if (XX.Find(tag) != -1)
                  {
                      //XX attr -> 1
                      (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(1);
                  }
                  else if (YY.Find(tag) != -1)
                  {
                      //YY attr -> 2
                      (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(2);
                  }
                  else if (ZZ.Find(tag) != -1)
                  {
                      //ZZ attr -> 3
                      (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(3);    
                  }
                  else 
                  {
                       cout << " !!! ERROR !!! boundary has no attribute : " << endl;
                  }
                  bdr_cnt++;
              }
          }
          pumi_mesh->end(itr);   
          pmesh->SetAttributes();          
          

          // 20. Update the FiniteElementSpace, Gridfunction, and bilinear form
          //fespace->Update();
          H1FESpace.Update();
          L2FESpace.Update();  
          //Update Essential true dofs
          oper.UpdateEssentialTrueDofs();
          
          //x.Update();
          Vsize_l2 = L2FESpace.GetVSize();
          Vsize_h1 = H1FESpace.GetVSize();
          Array<int> updated_offset(4);
          updated_offset[0] = 0;
          updated_offset[1] = updated_offset[0] + Vsize_h1;
          updated_offset[2] = updated_offset[1] + Vsize_h1;
          updated_offset[3] = updated_offset[2] + Vsize_l2; 
          S.Update(updated_offset);
          x_gf.MakeRef(&H1FESpace, S, updated_offset[0]);
          v_gf.MakeRef(&H1FESpace, S, updated_offset[1]);
          int ll = L2FESpace.GetVSize();
          int kk = S.Size(); 
          e_gf.MakeRef(&L2FESpace, S, updated_offset[2]);      
          x_gf = 0.0;
          v_gf = 0.0;
          e_gf = 0.0;
          d_gf.Update();

          pPPmesh->VectorFieldPUMItoMFEM(pumi_mesh, Dis_field, &x_gf);
          pPPmesh->VectorFieldPUMItoMFEM(pumi_mesh, Vel_field, &v_gf);
          pPPmesh->VectorFieldPUMItoMFEM(pumi_mesh, U_field, &d_gf);      
          pPPmesh->DGFieldPUMItoMFEM(pumi_mesh, Enr_field, &e_gf);   
          S_old.Update(updated_offset);
          //a->Update();
          //b->Update();
          oper.MeshAdaptUpdate(S, d_gf);

          //Destroy fields
          apf::destroyField(Vel_field);
          apf::destroyField(Dis_field);      
          apf::destroyField(U_field);      
          apf::destroyField(Enr_field);      
          apf::destroyField(sizefield); 
      }
   }

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6;
   }
   oper.PrintTimingData(mpi.Root(), steps);

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh;
   delete material_pcf;
   //delete mat_gf_coeff;
   
   pumi_mesh->destroyNative();
   apf::destroyMesh(pumi_mesh);
   PCU_Comm_Free();

#ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
#endif   
   

   return 0;
}

namespace mfem
{

namespace hydrodynamics
{

double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: if (x(0) < 0.5) { return 1.0; }
         else { return 0.1; }
      case 3: if (x(0) > 1.0 && x(1) <= 1.5) { return 1.0; }
         else { return 0.125; }
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double gamma(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5./3.;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3: if (x(0) > 1.0 && x(1) <= 1.5) { return 1.4; }
         else { return 1.5; }
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

void v0(const Vector &x, Vector &v)
{
   switch (problem)
   {
      case 0:
         v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
         v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
         if (x.Size() == 3)
         {
            v(0) *= cos(M_PI*x(2));
            v(1) *= cos(M_PI*x(2));
            v(2) = 0.0;
         }
         break;
      case 1: v = 0.0; break;
      case 2: v = 0.0; break;
      case 3: v = 0.0; break;
      default: MFEM_ABORT("Bad number given for problem id!");
   }
}

double e0(const Vector &x)
{
   switch (problem)
   {
      case 0:
      {
         const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
         double val;
         if (x.Size() == 2)
         {
            val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
         }
         else
         {
            val = 100.0 + ((cos(2*M_PI*x(2)) + 2) *
                           (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
         }
         return val/denom;
      }
      case 1: return 0.0; // This case in initialized in main().
      case 2: if (x(0) < 0.5) { return 1.0 / rho0(x) / (gamma(x) - 1.0); }
         else { return 0.1 / rho0(x) / (gamma(x) - 1.0); }
      case 3: if (x(0) > 1.0) { return 0.1 / rho0(x) / (gamma(x) - 1.0); }
         else { return 1.0 / rho0(x) / (gamma(x) - 1.0); }
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

} // namespace hydrodynamics

} // namespace mfem

void display_banner(ostream & os)
{
   os << endl
      << "       __                __                 " << endl
      << "      / /   ____  ____  / /_  ____  _____   " << endl
      << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
      << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
      << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
      << "               /____/                       " << endl << endl;
}
