      module ocnref
	IMPLICIT REAL (A - H,O - Z),
     &INTEGER (I - N) 
      !implicit none
      private
      public :: ocnref__NN_roughness_R
      public :: ocnref__NN_BRDF_ang_R
      public :: ocnref__NN_BRDF_vec_R

  ! Constants used in this module
      integer,   parameter :: R_    = selected_real_kind(13)  ! default precision
      real(R_),  parameter :: REPS_ = epsilon(1.0_R_)         ! 1.0e-13 for double precision
      real(R_),  parameter :: RSML_ = tiny(1.0_R_) * 3.0_R_   ! very small value
      real(R_),  parameter :: PI_   = 3.1415926535897932384626433832_R_   ! pi


      contains


  !+
  ! Ocean surface roughness, modeled by Nakajima and Tanaka's (1983)
  !-
      function ocnref__NN_roughness_R(u10) result(res)

      real, intent(in) :: u10  ! the wind velocity (m/s) at 10 m above the ocean surface
      real             :: res  ! sigma^2, surface roughness, variance of tangent

      res = 0.00534 * u10

      end function ocnref__NN_roughness_R


  !+
  ! BRDF of Nakajima and Tanaka's (1983) ocean surface model
  !  for a single set of incident and reflection directions
  !-
      function ocnref__NN_BRDF_ang_R(cr, ci, sigma2,
     & mu0, phi0, mu1, phi1) result(res)

      real, intent(in) :: cr      ! real part of refractive index
      real, intent(in) :: ci      ! imaginary part of refractive index
      real, intent(in) :: sigma2  ! sigma^2, surface roughness, variance of tangent
      real, intent(in) :: mu0     ! incident direction cosine (should be < 0 for downward)
      real, intent(in) :: phi0    ! incident direction azimuth angle (radian)
      real, intent(in) :: mu1     ! reflection direction cosine (should be > 0 for upward)
      real, intent(in) :: phi1    ! reflection direction azimuth angle (radian)
      real :: res   ! BRDF (/steradian)
      real :: vec0(3), vec1(3), s0, s1
	
   
    ! Direction cosines
      s0 = sqrt(1.0 - mu0**2)
      s1 = sqrt(1.0 - mu1**2)
	
      vec0(1) = s0 * cos(phi0)
      vec0(2) = s0 * sin(phi0)
      vec0(3) = mu0
      vec1(1) = s1 * cos(phi1)
      vec1(2) = s1 * sin(phi1)
      vec1(3) = mu1

    ! Compute the BRDF
      res = ocnref__NN_BRDF_vec_R(cr, ci, sigma2, vec0, vec1)

      end function ocnref__NN_BRDF_ang_R


  !+
  ! BRDF of Nakajima and Tanaka's (1983) ocean surface model
  !  for a single set of incident and reflection direction vectors
  !-
      function ocnref__NN_BRDF_vec_R(cr,ci,sigma2,vec0,vec1) result(res)

      real, intent(in) :: cr      ! real part of refractive index
      real, intent(in) :: ci      ! imaginary part of refractive index
      real, intent(in) :: sigma2  ! sigma^2, surface roughness, variance of tangent
      real, intent(in) :: vec0(:) ! incoming direction vector (downward motion to the plane)
      real, intent(in) :: vec1(:) ! outgoing direction vector (upward motion from the plane)
      real  :: res   ! BRDF (/steradian)

      real,  parameter :: RMAX = 1.0e+35
      real,  parameter :: FPI = 0.56418958
      real  :: s2, uz0, uz1, vv0, vv1, fshad0, fshad1, fshad, funcs
      real  :: uza, uzt, rho1, rho2, rho, uzn2, rnum, deno, vecn(3)
 
    ! Invalid directions
      if (vec0(3) * vec1(3) > -REPS_) then
       res = 0.0
       return
      endif

    ! Shadowing factors
      s2 = max(1.0e-4, sigma2)
      uz0 = -vec0(3)
      uz1 =  vec1(3)
      vv0 = max(REPS_, abs(uz0) / 
     &max(REPS_, sqrt(s2 * max(0.0, 1.0 - uz0**2))))
      vv1 = max(REPS_, abs(uz1) / 
     &max(REPS_, sqrt(s2 * max(0.0, 1.0 - uz1**2))))
      fshad0 = 0.0
      fshad1 = 0.0
      if (vv0 < 10.0) fshad0 = 
     &max(0.0, 0.5 * (exp(-vv0**2) / vv0 * FPI - erfc_R(vv0)))
      if (vv1 < 10.0) fshad1 = 
     &max(0.0, 0.5 * (exp(-vv1**2) / vv1 * FPI - erfc_R(vv1)))

    ! Facet normal vector
      vecn(1:3) = vec1(1:3) - vec0(1:3)         ! facet normal vector (upward)
      vecn(:) = vecn(:) / sqrt(sum(vecn(:)**2)) ! normalize

    ! Other functions
      uza = vecn(1) * vec0(1) + vecn(2) * vec0(2) + vecn(3) * vec0(3)
      call fresnelRef1(cr, ci, -uza, uzt, rho1, rho2, rho) ! rho = Fresnel reflectance
      fshad = 1.0 / (1.0 + fshad0 + fshad1) ! shadowing factor for the bidirectional geometry
      funcs = rho * fshad / vecn(3)**4            ! function S
 
    ! BRDF
      uzn2 = vecn(3)**2
      rnum = funcs / (PI_ * s2) * exp( -(1.0 - uzn2) / (s2 * uzn2) )

      deno = 4.0 * (-vec0(3)) * vec1(3)
      if (deno * RMAX > rnum) then ! to avoid too large BRDF values
       res = rnum / deno

      else
       res = RMAX
      end if
      
      end function ocnref__NN_BRDF_vec_R


  !+
  ! Fresnel reflection for a complex refractive index
  !-
      subroutine fresnelRef1(rr, ri, uzi, uzt, rhov, rhoh, rho)

      real, intent(in)  :: rr   ! real      part of refractive index
      real, intent(in)  :: ri   ! imaginary part of refractive index (should be >= 0)
      real, intent(in)  :: uzi  ! cosine of incident vector   nadir angle (positive)
      real, intent(out) :: uzt  ! cosine of refraction vector nadir angle (positive)
      real, intent(out) :: rhoh ! V-component reflectance
      real, intent(out) :: rhov ! H-component reflectance
      real, intent(out) :: rho  ! average reflectance
      real  :: g2, ri2, rr2, u, uzi2, v, w1, w2, w3, wa

    ! Ill input
      if (uzi <= 0.0 .or. rr <= 0.0) then
       uzt  =  0.0
       rhov = -1.0
       rhoh = -1.0
       rho  = -1.0

      else
       uzi2 = uzi * uzi
       rr2 = rr * rr
       ri2 = ri * ri
       g2 = rr2 + uzi2 - 1.0

       ! Partial reflection
       if (g2 > RSML_) then
          uzt = sqrt(g2) / rr
          w1 = rr2 - ri2
          w2 = 2.0 * rr * abs(ri)
          w3 = g2 - ri2
          wa = sqrt(w3 * w3 + w2 * w2)
          u = sqrt(0.5 * abs(wa + w3))
          v = sqrt(0.5 * abs(wa - w3))
          rhov = ((uzi - u)**2 + v*v) / ((uzi + u)**2 + v*v)
          rhoh =   ((w1 * uzi - u)**2 + (w2 * uzi - v)**2) 
     & / ((w1 * uzi + u)**2 + (w2 * uzi + v)**2)
          rho = 0.5 * (rhov + rhoh)

          ! 100% reflection
       else
          uzt  = 0.0
          rhov = 1.0
          rhoh = 1.0
          rho  = 1.0
       end if
      end if

      end subroutine fresnelRef1


  !+
  ! Complementary error function
  !-
      function erfc_R(x) result(y)

      real, intent(in)  :: x    ! X
      real  :: y    ! Y = erfc(X)
    !y = erfc(x) ! erfc() is nonstandard!
      if (x > 0.0) then
       y = erfc_Cheb_R(x)
      else
       y = 2.0 - erfc_Cheb_R(-x)
      end if

      end function erfc_R


  !+
  ! Complementary error function using a Chebyshev approximation (single precision version)
  !  -with relative error less than 1.2e-7
  !-
      function erfc_Cheb_R(z) result(y)

      real, intent(in)  :: z ! abs(X) > 0
      real  :: y ! Y = erfc(z)
      real  :: t

      if (z < 0.0) stop 'erfc_Cheb_R: Z should be >= 0.'
      t = 2.0 / (2.0 + z)
      y = -z**2 - 1.26551223 + t*(1.00002368
     & +t*(0.37409196 + t*(0.09678418 
     &+ t*(-0.18628806 + t*(0.2788680 
     &+ t*(-1.13520398 + t*(1.4851587 
     &+ t*(-0.82215223 + t*0.17087277))))))))
      y = t * exp(y)

      end function erfc_Cheb_R

      end module ocnref
