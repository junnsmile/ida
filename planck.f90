MODULE PLANCKK
    implicit none
    CONTAINS
REAL FUNCTION PLKAVG( WNUMLO, WNUMHI, T )


!c        Computes Planck function integrated between two wavenumbers
!c
!c  INPUT :  WNUMLO : Lower wavenumber (inv cm) of spectral interval
!c
!c           WNUMHI : Upper wavenumber
!c
!c           T      : Temperature (K)
!c
!c  OUTPUT : PLKAVG : Integrated Planck function ( Watts/sq m )
!c                      = Integral (WNUMLO to WNUMHI) of
!c                        2h c**2  nu**3 / ( EXP(hc nu/kT) - 1)
!c                        (where h=Plancks constant, c=speed of
!c                         light, nu=wavenumber, T=temperature,
!c                         and k = Boltzmann constant)
!c
!c  Reference : Specifications of the Physical World: New Value
!c                 of the Fundamental Constants, Dimensions/N.B.S.,
!c                 Jan. 1974
!c
!c  Method :  For WNUMLO close to WNUMHI, a Simpson-rule quadrature
!c            is done to avoid ill-conditioning; otherwise
!c
!c            (1)  For WNUMLO or WNUMHI small,
!c                 integral(0 to WNUMLO/HI) is calculated by expanding
!c                 the integrand in a power series and integrating
!c                 term by term;
!c
!c            (2)  Otherwise, integral(WNUMLO/HI to INFINITY) is
!c                 calculated by expanding the denominator of the
!c                 integrand in powers of the exponential and
!c                 integrating term by term.
!c
!c  Accuracy :  At least 6 significant digits, assuming the
!c              physical constants are infinitely accurate
!c
!c  ERRORS WHICH ARE NOT TRAPPED:
!c
!c      * power or exponential series may underflow, giving no
!c        significant digits.  This may or may not be of concern,
!c        depending on the application.
!c
!c      * Simpson-rule special case is skipped when denominator of
!c        integrand will cause overflow.  In that case the normal
!c        procedure is used, which may be inaccurate if the
!c        wavenumber limits (WNUMLO, WNUMHI) are close together.
!c
!c  LOCAL VARIABLES
!c
!c        A1,2,... :  Power series coefficients
!c        C2       :  h * c / k, in units cm*K (h = Plancks constant,
!c                      c = speed of light, k = Boltzmann constant)
!c        D(I)     :  Exponential series expansion of integral of
!c                       Planck function from WNUMLO (i=1) or WNUMHI
!c                       (i=2) to infinity
!c        EPSIL    :  Smallest number such that 1+EPSIL .GT. 1 on
!c                       computer
!c        EX       :  EXP( - V(I) )
!c        EXM      :  EX**M
!c        MMAX     :  No. of terms to take in exponential series
!c        MV       :  Multiples of V(I)
!c        P(I)     :  Power series expansion of integral of
!c                       Planck function from zero to WNUMLO (I=1) or
!c                       WNUMHI (I=2)
!c        PI       :  3.14159...
!c        SIGMA    :  Stefan-Boltzmann constant (W/m**2/K**4)
!c        SIGDPI   :  SIGMA / PI
!c        SMALLV   :  Number of times the power series is used (0,1,2)
!c        V(I)     :  C2 * (WNUMLO(I=1) or WNUMHI(I=2)) / temperature
!c        VCUT     :  Power-series cutoff point
!c        VCP      :  Exponential series cutoff points
!c        VMAX     :  Largest allowable argument of EXP function
!c
!c   Called by- DISORT
!c   Calls- D1MACH, ERRMSG
!c ----------------------------------------------------------------------
!
!c     .. Parameters ..

      REAL      A1, A2, A3, A4, A5, A6
      PARAMETER ( A1 = 1. / 3., A2 = -1. / 8., A3 = 1. / 60.,A4 = -1. / 5040., A5 = 1. / 272160.,A6 = -1. / 13305600. )
!c     ..
!c     .. Scalar Arguments ..

      REAL      T, WNUMHI, WNUMLO
!c     ..
!c     .. Local Scalars ..

      INTEGER   I, K, M, MMAX, N, SMALLV
      REAL      C2, CONC, DEL, EPSIL, EX, EXM, HH, MV, OLDVAL, PI,SIGDPI, SIGMA, VAL, VAL0, VCUT, VMAX, VSQ, X
!c     ..
!c     .. Local Arrays ..

      REAL      D( 2 ), P( 2 ), V( 2 ), VCP( 7 )
!c     ..
!c     .. External Functions ..

      REAL      D1MACH
      EXTERNAL  D1MACH
!c     ..
!c     .. External Subroutines ..

      EXTERNAL  ERRMSG
!c     ..
!c     .. Intrinsic Functions ..

      INTRINSIC ABS, ASIN, EXP, LOG, MOD
!c     ..
!c     .. Statement Functions ..

      REAL      PLKF
!c     ..
      SAVE      PI, CONC, VMAX, EPSIL, SIGDPI

      DATA      C2 / 1.438786 / , SIGMA / 5.67032E-8 / , VCUT / 1.5 / ,VCP / 10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0 /
      DATA      PI / 0.0 /

!c     .. Statement Function definitions ..

      PLKF( X ) = X**3 / ( EXP( X ) - 1 )
!c     ..


      IF( PI .EQ. 0.0 ) THEN

         PI     = 2.*ASIN( 1.0 )
         VMAX   = LOG( D1MACH( 2 ) )
         EPSIL  = D1MACH( 4 )
         SIGDPI = SIGMA / PI
         CONC   = 15. / PI**4

      END IF


      IF( T.LT.0.0 .OR. WNUMHI.LE.WNUMLO .OR. WNUMLO.LT.0. ) then
          CALL ERRMSG('PLKAVG--temperature or wavenums. wrong',.TRUE.)
          END IF


      IF( T .LT. 1.E-4 ) THEN

         PLKAVG = 0.0
         RETURN

      END IF


      V( 1 ) = C2*WNUMLO / T
      V( 2 ) = C2*WNUMHI / T

      IF( V( 1 ).GT.EPSIL .AND. V( 2 ).LT.VMAX .AND.( WNUMHI - WNUMLO ) / WNUMHI .LT. 1.E-2 ) THEN

!c                          ** Wavenumbers are very close.  Get integral
!c                          ** by iterating Simpson rule to convergence.

         HH     = V( 2 ) - V( 1 )
         OLDVAL = 0.0
         VAL0   = PLKF( V( 1 ) ) + PLKF( V( 2 ) )

         DO 20 N = 1, 10

            DEL  = HH / ( 2*N )
            VAL  = VAL0

            DO 10 K = 1, 2*N - 1
               VAL  = VAL + 2*( 1 + MOD( K,2 ) )*PLKF( V( 1 ) + K*DEL )
   10       CONTINUE

            VAL  = DEL / 3.*VAL
            IF( ABS( ( VAL - OLDVAL ) / VAL ).LE.1.E-6 ) GO TO  30
            OLDVAL = VAL

   20    CONTINUE

         CALL ERRMSG( 'PLKAVG--Simpson rule didnt converge',.FALSE.)

   30    CONTINUE

         PLKAVG = SIGDPI * T**4 * CONC * VAL

         RETURN

      END IF

!c                          *** General case ***
      SMALLV = 0

      DO 60 I = 1, 2

         IF( V( I ).LT.VCUT ) THEN
!c                                   ** Use power series
            SMALLV = SMALLV + 1
            VSQ    = V( I )**2
            P( I ) = CONC*VSQ*V( I )*( A1 +V( I )*( A2 + V( I )*( A3 + VSQ*( A4 + VSQ*( A5 +VSQ*A6 ) ) ) ) )

         ELSE
!c                      ** Use exponential series
            MMAX  = 0
!c                                ** Find upper limit of series
   40       CONTINUE
            MMAX  = MMAX + 1

            IF( V(I) .LT. VCP( MMAX ) ) GO TO  40

            EX     = EXP( - V(I) )
            EXM    = 1.0
            D( I ) = 0.0

            DO 50 M = 1, MMAX
               MV     = M*V( I )
               EXM    = EX*EXM
               D( I ) = D( I ) + EXM*( 6.+ MV*( 6.+ MV*( 3.+ MV ) ) )/ M**4
   50       CONTINUE

            D( I ) = CONC*D( I )

         END IF

   60 CONTINUE

!c                              ** Handle ill-conditioning
      IF( SMALLV.EQ.2 ) THEN
!c                                    ** WNUMLO and WNUMHI both small
         PLKAVG = P( 2 ) - P( 1 )

      ELSE IF( SMALLV.EQ.1 ) THEN
!c                                    ** WNUMLO small, WNUMHI large
         PLKAVG = 1.- P( 1 ) - D( 2 )

      ELSE
!c                                    ** WNUMLO and WNUMHI both large
         PLKAVG = D( 1 ) - D( 2 )

      END IF

      PLKAVG = SIGDPI * T**4 * PLKAVG

      IF( PLKAVG.EQ.0.0 ) then
          CALL ERRMSG('PLKAVG--returns zero; possible underflow',.FALSE.)
          END IF


      RETURN

      END FUNCTION PLKAVG
END MODULE PLANCKK
