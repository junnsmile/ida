c -----------------------------------------------------------------
c NOTE:  R1MACH is invoked through a USE RDI1MACH_f90 statement,
c        which is Fortran-90.  The reasons are given in the file
c        README.RDI1MACH.  One can do things the old way by 
c        removing the USE statement and getting R1MACH from netlib.
c        Remember that MODULEs must precede the program unit USEing
c        them in Fortran-90, or be separately compiled and linked.
c -----------------------------------------------------------------

      PROGRAM MITEST

c         Run 19 test cases for MIEV0, testing all the pathways
c         through the code.  In parentheses beneath each result is
c         printed its ratio to the answer obtained on a Cray
c         computer using 14-digit precision.  If these ratios drift
c         away from unity as size parameter increases, your computer
c         probably is of lower precision than the Cray, OR it may
c         handle arithmetic differently (truncating instead of
c         rounding, for example).  Before becoming overly concerned
c         about non-unit ratios, re-run in double precision
c         (most compilers have an auto-doubling option) and see if
c         your results improve.

c     NOTES:

c        ** Set NoPMOM = True at the beginning of the executable
c           statements below if using NoPMOM version of MIEV0

c        ** Temporarily set PARAMETER MAXTRM in MIEV0 to 10,100 to
c           run these test problems.  (Be sure to lower it again to
c           some reasonable value when you are finished.)

c        ** Timing is done by calls to the Fortran-90 intrinsic
c           function SYSTEM_CLOCK; if you absolutely can't find an
c           f90 compiler, these calls can just be deleted, or replaced
c           using some local system-dependent routine.

c        ** To keep storage requirements reasonable (i.e. to keep from
c           having to set  MAXTRM = 10100 in LPCOEF also), and to avoid
c           enormous DATA statements to hold the correct answers, 
c           Legendre moments are not checked for the size parameter = 
c           10,000 cases.  Also, only the first two moments are checked 
c           for size parameter = 1000 cases.

c        ** Pay especial attention to the cases where two successive
c           values of size parameter are small and close.  The lower
c           value uses small-particle approximations, the upper value
c           uses the full Mie series.  The results should be close.

c        ** TBACK is the quantity most sensitive to precision.
c           This is because the coefficients in the TBACK series
c           are large, increasing, and alternating in sign,
c           making the order in which the terms are summed
c           important in finite-precision arithmetic.  SBACK
c           and S1, S2 near 180 deg are also sensitive.

c        ** High-order Legendre moments, being near zero, are also
c           subject to a lot of cancellation in their defining series,
c           and hence may also be sensitive to computer precision.

c     CALLING TREE:

c          MITEST
c             RATIO
c                R1MACH
c             MIEV0
c          CHEKMI


      IMPLICIT  NONE

c ----------------------------------------------------------------------
c -----------  I/O SPECIFICATIONS FOR SUBROUTINE  MIEV0  ---------------
c ----------------------------------------------------------------------
      INTEGER   MAXANG, MOMDIM
      PARAMETER  ( MAXANG = 7, MOMDIM = 300 )
      LOGICAL   ANYANG, PERFCT, PRNT( 2 )
      INTEGER   IPOLZN, NUMANG, NMOM
      REAL      GQSC, MIMCUT, PMOM( 0:MOMDIM, 4 ), QEXT, QSCA, SPIKE,
     $          XMU( MAXANG ), XX
      COMPLEX   CREFIN, SFORW, SBACK, S1( MAXANG ), S2( MAXANG ),
     $          TFORW( 2 ), TBACK( 2 )
c ----------------------------------------------------------------------

c --------------- LOCAL VARIABLES --------------------------------------

c     .. Local Scalars ..

      LOGICAL   NOPMOM
      INTEGER   I, J,JJ, K, NCAS, NPQUAN, time0, time1, cntrat, maxcnt
      REAL      DEGPOL, FNORM, I1, I2, INTEN, PI, QABS, TESTIN,KK
c     ..
c     .. Local Arrays ..

      REAL      ANGLE( MAXANG )
c     ..
c     .. External Functions ..

      REAL      RATIO
      EXTERNAL  RATIO
c     ..
c     .. External Subroutines ..

      EXTERNAL  MIEV0
c     ..

c ----------------------------------------------------------------------
c        Input specifications and 'correct' answers to test problems
c ----------------------------------------------------------------------

      INTEGER  NCASES
      PARAMETER  ( NCASES = 19 )
c      REAL ::BADDS(100) =(/0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
c     $        0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23,
c     $        0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
c     $        0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,
c     $        0.49,0.5,0.51,
c     $        0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,
c     $        0.63,0.64,0.65,
c     $        0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74, 0.75,
c     $        0.76, 0.77, 0.78, 0.79,
c     $        0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,
c     $        0.9,  0.91, 0.92, 0.93,
c     $        0.94,0.95,0.96,0.97,0.98,0.99,1.0/)

c     .. Arrays in Common ..

      LOGICAL  TESTAN( NCASES )
      INTEGER  TESTIP( NCASES )
      REAL     TESTXX( NCASES ), TESTQE( NCASES ), TESTQS( NCASES ),
     $         TESTGQ( NCASES ), TESTPM( 0:MOMDIM, 4, NCASES )
      COMPLEX  TESTCR( NCASES ), TESTSF( NCASES ), TESTSB( NCASES ),
     $         TESTTF( 2,NCASES ), TESTTB( 2,NCASES ),
     $         TESTS1( 7,NCASES ), TESTS2( 7,NCASES )

      COMMON / MICHEK / TESTAN, TESTIP, TESTXX, TESTCR,
     &                  TESTQE, TESTQS, TESTGQ, TESTSF, TESTSB,
     $                  TESTTF, TESTTB, TESTS1, TESTS2, TESTPM
c ----------------------------------------------------------------------
      KK = 2.333
      DO JJ = 1,3842
      CALL SYSTEM_CLOCK( COUNT_RATE=cntrat, COUNT_MAX=maxcnt)
      IF( maxcnt == 0 ) 
     &   WRITE(*,*) ' System has no clock.  Timing results meaningless.'

      PI     = 2.* ASIN( 1.0 )
c      write(*,*) PI
      NOPMOM = .False.
c                            ** Set MIEV0 input variables that are
c                            ** the same for every test case
      MIMCUT = 1.E-6
      NUMANG = MAXANG

      DO 10 I = 1, NUMANG
         ANGLE( I ) = ( I - 1 )*180. / ( NUMANG - 1 )
         XMU( I ) = COS( PI / 180.*ANGLE( I ) )
   10 CONTINUE

c               ** Call once for very small perfectly conducting sphere;
c               ** this does MIEV0 self-test, so that self-test does
c               ** not affect timing results, and tests print option
c      print * ,"_______"
c      write(*,*) BADDS(JJ), BADDS(10)
      XX     = (2*PI*KK/6.2)
c      WRITE(*,*) XX
      CREFIN = (1.3630,0.088000)
      PERFCT = .FALSE.
      ANYANG = .FALSE.
      IPOLZN = 0
!     NMOM=int(2.*XX)
!     IF( XX.LT.1. .OR. XX.GT.100. ) NMOM   = 1
!     IF( XX.GT.1000. )              NMOM   = 0
!     IF( NoPMOM )                   NMOM   = 0
      NMOM = 257


      PRNT( 1 ) = .FALSE.
      PRNT( 2 ) = .TRUE.

      CALL MIEV0( XX, CREFIN, PERFCT, MIMCUT, ANYANG, NUMANG, XMU, NMOM,
     &            IPOLZN, MOMDIM, PRNT, QEXT, QSCA, GQSC, PMOM, SFORW,
     &            SBACK, S1, S2, TFORW, TBACK, SPIKE )

      PRNT( 1 ) = .FALSE.
      PRNT( 2 ) = .FALSE.



!        WRITE( *, 9070 )
!        WRITE( *, 9080 ) 0.0, SFORW, TFORW( 1 ), TFORW( 2 )
!        WRITE( *, 9090 ) RATIO(  REAL(SFORW),  REAL(TestSF(NCas)) ),
!    $                   RATIO( AIMAG(SFORW), AIMAG(TestSF(NCas)) ),
!    $              RATIO(  REAL(TFORW(1)),  REAL(TestTF(1,NCas)) ),
!    $              RATIO( AIMAG(TFORW(1)), AIMAG(TestTF(1,NCas)) ),
!    $              RATIO(  REAL(TFORW(2)),  REAL(TestTF(2,NCas)) ),
!    $              RATIO( AIMAG(TFORW(2)), AIMAG(TestTF(2,NCas)) )
!!        WRITE( *, 9080 ) 180., SBACK, TBACK( 1 ), TBACK( 2 )
!        WRITE( *, 9090 ) RATIO(  REAL(SBACK),  REAL(TestSB(NCas)) ),
!    $                    RATIO( AIMAG(SBACK), AIMAG(TestSB(NCas)) ),
!    $              RATIO(  REAL(TBACK(1)),  REAL(TestTB(1,NCas)) ),
!    $              RATIO( AIMAG(TBACK(1)), AIMAG(TestTB(1,NCas)) ),
!    $              RATIO(  REAL(TBACK(2)),  REAL(TestTB(2,NCas)) ),
!    $              RATIO( AIMAG(TBACK(2)), AIMAG(TestTB(2,NCas)) )

         Qabs   = QEXT - QSCA
!        WRITE( *, 9100 ) QEXT, QSCA, Qabs, GQSC / QSCA
         print *,"_______________________"
         write(*,*) KK
         write(*,*) QEXT 
         write(*,*) QSCA
         write(*,*) Qabs
         write(*,*) GQSC / QSCA
         
!        WRITE( *, 9110 ) RATIO( QEXT, TESTQE( NCAS ) ),
!    &                    RATIO( QSCA, TESTQS( NCAS ) ),
!    &                    RATIO( Qabs, TESTQE( NCAS )-TESTQS( NCAS )),
!    &                    RATIO( GQSC, TESTGQ( NCAS ) )
!!        WRITE( *, 9120 ) SPIKE


         IF( NMOM.GT.0 ) THEN

            WRITE( *, '(/,A)' ) ' Normalized moments of :'

            IF( IPOLZN.EQ.0 ) THEN

               NPQUAN = 1
!              WRITE( *, '(''+'',26X,A)' ) 'Phase Fcn'

            ELSE IF( IPOLZN.GT.0 ) THEN

               NPQUAN = 4
!              WRITE( *, '(''+'',33X,A)' )
!    &            'M1            M2           S21           D21'

            ELSE

               NPQUAN = 4
!              WRITE( *, '(''+'',33X,A)' )
!    &            'R1            R2            R3            R4'
            END IF


            fNorm  = 4./ ( XX**2 * QSCA )

            DO 30  J = 0, NMOM
               write(*,9130) ( fNorm*PMOM( J,K ), K = 1, NPQUAN )
!              WRITE( *, 9140 ) ( RATIO( fNorm*PMOM( J,K ),
!    &                            TESTPM( J,K,NCAS ) ), K = 1, NPQUAN )
!              WRITE( *, '(A,I4)' ) '      Moment no.', J
!              WRITE( *, 9130 ) ( fNorm*PMOM( J,K ), K = 1, NPQUAN )
!              WRITE( *, 9140 ) ( RATIO( fNorm*PMOM( J,K ),
!    &                            TESTPM( J,K,NCAS ) ), K = 1, NPQUAN )
   30       CONTINUE

         END IF

c -----------------------------------------------------------------
!     IF( time1 <= time0 )
!    &    WRITE(*,*) ' Clock maxed out.  Timing results meaningless.'
!     WRITE(*,*) ' Elapsed system-clock time (sec) = ',
!    &           (time1 - time0) / REAL(cntrat)
c -----------------------------------------------------------------

   40 CONTINUE


!     STOP

!!9000 FORMAT( ////, 80('*'),/,' MIEV0 Test Case', I3,
!    &   ':  Perfectly conducting,  Mie size parameter =', F10.3 )
!9010 FORMAT( ////, 80('*'),/,' MIEV0 Test Case', I3,
!    &   ':  Refractive index:  real ', F8.3, '  imag ', 1P,E11.3,
!    &   ',  Mie size parameter =', 0P, F10.3 )
!9020 FORMAT( 30X, 'NUMANG =', I4, ' arbitrary angles' )
!9030 FORMAT( 20X, 'NUMANG =', I4, ' angles symmetric about 90 degrees')
!9040 FORMAT( / , '  Angle    Cosine           S-sub-1', 21X, 'S-sub-2',
!    &      15X, 'Intensity  Deg of Polzn' )
!9050 FORMAT( F7.2, F10.6, 1P, 5E14.5, 0P, F14.4 )
!9060 FORMAT( 17X, 5( '  (',F10.6,')' ) )
!9070 FORMAT( / , '  Angle', 10X, 'S-sub-1', 21X, 'T-sub-1', 21X,
!    &      'T-sub-2' )
!9080 FORMAT( 0P, F7.2, 1P, 6E14.5 )
!9090 FORMAT( 7X, 6( '  (',F10.6,')' ) )
!9100 FORMAT( / , 12X, 'Efficiency factors for             Asymmetry',
!    & / , '    Extinction    Scattering    Absorption        Factor',
!    &  / ,  4ES14.6 )
!9110 FORMAT( 4( '  (',F10.6,')' ) )
!9120 FORMAT( / , ' SPIKE = ', 1P, E14.6 )
 9130 FORMAT(4E14.5 )
!9140 FORMAT( 24X, 4( '  (',F10.6,')' ) )
      IF (KK .GT. 5) THEN
         KK = KK+0.2
      ELSE        
         KK=KK+0.001
      ENDIF
      END DO
      END

      REAL FUNCTION RATIO( A, B )

c        Calculate ratio  A/B  with over- and under-flow protection
c        (thanks to Prof. Jeff Dozier for some suggestions here).
c        Since this routine takes two logs, it is no speed demon,
c        but it is invaluable for comparing results from two runs
c        of a program under development.

      USE RDI1MACH_f90, ONLY : R1MACH

c     .. Scalar Arguments ..

      REAL      A, B
c     ..
c     .. Local Scalars ..

      LOGICAL   PASS1
      REAL      ABSA, ABSB, BIG, POWA, POWB, POWMAX, POWMIN, SMALL
c     ..
      SAVE      PASS1, SMALL, BIG, POWMAX, POWMIN
      DATA      PASS1 / .TRUE. /


      IF( PASS1 ) THEN

         SMALL   = R1MACH( 1 )
         BIG   = R1MACH( 2 )
         POWMAX = LOG10( BIG )
         POWMIN = LOG10( SMALL )
         PASS1  = .FALSE.
      END IF


      IF( A.EQ.0.0 ) THEN

         IF( B.EQ.0.0 ) THEN

            RATIO  = 1.0

         ELSE

            RATIO  = 0.0

         END IF


      ELSE IF( B.EQ.0.0 ) THEN

         RATIO  = SIGN( BIG, A )

      ELSE

         ABSA   = ABS( A )
         ABSB   = ABS( B )
         POWA   = LOG10( ABSA )
         POWB   = LOG10( ABSB )

         IF( ABSA.LT.SMALL .AND. ABSB.LT.SMALL ) THEN

            RATIO  = 1.0

         ELSE IF( POWA - POWB.GE.POWMAX ) THEN

            RATIO  = BIG

         ELSE IF( POWA - POWB.LE.POWMIN ) THEN

            RATIO  = SMALL

         ELSE

            RATIO  = ABSA / ABSB

         END IF
c                      ** DONT use old trick of determining sign
c                      ** from A*B because A*B may (over/under)flow

         IF( ( A.GT.0.0 .AND. B.LT.0.0 ) .OR.
     &       ( A.LT.0.0 .AND. B.GT.0.0 ) ) RATIO = -RATIO

      END IF

      END

      BLOCK DATA CHEKMI

c        Input specifications and 'correct' answers to test problems

c        ( Suffixes of 'Test' variables:  AN = ANYANG, IP = IPOLZN,
c          XX = XX, CR = CREFIN, QE = QEXT, QS = QSCA, GQ = GQSC,
c          SF = SFORW, SB = SBACK, TF = TFORW, TB = TBACK )

c     .. Parameters ..

      INTEGER   NCASES, MOMDIM
      PARAMETER ( NCASES = 19, MOMDIM = 200 )
c     ..
c     .. Arrays in Common ..

      LOGICAL  TestAN( NCASES )
      INTEGER  TestIP( NCASES )
      REAL     TestXX( NCASES ), TestQE( NCASES ), TestQS( NCASES ),
     $         TestGQ( NCASES ), TestPM( 0:MOMDIM, 4, NCASES )
      COMPLEX  TestCR( NCASES ), TestSF( NCASES ), TestSB( NCASES ),
     $         TestTF( 2,NCASES ), TestTB( 2,NCASES ),
     $         TestS1( 7,NCASES ), TestS2( 7,NCASES )

      COMMON / MiChek / TestAN, TestIP, TestXX, TestCR,
     &                  TestQE, TestQS, TestGQ, TestSF, TestSB,
     $                  TestTF, TestTB, TestS1, TestS2, TestPM
c     ..
c     .. Local Scalars ..

      INTEGER   I, J, K
c     ..

c ------------- Perfectly Conducting, Size Par = 0.099 -----------------

      DATA  TestXX(1) / 0.099 /
      DATA  TestCR(1) /(0.,0.)/
      DATA  TestAN(1) /.FALSE./
      DATA  TestIP(1) /-1234/

      DATA  TestQE(1) / 3.209674E-04 /,
     $      TestQS(1) / 3.209674E-04 /,
     $      TestGQ(1) /  -1.275386E-04 /,
     $      TestSF(1) / (  7.878422E-07,  4.911194E-04 ) /,
     $      TestSB(1) / (  4.773735E-07,  1.454155E-03 ) /,
     $      TestTF( 1,1 ) / (  1.552344E-07, -4.841595E-04 ) /,
     $      TestTF( 2,1 ) / (  6.326078E-07,  9.752789E-04 ) /,
     $      TestTB( 1,1 ) / (  1.552344E-07, -4.820462E-04 ) /,
     $      TestTB( 2,1 ) / (  6.326078E-07,  9.721090E-04 ) /

      DATA ( TestS1( I,1 ), TestS2( I,1 ), I = 1, 7 ) /
     $ (  7.878422E-07, 4.911194E-04 ), (  7.878422E-07, 4.911194E-04 ),
     $ (  7.670447E-07, 5.558947E-04 ), (  7.030888E-07, 3.604144E-04 ),
     $ (  7.102250E-07, 7.326708E-04 ), (  4.715383E-07, 3.612002E-06 ),
     $ (  6.326078E-07, 9.736939E-04 ), (  1.552344E-07,-4.831029E-04 ),
     $ (  5.549907E-07, 1.214189E-03 ), ( -1.610695E-07,-9.690253E-04 ),
     $ (  4.981709E-07, 1.389908E-03 ), ( -3.926201E-07,-1.324243E-03 ),
     $ (  4.773735E-07, 1.454155E-03 ), ( -4.773735E-07,-1.454155E-03 )/

      DATA ( ( TestPM( J,K,1 ), K = 1, 4 ), J = 0, 1 )
     $  / 2.967623E-01, 1.205519E+00, -5.981239E-01, 5.807937E-04,
     $    4.327235E-04, 1.308231E-03, -7.606198E-04, 3.876034E-07 /

c ------------- Perfectly Conducting, Size Par = 0.101 -----------------

      DATA  TestXX(2) / 0.101 /
      DATA  TestCR(2) /(0.,0.)/
      DATA  TestAN(2) /.FALSE./
      DATA  TestIP(2) /-1234/

      DATA  TestQE(2) / 3.477160E-04 /,
     $      TestQS(2) / 3.477160E-04 /,
     $      TestGQ(2) /  -1.381344E-04 /,
     $      TestSF(2) / (  8.867628E-07,  5.217024E-04 ) /,
     $      TestSB(2) / (  5.372092E-07,  1.543993E-03 ) /,
     $      TestTF( 1,2 ) / (  1.747765E-07, -5.140600E-04 ) /,
     $      TestTF( 2,2 ) / (  7.119864E-07,  1.035762E-03 ) /,
     $      TestTB( 1,2 ) / (  1.747759E-07, -5.117301E-04 ) /,
     $      TestTB( 2,2 ) / (  7.119851E-07,  1.032263E-03 ) /

      DATA ( TestS1( I,2 ), TestS2( I,2 ), I = 1, 7 ) /
     $ (  8.867628E-07, 5.217024E-04 ), (  8.867628E-07, 5.217024E-04 ),
     $ (  8.633471E-07, 5.904742E-04 ), (  7.913746E-07, 3.828897E-04 ),
     $ (  7.993742E-07, 7.781489E-04 ), (  5.307693E-07, 3.966328E-06 ),
     $ (  7.119857E-07, 1.034013E-03 ), (  1.747762E-07,-5.128950E-04 ),
     $ (  6.245974E-07, 1.289294E-03 ), ( -1.812167E-07,-1.028882E-03 ),
     $ (  5.606248E-07, 1.475804E-03 ), ( -4.418213E-07,-1.406055E-03 ),
     $ (  5.372092E-07, 1.543993E-03 ), ( -5.372092E-07,-1.543993E-03 )/

      DATA ( ( TestPM( J,K,2 ), K = 1, 4 ), J = 0, 1 )
     $  / 2.966540E-01, 1.205716E+00, -5.980636E-01, 6.156040E-04,
     $    4.491982E-04, 1.360057E-03, -7.901094E-04, 4.269486E-07 /

c ------------- Perfectly Conducting, Size Par = 100   -----------------

      DATA  TestXX(3) / 100. /
      DATA  TestCR(3) /(0.,0.)/
      DATA  TestAN(3) /.FALSE./
      DATA  TestIP(3) /-1234/

      DATA  TestQE(3) / 2.008102E+00 /,
     $      TestQS(3) / 2.008102E+00 /,
     $      TestGQ(3) / 1.005911E+00 /,
     $      TestSF(3) / (  5.020256E+03, -1.746518E+01 ) /,
     $      TestSB(3) / ( -4.352508E+01, -2.455873E+01 ) /,
     $      TestTF( 1,3 ) / (  4.936594E+05, -1.017289E+06 ) /,
     $      TestTF( 2,3 ) / ( -4.886391E+05,  1.017272E+06 ) /,
     $      TestTB( 1,3 ) / (  7.145455E+01,  5.865980E+01 ) /,
     $      TestTB( 2,3 ) / (  2.792947E+01,  3.410108E+01 ) /

      DATA ( TestS1( I,3 ), TestS2( I,3 ), I = 1, 7 ) /
     $ (  5.020256E+03,-1.746518E+01 ), (  5.020256E+03,-1.746518E+01 ),
     $ (  5.352638E+01,-1.123588E+01 ), ( -3.353061E+01,-7.285853E+00 ),
     $ ( -2.621449E+01,-4.290513E+01 ), (  2.171234E+01, 4.412456E+01 ),
     $ ( -2.489003E+00, 4.997035E+01 ), (  3.504192E+00,-4.991006E+01 ),
     $ ( -2.045134E+01, 4.563103E+01 ), (  2.055790E+01,-4.554104E+01 ),
     $ ( -4.999320E+01, 8.991646E-01 ), (  5.000835E+01,-8.611726E-01 ),
     $ ( -4.352508E+01,-2.455873E+01 ), (  4.352508E+01, 2.455873E+01 )/

      DATA ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 0, 2 )
     $  / 3.317086E+04, 3.307264E+04, -3.312151E+04, 1.011479E+02,
     $    3.316044E+04, 3.306223E+04, -3.311109E+04, 1.011061E+02,
     $    3.314038E+04, 3.304217E+04, -3.309103E+04, 1.010337E+02 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 3, 21 )
     $  / 3.311093E+04, 3.301274E+04, -3.306159E+04, 1.009362E+02,
     $    3.307234E+04, 3.297416E+04, -3.302301E+04, 1.008174E+02,
     $    3.302476E+04, 3.292663E+04, -3.297546E+04, 1.006789E+02,
     $    3.296839E+04, 3.287032E+04, -3.291911E+04, 1.005223E+02,
     $    3.290339E+04, 3.280539E+04, -3.285415E+04, 1.003478E+02,
     $    3.282992E+04, 3.273201E+04, -3.278073E+04, 1.001564E+02,
     $    3.274815E+04, 3.265035E+04, -3.269902E+04, 9.994804E+01,
     $    3.265824E+04, 3.256057E+04, -3.260917E+04, 9.972333E+01,
     $    3.256035E+04, 3.246283E+04, -3.251136E+04, 9.948226E+01,
     $    3.245465E+04, 3.235730E+04, -3.240574E+04, 9.922541E+01,
     $    3.234129E+04, 3.224412E+04, -3.229248E+04, 9.895272E+01,
     $    3.222043E+04, 3.212347E+04, -3.217173E+04, 9.866476E+01,
     $    3.209224E+04, 3.199550E+04, -3.204365E+04, 9.836145E+01,
     $    3.195688E+04, 3.186038E+04, -3.190841E+04, 9.804336E+01,
     $    3.181451E+04, 3.171826E+04, -3.176616E+04, 9.771041E+01,
     $    3.166528E+04, 3.156930E+04, -3.161707E+04, 9.736315E+01,
     $    3.150935E+04, 3.141366E+04, -3.146129E+04, 9.700153E+01,
     $    3.134688E+04, 3.125150E+04, -3.129897E+04, 9.662607E+01,
     $    3.117803E+04, 3.108297E+04, -3.113029E+04, 9.623673E+01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 22, 40 )
     $  / 3.100296E+04, 3.090824E+04, -3.095539E+04, 9.583402E+01,
     $    3.082183E+04, 3.072746E+04, -3.077443E+04, 9.541790E+01,
     $    3.063478E+04, 3.054078E+04, -3.058757E+04, 9.498889E+01,
     $    3.044198E+04, 3.034836E+04, -3.039496E+04, 9.454694E+01,
     $    3.024358E+04, 3.015036E+04, -3.019676E+04, 9.409257E+01,
     $    3.003973E+04, 2.994692E+04, -2.999312E+04, 9.362572E+01,
     $    2.983059E+04, 2.973821E+04, -2.978420E+04, 9.314692E+01,
     $    2.961631E+04, 2.952438E+04, -2.957014E+04, 9.265612E+01,
     $    2.939705E+04, 2.930557E+04, -2.935111E+04, 9.215384E+01,
     $    2.917296E+04, 2.908194E+04, -2.912725E+04, 9.164002E+01,
     $    2.894418E+04, 2.885365E+04, -2.889872E+04, 9.111517E+01,
     $    2.871087E+04, 2.862083E+04, -2.866566E+04, 9.057926E+01,
     $    2.847318E+04, 2.838365E+04, -2.842822E+04, 9.003278E+01,
     $    2.823126E+04, 2.814224E+04, -2.818656E+04, 8.947570E+01,
     $    2.798525E+04, 2.789676E+04, -2.794081E+04, 8.890852E+01,
     $    2.773530E+04, 2.764736E+04, -2.769114E+04, 8.833120E+01,
     $    2.748156E+04, 2.739417E+04, -2.743768E+04, 8.774423E+01,
     $    2.722417E+04, 2.713735E+04, -2.718058E+04, 8.714758E+01,
     $    2.696328E+04, 2.687704E+04, -2.691998E+04, 8.654174E+01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 41, 59 )
     $  / 2.669904E+04, 2.661339E+04, -2.665603E+04, 8.592669E+01,
     $    2.643158E+04, 2.634653E+04, -2.638887E+04, 8.530289E+01,
     $    2.616104E+04, 2.607660E+04, -2.611864E+04, 8.467034E+01,
     $    2.588758E+04, 2.580375E+04, -2.584549E+04, 8.402950E+01,
     $    2.561132E+04, 2.552812E+04, -2.556954E+04, 8.338035E+01,
     $    2.533241E+04, 2.524985E+04, -2.529095E+04, 8.272336E+01,
     $    2.505098E+04, 2.496907E+04, -2.500985E+04, 8.205853E+01,
     $    2.476717E+04, 2.468592E+04, -2.472637E+04, 8.138630E+01,
     $    2.448112E+04, 2.440054E+04, -2.444066E+04, 8.070668E+01,
     $    2.419296E+04, 2.411305E+04, -2.415283E+04, 8.002011E+01,
     $    2.390282E+04, 2.382360E+04, -2.386304E+04, 7.932658E+01,
     $    2.361085E+04, 2.353231E+04, -2.357141E+04, 7.862655E+01,
     $    2.331715E+04, 2.323932E+04, -2.327807E+04, 7.792002E+01,
     $    2.302188E+04, 2.294475E+04, -2.298315E+04, 7.720742E+01,
     $    2.272515E+04, 2.264873E+04, -2.268678E+04, 7.648877E+01,
     $    2.242710E+04, 2.235140E+04, -2.238909E+04, 7.576448E+01,
     $    2.212785E+04, 2.205287E+04, -2.209020E+04, 7.503458E+01,
     $    2.182752E+04, 2.175328E+04, -2.179024E+04, 7.429947E+01,
     $    2.152624E+04, 2.145274E+04, -2.148933E+04, 7.355919E+01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 60, 78 )
     $  / 2.122413E+04, 2.115138E+04, -2.118760E+04, 7.281415E+01,
     $    2.092132E+04, 2.084931E+04, -2.088516E+04, 7.206436E+01,
     $    2.061791E+04, 2.054666E+04, -2.058214E+04, 7.131023E+01,
     $    2.031404E+04, 2.024355E+04, -2.027864E+04, 7.055179E+01,
     $    2.000981E+04, 1.994009E+04, -1.997480E+04, 6.978944E+01,
     $    1.970535E+04, 1.963640E+04, -1.967073E+04, 6.902321E+01,
     $    1.940076E+04, 1.933259E+04, -1.936653E+04, 6.825348E+01,
     $    1.909616E+04, 1.902877E+04, -1.906232E+04, 6.748030E+01,
     $    1.879167E+04, 1.872506E+04, -1.875822E+04, 6.670404E+01,
     $    1.848738E+04, 1.842156E+04, -1.845433E+04, 6.592476E+01,
     $    1.818342E+04, 1.811839E+04, -1.815076E+04, 6.514281E+01,
     $    1.787988E+04, 1.781565E+04, -1.784762E+04, 6.435826E+01,
     $    1.757687E+04, 1.751344E+04, -1.754502E+04, 6.357144E+01,
     $    1.727450E+04, 1.721186E+04, -1.724305E+04, 6.278245E+01,
     $    1.697286E+04, 1.691103E+04, -1.694181E+04, 6.199159E+01,
     $    1.667207E+04, 1.661104E+04, -1.664142E+04, 6.119897E+01,
     $    1.637221E+04, 1.631199E+04, -1.634196E+04, 6.040489E+01,
     $    1.607338E+04, 1.601397E+04, -1.604354E+04, 5.960945E+01,
     $    1.577568E+04, 1.571709E+04, -1.574626E+04, 5.881296E+01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 79, 97 )
     $  / 1.547921E+04, 1.542143E+04, -1.545019E+04, 5.801551E+01,
     $    1.518406E+04, 1.512709E+04, -1.515545E+04, 5.721740E+01,
     $    1.489031E+04, 1.483416E+04, -1.486211E+04, 5.641874E+01,
     $    1.459807E+04, 1.454273E+04, -1.457027E+04, 5.561980E+01,
     $    1.430740E+04, 1.425288E+04, -1.428002E+04, 5.482071E+01,
     $    1.401841E+04, 1.396470E+04, -1.399144E+04, 5.402173E+01,
     $    1.373117E+04, 1.367828E+04, -1.370461E+04, 5.322298E+01,
     $    1.344577E+04, 1.339370E+04, -1.341962E+04, 5.242473E+01,
     $    1.316229E+04, 1.311103E+04, -1.313654E+04, 5.162711E+01,
     $    1.288080E+04, 1.283036E+04, -1.285547E+04, 5.083035E+01,
     $    1.260139E+04, 1.255176E+04, -1.257647E+04, 5.003460E+01,
     $    1.232413E+04, 1.227532E+04, -1.229961E+04, 4.924009E+01,
     $    1.204909E+04, 1.200109E+04, -1.202498E+04, 4.844697E+01,
     $    1.177635E+04, 1.172915E+04, -1.175264E+04, 4.765544E+01,
     $    1.150597E+04, 1.145958E+04, -1.148267E+04, 4.686569E+01,
     $    1.123802E+04, 1.119244E+04, -1.121512E+04, 4.607790E+01,
     $    1.097257E+04, 1.092780E+04, -1.095008E+04, 4.529224E+01,
     $    1.070968E+04, 1.066571E+04, -1.068760E+04, 4.450889E+01,
     $    1.044942E+04, 1.040626E+04, -1.042774E+04, 4.372806E+01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 98, 116 )
     $  / 1.019185E+04, 1.014948E+04, -1.017056E+04, 4.294987E+01,
     $    9.937019E+03, 9.895445E+03, -9.916133E+03, 4.217455E+01,
     $    9.684992E+03, 9.644210E+03, -9.664504E+03, 4.140224E+01,
     $    9.435822E+03, 9.395829E+03, -9.415729E+03, 4.063314E+01,
     $    9.189562E+03, 9.150354E+03, -9.169864E+03, 3.986738E+01,
     $    8.946263E+03, 8.907837E+03, -8.926957E+03, 3.910518E+01,
     $    8.705975E+03, 8.668327E+03, -8.687059E+03, 3.834666E+01,
     $    8.468745E+03, 8.431870E+03, -8.450217E+03, 3.759204E+01,
     $    8.234618E+03, 8.198513E+03, -8.216476E+03, 3.684142E+01,
     $    8.003638E+03, 7.968297E+03, -7.985880E+03, 3.609503E+01,
     $    7.775846E+03, 7.741266E+03, -7.758469E+03, 3.535298E+01,
     $    7.551281E+03, 7.517456E+03, -7.534284E+03, 3.461546E+01,
     $    7.329982E+03, 7.296908E+03, -7.313361E+03, 3.388261E+01,
     $    7.111983E+03, 7.079654E+03, -7.095736E+03, 3.315461E+01,
     $    6.897319E+03, 6.865730E+03, -6.881443E+03, 3.243159E+01,
     $    6.686021E+03, 6.655167E+03, -6.670514E+03, 3.171373E+01,
     $    6.478118E+03, 6.447994E+03, -6.462977E+03, 3.100116E+01,
     $    6.273640E+03, 6.244238E+03, -6.258862E+03, 3.029403E+01,
     $    6.072610E+03, 6.043925E+03, -6.058192E+03, 2.959250E+01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 117, 135 )
     $  / 5.875053E+03, 5.847080E+03, -5.860992E+03, 2.889671E+01,
     $    5.680991E+03, 5.653722E+03, -5.667283E+03, 2.820681E+01,
     $    5.490443E+03, 5.463873E+03, -5.477086E+03, 2.752293E+01,
     $    5.303428E+03, 5.277549E+03, -5.290417E+03, 2.684523E+01,
     $    5.119961E+03, 5.094766E+03, -5.107294E+03, 2.617382E+01,
     $    4.940055E+03, 4.915538E+03, -4.927728E+03, 2.550887E+01,
     $    4.763723E+03, 4.739876E+03, -4.751732E+03, 2.485050E+01,
     $    4.590974E+03, 4.567790E+03, -4.579316E+03, 2.419883E+01,
     $    4.421816E+03, 4.399289E+03, -4.410488E+03, 2.355401E+01,
     $    4.256256E+03, 4.234376E+03, -4.245253E+03, 2.291615E+01,
     $    4.094296E+03, 4.073057E+03, -4.083614E+03, 2.228540E+01,
     $    3.935939E+03, 3.915333E+03, -3.925575E+03, 2.166186E+01,
     $    3.781185E+03, 3.761204E+03, -3.771135E+03, 2.104566E+01,
     $    3.630031E+03, 3.610667E+03, -3.620291E+03, 2.043693E+01,
     $    3.482475E+03, 3.463719E+03, -3.473039E+03, 1.983576E+01,
     $    3.338509E+03, 3.320353E+03, -3.329375E+03, 1.924230E+01,
     $    3.198126E+03, 3.180561E+03, -3.189289E+03, 1.865664E+01,
     $    3.061316E+03, 3.044334E+03, -3.052771E+03, 1.807890E+01,
     $    2.928067E+03, 2.911660E+03, -2.919811E+03, 1.750918E+01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 136, 154 )
     $  / 2.798366E+03, 2.782524E+03, -2.790394E+03, 1.694757E+01,
     $    2.672197E+03, 2.656912E+03, -2.664504E+03, 1.639421E+01,
     $    2.549543E+03, 2.534805E+03, -2.542125E+03, 1.584917E+01,
     $    2.430384E+03, 2.416184E+03, -2.423236E+03, 1.531256E+01,
     $    2.314698E+03, 2.301027E+03, -2.307816E+03, 1.478448E+01,
     $    2.202464E+03, 2.189312E+03, -2.195843E+03, 1.426499E+01,
     $    2.093656E+03, 2.081014E+03, -2.087290E+03, 1.375422E+01,
     $    1.988246E+03, 1.976105E+03, -1.982132E+03, 1.325223E+01,
     $    1.886208E+03, 1.874557E+03, -1.880340E+03, 1.275910E+01,
     $    1.787509E+03, 1.776339E+03, -1.781883E+03, 1.227493E+01,
     $    1.692118E+03, 1.681420E+03, -1.686729E+03, 1.179979E+01,
     $    1.600002E+03, 1.589764E+03, -1.594844E+03, 1.133373E+01,
     $    1.511123E+03, 1.501337E+03, -1.506192E+03, 1.087687E+01,
     $    1.425446E+03, 1.416101E+03, -1.420737E+03, 1.042922E+01,
     $    1.342930E+03, 1.334017E+03, -1.338438E+03, 9.990884E+00,
     $    1.263536E+03, 1.255043E+03, -1.259255E+03, 9.561919E+00,
     $    1.187220E+03, 1.179138E+03, -1.183145E+03, 9.142360E+00,
     $    1.113939E+03, 1.106257E+03, -1.110065E+03, 8.732276E+00,
     $    1.043646E+03, 1.036355E+03, -1.039969E+03, 8.331728E+00 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 155, 173 )
     $  / 9.762952E+02, 9.693834E+02, -9.728086E+02, 7.940736E+00,
     $    9.118373E+02, 9.052947E+02, -9.085363E+02, 7.559355E+00,
     $    8.502219E+02, 8.440382E+02, -8.471013E+02, 7.187642E+00,
     $    7.913973E+02, 7.855619E+02, -7.884518E+02, 6.825601E+00,
     $    7.353103E+02, 7.298127E+02, -7.325347E+02, 6.473269E+00,
     $    6.819062E+02, 6.767359E+02, -6.792952E+02, 6.130700E+00,
     $    6.311290E+02, 6.262754E+02, -6.286773E+02, 5.797890E+00,
     $    5.829214E+02, 5.783740E+02, -5.806237E+02, 5.474844E+00,
     $    5.372247E+02, 5.329729E+02, -5.350758E+02, 5.161612E+00,
     $    4.939790E+02, 4.900122E+02, -4.919735E+02, 4.858195E+00,
     $    4.531231E+02, 4.494309E+02, -4.512558E+02, 4.564561E+00,
     $    4.145949E+02, 4.111666E+02, -4.128604E+02, 4.280734E+00,
     $    3.783308E+02, 3.751560E+02, -3.767239E+02, 4.006736E+00,
     $    3.442664E+02, 3.413346E+02, -3.427819E+02, 3.742518E+00,
     $    3.123361E+02, 3.096370E+02, -3.109688E+02, 3.488043E+00,
     $    2.824737E+02, 2.799966E+02, -2.812182E+02, 3.243338E+00,
     $    2.546116E+02, 2.523463E+02, -2.534629E+02, 3.008383E+00,
     $    2.286816E+02, 2.266179E+02, -2.276344E+02, 2.783091E+00,
     $    2.046148E+02, 2.027424E+02, -2.036641E+02, 2.567416E+00 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 174, 192 )
     $  / 1.823415E+02, 1.806502E+02, -1.814822E+02, 2.361369E+00,
     $    1.617913E+02, 1.602710E+02, -1.610182E+02, 2.164906E+00,
     $    1.428932E+02, 1.415341E+02, -1.422015E+02, 1.977900E+00,
     $    1.255759E+02, 1.243680E+02, -1.249605E+02, 1.800266E+00,
     $    1.097676E+02, 1.087011E+02, -1.092237E+02, 1.631997E+00,
     $    9.539609E+01, 9.446129E+01, -9.491871E+01, 1.473049E+00,
     $    8.238871E+01, 8.157630E+01, -8.197322E+01, 1.323264E+00,
     $    7.067312E+01, 6.997376E+01, -7.031484E+01, 1.182473E+00,
     $    6.017693E+01, 5.958128E+01, -5.987117E+01, 1.050608E+00,
     $    5.082753E+01, 5.032644E+01, -5.056970E+01, 9.276491E-01,
     $    4.255225E+01, 4.213700E+01, -4.233796E+01, 8.134805E-01,
     $    3.527892E+01, 3.494109E+01, -3.510395E+01, 7.078544E-01,
     $    2.893623E+01, 2.866747E+01, -2.879639E+01, 6.105250E-01,
     $    2.345354E+01, 2.324545E+01, -2.334460E+01, 5.213733E-01,
     $    1.876049E+01, 1.860490E+01, -1.867835E+01, 4.403683E-01,
     $    1.478706E+01, 1.467636E+01, -1.472789E+01, 3.674050E-01,
     $    1.146435E+01, 1.139156E+01, -1.142463E+01, 3.021938E-01,
     $    8.725340E+00, 8.683876E+00, -8.701760E+00, 2.443164E-01,
     $    6.505261E+00, 6.488604E+00, -6.494540E+00, 1.933932E-01 /
      DATA  ( ( TestPM( J,K,3 ), K = 1, 4 ), J = 193, 200 )
     $  / 4.741188E+00, 4.742760E+00, -4.740013E+00, 1.492121E-01,
     $    3.371389E+00, 3.384791E+00, -3.376531E+00, 1.117126E-01,
     $    2.335117E+00, 2.354625E+00, -2.343675E+00, 8.083981E-02,
     $    1.573323E+00, 1.594384E+00, -1.582971E+00, 5.636657E-02,
     $    1.030227E+00, 1.049735E+00, -1.039355E+00, 3.778627E-02,
     $    6.552545E-01, 6.715251E-01, -6.629643E-01, 2.432051E-02,
     $    4.047166E-01, 4.172148E-01, -4.106883E-01, 1.501946E-02,
     $    2.427617E-01, 2.517182E-01, -2.470665E-01, 8.898844E-03 /

c ------------- Perfectly Conducting, Size Par = 10000 -----------------

      DATA  TestXX(4) / 10000. /
      DATA  TestCR(4) /(0.,0.)/
      DATA  TestAN(4) /.FALSE./
      DATA  TestIP(4) /-1234/

      DATA  TestQE(4) / 2.000289E+00 /,
     $      TestQS(4) / 2.000289E+00 /,
     $      TestGQ(4) / 1.000284E+00 /,
     $      TestSF(4) / (  5.000722E+07, -1.211372E+04 ) /,
     $      TestSB(4) / (  2.910127E+03, -4.065854E+03 ) /,
     $      TestTF( 1,4 ) / (  2.494890E+12, -4.356770E+12 ) /,
     $      TestTF( 2,4 ) / ( -2.494840E+12,  4.356770E+12 ) /,
     $      TestTB( 1,4 ) / ( -1.454564E+03,  1.661631E+03 ) /,
     $      TestTB( 2,4 ) / (  1.455564E+03, -2.404222E+03 ) /

      DATA ( TestS1( I,4 ), TestS2( I,4 ), I = 1, 7 ) /
     $ (  5.000722E+07,-1.211371E+04 ), (  5.000722E+07,-1.211373E+04 ),
     $ ( -4.114082E+03,-2.841914E+03 ), (  4.097371E+03, 2.864462E+03 ),
     $ ( -1.527121E+03, 4.761086E+03 ), (  1.529975E+03,-4.760161E+03 ),
     $ ( -4.836681E+03,-1.267485E+03 ), (  4.836501E+03, 1.268169E+03 ),
     $ ( -3.940804E+03, 3.077347E+03 ), (  3.940922E+03,-3.077195E+03 ),
     $ ( -3.800110E+03, 3.249487E+03 ), (  3.800134E+03,-3.249459E+03 ),
     $ (  2.910127E+03,-4.065854E+03 ), ( -2.910127E+03, 4.065854E+03 )/

c ---------------- Refr Index = 0.75, Size Par = 0.099 -----------------

      DATA  TestXX(5) / 0.099 /
      DATA  TestCR(5) /(0.75,0.)/
      DATA  TestAN(5) /.TRUE./
      DATA  TestIP(5) /0/

      DATA  TestQE(5) / 7.417859E-06 /,
     $      TestQS(5) / 7.417859E-06 /,
     $      TestGQ(5) / 1.074279E-08 /,
     $      TestSF(5) / (  1.817558E-08, -1.654225E-04 ) /,
     $      TestSB(5) / (  1.817558E-08, -1.648100E-04 ) /,
     $      TestTF( 1,5 ) / (  0.000000E+00,  2.938374E-08 ) /,
     $      TestTF( 2,5 ) / (  1.817558E-08, -1.654519E-04 ) /,
     $      TestTB( 1,5 ) / (  0.000000E+00,  2.938374E-08 ) /,
     $      TestTB( 2,5 ) / (  1.817558E-08, -1.647806E-04 ) /

      DATA ( TestS1( I,5 ), TestS2( I,5 ), I = 1, 7 ) /
     $ (  1.817558E-08,-1.654225E-04 ), (  1.817558E-08,-1.654225E-04 ),
     $ (  1.817558E-08,-1.653815E-04 ), (  1.574051E-08,-1.432172E-04 ),
     $ (  1.817558E-08,-1.652694E-04 ), (  9.087788E-09,-8.261265E-05 ),
     $ (  1.817558E-08,-1.651163E-04 ), (  9.797186E-23, 2.938374E-08 ),
     $ (  1.817558E-08,-1.649631E-04 ), ( -9.087788E-09, 8.250360E-05 ),
     $ (  1.817558E-08,-1.648510E-04 ), ( -1.574051E-08, 1.427725E-04 ),
     $ (  1.817558E-08,-1.648100E-04 ), ( -1.817558E-08, 1.648100E-04 )/

      DATA ( TestPM( J,1,5 ), J = 0, 1 ) / 1.000000E+00, 1.448233E-03 /

c ---------------- Refr Index = 0.75, Size Par = 0.101 -----------------

      DATA  TestXX(6) / 0.101 /
      DATA  TestCR(6) /(0.75,0.)/
      DATA  TestAN(6) /.TRUE./
      DATA  TestIP(6) /0/

      DATA  TestQE(6) / 8.033542E-06 /,
     $      TestQS(6) / 8.033542E-06 /,
     $      TestGQ(6) / 1.211000E-08 /,
     $      TestSF(6) / (  2.048754E-08, -1.756419E-04 ) /,
     $      TestSB(6) / (  2.048749E-08, -1.749650E-04 ) /,
     $      TestTF(1,6) / (  1.845061E-15,  3.232399E-08 ) /,
     $      TestTF(2,6) / (  2.048754E-08, -1.756742E-04 ) /,
     $      TestTB(1,6) / (  1.845052E-15,  3.262141E-08 ) /,
     $      TestTB(2,6) / (  2.048749E-08, -1.749324E-04 ) /

      DATA ( TestS1( I, 6), TestS2( I, 6), I = 1, 7 ) /
     $ (  2.048754E-08,-1.756419E-04 ), (  2.048754E-08,-1.756419E-04 ),
     $ (  2.048754E-08,-1.755965E-04 ), (  1.774273E-08,-1.520629E-04 ),
     $ (  2.048753E-08,-1.754726E-04 ), (  1.024377E-08,-8.771198E-05 ),
     $ (  2.048751E-08,-1.753033E-04 ), (  1.845057E-15, 3.247270E-08 ),
     $ (  2.048750E-08,-1.751341E-04 ), ( -1.024375E-08, 8.759147E-05 ),
     $ (  2.048749E-08,-1.750103E-04 ), ( -1.774269E-08, 1.515715E-04 ),
     $ (  2.048749E-08,-1.749650E-04 ), ( -2.048749E-08, 1.749650E-04 )/

      DATA ( TestPM( J,1,6 ), J = 0, 1 ) / 1.000000E+00, 1.507430E-03 /

c ---------------- Refr Index = 0.75, Size Par = 10    -----------------

      DATA  TestXX(7) / 10. /
      DATA  TestCR(7) /(0.75,0.)/
      DATA  TestAN(7) /.TRUE./
      DATA  TestIP(7) /0/

      DATA  TestQE(7) / 2.232265E+00 /,
     $      TestQS(7) / 2.232265E+00 /,
     $      TestGQ(7) / 2.001164E+00 /,
     $      TestSF(7) / (  5.580662E+01, -9.758097E+00 ) /,
     $      TestSB(7) / ( -1.078568E+00, -3.608807E-02 ) /,
     $      TestTF(1,7) / ( -6.931601E+01,  6.739155E+01 ) /,
     $      TestTF(2,7) / (  1.251226E+02, -7.714965E+01 ) /,
     $      TestTB(1,7) / (  1.061339E+00, -5.510357E-01 ) /,
     $      TestTB(2,7) / ( -1.722890E-02, -5.871237E-01 ) /

      DATA ( TestS1( I, 7), TestS2( I, 7), I = 1, 7 ) /
     $ (  5.580662E+01,-9.758097E+00 ), (  5.580662E+01,-9.758097E+00 ),
     $ ( -7.672879E+00, 1.087317E+01 ), ( -1.092923E+01, 9.629667E+00 ),
     $ (  3.587894E+00,-1.756177E+00 ), (  3.427411E+00, 8.082691E-02 ),
     $ ( -1.785905E+00,-5.232828E-02 ), ( -5.148748E-01,-7.027288E-01 ),
     $ (  1.537971E+00,-8.329374E-02 ), ( -6.908338E-01, 2.152693E-01 ),
     $ ( -4.140427E-01, 1.876851E-01 ), (  5.247557E-01,-1.923391E-01 ),
     $ ( -1.078568E+00,-3.608807E-02 ), (  1.078568E+00, 3.608807E-02 )/

      DATA  ( TestPM( J,1,7 ), J = 0, 20 ) /
     $    1.000000E+00,  8.964726E-01,   7.562458E-01,   6.160459E-01,
     $    4.968071E-01,  4.038391E-01,   3.351603E-01,   2.806743E-01,
     $    2.394767E-01,  2.053136E-01,   1.738309E-01,   1.468737E-01,
     $    1.239173E-01,  1.022173E-01,   8.059518E-02,   5.964501E-02,
     $    4.071085E-02,  2.520723E-02,   1.396914E-02,   6.861831E-03,
     $    2.969611E-03 /

c ---------------- Refr Index = 0.75, Size Par = 1000  -----------------

      DATA  TestXX(8) / 1000. /
      DATA  TestCR(8) /(0.75,0.)/
      DATA  TestAN(8) /.TRUE./
      DATA  TestIP(8) /0/

      DATA  TestQE(8) / 1.997908E+00 /,
     $      TestQS(8) / 1.997908E+00 /,
     $      TestGQ(8) / 1.688121E+00 /,
     $      TestSF(8) / (  4.994770E+05, -1.336502E+04 ) /,
     $      TestSB(8) / (  1.705778E+01,  4.842510E+02 ) /,
     $      TestTF(1,8) / ( -3.111072E+08, -1.536383E+08 ) /,
     $      TestTF(2,8) / (  3.116067E+08,  1.536249E+08 ) /,
     $      TestTB(1,8) / ( -1.825940E+06, -4.512978E+07 ) /,
     $      TestTB(2,8) / ( -1.825922E+06, -4.512930E+07 ) /

      DATA ( TestS1( I, 8), TestS2( I, 8), I = 1, 7 ) /
     $ (  4.994770E+05,-1.336502E+04 ), (  4.994770E+05,-1.336502E+04 ),
     $ ( -3.999296E+02,-3.316361E+02 ), ( -3.946018E+02,-1.147791E+02 ),
     $ ( -5.209852E+02,-5.776614E+02 ), ( -1.970767E+02,-6.937470E+02 ),
     $ ( -1.600887E+02, 1.348013E+02 ), ( -4.152365E+01, 1.143000E+02 ),
     $ (  8.431720E+01,-1.209493E+02 ), ( -4.261732E+01, 5.535055E+01 ),
     $ ( -7.556092E+01,-8.134810E+01 ), (  4.218303E+01, 9.100831E+01 ),
     $ (  1.705778E+01, 4.842510E+02 ), ( -1.705778E+01,-4.842510E+02 )/

      DATA ( TestPM( J,1,8 ), J = 0, 1 ) / 1.000000E+00, 8.449443E-01 /

c ----------- Refr Index = 1.33 + 1.E-5 I, Size Par = 1     ------------

      DATA  TestXX(9) / 1.0 /
      DATA  TestCR(9) /(1.33,-1.E-5)/
      DATA  TestAN(9) /.TRUE./
      DATA  TestIP(9) /1234/

      DATA  TestQE(9) / 9.395198E-02 /,
     $      TestQS(9) / 9.392330E-02 /,
     $      TestGQ(9) / 1.733048E-02 /,
     $      TestSF(9) / (  2.348800E-02,  2.281705E-01 ) /,
     $      TestSB(9) / (  2.243622E-02,  1.437106E-01 ) /,
     $      TestTF( 1,9 ) / (  2.737048E-04,  7.328549E-03 ) /,
     $      TestTF( 2,9 ) / (  2.321429E-02,  2.208419E-01 ) /,
     $      TestTB( 1,9 ) / (  2.722058E-04,  6.127286E-03 ) /,
     $      TestTB( 2,9 ) / (  2.270843E-02,  1.498379E-01 ) /

      DATA ( TestS1( I,9 ), TestS2( I,9 ), I = 1, 7 ) /
     $ (  2.348800E-02, 2.281705E-01 ), (  2.348800E-02, 2.281705E-01 ),
     $ (  2.341722E-02, 2.217102E-01 ), (  2.034830E-02, 1.938171E-01 ),
     $ (  2.322408E-02, 2.046815E-01 ), (  1.181704E-02, 1.075976E-01 ),
     $ (  2.296081E-02, 1.828349E-01 ), (  2.729533E-04, 6.702879E-03 ),
     $ (  2.269820E-02, 1.625401E-01 ), ( -1.114466E-02,-7.646326E-02 ),
     $ (  2.250635E-02, 1.486170E-01 ), ( -1.942300E-02,-1.271557E-01 ),
     $ (  2.243622E-02, 1.437106E-01 ), ( -2.243622E-02,-1.437106E-01 )/

      DATA ( ( TestPM( J,K,9 ), K = 1, 4 ), J = 0, 2 )
     $  / 1.487570E+00, 5.124299E-01, 2.568077E-01, 2.952825E-03,
     $    2.215468E-01, 1.474879E-01, 5.091695E-01, 3.218183E-05,
     $    1.661522E-02, 2.074039E-01, 8.211163E-02,-5.905264E-04 /

c ----------- Refr Index = 1.33 + 1.E-5 I, Size Par = 100   ------------

      DATA  TestXX(10) / 100. /
      DATA  TestCR(10) /(1.33,-1.E-5)/
      DATA  TestAN(10) /.TRUE./
      DATA  TestIP(10) /1234/

      DATA  TestQE(10) / 2.101321E+00 /,
     $      TestQS(10) / 2.096594E+00 /,
     $      TestGQ(10) / 1.821854E+00 /,
     $      TestSF(10) / (  5.253302E+03, -1.243188E+02 ) /,
     $      TestSB(10) / ( -5.659205E+01,  4.650974E+01 ) /,
     $      TestTF(1,10) / (  1.191653E+05, -2.036153E+05 ) /,
     $      TestTF(2,10) / ( -1.139120E+05,  2.034909E+05 ) /,
     $      TestTB(1,10) / ( -1.050411E+05, -5.686879E+04 ) /,
     $      TestTB(2,10) / ( -1.050976E+05, -5.682228E+04 ) /

      DATA ( TestS1( I,10 ), TestS2( I,10 ), I = 1, 7 ) /
     $ (  5.253302E+03,-1.243188E+02 ), (  5.253302E+03,-1.243188E+02 ),
     $ ( -5.534573E+01,-2.971881E+01 ), ( -8.467204E+01,-1.999470E+01 ),
     $ (  1.710488E+01,-1.520096E+01 ), (  3.310764E+01,-2.709787E+00 ),
     $ ( -3.655758E+00, 8.769860E+00 ), ( -6.550512E+00,-4.675370E+00 ),
     $ (  2.414318E+00, 5.380874E-01 ), (  6.039011E+00,-1.169971E+01 ),
     $ ( -1.222996E+00, 3.283917E+01 ), ( -9.653812E+00, 1.474455E+01 ),
     $ ( -5.659205E+01, 4.650974E+01 ), (  5.659205E+01,-4.650974E+01 )/

      DATA ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 0, 2 )
     $  / 9.995849E-01, 1.000415E+00, 9.523065E-01,  3.582337E-04,
     $    8.447503E-01, 8.931682E-01, 8.799091E-01, -1.045906E-02,
     $    7.945251E-01, 8.027264E-01, 7.877438E-01, -7.806765E-03 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 3, 21 )
     $  / 6.789430E-01, 6.815073E-01, 6.840159E-01, -1.032598E-02,
     $    5.972651E-01, 6.059801E-01, 6.028274E-01, -1.395129E-02,
     $    5.624393E-01, 5.553640E-01, 5.586651E-01, -1.061297E-02,
     $    5.120082E-01, 5.253760E-01, 5.209300E-01, -1.241309E-02,
     $    5.021030E-01, 5.101969E-01, 5.054351E-01, -1.181134E-02,
     $    4.889266E-01, 4.965281E-01, 4.910913E-01, -8.660254E-03,
     $    4.800820E-01, 4.873224E-01, 4.829032E-01, -7.373238E-03,
     $    4.839518E-01, 4.799784E-01, 4.794064E-01, -7.360156E-03,
     $    4.720456E-01, 4.728458E-01, 4.725975E-01, -6.717067E-03,
     $    4.696421E-01, 4.693685E-01, 4.689486E-01, -1.055553E-02,
     $    4.612040E-01, 4.651512E-01, 4.626937E-01, -8.855766E-03,
     $    4.506855E-01, 4.625771E-01, 4.571017E-01, -9.455134E-03,
     $    4.510960E-01, 4.595032E-01, 4.547127E-01, -7.842258E-03,
     $    4.447861E-01, 4.563520E-01, 4.506138E-01, -5.082292E-03,
     $    4.467789E-01, 4.530136E-01, 4.494322E-01, -5.755999E-03,
     $    4.476215E-01, 4.493673E-01, 4.479687E-01, -4.694914E-03,
     $    4.415127E-01, 4.459308E-01, 4.439251E-01, -5.816043E-03,
     $    4.423395E-01, 4.423087E-01, 4.416089E-01, -7.792757E-03,
     $    4.336742E-01, 4.388880E-01, 4.367592E-01, -6.992143E-03 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 22, 40 )
     $  / 4.285400E-01, 4.359443E-01, 4.322145E-01, -8.822026E-03,
     $    4.270653E-01, 4.322202E-01, 4.290776E-01, -7.114439E-03,
     $    4.193550E-01, 4.298165E-01, 4.247034E-01, -6.576343E-03,
     $    4.196903E-01, 4.260331E-01, 4.222957E-01, -6.386136E-03,
     $    4.162984E-01, 4.233608E-01, 4.192999E-01, -4.090015E-03,
     $    4.124746E-01, 4.202232E-01, 4.170137E-01, -4.937778E-03,
     $    4.142517E-01, 4.169612E-01, 4.150707E-01, -4.197578E-03,
     $    4.094004E-01, 4.141994E-01, 4.121591E-01, -4.381808E-03,
     $    4.090947E-01, 4.111617E-01, 4.095052E-01, -6.131894E-03,
     $    4.045404E-01, 4.078631E-01, 4.066631E-01, -4.941448E-03,
     $    4.003085E-01, 4.054457E-01, 4.026587E-01, -6.566076E-03,
     $    3.985000E-01, 4.012164E-01, 3.999551E-01, -6.064701E-03,
     $    3.921546E-01, 3.991089E-01, 3.953624E-01, -5.783780E-03,
     $    3.896156E-01, 3.948902E-01, 3.921516E-01, -6.169847E-03,
     $    3.855280E-01, 3.918521E-01, 3.885351E-01, -4.293832E-03,
     $    3.812036E-01, 3.883826E-01, 3.851642E-01, -4.994268E-03,
     $    3.807302E-01, 3.846780E-01, 3.818572E-01, -4.039497E-03,
     $    3.750872E-01, 3.818950E-01, 3.788207E-01, -3.460506E-03,
     $    3.746271E-01, 3.779513E-01, 3.763690E-01, -4.039379E-03 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 41, 59 )
     $  / 3.721309E-01, 3.752643E-01, 3.734249E-01, -3.127165E-03,
     $    3.690160E-01, 3.725264E-01, 3.709203E-01, -4.350159E-03,
     $    3.683538E-01, 3.691432E-01, 3.684734E-01, -3.421263E-03,
     $    3.637944E-01, 3.670758E-01, 3.657826E-01, -4.406350E-03,
     $    3.626010E-01, 3.640914E-01, 3.630660E-01, -4.814285E-03,
     $    3.590485E-01, 3.617985E-01, 3.603798E-01, -4.035500E-03,
     $    3.561556E-01, 3.589323E-01, 3.576552E-01, -5.568425E-03,
     $    3.532404E-01, 3.563548E-01, 3.546944E-01, -4.385983E-03,
     $    3.493035E-01, 3.538534E-01, 3.514895E-01, -4.732075E-03,
     $    3.471550E-01, 3.501964E-01, 3.485423E-01, -4.714461E-03,
     $    3.418253E-01, 3.479135E-01, 3.451915E-01, -4.013773E-03,
     $    3.403170E-01, 3.441176E-01, 3.416255E-01, -4.189094E-03,
     $    3.359575E-01, 3.406880E-01, 3.386390E-01, -3.398879E-03,
     $    3.323424E-01, 3.376832E-01, 3.349504E-01, -3.719924E-03,
     $    3.304026E-01, 3.333423E-01, 3.316171E-01, -2.612400E-03,
     $    3.257603E-01, 3.304013E-01, 3.281255E-01, -3.204250E-03,
     $    3.232661E-01, 3.260713E-01, 3.248287E-01, -2.583383E-03,
     $    3.200688E-01, 3.229935E-01, 3.211107E-01, -2.341935E-03,
     $    3.167832E-01, 3.188905E-01, 3.182038E-01, -3.019105E-03 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 60, 78 )
     $  / 3.139395E-01, 3.158611E-01, 3.147032E-01, -2.068368E-03,
     $    3.111474E-01, 3.123057E-01, 3.116391E-01, -3.044328E-03,
     $    3.081761E-01, 3.089873E-01, 3.087771E-01, -2.566343E-03,
     $    3.051895E-01, 3.063833E-01, 3.055811E-01, -2.924465E-03,
     $    3.030173E-01, 3.027913E-01, 3.030020E-01, -3.066068E-03,
     $    2.994814E-01, 3.008100E-01, 3.001103E-01, -3.095194E-03,
     $    2.976411E-01, 2.975463E-01, 2.975639E-01, -3.376917E-03,
     $    2.944765E-01, 2.955194E-01, 2.950285E-01, -3.291160E-03,
     $    2.921859E-01, 2.930186E-01, 2.925503E-01, -3.703656E-03,
     $    2.897395E-01, 2.906970E-01, 2.902975E-01, -3.340017E-03,
     $    2.872493E-01, 2.887943E-01, 2.878840E-01, -3.925839E-03,
     $    2.850228E-01, 2.863315E-01, 2.858330E-01, -3.367139E-03,
     $    2.825949E-01, 2.846208E-01, 2.834324E-01, -3.806079E-03,
     $    2.803696E-01, 2.821584E-01, 2.813483E-01, -3.428502E-03,
     $    2.777169E-01, 2.803622E-01, 2.790138E-01, -3.441842E-03,
     $    2.756625E-01, 2.778702E-01, 2.767013E-01, -3.417522E-03,
     $    2.727181E-01, 2.759331E-01, 2.743935E-01, -3.078134E-03,
     $    2.706746E-01, 2.732882E-01, 2.719591E-01, -3.238887E-03,
     $    2.678559E-01, 2.712530E-01, 2.694449E-01, -2.794093E-03 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 79, 97 )
     $  / 2.653419E-01, 2.683880E-01, 2.671205E-01, -2.917043E-03,
     $    2.630181E-01, 2.662351E-01, 2.642137E-01, -2.524618E-03,
     $    2.598357E-01, 2.632354E-01, 2.620197E-01, -2.542145E-03,
     $    2.578496E-01, 2.608216E-01, 2.588472E-01, -2.229440E-03,
     $    2.543848E-01, 2.578829E-01, 2.565092E-01, -2.191007E-03,
     $    2.521887E-01, 2.550455E-01, 2.534196E-01, -1.914809E-03,
     $    2.490068E-01, 2.523084E-01, 2.506351E-01, -1.899736E-03,
     $    2.462057E-01, 2.490255E-01, 2.478456E-01, -1.600144E-03,
     $    2.434746E-01, 2.464241E-01, 2.445867E-01, -1.657179E-03,
     $    2.401875E-01, 2.428752E-01, 2.419426E-01, -1.309921E-03,
     $    2.375180E-01, 2.401586E-01, 2.384939E-01, -1.420600E-03,
     $    2.342026E-01, 2.365834E-01, 2.356295E-01, -1.075575E-03,
     $    2.311456E-01, 2.335572E-01, 2.322609E-01, -1.148854E-03,
     $    2.280056E-01, 2.300299E-01, 2.290324E-01, -9.140005E-04,
     $    2.246089E-01, 2.267461E-01, 2.257037E-01, -8.487138E-04,
     $    2.213884E-01, 2.231696E-01, 2.223289E-01, -7.911832E-04,
     $    2.180887E-01, 2.198334E-01, 2.188205E-01, -5.940307E-04,
     $    2.144702E-01, 2.161396E-01, 2.155878E-01, -6.363867E-04,
     $    2.115096E-01, 2.128669E-01, 2.118645E-01, -4.414654E-04 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 98, 116 )
     $  / 2.076785E-01, 2.091643E-01, 2.087427E-01, -4.267700E-04,
     $    2.046785E-01, 2.058644E-01, 2.051055E-01, -3.464121E-04,
     $    2.012085E-01, 2.023197E-01, 2.017778E-01, -2.299778E-04,
     $    1.976799E-01, 1.988720E-01, 1.984689E-01, -1.979954E-04,
     $    1.946983E-01, 1.954671E-01, 1.948192E-01, -1.097558E-04,
     $    1.908021E-01, 1.919000E-01, 1.916519E-01, 3.393462E-05,
     $    1.877404E-01, 1.884691E-01, 1.879483E-01, -1.407312E-05,
     $    1.841257E-01, 1.849111E-01, 1.845484E-01, 2.224420E-04,
     $    1.804708E-01, 1.813980E-01, 1.810905E-01, 1.714146E-04,
     $    1.773579E-01, 1.779139E-01, 1.774681E-01, 2.664812E-04,
     $    1.735387E-01, 1.744809E-01, 1.741551E-01, 4.274608E-04,
     $    1.703423E-01, 1.710595E-01, 1.707466E-01, 2.614075E-04,
     $    1.671323E-01, 1.678353E-01, 1.673135E-01, 5.970349E-04,
     $    1.634673E-01, 1.644989E-01, 1.643038E-01, 3.759911E-04,
     $    1.607455E-01, 1.613880E-01, 1.608152E-01, 6.334488E-04,
     $    1.571618E-01, 1.581972E-01, 1.578278E-01, 5.519724E-04,
     $    1.540164E-01, 1.550354E-01, 1.546420E-01, 7.068248E-04,
     $    1.511567E-01, 1.520652E-01, 1.513660E-01, 5.631457E-04,
     $    1.475722E-01, 1.488521E-01, 1.485803E-01, 9.010070E-04 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 117, 135 )
     $  / 1.449985E-01, 1.462536E-01, 1.453569E-01, 4.227853E-04,
     $    1.420848E-01, 1.430852E-01, 1.427560E-01, 9.543891E-04,
     $    1.390428E-01, 1.409933E-01, 1.400431E-01, 4.709293E-04,
     $    1.372080E-01, 1.379679E-01, 1.375553E-01, 6.427734E-04,
     $    1.341746E-01, 1.361658E-01, 1.351846E-01, 7.770630E-04,
     $    1.321492E-01, 1.335398E-01, 1.330344E-01, 2.582762E-04,
     $    1.303520E-01, 1.315231E-01, 1.306440E-01, 8.565113E-04,
     $    1.273212E-01, 1.296798E-01, 1.289174E-01, 2.708071E-04,
     $    1.267682E-01, 1.273624E-01, 1.267611E-01, 3.933092E-04,
     $    1.239231E-01, 1.262468E-01, 1.252486E-01, 5.456689E-04,
     $    1.231692E-01, 1.242379E-01, 1.238012E-01, -1.115411E-04,
     $    1.219090E-01, 1.231646E-01, 1.223398E-01, 5.221940E-04,
     $    1.202426E-01, 1.219258E-01, 1.213895E-01, -4.848166E-05,
     $    1.199982E-01, 1.204752E-01, 1.200574E-01, 9.702411E-05,
     $    1.183344E-01, 1.196218E-01, 1.190994E-01, 2.810419E-04,
     $    1.177012E-01, 1.182852E-01, 1.179996E-01, -2.418280E-04,
     $    1.168562E-01, 1.172056E-01, 1.170749E-01, 2.394537E-04,
     $    1.158114E-01, 1.163717E-01, 1.160336E-01, -1.472556E-04,
     $    1.150709E-01, 1.149709E-01, 1.152369E-01, -2.253299E-05 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 136, 154 )
     $  / 1.141796E-01, 1.140186E-01, 1.139056E-01, 1.415609E-04,
     $    1.126489E-01, 1.124908E-01, 1.128068E-01, -9.784970E-06,
     $    1.116178E-01, 1.108055E-01, 1.111221E-01, 2.598811E-04,
     $    1.096126E-01, 1.091337E-01, 1.094518E-01, 1.864455E-04,
     $    1.079839E-01, 1.069198E-01, 1.074750E-01, 3.113435E-04,
     $    1.057828E-01, 1.047209E-01, 1.053099E-01, 3.546645E-04,
     $    1.033632E-01, 1.021174E-01, 1.027397E-01, 5.487967E-04,
     $    1.005870E-01, 9.915312E-02, 9.992035E-02, 5.744165E-04,
     $    9.733366E-02, 9.598357E-02, 9.675418E-02, 8.066974E-04,
     $    9.423961E-02, 9.259746E-02, 9.330841E-02, 8.186609E-04,
     $    9.043551E-02, 8.903802E-02, 8.996492E-02, 9.730151E-04,
     $    8.724657E-02, 8.547441E-02, 8.627434E-02, 1.074889E-03,
     $    8.341788E-02, 8.187046E-02, 8.267164E-02, 1.214905E-03,
     $    7.972947E-02, 7.825409E-02, 7.919444E-02, 1.327325E-03,
     $    7.658993E-02, 7.502436E-02, 7.559587E-02, 1.326868E-03,
     $    7.312210E-02, 7.193637E-02, 7.275659E-02, 1.449368E-03,
     $    7.061673E-02, 6.939711E-02, 6.994805E-02, 1.391364E-03,
     $    6.819616E-02, 6.719189E-02, 6.776884E-02, 1.431023E-03,
     $    6.635408E-02, 6.545222E-02, 6.588103E-02, 1.444612E-03 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 155, 173 )
     $  / 6.478013E-02, 6.420487E-02, 6.460952E-02, 1.243774E-03,
     $    6.407945E-02, 6.330800E-02, 6.364697E-02, 1.227181E-03,
     $    6.342738E-02, 6.276716E-02, 6.312223E-02, 1.119597E-03,
     $    6.288886E-02, 6.228993E-02, 6.274136E-02, 1.062416E-03,
     $    6.293332E-02, 6.180225E-02, 6.222439E-02, 9.328585E-04,
     $    6.214278E-02, 6.112306E-02, 6.169997E-02, 1.022651E-03,
     $    6.127478E-02, 6.003827E-02, 6.082519E-02, 9.934555E-04,
     $    6.018617E-02, 5.854743E-02, 5.922570E-02, 1.116725E-03,
     $    5.795936E-02, 5.628836E-02, 5.731062E-02, 1.373701E-03,
     $    5.530106E-02, 5.369662E-02, 5.451291E-02, 1.479113E-03,
     $    5.209658E-02, 5.057324E-02, 5.141134E-02, 1.652876E-03,
     $    4.865372E-02, 4.714350E-02, 4.790925E-02, 1.760545E-03,
     $    4.481607E-02, 4.379385E-02, 4.442044E-02, 1.858751E-03,
     $    4.154238E-02, 4.072577E-02, 4.112842E-02, 1.687397E-03,
     $    3.900776E-02, 3.817058E-02, 3.850909E-02, 1.308682E-03,
     $    3.716128E-02, 3.627569E-02, 3.677365E-02, 1.243747E-03,
     $    3.621635E-02, 3.511295E-02, 3.579543E-02, 8.889020E-04,
     $    3.640263E-02, 3.477418E-02, 3.542574E-02, 6.748867E-04,
     $    3.708752E-02, 3.453406E-02, 3.587837E-02, 6.532192E-04 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 174, 192 )
     $  / 3.740174E-02, 3.435052E-02, 3.606735E-02, 9.399562E-04,
     $    3.717382E-02, 3.381498E-02, 3.563995E-02, 1.741458E-03,
     $    3.571078E-02, 3.265442E-02, 3.413288E-02, 2.083867E-03,
     $    3.287571E-02, 3.061930E-02, 3.187954E-02, 2.443558E-03,
     $    2.953252E-02, 2.786366E-02, 2.884003E-02, 2.907303E-03,
     $    2.510936E-02, 2.476234E-02, 2.505902E-02, 2.806322E-03,
     $    2.136666E-02, 2.217549E-02, 2.146864E-02, 2.283584E-03,
     $    1.924913E-02, 2.024489E-02, 1.950213E-02, 7.567186E-04,
     $    1.946434E-02, 1.929070E-02, 1.936778E-02, -5.678368E-04,
     $    2.184175E-02, 1.886144E-02, 2.058025E-02, -3.934122E-04,
     $    2.301588E-02, 1.841400E-02, 2.099236E-02, 5.088029E-04,
     $    2.225193E-02, 1.745389E-02, 2.004729E-02, 2.015813E-03,
     $    1.981356E-02, 1.615055E-02, 1.790569E-02, 2.867106E-03,
     $    1.594732E-02, 1.440932E-02, 1.529377E-02, 2.631456E-03,
     $    1.391439E-02, 1.307891E-02, 1.329721E-02, 2.320481E-03,
     $    1.218588E-02, 1.175814E-02, 1.185775E-02, 1.588275E-03,
     $    1.065619E-02, 1.057426E-02, 1.066656E-02, 1.025125E-03,
     $    1.048762E-02, 9.430109E-03, 9.993569E-03, 9.751590E-04,
     $    9.701559E-03, 8.171617E-03, 9.038013E-03, 6.089439E-04 /
      DATA  ( ( TestPM( J,K,10 ), K = 1, 4 ), J = 193, 200 )
     $  / 9.475186E-03, 7.013667E-03, 8.450630E-03, 9.389697E-04,
     $    9.763063E-03, 6.328942E-03, 7.833635E-03, 1.236078E-03,
     $    8.369626E-03, 5.347505E-03, 6.842162E-03, 1.380403E-03,
     $    7.152996E-03, 4.936310E-03, 5.880238E-03, 1.711945E-03,
     $    5.456549E-03, 3.951940E-03, 4.593511E-03, 1.478353E-03,
     $    3.399460E-03, 3.085708E-03, 3.306811E-03, 1.122921E-03,
     $    2.777909E-03, 2.301019E-03, 2.650776E-03, 9.293232E-04,
     $    2.720015E-03, 1.588991E-03, 2.149507E-03, 5.267358E-04 /

c ----------- Refr Index = 1.33 + 1.E-5 I, Size Par = 10000 ------------

      DATA  TestXX(11) / 10000. /
      DATA  TestCR(11) /(1.33,-1.E-5)/
      DATA  TestAN(11) /.TRUE./
      DATA  TestIP(11) /1234/

      DATA  TestQE(11) / 2.004089E+00 /,
     $      TestQS(11) / 1.723857E+00 /,
     $      TestGQ(11) / 1.564987E+00 /,
     $      TestSF(11) / (  5.010222E+07, -1.535815E+05 ) /,
     $      TestSB(11) / ( -1.821194E+02, -9.519122E+02 ) /,
     $      TestTF(1,11) / (  2.704468E+10, -6.309326E+10 ) /,
     $      TestTF(2,11) / ( -2.699457E+10,  6.309311E+10 ) /,
     $      TestTB(1,11) / (  1.765444E+10,  1.708970E+10 ) /,
     $      TestTB(2,11) / (  1.765444E+10,  1.708970E+10 ) /

      DATA ( TestS1( I,11 ), TestS2( I,11 ), I = 1, 7 ) /
     $ (  5.010222E+07,-1.535815E+05 ), (  5.010222E+07,-1.535815E+05 ),
     $ (  3.786814E+03,-7.654293E+03 ), (  5.074755E+03,-7.515986E+03 ),
     $ ( -2.731172E+03, 1.326633E+03 ), ( -3.076558E+03,-1.775975E+02 ),
     $ ( -1.061003E+03,-1.930155E+02 ), (  2.430920E+02, 8.409836E+01 ),
     $ ( -1.058140E+03, 2.298414E+01 ), (  5.906487E+01,-5.370283E+02 ),
     $ ( -2.748855E+03, 2.298181E+03 ), ( -8.036201E+01,-4.939186E+00 ),
     $ ( -1.821193E+02,-9.519122E+02 ), (  1.821194E+02, 9.519123E+02 )/

      DATA ( ( TestPM( J,K,11 ), K = 1, 4 ), J = 0, 1 )
     $  / 1.001787E+00, 9.982126E-01, 9.673274E-01, -5.900902E-05,
     $    8.933875E-01, 9.222932E-01, 9.156430E-01, -8.957813E-05 /

c -------------- Refr Index = 1.50 + I, Size Par = 0.055 ---------------

      DATA  TestXX(12) / 0.055 /
      DATA  TestCR(12) /(1.5,-1.0)/
      DATA  TestAN(12) /.FALSE./
      DATA  TestIP(12) /-1234/

      DATA  TestQE(12) / 1.014910E-01 /,
     $      TestQS(12) / 1.131687E-05 /,
     $      TestGQ(12) / 5.558541E-09 /,
     $      TestSF(12) / (  7.675259E-05,  8.343879E-05 ) /,
     $      TestSB(12) / (  7.661398E-05,  8.338145E-05 ) /,
     $      TestTF(1,12) / (  3.132066E-08, -2.037399E-08 ) /,
     $      TestTF(2,12) / (  7.672127E-05,  8.345916E-05 ) /,
     $      TestTB(1,12) / (  3.132066E-08, -2.037399E-08 ) /,
     $      TestTB(2,12) / (  7.664530E-05,  8.336107E-05 ) /

      DATA ( TestS1( I,12 ), TestS2( I,12 ), I = 1, 7 ) /
     $ (  7.675259E-05, 8.343879E-05 ), (  7.675259E-05, 8.343879E-05 ),
     $ (  7.674331E-05, 8.343495E-05 ), (  6.646948E-05, 7.225169E-05 ),
     $ (  7.671794E-05, 8.342445E-05 ), (  3.838246E-05, 4.169695E-05 ),
     $ (  7.668328E-05, 8.341012E-05 ), (  3.132066E-08,-2.037399E-08 ),
     $ (  7.664863E-05, 8.339578E-05 ), ( -3.830082E-05,-4.171317E-05 ),
     $ (  7.662326E-05, 8.338529E-05 ), ( -6.634986E-05,-7.221887E-05 ),
     $ (  7.661398E-05, 8.338145E-05 ), ( -7.661398E-05,-8.338145E-05 )/

      DATA ( ( TestPM( J,K,12 ), K = 1, 4 ), J = 0, 1 )
     $  / 1.631246E-07, 1.500000E+00, 8.206852E-05, 4.878029E-04,
     $    0.000000E+00, 5.455757E-04, 7.419554E-09, 8.997385E-08 /

c -------------- Refr Index = 1.50 + I, Size Par = 0.056 ---------------

      DATA  TestXX(13) / 0.056 /
      DATA  TestCR(13) /(1.5,-1.0)/
      DATA  TestAN(13) /.FALSE./
      DATA  TestIP(13) /-1234/

      DATA  TestQE(13) / 1.033467E-01 /,
     $      TestQS(13) / 1.216311E-05 /,
     $      TestGQ(13) / 6.193255E-09 /,
     $      TestSF(13) / (  8.102381E-05,  8.807251E-05 ) /,
     $      TestSB(13) / (  8.087213E-05,  8.800976E-05 ) /,
     $      TestTF(1,13) / (  3.428921E-08, -2.229495E-08 ) /,
     $      TestTF(2,13) / (  8.098952E-05,  8.809480E-05 ) /,
     $      TestTB(1,13) / (  3.425632E-08, -2.229767E-08 ) /,
     $      TestTB(2,13) / (  8.090638E-05,  8.798746E-05 ) /

      DATA ( TestS1( I,13 ), TestS2( I,13 ), I = 1, 7 ) /
     $ (  8.102381E-05, 8.807251E-05 ), (  8.102381E-05, 8.807251E-05 ),
     $ (  8.101364E-05, 8.806830E-05 ), (  7.016844E-05, 7.626381E-05 ),
     $ (  8.098587E-05, 8.805682E-05 ), (  4.051865E-05, 4.401169E-05 ),
     $ (  8.094795E-05, 8.804113E-05 ), (  3.427277E-08,-2.229631E-08 ),
     $ (  8.091003E-05, 8.802545E-05 ), ( -4.042932E-05,-4.402945E-05 ),
     $ (  8.088228E-05, 8.801396E-05 ), ( -7.003755E-05,-7.622790E-05 ),
     $ (  8.087213E-05, 8.800976E-05 ), ( -8.087213E-05,-8.800976E-05 )/

      DATA ( ( TestPM( J,K,13 ), K = 1, 4 ), J = 0, 1 )
     $  / 1.753114E-07, 1.500000E+00, 8.508061E-05, 5.056958E-04,
     $    3.727529E-11, 5.655788E-04, 5.869632E-08, 1.434434E-07 /

c -------------- Refr Index = 1.50 + I, Size Par = 1     ---------------

      DATA  TestXX(14) / 1.0 /
      DATA  TestCR(14) /(1.5,-1.0)/
      DATA  TestAN(14) /.FALSE./
      DATA  TestIP(14) /-1234/

      DATA  TestQE(14) / 2.336321E+00 /,
     $      TestQS(14) / 6.634538E-01 /,
     $      TestGQ(14) / 1.274736E-01 /,
     $      TestSF(14) / (  5.840802E-01,  1.905153E-01 ) /,
     $      TestSB(14) / (  3.488438E-01,  1.468286E-01 ) /,
     $      TestTF(1,14) / (  4.176586E-02, -6.670919E-02 ) /,
     $      TestTF(2,14) / (  5.423144E-01,  2.572245E-01 ) /,
     $      TestTB(1,14) / (  3.116882E-02, -5.728985E-02 ) /,
     $      TestTB(2,14) / (  3.800126E-01,  8.953879E-02 ) /

      DATA ( TestS1( I,14 ), TestS2( I,14 ), I = 1, 7 ) /
     $ (  5.840802E-01, 1.905153E-01 ), (  5.840802E-01, 1.905153E-01 ),
     $ (  5.657020E-01, 1.871997E-01 ), (  5.001610E-01, 1.456112E-01 ),
     $ (  5.175251E-01, 1.784426E-01 ), (  2.879639E-01, 4.105398E-02 ),
     $ (  4.563396E-01, 1.671665E-01 ), (  3.622847E-02,-6.182646E-02 ),
     $ (  4.002117E-01, 1.566427E-01 ), ( -1.748750E-01,-1.229586E-01 ),
     $ (  3.621572E-01, 1.493910E-01 ), ( -3.056823E-01,-1.438460E-01 ),
     $ (  3.488438E-01, 1.468286E-01 ), ( -3.488438E-01,-1.468286E-01 )/

      DATA ( ( TestPM( J,K,14 ), K = 1, 4 ), J = 0, 2 )
     $  / 3.113808E-02, 1.464409E+00, 3.717382E-02, 2.095679E-01,
     $    1.946190E-03, 2.071948E-01, -1.225825E-03, 2.237764E-02,
     $    7.156107E-05, 1.618887E-02, -7.574185E-05, 1.181909E-03 /

c -------------- Refr Index = 1.50 + I, Size Par = 100   ---------------

      DATA  TestXX(15) / 100. /
      DATA  TestCR(15) /(1.5,-1.0)/
      DATA  TestAN(15) /.FALSE./
      DATA  TestIP(15) /-1234/

      DATA  TestQE(15) / 2.097502E+00 /,
     $      TestQS(15) / 1.283697E+00 /,
     $      TestGQ(15) / 1.091466E+00 /,
     $      TestSF(15) / (  5.243754E+03, -2.934167E+02 ) /,
     $      TestSB(15) / ( -2.029360E+01,  4.384435E+00 ) /,
     $      TestTF(1,15) / ( -1.516411E+05, -1.696388E+05 ) /,
     $      TestTF(2,15) / (  1.568849E+05,  1.693454E+05 ) /,
     $      TestTB(1,15) / (  4.785763E+00, -4.253510E+00 ) /,
     $      TestTB(2,15) / ( -1.550784E+01,  1.309257E-01 ) /

      DATA ( TestS1( I,15 ), TestS2( I,15 ), I = 1, 7 ) /
     $ (  5.243754E+03,-2.934167E+02 ), (  5.243754E+03,-2.934167E+02 ),
     $ (  4.049055E+01,-1.898456E+01 ), (  2.019198E+01, 3.110731E+00 ),
     $ ( -2.646835E+01,-1.929564E+01 ), (  9.152743E+00,-7.470202E+00 ),
     $ (  1.268890E+01, 2.397474E+01 ), ( -1.232914E+01,-7.823167E+00 ),
     $ (  5.149886E+00, 2.290736E+01 ), ( -7.173357E+00,-1.655464E+01 ),
     $ ( -1.605395E+01, 1.418642E+01 ), (  1.448052E+01,-1.393594E+01 ),
     $ ( -2.029360E+01, 4.384435E+00 ), (  2.029360E+01,-4.384435E+00 )/

      DATA ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 0, 2 )
     $  / 2.066055E+03, 2.114566E+03, -2.089842E+03, 2.646901E+01,
     $    2.065349E+03, 2.113673E+03, -2.089080E+03, 2.653086E+01,
     $    2.064031E+03, 2.112208E+03, -2.087703E+03, 2.657271E+01 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 3, 21 )
     $  / 2.062136E+03, 2.110184E+03, -2.085751E+03, 2.659845E+01,
     $    2.059684E+03, 2.107606E+03, -2.083241E+03, 2.661326E+01,
     $    2.056686E+03, 2.104482E+03, -2.080183E+03, 2.661989E+01,
     $    2.053152E+03, 2.100818E+03, -2.076588E+03, 2.661970E+01,
     $    2.049092E+03, 2.096624E+03, -2.072464E+03, 2.661337E+01,
     $    2.044517E+03, 2.091909E+03, -2.067822E+03, 2.660129E+01,
     $    2.039437E+03, 2.086683E+03, -2.062672E+03, 2.658370E+01,
     $    2.033860E+03, 2.080954E+03, -2.057022E+03, 2.656077E+01,
     $    2.027797E+03, 2.074733E+03, -2.050883E+03, 2.653263E+01,
     $    2.021257E+03, 2.068030E+03, -2.044264E+03, 2.649939E+01,
     $    2.014251E+03, 2.060853E+03, -2.037175E+03, 2.646116E+01,
     $    2.006787E+03, 2.053214E+03, -2.029627E+03, 2.641801E+01,
     $    1.998876E+03, 2.045122E+03, -2.021627E+03, 2.637005E+01,
     $    1.990527E+03, 2.036586E+03, -2.013188E+03, 2.631734E+01,
     $    1.981750E+03, 2.027616E+03, -2.004317E+03, 2.625996E+01,
     $    1.972554E+03, 2.018222E+03, -1.995025E+03, 2.619800E+01,
     $    1.962949E+03, 2.008414E+03, -1.985321E+03, 2.613152E+01,
     $    1.952945E+03, 1.998200E+03, -1.975215E+03, 2.606060E+01,
     $    1.942551E+03, 1.987592E+03, -1.964716E+03, 2.598532E+01 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 22, 40 )
     $  / 1.931777E+03, 1.976598E+03, -1.953834E+03, 2.590573E+01,
     $    1.920632E+03, 1.965228E+03, -1.942579E+03, 2.582192E+01,
     $    1.909125E+03, 1.953491E+03, -1.930960E+03, 2.573396E+01,
     $    1.897266E+03, 1.941398E+03, -1.918987E+03, 2.564191E+01,
     $    1.885064E+03, 1.928957E+03, -1.906668E+03, 2.554585E+01,
     $    1.872530E+03, 1.916177E+03, -1.894013E+03, 2.544585E+01,
     $    1.859671E+03, 1.903070E+03, -1.881033E+03, 2.534196E+01,
     $    1.846497E+03, 1.889642E+03, -1.867735E+03, 2.523428E+01,
     $    1.833018E+03, 1.875905E+03, -1.854129E+03, 2.512286E+01,
     $    1.819242E+03, 1.861867E+03, -1.840225E+03, 2.500776E+01,
     $    1.805179E+03, 1.847537E+03, -1.826031E+03, 2.488907E+01,
     $    1.790838E+03, 1.832925E+03, -1.811557E+03, 2.476685E+01,
     $    1.776228E+03, 1.818040E+03, -1.796812E+03, 2.464117E+01,
     $    1.761358E+03, 1.802891E+03, -1.781804E+03, 2.451209E+01,
     $    1.746236E+03, 1.787486E+03, -1.766544E+03, 2.437969E+01,
     $    1.730873E+03, 1.771836E+03, -1.751040E+03, 2.424403E+01,
     $    1.715276E+03, 1.755948E+03, -1.735300E+03, 2.410517E+01,
     $    1.699454E+03, 1.739833E+03, -1.719334E+03, 2.396319E+01,
     $    1.683417E+03, 1.723497E+03, -1.703150E+03, 2.381816E+01 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 41, 59 )
     $  / 1.667173E+03, 1.706952E+03, -1.686758E+03, 2.367014E+01,
     $    1.650730E+03, 1.690204E+03, -1.670165E+03, 2.351920E+01,
     $    1.634097E+03, 1.673264E+03, -1.653381E+03, 2.336540E+01,
     $    1.617283E+03, 1.656138E+03, -1.636413E+03, 2.320881E+01,
     $    1.600296E+03, 1.638837E+03, -1.619272E+03, 2.304950E+01,
     $    1.583144E+03, 1.621369E+03, -1.601964E+03, 2.288753E+01,
     $    1.565836E+03, 1.603741E+03, -1.584499E+03, 2.272298E+01,
     $    1.548381E+03, 1.585963E+03, -1.566885E+03, 2.255590E+01,
     $    1.530786E+03, 1.568042E+03, -1.549129E+03, 2.238637E+01,
     $    1.513059E+03, 1.549987E+03, -1.531241E+03, 2.221444E+01,
     $    1.495209E+03, 1.531806E+03, -1.513228E+03, 2.204019E+01,
     $    1.477244E+03, 1.513508E+03, -1.495098E+03, 2.186367E+01,
     $    1.459171E+03, 1.495099E+03, -1.476860E+03, 2.168497E+01,
     $    1.440999E+03, 1.476589E+03, -1.458522E+03, 2.150413E+01,
     $    1.422734E+03, 1.457985E+03, -1.440090E+03, 2.132122E+01,
     $    1.404386E+03, 1.439295E+03, -1.421573E+03, 2.113631E+01,
     $    1.385962E+03, 1.420527E+03, -1.402979E+03, 2.094947E+01,
     $    1.367469E+03, 1.401687E+03, -1.384315E+03, 2.076075E+01,
     $    1.348914E+03, 1.382785E+03, -1.365590E+03, 2.057023E+01 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 60, 78 )
     $  / 1.330306E+03, 1.363827E+03, -1.346809E+03, 2.037796E+01,
     $    1.311651E+03, 1.344821E+03, -1.327981E+03, 2.018400E+01,
     $    1.292957E+03, 1.325775E+03, -1.309113E+03, 1.998843E+01,
     $    1.274231E+03, 1.306694E+03, -1.290212E+03, 1.979131E+01,
     $    1.255480E+03, 1.287588E+03, -1.271286E+03, 1.959269E+01,
     $    1.236712E+03, 1.268462E+03, -1.252341E+03, 1.939264E+01,
     $    1.217932E+03, 1.249325E+03, -1.233385E+03, 1.919123E+01,
     $    1.199148E+03, 1.230182E+03, -1.214424E+03, 1.898851E+01,
     $    1.180367E+03, 1.211040E+03, -1.195466E+03, 1.878455E+01,
     $    1.161596E+03, 1.191908E+03, -1.176516E+03, 1.857941E+01,
     $    1.142840E+03, 1.172790E+03, -1.157581E+03, 1.837315E+01,
     $    1.124107E+03, 1.153694E+03, -1.138669E+03, 1.816583E+01,
     $    1.105403E+03, 1.134626E+03, -1.119786E+03, 1.795751E+01,
     $    1.086734E+03, 1.115593E+03, -1.100937E+03, 1.774825E+01,
     $    1.068106E+03, 1.096601E+03, -1.082129E+03, 1.753812E+01,
     $    1.049526E+03, 1.077656E+03, -1.063369E+03, 1.732718E+01,
     $    1.031000E+03, 1.058764E+03, -1.044663E+03, 1.711547E+01,
     $    1.012533E+03, 1.039931E+03, -1.026015E+03, 1.690308E+01,
     $    9.941319E+02, 1.021164E+03, -1.007433E+03, 1.669004E+01 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 79, 97 )
     $  / 9.758019E+02, 1.002468E+03, -9.889227E+02, 1.647643E+01,
     $    9.575488E+02, 9.838487E+02, -9.704888E+02, 1.626229E+01,
     $    9.393781E+02, 9.653118E+02, -9.521374E+02, 1.604770E+01,
     $    9.212952E+02, 9.468630E+02, -9.338739E+02, 1.583270E+01,
     $    9.033057E+02, 9.285076E+02, -9.157038E+02, 1.561736E+01,
     $    8.854146E+02, 9.102511E+02, -8.976323E+02, 1.540173E+01,
     $    8.676272E+02, 8.920985E+02, -8.796646E+02, 1.518587E+01,
     $    8.499485E+02, 8.740551E+02, -8.618059E+02, 1.496984E+01,
     $    8.323835E+02, 8.561259E+02, -8.440611E+02, 1.475370E+01,
     $    8.149370E+02, 8.383158E+02, -8.264350E+02, 1.453749E+01,
     $    7.976137E+02, 8.206296E+02, -8.089326E+02, 1.432128E+01,
     $    7.804183E+02, 8.030720E+02, -7.915584E+02, 1.410512E+01,
     $    7.633553E+02, 7.856477E+02, -7.743170E+02, 1.388906E+01,
     $    7.464291E+02, 7.683611E+02, -7.572129E+02, 1.367317E+01,
     $    7.296441E+02, 7.512166E+02, -7.402504E+02, 1.345750E+01,
     $    7.130044E+02, 7.342185E+02, -7.234338E+02, 1.324209E+01,
     $    6.965141E+02, 7.173709E+02, -7.067671E+02, 1.302701E+01,
     $    6.801773E+02, 7.006780E+02, -6.902544E+02, 1.281231E+01,
     $    6.639977E+02, 6.841435E+02, -6.738997E+02, 1.259804E+01 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 98, 116 )
     $  / 6.479791E+02, 6.677715E+02, -6.577066E+02, 1.238425E+01,
     $    6.321252E+02, 6.515655E+02, -6.416789E+02, 1.217099E+01,
     $    6.164395E+02, 6.355293E+02, -6.258202E+02, 1.195833E+01,
     $    6.009255E+02, 6.196662E+02, -6.101338E+02, 1.174630E+01,
     $    5.855864E+02, 6.039797E+02, -5.946232E+02, 1.153497E+01,
     $    5.704254E+02, 5.884730E+02, -5.792916E+02, 1.132437E+01,
     $    5.554456E+02, 5.731493E+02, -5.641420E+02, 1.111457E+01,
     $    5.406500E+02, 5.580116E+02, -5.491776E+02, 1.090560E+01,
     $    5.260414E+02, 5.430628E+02, -5.344011E+02, 1.069753E+01,
     $    5.116226E+02, 5.283058E+02, -5.198153E+02, 1.049040E+01,
     $    4.973961E+02, 5.137431E+02, -5.054228E+02, 1.028425E+01,
     $    4.833645E+02, 4.993774E+02, -4.912263E+02, 1.007915E+01,
     $    4.695301E+02, 4.852110E+02, -4.772281E+02, 9.875120E+00,
     $    4.558952E+02, 4.712464E+02, -4.634305E+02, 9.672225E+00,
     $    4.424620E+02, 4.574858E+02, -4.498357E+02, 9.470508E+00,
     $    4.292324E+02, 4.439312E+02, -4.364457E+02, 9.270014E+00,
     $    4.162084E+02, 4.305845E+02, -4.232625E+02, 9.070790E+00,
     $    4.033918E+02, 4.174477E+02, -4.102879E+02, 8.872880E+00,
     $    3.907842E+02, 4.045225E+02, -3.975236E+02, 8.676329E+00 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 117, 135 )
     $  / 3.783872E+02, 3.918105E+02, -3.849712E+02, 8.481182E+00,
     $    3.662023E+02, 3.793132E+02, -3.726322E+02, 8.287481E+00,
     $    3.542307E+02, 3.670320E+02, -3.605078E+02, 8.095271E+00,
     $    3.424736E+02, 3.549681E+02, -3.485994E+02, 7.904594E+00,
     $    3.309322E+02, 3.431226E+02, -3.369080E+02, 7.715491E+00,
     $    3.196074E+02, 3.314966E+02, -3.254346E+02, 7.528005E+00,
     $    3.084999E+02, 3.200909E+02, -3.141801E+02, 7.342176E+00,
     $    2.976106E+02, 3.089064E+02, -3.031452E+02, 7.158044E+00,
     $    2.869400E+02, 2.979437E+02, -2.923306E+02, 6.975650E+00,
     $    2.764887E+02, 2.872033E+02, -2.817367E+02, 6.795033E+00,
     $    2.662569E+02, 2.766856E+02, -2.713640E+02, 6.616231E+00,
     $    2.562450E+02, 2.663910E+02, -2.612127E+02, 6.439283E+00,
     $    2.464531E+02, 2.563196E+02, -2.512830E+02, 6.264225E+00,
     $    2.368811E+02, 2.464715E+02, -2.415749E+02, 6.091094E+00,
     $    2.275289E+02, 2.368465E+02, -2.320883E+02, 5.919927E+00,
     $    2.183964E+02, 2.274446E+02, -2.228230E+02, 5.750760E+00,
     $    2.094832E+02, 2.182655E+02, -2.137788E+02, 5.583626E+00,
     $    2.007888E+02, 2.093087E+02, -2.049551E+02, 5.418559E+00,
     $    1.923127E+02, 2.005736E+02, -1.963514E+02, 5.255595E+00 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 136, 154 )
     $  / 1.840541E+02, 1.920597E+02, -1.879671E+02, 5.094765E+00,
     $    1.760123E+02, 1.837662E+02, -1.798013E+02, 4.936101E+00,
     $    1.681863E+02, 1.756922E+02, -1.718532E+02, 4.779634E+00,
     $    1.605751E+02, 1.678367E+02, -1.641217E+02, 4.625396E+00,
     $    1.531776E+02, 1.601986E+02, -1.566058E+02, 4.473416E+00,
     $    1.459925E+02, 1.527766E+02, -1.493041E+02, 4.323723E+00,
     $    1.390184E+02, 1.455696E+02, -1.422154E+02, 4.176346E+00,
     $    1.322539E+02, 1.385759E+02, -1.353381E+02, 4.031311E+00,
     $    1.256973E+02, 1.317940E+02, -1.286707E+02, 3.888647E+00,
     $    1.193470E+02, 1.252224E+02, -1.222115E+02, 3.748378E+00,
     $    1.132011E+02, 1.188591E+02, -1.159587E+02, 3.610531E+00,
     $    1.072578E+02, 1.127023E+02, -1.099104E+02, 3.475128E+00,
     $    1.015149E+02, 1.067501E+02, -1.040646E+02, 3.342194E+00,
     $    9.597044E+01, 1.010002E+02, -9.841918E+01, 3.211750E+00,
     $    9.062212E+01, 9.545064E+01, -9.297195E+01, 3.083820E+00,
     $    8.546763E+01, 9.009894E+01, -8.772057E+01, 2.958423E+00,
     $    8.050452E+01, 8.494275E+01, -8.266263E+01, 2.835578E+00,
     $    7.573026E+01, 7.997955E+01, -7.779559E+01, 2.715306E+00,
     $    7.114223E+01, 7.520674E+01, -7.311684E+01, 2.597623E+00 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 155, 173 )
     $  / 6.673769E+01, 7.062161E+01, -6.862367E+01, 2.482546E+00,
     $    6.251383E+01, 6.622135E+01, -6.431326E+01, 2.370092E+00,
     $    5.846774E+01, 6.200307E+01, -6.018270E+01, 2.260274E+00,
     $    5.459639E+01, 5.796377E+01, -5.622900E+01, 2.153107E+00,
     $    5.089671E+01, 5.410038E+01, -5.244907E+01, 2.048603E+00,
     $    4.736551E+01, 5.040973E+01, -4.883972E+01, 1.946774E+00,
     $    4.399951E+01, 4.688855E+01, -4.539771E+01, 1.847629E+00,
     $    4.079537E+01, 4.353350E+01, -4.211967E+01, 1.751178E+00,
     $    3.774966E+01, 4.034116E+01, -3.900218E+01, 1.657429E+00,
     $    3.485886E+01, 3.730802E+01, -3.604174E+01, 1.566388E+00,
     $    3.211939E+01, 3.443050E+01, -3.323474E+01, 1.478061E+00,
     $    2.952758E+01, 3.170493E+01, -3.057754E+01, 1.392451E+00,
     $    2.707970E+01, 2.912758E+01, -2.806640E+01, 1.309562E+00,
     $    2.477195E+01, 2.669465E+01, -2.569752E+01, 1.229395E+00,
     $    2.260047E+01, 2.440227E+01, -2.346702E+01, 1.151949E+00,
     $    2.056132E+01, 2.224650E+01, -2.137097E+01, 1.077224E+00,
     $    1.865051E+01, 2.022333E+01, -1.940539E+01, 1.005214E+00,
     $    1.686401E+01, 1.832873E+01, -1.756622E+01, 9.359171E-01,
     $    1.519772E+01, 1.655857E+01, -1.584935E+01, 8.693257E-01 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 174, 192 )
     $  / 1.364749E+01, 1.490869E+01, -1.425063E+01, 8.054318E-01,
     $    1.220912E+01, 1.337489E+01, -1.276587E+01, 7.442252E-01,
     $    1.087840E+01, 1.195290E+01, -1.139082E+01, 6.856947E-01,
     $    9.651052E+00, 1.063845E+01, -1.012120E+01, 6.298273E-01,
     $    8.522775E+00, 9.427200E+00, -8.952690E+00, 5.766067E-01,
     $    7.489244E+00, 8.314792E+00, -7.880956E+00, 5.260149E-01,
     $    6.546110E+00, 7.296845E+00, -6.901627E+00, 4.780327E-01,
     $    5.689004E+00, 6.368952E+00, -6.010317E+00, 4.326392E-01,
     $    4.913545E+00, 5.526694E+00, -5.202625E+00, 3.898095E-01,
     $    4.215351E+00, 4.765642E+00, -4.474142E+00, 3.495151E-01,
     $    3.590038E+00, 4.081362E+00, -3.820461E+00, 3.117268E-01,
     $    3.033225E+00, 3.469424E+00, -3.237177E+00, 2.764146E-01,
     $    2.540545E+00, 2.925406E+00, -2.719891E+00, 2.435445E-01,
     $    2.107653E+00, 2.444899E+00, -2.264228E+00, 2.130761E-01,
     $    1.730239E+00, 2.023525E+00, -1.865840E+00, 1.849644E-01,
     $    1.404030E+00, 1.656939E+00, -1.520420E+00, 1.591650E-01,
     $    1.124794E+00, 1.340834E+00, -1.223699E+00, 1.356336E-01,
     $    8.883479E-01, 1.070948E+00, -9.714523E-01, 1.143214E-01,
     $    6.905777E-01, 8.430805E-01, -7.595178E-01, 9.516821E-02 /
      DATA  ( ( TestPM( J,K,15 ), K = 1, 4 ), J = 193, 200 )
     $  / 5.274665E-01, 6.531074E-01, -5.838220E-01, 7.810233E-02,
     $    3.951048E-01, 4.970033E-01, -4.403971E-01, 6.304792E-02,
     $    2.896890E-01, 3.708482E-01, -3.253840E-01, 4.993323E-02,
     $    2.075141E-01, 2.708263E-01, -2.350242E-01, 3.869198E-02,
     $    1.449839E-01, 1.932274E-01, -1.656624E-01, 2.925373E-02,
     $    9.865233E-02, 1.344689E-01, -1.137753E-01, 2.152747E-02,
     $    6.529631E-02, 9.114575E-02, -7.603556E-02, 1.538685E-02,
     $    4.200180E-02, 6.010505E-02, -4.939533E-02, 1.066472E-02 /

c -------------- Refr Index = 1.50 + I, Size Par = 10000 ---------------

      DATA  TestXX(16) / 10000. /
      DATA  TestCR(16) /(1.5,-1.0)/
      DATA  TestAN(16) /.FALSE./
      DATA  TestIP(16) /-1234/

      DATA  TestQE(16) / 2.004368E+00 /,
     $      TestQS(16) / 1.236574E+00 /,
     $      TestGQ(16) / 1.046525E+00 /,
     $      TestSF(16) / (  5.010919E+07, -1.753404E+05 ) /,
     $      TestSB(16) / ( -2.184719E+02, -2.064610E+03 ) /,
     $      TestTF(1,16) / ( -1.474433E+11, -1.602157E+11 ) /,
     $      TestTF(2,16) / (  1.474934E+11,  1.602155E+11 ) /,
     $      TestTB(1,16) / ( -4.935264E+03,  1.723872E+04 ) /,
     $      TestTB(2,16) / ( -5.153736E+03,  1.517411E+04 ) /

      DATA ( TestS1( I,16 ), TestS2( I,16 ), I = 1, 7 ) /
     $ (  5.010919E+07,-1.753404E+05 ), (  5.010919E+07,-1.753404E+05 ),
     $ ( -3.690394E+03,-1.573897E+03 ), ( -9.333175E+02,-1.839736E+03 ),
     $ (  2.391551E+02, 3.247786E+03 ), ( -1.202951E+03,-1.899647E+02 ),
     $ ( -2.607463E+03, 7.414859E+02 ), (  1.013073E+03,-1.064666E+03 ),
     $ ( -6.183154E+02, 2.264970E+03 ), (  1.334826E+02,-1.800859E+03 ),
     $ ( -3.368019E+02, 2.115750E+03 ), (  2.293862E+02,-1.996754E+03 ),
     $ ( -2.184719E+02,-2.064610E+03 ), (  2.184719E+02, 2.064610E+03 )/

c -------------- Refr Index = 10 + 10 I, Size Par = 1     --------------

      DATA  TestXX(17) / 1.0 /
      DATA  TestCR(17) /(10.,-10.)/
      DATA  TestAN(17) /.TRUE./
      DATA  TestIP(17) /0/

      DATA  TestQE(17) / 2.532993E+00 /,
     $      TestQS(17) / 2.049405E+00 /,
     $      TestGQ(17) /  -2.267961E-01 /,
     $      TestSF(17) / (  6.332483E-01,  4.179305E-01 ) /,
     $      TestSB(17) / (  4.485464E-01,  7.912365E-01 ) /,
     $      TestTF(1,17) / (  9.729778E-02, -4.218849E-01 ) /,
     $      TestTF(2,17) / (  5.359505E-01,  8.398154E-01 ) /,
     $      TestTB(1,17) / (  6.377872E-02, -2.752983E-01 ) /,
     $      TestTB(2,17) / (  5.123252E-01,  5.159382E-01 ) /

      DATA ( TestS1( I,17 ), TestS2( I,17 ), I = 1, 7 ) /
     $ (  6.332483E-01, 4.179305E-01 ), (  6.332483E-01, 4.179305E-01 ),
     $ (  6.162264E-01, 4.597163E-01 ), (  5.573186E-01, 2.954338E-01 ),
     $ (  5.736317E-01, 5.602514E-01 ), (  3.525107E-01,-5.921611E-03 ),
     $ (  5.238628E-01, 6.675352E-01 ), (  7.881172E-02,-3.435544E-01 ),
     $ (  4.825816E-01, 7.434033E-01 ), ( -1.881212E-01,-6.028739E-01 ),
     $ (  4.570214E-01, 7.809867E-01 ), ( -3.793898E-01,-7.473279E-01 ),
     $ (  4.485464E-01, 7.912365E-01 ), ( -4.485464E-01,-7.912365E-01 )/

      DATA ( TestPM( J,1,17 ), J = 0, 2 ) /
     $    1.000000E+00, -1.106644E-01, 7.188010E-02 /

c -------------- Refr Index = 10 + 10 I, Size Par = 100   --------------

      DATA  TestXX(18) / 100. /
      DATA  TestCR(18) /(10.,-10.)/
      DATA  TestAN(18) /.TRUE./
      DATA  TestIP(18) /0/

      DATA  TestQE(18) / 2.071124E+00 /,
     $      TestQS(18) / 1.836785E+00 /,
     $      TestGQ(18) / 1.021648E+00 /,
     $      TestSF(18) / (  5.177811E+03, -2.633811E+01 ) /,
     $      TestSB(18) / ( -4.145383E+01, -1.821808E+01 ) /,
     $      TestTF(1,18) / (  8.146500E+04, -9.742027E+05 ) /,
     $      TestTF(2,18) / ( -7.628719E+04,  9.741764E+05 ) /,
     $      TestTB(1,18) / ( -3.989342E+01, -6.134686E+01 ) /,
     $      TestTB(2,18) / ( -8.134725E+01, -7.956494E+01 ) /

      DATA ( TestS1( I,18 ), TestS2( I,18 ), I = 1, 7 ) /
     $ (  5.177811E+03,-2.633811E+01 ), (  5.177811E+03,-2.633811E+01 ),
     $ (  5.227436E+01,-1.270012E+01 ), ( -2.380252E+01,-3.872567E-01 ),
     $ ( -2.705712E+01,-3.951751E+01 ), (  2.585821E+01, 3.323624E+01 ),
     $ (  1.008860E+00, 4.663027E+01 ), ( -3.479935E+00,-4.364245E+01 ),
     $ ( -1.505640E+01, 4.333057E+01 ), (  1.360634E+01,-4.238302E+01 ),
     $ ( -4.510770E+01, 5.199554E+00 ), (  4.474564E+01,-5.452513E+00 ),
     $ ( -4.145383E+01,-1.821808E+01 ), (  4.145383E+01, 1.821808E+01 )/

      DATA  ( TestPM( J,1,18 ), J = 0, 59 )
     $  / 1.000000E+00,  5.562155E-01,   5.574387E-01,   5.561997E-01,
     $    5.532963E-01,  5.497068E-01,   5.458596E-01,   5.419399E-01,
     $    5.380247E-01,  5.341422E-01,   5.302998E-01,   5.264964E-01,
     $    5.227281E-01,  5.189900E-01,   5.152780E-01,   5.115884E-01,
     $    5.079181E-01,  5.042644E-01,   5.006253E-01,   4.969991E-01,
     $    4.933844E-01,  4.897800E-01,
     $    4.861848E-01,  4.825982E-01,   4.790194E-01,   4.754479E-01,
     $    4.718832E-01,  4.683249E-01,   4.647726E-01,   4.612262E-01,
     $    4.576854E-01,  4.541500E-01,   4.506198E-01,   4.470947E-01,
     $    4.435746E-01,  4.400594E-01,   4.365491E-01,   4.330436E-01,
     $    4.295429E-01,  4.260469E-01,   4.225557E-01,
     $    4.190692E-01,  4.155875E-01,   4.121105E-01,   4.086384E-01,
     $    4.051710E-01,  4.017085E-01,   3.982510E-01,   3.947983E-01,
     $    3.913507E-01,  3.879081E-01,   3.844705E-01,   3.810382E-01,
     $    3.776111E-01,  3.741892E-01,   3.707728E-01,   3.673617E-01,
     $    3.639562E-01,  3.605562E-01,   3.571619E-01 /
      DATA  ( TestPM( J,1,18 ), J = 60, 116 )
     $  / 3.537733E-01,  3.503905E-01,   3.470136E-01,   3.436427E-01,
     $    3.402778E-01,  3.369191E-01,   3.335667E-01,   3.302205E-01,
     $    3.268807E-01,  3.235475E-01,   3.202208E-01,   3.169009E-01,
     $    3.135877E-01,  3.102814E-01,   3.069822E-01,   3.036899E-01,
     $    3.004049E-01,  2.971271E-01,   2.938567E-01,
     $    2.905938E-01,  2.873384E-01,   2.840909E-01,   2.808510E-01,
     $    2.776192E-01,  2.743952E-01,   2.711794E-01,   2.679719E-01,
     $    2.647726E-01,  2.615820E-01,   2.583996E-01,   2.552263E-01,
     $    2.520614E-01,  2.489058E-01,   2.457589E-01,   2.426215E-01,
     $    2.394930E-01,  2.363742E-01,   2.332647E-01,
     $    2.301651E-01,  2.270750E-01,   2.239950E-01,   2.209248E-01,
     $    2.178650E-01,  2.148153E-01,   2.117762E-01,   2.087474E-01,
     $    2.057296E-01,  2.027223E-01,   1.997264E-01,   1.967410E-01,
     $    1.937676E-01,  1.908047E-01,   1.878543E-01,   1.849145E-01,
     $    1.819878E-01,  1.790717E-01,   1.761691E-01 /
      DATA  ( TestPM( J,1,18 ), J = 117, 173 )
     $  / 1.732775E-01,   1.703995E-01,   1.675333E-01,   1.646801E-01,
     $    1.618402E-01,   1.590124E-01,   1.561996E-01,   1.533979E-01,
     $    1.506123E-01,   1.478383E-01,   1.450796E-01,   1.423351E-01,
     $    1.396033E-01,   1.368889E-01,   1.341857E-01,   1.315004E-01,
     $    1.288289E-01,   1.261716E-01,   1.235332E-01,
     $    1.209059E-01,   1.182984E-01,   1.157059E-01,   1.131271E-01,
     $    1.105704E-01,   1.080249E-01,   1.054987E-01,   1.029928E-01,
     $    1.004974E-01,   9.802628E-02,   9.557211E-02,   9.313022E-02,
     $    9.071602E-02,   8.831573E-02,   8.593020E-02,   8.357484E-02,
     $    8.123152E-02,   7.890444E-02,   7.661032E-02,
     $    7.432789E-02,   7.206079E-02,   6.983035E-02,   6.761393E-02,
     $    6.540846E-02,   6.324279E-02,   6.109900E-02,   5.895903E-02,
     $    5.685623E-02,   5.479080E-02,   5.272664E-02,   5.068375E-02,
     $    4.869383E-02,   4.672252E-02,   4.474852E-02,   4.281780E-02,
     $    4.094328E-02,   3.907284E-02,   3.720106E-02 /
      DATA  ( TestPM( J,1,18 ), J = 174, 200 )
     $  / 3.538717E-02,   3.363679E-02,   3.188645E-02,   3.012554E-02,
     $    2.842342E-02,   2.680798E-02,   2.521286E-02,   2.358672E-02,
     $    2.198245E-02,   2.048203E-02,   1.907381E-02,   1.766455E-02,
     $    1.620655E-02,   1.476618E-02,   1.344893E-02,   1.227834E-02,
     $    1.116845E-02,   1.001140E-02,   8.782895E-03,
     $    7.567168E-03,   6.491992E-03,   5.639304E-03,   5.000030E-03,
     $    4.492685E-03,   4.017973E-03,   3.508363E-03,   2.948243E-03 /

c -------------- Refr Index = 10 + 10 I, Size Par = 10000 --------------

      DATA  TestXX(19) / 10000. /
      DATA  TestCR(19) /(10.,-10.)/
      DATA  TestAN(19) /.TRUE./
      DATA  TestIP(19) /0/

      DATA  TestQE(19) / 2.005914E+00 /,
     $      TestQS(19) / 1.795393E+00 /,
     $      TestGQ(19) / 9.842238E-01 /,
     $      TestSF(19) / (  5.014786E+07, -1.206004E+05 ) /,
     $      TestSB(19) / (  2.252480E+03, -3.924468E+03 ) /,
     $      TestTF(1,19) / ( -1.034867E+12, -1.624318E+12 ) /,
     $      TestTF(2,19) / (  1.034918E+12,  1.624318E+12 ) /,
     $      TestTB(1,19) / ( -1.347624E+04, -1.445821E+03 ) /,
     $      TestTB(2,19) / ( -1.122376E+04, -5.370289E+03 ) /

      DATA ( TestS1( I,19 ), TestS2( I,19 ), I = 1, 7 ) /
     $ (  5.014786E+07,-1.206004E+05 ), (  5.014786E+07,-1.206004E+05 ),
     $ ( -4.080090E+03,-2.664399E+03 ), (  3.351286E+03, 7.291906E+02 ),
     $ ( -1.224040E+03, 4.596569E+03 ), (  4.497446E+02,-4.072999E+03 ),
     $ ( -4.579490E+03,-8.590486E+02 ), (  4.313394E+03, 4.969719E+02 ),
     $ ( -3.356286E+03, 3.125121E+03 ), (  3.171910E+03,-3.129068E+03 ),
     $ ( -3.149584E+03, 3.270358E+03 ), (  3.105243E+03,-3.269355E+03 ),
     $ (  2.252480E+03,-3.924468E+03 ), ( -2.252480E+03, 3.924468E+03 )/

      END

