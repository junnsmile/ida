program main
    integer i,j,iband,m,n,K,POINT
    !H8BAND:葵花八号模拟波段数；ARENETP:研究区内aeronet输入站点个数
    integer,parameter::H8BAND=7,nlon= 501,nlat = 401,ARENETP=3
!    integer,parameter::AODPOINTS_NUM=7,SSAAFCpoint_num =4,H8BAND=7,nlon= 501,nlat = 401,ARENETP=3
    real aod_result(H8BAND,ARENETP),ssa_result(H8BAND,ARENETP),afc_result(H8BAND,ARENETP),aod_tem(H8BAND),ssa_tem(H8BAND),afc_tem(H8BAND),aod_z(H8BAND),ssa_z(H8BAND),afc_z(H8BAND)
    real lon(nlon),lat(nlat),lonh,lath,tem
    real(kind =8) aod_point,bi
    character(len=100) filename
    integer::start2(2)=(/1,1/)
    integer::counts2(2)=(/nlon,nlat/)
    integer::strid2(2)=(/1,1/)
    integer::start3(3)=(/1,1,1/)
    integer::counts3(3)=(/nlon,nlat,H8BAND/)
    integer::strid3(3)=(/1,1,1/)
    real(8) aod_all(nlon,nlat),dis(ARENETP)
    real::lons(ARENETP) = (/121.185,120.496,120.545/)
    real::lats(ARENETP)=(/24.968,23.49,23.712/)
    integer::AODPOINTS_NUM(ARENETP) = (/7,7,7/)
    integer::SSApoint_num(ARENETP) = (/4,4,4/)
    integer::AFCpoint_num(ARENETP) = (/12,12,12/)
    real wave_len(16,ARENETP),aod(16,ARENETP),wave_len2(16,ARENETP),ssa(16,ARENETP),wave_len3(16,ARENETP),afc(16,ARENETP)
    real waod(16),wssa(16),wafc(16)
    integer ierr,ncid,varid,latvarid,lonvarid,bandvarid,lon_dimid,lat_dimid,band_dimid,dimids(3)
    integer::band(7)=(/1,2,3,4,5,6,7 /)
    real (8) AODD_ALL(H8BAND,nlon,nlat),SSA_ALL(H8BAND,nlon,nlat),AFC_ALL(H8BAND,nlon,nlat)
    include '/usr/local/netcdf4/include/netcdf.inc'
    call readarenet(ARENETP,AODPOINTS_NUM,SSApoint_num,AFCpoint_num,wave_len,wave_len2,wave_len3,aod,ssa,afc)
!    write(*,*) wave_len,afc
    do i = 1,ARENETP
        call toh8(wave_len(:,i),aod(:,i),wave_len2(:,i),ssa(:,i),wave_len3(:,i),afc(:,i),AODPOINTS_NUM(i),SSApoint_num(i),AFCpoint_num(i),H8BAND,aod_result(:,i),ssa_result(:,i),afc_result(:,i))
!        write(*,*) aod_result,ssa_result,afc_result
        end do
!    write(*,*) aod_result(:,1)
!    write(*,*)ssa_result(:,1)
!    write(*,*)afc_result(:,1)
!    write(*,*) aod_result(:,2)
!    write(*,*)ssa_result(:,2)
!    write(*,*)afc_result(:,2)
!    write(*,*) aod_result(:,3)
!    write(*,*)ssa_result(:,3)
!    write(*,*)afc_result(:,3)
    !!!!!!!!读取h8 数据
    filename ='/home/lij/code/aerosol/h8/2015100306.nc'
    ierr=nf_open(trim(filename),nf_nowrite,ncid)
    ierr=nf_inq_varid (ncid, 'aot', varid)                ! Temperature
    ierr=nf_get_vars_double(ncid,varid,start2,counts2,strid2,aod_all)

    ierr=nf_inq_varid (ncid, 'lat', varid)
    ierr=nf_get_var_real(ncid,varid, lat)
    ierr=nf_inq_varid (ncid, 'lon', varid)
    ierr=nf_get_var_real(ncid,varid, lon)
    !write(*,*)aod_all
    !!! 遍历矩阵每个点  找离aeronet最近站点坐标
    do m=1,nlon
    do n=1,nlat
        lonh = lon(m)
        lath = lat(n)
        
        do K=1,ARENETP
            call distance(lonh,lath,lons(K),lats(K),dis(K))
        end do
        tem = dis(1)
        POINT =1
!        write(*,*) tem
!        找距离最近的点
        do K=1,ARENETP-1
            if (tem .gt. dis(K+1)) THEN
                POINT = POINT+1
                tem = dis(POINT+1)
                ELSE
                POINT = POINT
                tem = dis(POINT)
            end if
        end do
        if (aod_all(m,n).EQ.0.00000)then
            do i=1,H8BAND
                AODD_ALL(i,m,n) = 0.0
                SSA_ALL(i,m,n) = 0.0
                AFC_ALL(i,m,n) = 0.0
            end do
           ! write(*,*) aod_all(m,n)
            else
            aod_point= aod_all(m,n)
            aod_tem = aod_result(:,POINT)
            ssa_tem = ssa_result(:,POINT)
            afc_tem = afc_result(:,POINT)
            bi = aod_point/aod_tem(2)
            do i=1,H8BAND
                aod_z(i)=aod_tem(i)*bi
                AODD_ALL(i,m,n) = aod_tem(i)*bi
                SSA_ALL(i,m,n) = ssa_tem(i)
                AFC_ALL(i,m,n) = afc_tem(i)
            end do
!            waod = wave_len(:,POINT)
!            wssa = wave_len2(:,POINT)
!            wafc = wave_len3(:,POINT)
            write(*,*) aod_z
!            write(*,*) ssa_z
!            write(*,*) afc_z
            write(*,*) aod_tem
            write(*,*) ssa_tem
            write(*,*) afc_tem
            write(*,*) AODD_ALL(:,m,n)
!            write(*,*) SSA_ALL
            !write(*,*) AFC_ALL(0,:,:)
        end if
!        write(*,*) tem,POINT
!        write(*,*) dis,lonh,lath
        end do
        end do

!    aod,ssa,afc写入nc文件
!    filename='/home/lij/code/aerosol/result/2015100306.nc'
!    call check( nf_create(filename, nf_clobber, ncid))
!    call check( nf_def_dim(ncid, 'lon', nlon, lon_dimid) )
!    call check( nf_def_dim(ncid, 'lat', nlat, lat_dimid) )
!    call check( nf_def_dim(ncid, 'BAND',H8BAND,band_dimid) )
!    dimids = (/ lon_dimid, lat_dimid, band_dimid/)
!
!    call check( nf_def_var(ncid, 'lon', nf_float,1,lon_dimid, lonvarid) )
!    call check( nf_def_var(ncid, 'lat', nf_float,1,lat_dimid, latvarid) )
!    call check( nf_def_var(ncid, 'BAND',nf_int,1,band_dimid,bandvarid) )
!    call check( nf_def_var(ncid,'AOD',nf_double,3,dimids,varid))
!    call check( nf_def_var(ncid,'SSA',nf_double,3,dimids,varid))
!    call check( nf_def_var(ncid,'AFC',nf_double,3,dimids,varid))
!
!    call check(nf_enddef(ncid))
!    call check( nf_put_var_real(ncid,lonvarid,lon) )
!    call check( nf_put_var_real(ncid,latvarid,lat) )
!    call check( nf_put_var_int(ncid,bandvarid,band) )
!!    counts3(3)=nband
!    call check(nf_put_vars_double(ncid, varid,start3,counts3,strid3,AODD_ALL))
!    call check(nf_put_vars_double(ncid, varid,start3,counts3,strid3,SSA_ALL))
!    call check(nf_put_vars_double(ncid, varid,start3,counts3,strid3,AFC_ALL))
!    call check( nf_close(ncid) )
!    print *, "SUCCESS writing file.nc! "


end program main

subroutine readarenet(ARENETP,AODPOINTS_NUM,SSApoint_num,AFCpoint_num,wave_len,wave_len2,wave_len3,aod,ssa,afc)
    integer ARENETP
    integer number,AODPOINTS_NUM(ARENETP),SSApoint_num(ARENETP),AFCpoint_num(ARENETP)
    real wave_len(16,ARENETP),aod(16,ARENETP),wave_len2(16,ARENETP),ssa(16,ARENETP),wave_len3(16,ARENETP),afc(16,ARENETP)
    character(5) num
    do number =1,ARENETP
        write(num,'(I1.1)') number
        !读取ARONET站点AOD wav_len:波长，aod:光学厚度
    open(1,file='/home/lij/code/aerosol/aod/aod_'//Trim(AdjustL(num))//'.txt',status='old')
  	do i=1,AODPOINTS_NUM(number)
	   read(1,*) wave_len(i,number),aod(i,number)
       !write(*,*) wave_len(i)
       !write(*,*) aod(i)
    end do
    close(1)
    !读取ARONET站点SSA wav_len:波长，ssa:单次散射反照率
    open(2,file='/home/lij/code/aerosol/ssa/ssa_'//Trim(AdjustL(num))//'.txt',status='old')
  	do i=1,SSApoint_num(number)
	   read(2,*) wave_len2(i,number),ssa(i,number)
       !write(*,*) wave_len2(i)
       !write(*,*) ssa(i)
    end do
    close(2)
     !读取ARONET站点AFC wav_len:波长，afc:非对称因子
    open(3,file='/home/lij/code/aerosol/afc/afc_'//Trim(AdjustL(num))//'.txt',status='old')
  	do i=1,AFCpoint_num(number)
	   read(3,*) wave_len3(i,number),afc(i,number)
       !write(*,*) wave_len3(i)
       !write(*,*) afc(i)
    end do
    close(3)
    end do
    endsubroutine


subroutine distance(lonh,lath,lonss,latss,dis)
    real(kind=8) dis,diss
    real lonh,lath,lonss,latss
    diss = (((lonh-lonss)**2)+((lath-latss)**2))
    dis=sqrt(diss)
!    write(*,*) diss,dis,lonh,lath,lonss,latss
    endsubroutine

SUBROUTINE toh8(wave_len,aod,wave_len2,ssa,wave_len3,afc,AODPOINTS_NUM,SSApoint_num,AFCpoint_num,H8BAND,aod_result,ssa_result,afc_result)
     integer i,j,iband,AODPOINTS_NUM,SSApoint_num,H8BAND,AFCpoint_num
    real wave_len(AODPOINTS_NUM),aod(AODPOINTS_NUM),wave_len2(SSApoint_num),ssa(SSApoint_num),wave_len3(AFCpoint_num),afc(AFCpoint_num)
    real aod_fun,k,af,wave_in,ii,bb
    real wav_len_all(11,16),srf_all(11,16),rate_all(11,16)
    REAL X1,X2,Y1,Y2,XMAX,XMIN,aod_result(H8BAND),aod_temp,rate_temp,ssa_temp,ssa_result(H8BAND),afc_temp,afc_result(H8BAND)
    real::srf_num(16)=(/10,10,11,10,10,10,10,11,11,11,11,11,11,11,11,11/)
    character(5) fname_iband
    real::wav_cen(16)=(/0.4703,0.5105,0.6399,0.8563,1.6098,2.257,3.8848,6.2383,6.9395,7.3471,8.5905,9.6347,10.4029,11.2432,12.3828,13.2844/)
    ! 读取H8 16个波段波长（wav_len_all）及光波谱响应值（srf_all），及该波长在通道中占有的权重（rate_all）
    do iband=1,16
        num=srf_num(iband)
        !chan=chan_num(iband)
   	    write(fname_iband,'(I2.2)') iband
	    open(1,file='/home/lij/code/aerosol/srf/S'//Trim(AdjustL(fname_iband))//'.txt',status='old')
        do jj=1,num
            read(1,*) wav_len_all(jj,iband),srf_all(jj,iband),rate_all(jj,iband)
        enddo
        close(1)
    enddo
    !write(*,*) wav_len_all(:,:)
!    write(*,*) H8BAND,AODPOINTS_NUM,SSApoint_num

   ! 循环求出H8 16个波段分别对应的AOD
   do iband = 1,H8BAND
   ii = 0
    num = srf_num(iband)
   ! 遍历ARONET站点 找位置求K,AF
    do j = 1,AODPOINTS_NUM-1
        !write(*,*) wave_len(1)
        XMAX = wave_len(1)
        !write(*,*)XMAX,wave_len(1)
        XMIN = wave_len(AODPOINTS_NUM)
        X1 = wave_len(j)
        Y1 = aod(j)
        X2 =  wave_len(j+1)
        Y2 = aod(j+1)
        !write(*,*) X1,wav_cen(iband)
        ! 判断H8中心波长位于ARONET站点各波长的哪个位置
        if ((XMAX) .lt. (wav_cen(iband))) then
            ii = iband
            X1 = XMAX
            Y1 = aod(1)
            X2 =  wave_len(2)
            Y2 = aod(2)
        else if (((X1 .gt. wav_cen(iband)) .and. (X2 .lt. wav_cen(iband)))) then
            ii = iband
        else if ((XMIN) .gt. (wav_cen(iband))) then
            ii = iband
            X1 = wave_len(AODPOINTS_NUM-1)
            Y1 = aod(AODPOINTS_NUM-1)
            ii =0
        endif
        if (ii .ne. 0) then
            !求k和AF
            CALL LSINTERPOLY(AODPOINTS_NUM,X1,Y1,X2,Y2,af,k,aod_fun,wave_in)
            !write(*,*) af,k
            aod_temp =0
            rate_temp=0
            ! 中心波长附近波长位置求AOD，按照波谱响应函数加权平均求中心波长AOD
            do n = 1,num
                !write(*,*) num,wav_len_all(n,ii)
                aod_temp = k*wav_len_all(n,ii)**(-af)*srf_all(n,ii)*rate_all(n,ii)+aod_temp
                rate_temp = rate_all(n,ii)+rate_temp
            enddo
            aod_result(ii) = aod_temp/rate_temp
            !write(*,*) aod_result
        endif
     enddo
  enddo

!  open(4,file='/home/lij/code/aerosol/aod_result.txt',status='new')
!  	do i=1,H8BAND
!	   write(4,*) aod_result(i)
!    end do
!    close(4)

   ! 循环求出H8 16个波段分别对应的SSA
   do iband = 1,H8BAND
   ii = 0
    num = srf_num(iband)
   ! 遍历ARONET站点 找位置求K,BB
    do j = 1,SSApoint_num-1
        XMAX = wave_len2(SSApoint_num)
        XMIN = wave_len2(1)
        X1 = wave_len2(j)
        Y1 = ssa(j)
        X2 =  wave_len2(j+1)
        Y2 = ssa(j+1)
        !write(*,*) X1,wav_cen(iband)
        ! 判断H8中心波长位于ARONET站点各波长的哪个位置
        if ((XMAX) .lt. (wav_cen(iband))) then
            ii = iband
            X1 = wave_len2(SSAApoint_num-1)
            Y1 = ssa(SSApoint_num-1)
            X2 =  XMAX
            Y2 = ssa(SSApoint_num)
        else if (((X1 .lt. wav_cen(iband)) .and. (X2 .gt. wav_cen(iband)))) then
            ii = iband
        else if ((XMIN) .gt. (wav_cen(iband))) then
            ii = iband
            X1 = XMIN
            Y1 = ssa(1)
            X2 =  wave_len2(2)
            Y2 = ssa(2)
        else
            ii =0
        endif

        if (ii .ne. 0) then
            !求k和B
            CALL Linearinter(AODPOINTS_NUM,X1,Y1,X2,Y2,bb,k)
            !write(*,*) K,B
            ssa_temp =0
            rate_temp=0
            ! 中心波长附近波长位置求SSA，按照波谱响应函数加权平均求中心波长SSA
            do n = 1,num
                !write(*,*) num,wav_len_all(n,ii)
                ssa_temp = (k*wav_len_all(n,ii)+bb)*srf_all(n,ii)*rate_all(n,ii)+ssa_temp
                rate_temp = rate_all(n,ii)+rate_temp
            enddo
            ssa_result(ii) = ssa_temp/rate_temp
            !write(*,*) rate_temp
        endif
     enddo
  enddo
!  write(*,*) ssa_result

!  open(5,file='/home/lij/code/aerosol/ssa_result.txt',status='new')
!  	do i=1,H8BAND
!	   write(5,*) ssa_result(i)
!    end do
!    close(5)

     ! 循环求出H8 16个波段分别对应的AFC
   do iband = 1,H8BAND
   ii = 0
    num = srf_num(iband)
   ! 遍历ARONET站点 找位置求K,B
    do j = 1,AFCpoint_num-1
        XMAX = wave_len3(AFCpoint_num)
        XMIN = wave_len3(1)
        X1 = wave_len3(j)
        Y1 = afc(j)
        X2 =  wave_len3(j+1)
        Y2 = afc(j+1)
        !write(*,*) X1,wav_cen(iband)
        ! 判断H8中心波长位于ARONET站点各波长的哪个位置
        if ((XMAX) .lt. (wav_cen(iband))) then
            ii = iband
            X1 = wave_len3(AFCpoint_num-1)
            Y1 = afc(AFCpoint_num-1)
            X2 =  XMAX
            Y2 = afc(AFCpoint_num)
        else if (((X1 .lt. wav_cen(iband)) .and. (X2 .gt. wav_cen(iband)))) then
            ii = iband
        else if ((XMIN) .gt. (wav_cen(iband))) then
            ii = iband
            X1 = XMIN
            Y1 = afc(1)
            X2 =  wave_len3(2)
            Y2 = afc(2)
            X1 = XMIN
            Y1 = afc(1)
            X2 =  wave_len3(2)
            Y2 = afc(2)
        else
            ii =0
        endif

        if (ii .ne. 0) then
            !求k和B
            CALL Linearinter(AODPOINTS_NUM,X1,Y1,X2,Y2,bb,k)
            !write(*,*) B,k
            afc_temp =0
            rate_temp=0
            ! 中心波长附近波长位置求AFC，按照波谱响应函数加权平均求中心波长AFC
            do n = 1,num
                !write(*,*) num,wav_len_all(n,ii)
                afc_temp = (k*wav_len_all(n,ii)+bb)*srf_all(n,ii)*rate_all(n,ii)+afc_temp
                rate_temp = rate_all(n,ii)+rate_temp
            enddo
            afc_result(ii) = afc_temp/rate_temp
            !write(*,*) rate_temp
        endif
     enddo
  enddo
!  write(*,*) afc_result

!  open(6,file='/home/lij/code/aerosol/afc_result.txt',status='new')
!  	do i=1,H8BAND
!	   write(6,*) afc_result(i)
!    end do
!    close(6)
END SUBROUTINE
!---------------------------------------------------------------------------
! AOD插值函数
SUBROUTINE LSINTERPOLY(AODPOINTS_NUM& !测试数据的个数
        ,X1& !测试数据横坐标
        ,Y1& !测试数据纵坐标
        ,X2,Y2,af& !多项式阶数
        ,k& !多项式系数
        ,aod_fun,wave_in)!求多项式系数
    IMPLICIT NONE
    integer AODPOINTS_NUM
    REAL X1,X2,Y1,Y2
    real aod_fun,k,af,wave_in
    
    af = log(Y1/Y2)/log(X2/X1)
    k = Y1/(X1**(-(af)))
    !write(*,*)af,k

END SUBROUTINE
!---------------------------------------------------------------------------

!---------------------------------------------------------------------------
! 线性插值
SUBROUTINE  Linearinter(AODPOINTS_NUM& !测试数据的个数
        ,X1& !测试数据横坐标
        ,Y1& !测试数据纵坐标
        ,X2,Y2,bb& !多项式阶数
        ,k& !多项式系数
        )!求多项式系数
    IMPLICIT NONE
    integer AODPOINTS_NUM
    REAL X1,X2,Y1,Y2
    real aod_fun,k,bb,wave_in
    
    k = (Y2-Y1)/(X2-X1)
    bb = Y1-k*X1
    !write(*,*)af,k

END SUBROUTINE
!---------------------------------------------------------------------------
subroutine check(status)
   integer, intent ( in) :: status
   if(status /= nf_noerr) then
      print *, 'trim(nf_strerror(status))'
   end if
end subroutine check
