

program ocean_main
    integer::start2(2)=(/1,1/)
    integer::counts2(2)=(/nlon,nlat/)
    integer::strid2(2)=(/1,1/)
    integer ncid,varid

    USE ocnref, ONLY :     &
            ocnref__NN_roughness_R, &
            ocnref__NN_BRDF_ang_R,  &
            ocnref__NN_BRDF_vec_R,  &
            fresnelRef1,            &
            erfc_R,                 &
            erfc_Cheb_R

    filename = '/home/data/ERA5/era5_w10t2/era5_20160305.nc'
    ierr = nf_open(trim(filename),nf_nowrite,ncid)
    ierr=nf_inq_varid (ncid, 'u10', varid)                ! Temperature
    ierr=nf_get_vars_double(ncid,varid,start2,counts2,strid2,uwind)
    write(*,*) uwind

end program ocean_main