;+
;rd_probclass
;	read in the csv files from Samuel listing best, second-best, and worst matches and do some rudimentary plots
;
;vinay k (2024-jun-10)
;-

;	set up displays and colors
peasecolr & loadct,3 & blindcolr,/white & thk=4.0 & csz=3.0 & !p.background=255 & poaintsym,'circle',/pfil,psiz=0.75
sep=string(9b)
window,0,xsize=1000,ysize=1000
window,2,xsize=1400,ysize=700

if not keyword_set(nnn) then nnn=500L	;half-window for p-value estimation

;	Samuel's recompilation of Dong-Woo's matches
bestfil='most_prob_class_gaia_props.csv'
secondfil='second_most_prob_class_gaia_props.csv'
worstfil='last_prob_class_gaia_props.csv'

if n_tags(tb) eq 0 then tb=read_csv(bestfil,count=nb,header=hdrb)	;read best matches
for i=0L,n_elements(hdrb)-1 do jnk=execute(hdrb[i]+"_b=tb.(i)")		;extract columns into arrays named FEATURE_b
sepb=deltang(csc21_ra_b,csc21_dec_b,gaia3_ra_b,gaia3_dec_b)*3600.	;compute separations between CSC and Gaia coordinates [arcsec]
if n_tags(t2) eq 0 then t2=read_csv(secondfil,count=n2,header=hdr2)	;read 2nd best matches
for i=0L,n_elements(hdr2)-1 do jnk=execute(hdr2[i]+"_2=t2.(i)")		;extract columns into arrays named FEATURE_2
sep2=deltang(csc21_ra_2,csc21_dec_2,gaia3_ra_2,gaia3_dec_2)*3600.	;compute separations between CSC and Gaia coordinates [arcsec]
if n_tags(tx) eq 0 then tx=read_csv(worstfil,count=nx,header=hdrx)	;read worst matches within 15"
for i=0L,n_elements(hdrx)-1 do jnk=execute(hdrx[i]+"_x=tx.(i)")		;extract columns into arrays named FEATURE_x
sepx=deltang(csc21_ra_x,csc21_dec_x,gaia3_ra_x,gaia3_dec_x)*3600.	;compute separations between CSC and Gaia coordinates [arcsec]

;	for those missing in worst, find the nearest optical match to best from 2nd best and use that
cb=csc21_name_b & c2=csc21_name_2 & cx=csc21_name_x & jbx=lonarr(nb)-1L & jb2=jbx
rab=csc21_ra_b & decb=csc21_dec_b & ra2=csc21_ra_2 & dec2=csc21_dec_2 & rax=csc21_ra_x & decx=csc21_dec_x
extrafil=file_search('inb_in2_inx.save',count=nextra)
if nextra eq 1 then restore,extrafil,/verbose	;no need to keep redoing this, only need to do it once
if n_elements(jbx) ne nb then begin
  for i=0L,nb-1L do begin
    if i eq 1000*long(i/1000) then kilroy,dot=strtrim(i,2)
    oo=where(cx eq cb[i],moo)
    if moo eq 0 then begin
      kilroy
      dd=deltang(ra2,dec2,rab[i],decb[i]) & jnk=min(dd,imn)
      jb2[i]=imn
    endif else jbx[i]=oo[0]
  endfor
  save,file='inb_in2_inx.save',jb2,jbx
endif
;	build up the new worst match database
tz=tb & o2=where(jb2 ge 0) & ox=where(jbx ge 0)
;	but first protect the X-ray properties
ixray=lonarr(n_elements(hdrb))-1
for i=0L,n_elements(hdrb)-1L do begin
  if strpos(hdrb[i],'csc21') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'hard_') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'var_int') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'extent_flag') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'pileup_flag') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'var_flag') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'src_area') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'photflux_aper') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'acis_') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'hrc_') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'min_theta_mean') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'yangetal_') ge 0 then ixray[i]=1
  if strpos(hdrb[i],'perezdiazetal_') ge 0 then ixray[i]=1
endfor
for i=0L,n_elements(hdrb)-1L do begin
  if ixray[i] lt 0 then begin
    tmpz=tz.(i) & tmp2=t2.(i) & tmpx=tx.(i)
    tmpz[ox]=tmpx[jbx[ox]]
    tmpz[o2]=tmp2[jb2[o2]]
    tz.(i)=tmpz
  endif
endfor
;	sanity check --
;k=2 & plot,(tz.(k))[ox]-(tx.(k))[jbx[ox]],psym=1,col=0	;should be identically 0 for k=1,2,4,5,6,7,etc
;k=2 & plot,(tz.(k))[o2]-(t2.(k))[jb2[o2]],psym=1,col=0	;should be identically 0 for k=1,2,4,5,6,7,etc
;	make the augmented worst matches dataset
for i=0L,n_elements(hdrx)-1 do jnk=execute(hdrx[i]+"_z=tz.(i)")		;extract columns into arrays named FEATURE_z
sepz=deltang(csc21_ra_z,csc21_dec_z,gaia3_ra_z,gaia3_dec_z)*3600.	;compute separations between CSC and Gaia coordinates [arcsec]

;	convert X-ray coordinates to galactic and then to Aitoff projection and make a plot
glactc,csc21_ra_b,csc21_dec_b,2000.,glb,gbb,1,/degree
wcssph2xy,glb,gbb,xlb,ylb,21
plot,xlb,ylb,psym=3,xstyle=5,ystyle=5,title='CSC with Gaia Matches',thick=thk,xthick=thk,ythick=thk,charthick=thk,charsize=csz,col=0
zlon=fltarr(180)+359.99-180 & zlat=findgen(180)-90 & wcssph2xy,zlon,zlat,zx,zy,21 & oplot,zx,zy,col=0
for k=0,359,20 do begin & zlon=fltarr(180)+k-180 & zlat=findgen(180)-90 & wcssph2xy,zlon,zlat,zx,zy,21 & oplot,zx,zy,col=0 & endfor
for k=0,180,20 do begin & zlon=findgen(360)-180 & zlat=fltarr(360)+k-90 & wcssph2xy,zlon,zlat,zx,zy,21 & oplot,zx,zy,col=0 & endfor
oplot,xlb,ylb,psym=8,col=1
zlon=findgen(19)*20-0.1-180 & zlat=fltarr(19) & wcssph2xy,zlon,zlat,zx,zy,21 & xyouts,zx,zy-5,strtrim(ceil(zlon),2),charthick=thk/2,charsize=csz/2,col=2,align=0.5
zlon=fltarr(10) & zlat=findgen(10)*20-90 & wcssph2xy,zlon,zlat,zx,zy,21 & xyouts,zx+10,zy,strtrim(ceil(zlat),2),charthick=thk/2,charsize=csz/2,col=2,align=1
write_png,'CSCGDR3_l2b2.png',tvrd(/true)

;	find the areas by doing VTP
;	set it up for a periodic reflection to avoid edge effects -- only the first NB points will be used
vorfil=file_search('CSCwGaia_voronoi.save',count=nvor)
if nvor eq 1 then restore,vorfil[0],/verbose
if n_elements(zra) ne 9L*nb then begin
  zra=[csc21_ra_b, -csc21_ra_b, -csc21_ra_b+2*360.,$
     csc21_ra_b, -csc21_ra_b, -csc21_ra_b+2*360.,$
     csc21_ra_b, -csc21_ra_b, -csc21_ra_b+2*360.]
  zdec=[csc21_dec_b, csc21_dec_b, csc21_dec_b,$
      -csc21_dec_b+180., -csc21_dec_b+180., -csc21_dec_b+180.,$
      -csc21_dec_b-180., -csc21_dec_b-180., -csc21_dec_b-180.]
  plot,zra,zdec,psym=3,/xs,/ys,col=0
  zarea=0.*zra
  triangulate,100.*zra,100.*zdec,tr,conn=c
  plot,zra[0:nb-1],zdec[0:nb-1],/xs,/ys,psym=3,col=0
  for i=0L,nb-1 do begin
    voronoi,100.*zra,100.*zdec,i,c,Xp,Yp
    Xp=Xp/100. & Yp=Yp/100.
    oplot,Xp,Yp,col=1
    zarea[i]=((areapoly(Xp,Yp)*cos(zdec[i]*!pi/180.))>(1e-7))<(180.*360.)
  endfor
  save,file='CSCwGaia_voronoi.save',zra,zdec,tr,c,zarea
endif
zarea=zarea>1e-7	;so the area is at least the size of the on-axis PSF

;	group all nearby X-ray sources into clusters
grpfil=file_search('CSCwGaia_clusters.save',count=ngrp)
if ngrp eq 1 then restore,grpfil[0],/verbose
if n_elements(igrp) ne nb then begin
  igrp=lindgen(nb)+1L
  zra=csc21_ra_b & zdec=csc21_dec_b
  for i=0L,nb-1L do begin
    if i eq 500L*(i/500L) then kilroy,dot=strtrim(i,2)+' '
    oo=where(abs(zra-zra[i]) lt 1 and abs(zdec-zdec[i]) lt 1,moo)	;just to speed things up
    if moo gt 1 then begin
      dd=deltang(zra[oo],zdec[oo],zra[i],zdec[i])*60.	;[arcmin]
      ok=where(dd lt 8.,mok)
      if mok gt 0 then igrp[oo[ok]]=min(igrp[oo[ok]])
      if mok gt 0 then kilroy,dot=strtrim(i,2)+':'+strtrim(mok,2)+' '
    endif
  endfor
  save,file='CSCwGaia_clusters.save',zra,zdec,igrp
endif
uig=igrp[uniq(igrp,sort(igrp))] & nuig=n_elements(uig)	;unique groups
huig=lonarr(nuig) & for i=0L,nuig-1L do begin & oo=where(igrp eq uig[i],moo) & huig[i]=moo & endfor

;	for each cluster, find the total detector area as that from the convex hull and the grasp as combination of voronoi areas and exposure times
;	and write it out as a table for Farah to use
;	NOTE: this will be superseded by the sensitivity maps that Samuel is constructing
wra=fltarr(nuig) & wdec=wra & wexp=wra & warea=wra & wgrasp=wra & wsky=wra & wfx=wra
openw,uo,'CSC2.1_pointings_for_GUMS.txt',/get_lun
printf,uo,'# '+systime()+' -- DO NOT EDIT -- made with CSC-GDR3/Matches_2024jun10/rd_probclass.pro'
printf,uo,'# RA'+sep+'DEC'+sep+'skyarea'+sep+'fluxlimit'+sep+'numX'+sep+'CHarea'+sep+'grasp'+sep+'exptime'
for k=0L,nuig-1L do begin
  oo=where(igrp eq uig[k],moo)
  xx=csc21_ra_b[oo] & yy=csc21_dec_b[oo]
  if moo gt 2 then begin
    qhull,xx,yy,tr & jj=reform(tr[0,*])
    wra[k]=mean(xx[jj]) & wdec[k]=mean(yy[jj]) & warea[k]=areapoly(xx[jj],yy[jj])*cos(wdec[k]*!pi/180.) & wexp[k]=median(acis_time_b[oo])
  endif else begin
    if moo eq 1 then begin & wra[k]=xx[0] & wdec[k]=yy[0] & warea[k]=zarea[oo[0]] & wexp[k]=acis_time_b[oo[0]] & endif
    if moo eq 2 then begin & wra[k]=mean(xx) & wdec[k]=mean(yy) & warea[k]=total(zarea[oo]) & wexp[k]=mean(acis_time_b[oo]) & endif
  endelse
  wgrasp[k]=total(acis_time_b[oo]*(zarea[oo]<(2*warea[k]/huig[k])),/nan)
  wsky[k]=wgrasp[k]/wexp[k]
  wfx[k]=5.*1.602e-9/400./wexp[k]	;expected flux from 5 photons at 1 keV for an EA of 400 cm^2
  cc=strtrim(string(wra[k],'(f12.7)'),2)+sep+$
  	strtrim(string(wdec[k],'(f12.7)'),2)+sep+$
	strtrim(wsky[k],2)+sep+$
	strtrim(wfx[k])+sep+$
	strtrim(huig[k],2)+sep+$
	strtrim(warea[k],2)+sep+$
	strtrim(string(wgrasp[k],'(f12.3)'),2)+sep+$
	strtrim(string(wexp[k]/1e3,'(f12.3)'),2)+sep
  print,cc
  ;if huig[k] gt 1 then begin
  ;  plot,csc21_ra_b,csc21_dec_b,psym=4,col=0,symsize=3,thick=thk,charsize=csz,xr=wra[k]+[-1,1],yr=wdec[k]+[-1,1],/xs,/ys
  ;  oplot,xx,yy,psym=8,symsize=2,col=2
  ;  stop,zarea[oo],warea[k],wgrasp[k],wexp[k],wsky[k],moo
  ;endif
  printf,uo,cc
endfor
close,uo & free_lun,uo

;	stratification
o0b=where(min_theta_mean_b lt 3.,mo0b) & o02=where(min_theta_mean_2 lt 3.,mo02) & o0x=where(min_theta_mean_x lt 3.,mo0x) & o0z=where(min_theta_mean_z lt 3.,mo0z)	;near aimpoint, 0-3 arcmin
o3b=where(min_theta_mean_b ge 3. and min_theta_mean_b lt 6.,mo3b) & o32=where(min_theta_mean_2 ge 3. and min_theta_mean_2 lt 6.,mo32) & o3x=where(min_theta_mean_x ge 3. and min_theta_mean_x lt 6.,mo3x) & o3z=where(min_theta_mean_z ge 3. and min_theta_mean_z lt 6.,mo3z)	;mid range off-axis, 3-6 arcmin
o6b=where(min_theta_mean_b ge 6.,mo6b) & o62=where(min_theta_mean_2 ge 6.,mo62) & o6x=where(min_theta_mean_x ge 6.,mo6x) & o6z=where(min_theta_mean_z ge 6.,mo6z)	;large off-axis, >6 arcmin

;	compute p-values for varying separations and make plots, for each of the stratified cases, for each of gmag,rpmag,bpmag

;	near aimpoint
pvalsav0=file_search('stratified0.save',count=nstrat0) & mmm=nnn
if nstrat0 eq 1 then restore,pvalsav0[0],/verbose
if n_elements(xsepb0) ne mo0b or mmm ne nnn then begin
  xsepb0=sepb[o0b] & xsepz0=sepz[o0b] & os=sort(xsepb0) & xsepb0=xsepb0[os] & xsepz0=xsepz0[os]
  gmagb0=phot_g_mean_mag_b[o0b] & gmagz0=phot_g_mean_mag_z[o0b] & gmagb0=gmagb0[os] & gmagz0=gmagz0[os] & gpval0=0.*xsepb0 & gsepmin0=gpval0 & gsepmax0=gpval0 & gdks0=gpval0
  rpmagb0=phot_rp_mean_mag_b[o0b] & rpmagz0=phot_rp_mean_mag_z[o0b] & rpmagb0=rpmagb0[os] & rpmagz0=rpmagz0[os] & rppval0=0.*xsepb0 & rpsepmin0=rppval0 & rpsepmax0=rppval0 & rpdks0=rppval0
  bpmagb0=phot_bp_mean_mag_b[o0b] & bpmagz0=phot_bp_mean_mag_z[o0b] & bpmagb0=bpmagb0[os] & bpmagz0=bpmagz0[os] & bppval0=0.*xsepb0 & bpsepmin0=bppval0 & bpsepmax0=bppval0 & bpdks0=bppval0
  mmm=nnn
  for k=0L,mo0b-1L do begin
    k0=(k-mmm)>0L
    k1=(k+mmm)<(mo0b-1L)
    kstwo,gmagb0[k0:k1],gmagz0[k0:k1],dks,pks & gpval0[k]=pks & gdks0[k]=dks & gsepmin0[k]=xsepb0[k0] & gsepmax0[k]=xsepb0[k1]
    kstwo,rpmagb0[k0:k1],rpmagz0[k0:k1],dks,pks & rppval0[k]=pks & rpdks0[k]=dks & rpsepmin0[k]=xsepb0[k0] & rpsepmax0[k]=xsepb0[k1]
    kstwo,bpmagb0[k0:k1],bpmagz0[k0:k1],dks,pks & bppval0[k]=pks & bpdks0[k]=dks & bpsepmin0[k]=xsepb0[k0] & bpsepmax0[k]=xsepb0[k1]
    if k eq 1000*long(k/1000) then kilroy,dot=strtrim(k,2)+' '
  endfor
  save,file='stratified0.save',xsepb0,xsepz0,mmm,$
	gmagb0,gmagz0,gpval0,gdks0,gsepmin0,gsepmax0,$
	rpmagb0,rpmagz0,rppval0,rpdks0,rpsepmin0,rpsepmax0,$
	bpmagb0,bpmagz0,bppval0,bpdks0,bpsepmin0,bpsepmax0
endif
!p.multi=[0,1,2]
plot,xsepb0,gpval0,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='gmag [0-3 arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo0b-1 do oplot,[gsepmin0[i]>(10.^!x.crange[0]),gsepmax0[i]],gpval0[i]*[1,1],col=1
plot,xsepb0,gdks0,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo0b-1 do oplot,[gsepmin0[i]>(10.^!x.crange[0]),gsepmax0[i]],gdks0[i]*[1,1],col=1
write_png,'separation_pvalue_gmag_0t3.png',tvrd(/true)
;
plot,xsepb0,rppval0,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='rpmag [0-3 arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo0b-1 do oplot,[rpsepmin0[i]>(10.^!x.crange[0]),rpsepmax0[i]],rppval0[i]*[1,1],col=1
plot,xsepb0,rpdks0,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo0b-1 do oplot,[rpsepmin0[i]>(10.^!x.crange[0]),rpsepmax0[i]],rpdks0[i]*[1,1],col=1
write_png,'separation_pvalue_rpmag_0t3.png',tvrd(/true)
;
plot,xsepb0,bppval0,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='bpmag [0-3 arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo0b-1 do oplot,[bpsepmin0[i]>(10.^!x.crange[0]),bpsepmax0[i]],bppval0[i]*[1,1],col=1
plot,xsepb0,bpdks0,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo0b-1 do oplot,[bpsepmin0[i]>(10.^!x.crange[0]),bpsepmax0[i]],bpdks0[i]*[1,1],col=1
write_png,'separation_pvalue_bpmag_0t3.png',tvrd(/true)
!p.multi=0

;	mid-range offaxis
pvalsav3=file_search('stratified3.save',count=nstrat3) & mmm=nnn
if nstrat3 eq 1 then restore,pvalsav3[0],/verbose
if n_elements(xsepb3) ne mo3b or mmm ne nnn then begin
  xsepb3=sepb[o3b] & xsepz3=sepz[o3b] & os=sort(xsepb3) & xsepb3=xsepb3[os] & xsepz3=xsepz3[os]
  gmagb3=phot_g_mean_mag_b[o3b] & gmagz3=phot_g_mean_mag_z[o3b] & gmagb3=gmagb3[os] & gmagz3=gmagz3[os] & gpval3=0.*xsepb3 & gsepmin3=gpval3 & gsepmax3=gpval3 & gdks3=gpval3
  rpmagb3=phot_rp_mean_mag_b[o3b] & rpmagz3=phot_rp_mean_mag_z[o3b] & rpmagb3=rpmagb3[os] & rpmagz3=rpmagz3[os] & rppval3=0.*xsepb3 & rpsepmin3=rppval3 & rpsepmax3=rppval3 & rpdks3=rppval3
  bpmagb3=phot_bp_mean_mag_b[o3b] & bpmagz3=phot_bp_mean_mag_z[o3b] & bpmagb3=bpmagb3[os] & bpmagz3=bpmagz3[os] & bppval3=0.*xsepb3 & bpsepmin3=bppval3 & bpsepmax3=bppval3 & bpdks3=bppval3
  mmm=nnn
  for k=0L,mo3b-1L do begin
    k0=(k-mmm)>0L
    k1=(k+mmm)<(mo3b-1L)
    kstwo,gmagb3[k0:k1],gmagz3[k0:k1],dks,pks & gpval3[k]=pks & gdks3[k]=dks & gsepmin3[k]=xsepb3[k0] & gsepmax3[k]=xsepb3[k1]
    kstwo,rpmagb3[k0:k1],rpmagz3[k0:k1],dks,pks & rppval3[k]=pks & rpdks3[k]=dks & rpsepmin3[k]=xsepb3[k0] & rpsepmax3[k]=xsepb3[k1]
    kstwo,bpmagb3[k0:k1],bpmagz3[k0:k1],dks,pks & bppval3[k]=pks & bpdks3[k]=dks & bpsepmin3[k]=xsepb3[k0] & bpsepmax3[k]=xsepb3[k1]
    if k eq 1000*long(k/1000) then kilroy,dot=strtrim(k,2)+' '
  endfor
  save,file='stratified3.save',xsepb3,xsepz3,mmm,$
	gmagb3,gmagz3,gpval3,gdks3,gsepmin3,gsepmax3,$
	rpmagb3,rpmagz3,rppval3,rpdks3,rpsepmin3,rpsepmax3,$
	bpmagb3,bpmagz3,bppval3,bpdks3,bpsepmin3,bpsepmax3
endif
!p.multi=[0,1,2]
plot,xsepb3,gpval3,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='gmag [3-6 arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo3b-1 do oplot,[gsepmin3[i]>(10.^!x.crange[0]),gsepmax3[i]],gpval3[i]*[1,1],col=3
plot,xsepb3,gdks3,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo3b-1 do oplot,[gsepmin3[i]>(10.^!x.crange[0]),gsepmax3[i]],gdks3[i]*[1,1],col=3
write_png,'separation_pvalue_gmag_3t6.png',tvrd(/true)
;
plot,xsepb3,rppval3,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='rpmag [3-6 arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo3b-1 do oplot,[rpsepmin3[i]>(10.^!x.crange[0]),rpsepmax3[i]],rppval3[i]*[1,1],col=3
plot,xsepb3,rpdks3,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo3b-1 do oplot,[rpsepmin3[i]>(10.^!x.crange[0]),rpsepmax3[i]],rpdks3[i]*[1,1],col=3
write_png,'separation_pvalue_rpmag_3t6.png',tvrd(/true)
;
plot,xsepb3,bppval3,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='bpmag [3-6 arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo3b-1 do oplot,[bpsepmin3[i]>(10.^!x.crange[0]),bpsepmax3[i]],bppval3[i]*[1,1],col=3
plot,xsepb3,bpdks3,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo3b-1 do oplot,[bpsepmin3[i]>(10.^!x.crange[0]),bpsepmax3[i]],bpdks3[i]*[1,1],col=3
write_png,'separation_pvalue_bpmag_3t6.png',tvrd(/true)
!p.multi=0

;	large off-axis
pvalsav6=file_search('stratified6.save',count=nstrat6) & mmm=nnn
if nstrat6 eq 1 then restore,pvalsav6[0],/verbose
if n_elements(xsepb6) ne mo6b or mmm ne nnn then begin
  xsepb6=sepb[o6b] & xsepz6=sepz[o6b] & os=sort(xsepb6) & xsepb6=xsepb6[os] & xsepz6=xsepz6[os]
  gmagb6=phot_g_mean_mag_b[o6b] & gmagz6=phot_g_mean_mag_z[o6b] & gmagb6=gmagb6[os] & gmagz6=gmagz6[os] & gpval6=0.*xsepb6 & gsepmin6=gpval6 & gsepmax6=gpval6 & gdks6=gpval6
  rpmagb6=phot_rp_mean_mag_b[o6b] & rpmagz6=phot_rp_mean_mag_z[o6b] & rpmagb6=rpmagb6[os] & rpmagz6=rpmagz6[os] & rppval6=0.*xsepb6 & rpsepmin6=rppval6 & rpsepmax6=rppval6 & rpdks6=rppval6
  bpmagb6=phot_bp_mean_mag_b[o6b] & bpmagz6=phot_bp_mean_mag_z[o6b] & bpmagb6=bpmagb6[os] & bpmagz6=bpmagz6[os] & bppval6=0.*xsepb6 & bpsepmin6=bppval6 & bpsepmax6=bppval6 & bpdks6=bppval6
  mmm=nnn
  for k=0L,mo6b-1L do begin
    k0=(k-mmm)>0L
    k1=(k+mmm)<(mo6b-1L)
    kstwo,gmagb6[k0:k1],gmagz6[k0:k1],dks,pks & gpval6[k]=pks & gdks6[k]=dks & gsepmin6[k]=xsepb6[k0] & gsepmax6[k]=xsepb6[k1]
    kstwo,rpmagb6[k0:k1],rpmagz6[k0:k1],dks,pks & rppval6[k]=pks & rpdks6[k]=dks & rpsepmin6[k]=xsepb6[k0] & rpsepmax6[k]=xsepb6[k1]
    kstwo,bpmagb6[k0:k1],bpmagz6[k0:k1],dks,pks & bppval6[k]=pks & bpdks6[k]=dks & bpsepmin6[k]=xsepb6[k0] & bpsepmax6[k]=xsepb6[k1]
    if k eq 1000*long(k/1000) then kilroy,dot=strtrim(k,2)+' '
  endfor
  save,file='stratified6.save',xsepb6,xsepz6,mmm,$
	gmagb6,gmagz6,gpval6,gdks6,gsepmin6,gsepmax6,$
	rpmagb6,rpmagz6,rppval6,rpdks6,rpsepmin6,rpsepmax6,$
	bpmagb6,bpmagz6,bppval6,bpdks6,bpsepmin6,bpsepmax6
endif
!p.multi=[0,1,2]
plot,xsepb6,gpval6,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='gmag [6+ arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo6b-1 do oplot,[gsepmin6[i]>(10.^!x.crange[0]),gsepmax6[i]],gpval6[i]*[1,1],col=6
plot,xsepb6,gdks6,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo6b-1 do oplot,[gsepmin6[i]>(10.^!x.crange[0]),gsepmax6[i]],gdks6[i]*[1,1],col=6
write_png,'separation_pvalue_gmag_6p.png',tvrd(/true)
;
plot,xsepb6,rppval6,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='rpmag [6+ arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo6b-1 do oplot,[rpsepmin6[i]>(10.^!x.crange[0]),rpsepmax6[i]],rppval6[i]*[1,1],col=6
plot,xsepb6,rpdks6,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo6b-1 do oplot,[rpsepmin6[i]>(10.^!x.crange[0]),rpsepmax6[i]],rpdks6[i]*[1,1],col=6
write_png,'separation_pvalue_rpmag_6p.png',tvrd(/true)
;
plot,xsepb6,bppval6,psym=1,/xs,/ys,/xl,xr=[1e-2,15],/yl,yr=[1e-10,1.05],xtitle='SEPARATION [arcsec]',ytitle='p-value KS',title='bpmag [6+ arcmin] best vs worst',subtitle='+-'+strtrim(mmm,2),xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo6b-1 do oplot,[bpsepmin6[i]>(10.^!x.crange[0]),bpsepmax6[i]],bppval6[i]*[1,1],col=6
plot,xsepb6,bpdks6,psym=1,/xs,/ys,/xl,xr=[1e-2,15],xtitle='SEPARATION [arcsec]',ytitle='d!dKS!n',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk/2,xthick=thk/2,ythick=thk/2,charthick=thk/2,charsize=csz/2,col=0
for i=0,mo6b-1 do oplot,[bpsepmin6[i]>(10.^!x.crange[0]),bpsepmax6[i]],bpdks6[i]*[1,1],col=6
write_png,'separation_pvalue_bpmag_6p.png',tvrd(/true)
!p.multi=0

;	make histograms for different separations
hsb=histogram(sepb,min=0,max=15,bin=0.1) & xhs=findgen(n_elements(hsb))*0.1+0.1/2.
hs2=histogram(sep2,min=0,max=15,bin=0.1)
hsz=histogram(sepz,min=0,max=15,bin=0.1)
hsb0=histogram(sepb[o0b],min=0,max=15,bin=0.1)
hs20=histogram(sep2[o02],min=0,max=15,bin=0.1)
hsz0=histogram(sepz[o0z],min=0,max=15,bin=0.1)
hsb3=histogram(sepb[o3b],min=0,max=15,bin=0.1)
hs23=histogram(sep2[o32],min=0,max=15,bin=0.1)
hsz3=histogram(sepz[o3z],min=0,max=15,bin=0.1)
hsb6=histogram(sepb[o6b],min=0,max=15,bin=0.1)
hs26=histogram(sep2[o62],min=0,max=15,bin=0.1)
hsz6=histogram(sepz[o6z],min=0,max=15,bin=0.1)

wset,0
plot,xhs,hsb,psym=10,/xl,/yl,yr=[0.9,max(hsb)*2],/ys,/xs,xtitle='separation [arcsec]',ytitle='# matches',title='cross-matching CSC to GDR3',xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk,xthick=thk,ythick=thk,charthick=thk,charsize=csz,col=0
oplot,xhs,hs2,thick=thk,col=2 & oplot,xhs,hsz,thick=thk,col=3
oplot,xhs,hsb0,col=0,thick=thk,line=1,psym=10 & oplot,xhs,hsb3,col=0,thick=thk,line=2,psym=10 & oplot,xhs,hsb6,col=0,thick=thk,line=3,psym=10
oplot,xhs,hs20,col=2,thick=thk,line=1 & oplot,xhs,hs23,col=2,thick=thk,line=2 & oplot,xhs,hs26,col=2,thick=thk,line=3
oplot,xhs,hsz0,col=3,thick=thk,line=1 & oplot,xhs,hsz3,col=3,thick=thk,line=2 & oplot,xhs,hsz6,col=3,thick=thk,line=3
xyouts,1.1,max(hsb)*1.2,'BEST',charthick=thk,charsize=csz,col=0
xyouts,2.1,max(hsb)*1.2,'2!uND!n',charthick=thk,charsize=csz,col=2
xyouts,4.1,max(hsb)*1.2,'WORST',charthick=thk,charsize=csz,col=3
oplot,[1.5,3],1.05*1.5*1.5*1.5*[1,1],thick=thk,line=1,col=1 & xyouts,3.5,1.5*1.5*1.5,"!4h!X<3'",charthick=thk,charsize=csz,col=1
oplot,[1.5,3],1.05*1.5*1.5*[1,1],thick=thk,line=2,col=1 & xyouts,3.5,1.5*1.5,"3'<!4h!X<6'",charthick=thk,charsize=csz,col=1
oplot,[1.5,3],1.05*1.5*[1,1],thick=thk,line=3,col=1 & xyouts,3.5,1.5,"!4h!X>6'",charthick=thk,charsize=csz,col=1
write_png,'CSCGDR3_NWAY_xmatch_bysep.png',tvrd(/true)

;	make plots showing the distributions changing between best small-separation vs large-separation and best small-separation vs worst all
wset,2
oo1b=where(min_theta_mean_b lt 3 and sepb lt 1.2,moo1b)
oo2b=where(min_theta_mean_b lt 3 and sepb gt 2 and sepb lt 5,moo2b)
oo2z=where(min_theta_mean_z lt 3,moo2z)

gbin=0.2 & gmin=5. & gmax=22.3
hg1b=histogram(phot_g_mean_mag_b[oo1b],min=gmin,max=gmax,bin=gbin)
hg2b=histogram(phot_g_mean_mag_b[oo2b],min=gmin,max=gmax,bin=gbin)
hg2z=histogram(phot_g_mean_mag_z[oo2z],min=gmin,max=gmax,bin=gbin)
xhg=findgen(n_elements(hg1b))*gbin+gmin
plot,xhg,hg2b/total(hg2b),psym=10,xtitle='g',ytitle='#',title="!4h!X<3'",xr=[10,22.3],yr=[0,0.085],/xs,/ys,xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk,xthick=thk,ythick=thk,charthick=thk,charsize=csz,col=0
oplot,xhg,hg2z/total(hg2z),psym=10,thick=thk,col=0
oplot,xhg,hg2b/total(hg2b),psym=10,thick=thk,col=2
oplot,xhg,hg1b/total(hg1b),psym=10,thick=thk,col=6
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.1*(!y.crange[1]-!y.crange[0]),'[+ve set] best !4d!Xpos<0.8"',charthick=thk,charsize=csz,col=6
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.2*(!y.crange[1]-!y.crange[0]),'best 2"<!4d!Xpos<5"',charthick=thk,charsize=csz,col=2
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.3*(!y.crange[1]-!y.crange[0]),'[-ve set] all worst',charthick=thk,charsize=csz,col=0
write_png,'CSCGDR3_NWAY_xmatch_gmag_posVneg.png',tvrd(/true)

rpbin=0.2 & rpmin=5. & rpmax=23.05
hrp1b=histogram(phot_rp_mean_mag_b[oo1b],min=rpmin,max=rpmax,bin=rpbin)
hrp2b=histogram(phot_rp_mean_mag_b[oo2b],min=rpmin,max=rpmax,bin=rpbin)
hrp2z=histogram(phot_rp_mean_mag_z[oo2z],min=rpmin,max=rpmax,bin=rpbin)
xhrp=findgen(n_elements(hrp1b))*rpbin+rpmin
plot,xhrp,hrp2b/total(hrp2b),psym=10,xtitle='rp',ytitle='#',title="!4h!X<3'",xr=[10,22.3],yr=[0,0.085],/xs,/ys,xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk,xthick=thk,ythick=thk,charthick=thk,charsize=csz,col=0
oplot,xhrp,hrp2z/total(hrp2z),psym=10,thick=thk,col=0
oplot,xhrp,hrp2b/total(hrp2b),psym=10,thick=thk,col=2
oplot,xhrp,hrp1b/total(hrp1b),psym=10,thick=thk,col=6
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.1*(!y.crange[1]-!y.crange[0]),'[+ve set] best !4d!Xpos<0.8" (+ve set)',charthick=thk,charsize=csz,col=6
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.2*(!y.crange[1]-!y.crange[0]),'best 2"<!4d!Xpos<5"',charthick=thk,charsize=csz,col=2
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.3*(!y.crange[1]-!y.crange[0]),'[-ve set] all worst',charthick=thk,charsize=csz,col=0
write_png,'CSCGDR3_NWAY_xmatch_rpmag_posVneg.png',tvrd(/true)

bpbin=0.2 & bpmin=5. & bpmax=23.05
hbp1b=histogram(phot_bp_mean_mag_b[oo1b],min=bpmin,max=bpmax,bin=bpbin)
hbp2b=histogram(phot_bp_mean_mag_b[oo2b],min=bpmin,max=bpmax,bin=bpbin)
hbp2z=histogram(phot_bp_mean_mag_z[oo2z],min=bpmin,max=bpmax,bin=bpbin)
xhbp=findgen(n_elements(hbp1b))*bpbin+bpmin
plot,xhbp,hbp2b/total(hbp2b),psym=10,xtitle='bp',ytitle='#',title="!4h!X<3'",xr=[10,22.3],yr=[0,0.085],/xs,/ys,xticklen=1,yticklen=1,xgridstyle=1,ygridstyle=1,thick=thk,xthick=thk,ythick=thk,charthick=thk,charsize=csz,col=0
oplot,xhbp,hbp2z/total(hbp2z),psym=10,thick=thk,col=0
oplot,xhbp,hbp2b/total(hbp2b),psym=10,thick=thk,col=2
oplot,xhbp,hbp1b/total(hbp1b),psym=10,thick=thk,col=6
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.1*(!y.crange[1]-!y.crange[0]),'[+ve set] best !4d!Xpos<0.8"',charthick=thk,charsize=csz,col=6
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.2*(!y.crange[1]-!y.crange[0]),'2 best"<!4d!Xpos<5"',charthick=thk,charsize=csz,col=2
xyouts,!x.crange[0]+0.05*(!x.crange[1]-!x.crange[0]),!y.crange[1]-0.3*(!y.crange[1]-!y.crange[0]),'[-ve set] all worst',charthick=thk,charsize=csz,col=0
write_png,'CSCGDR3_NWAY_xmatch_bpmag_posVneg.png',tvrd(/true)

;	make images in gmag-rpmag vs g for the same cases as above to demonstrate the differences in 2D
;	should rightly be contour plots
grp_g_img1b=hist_2d(g_rp_b[oo1b],phot_g_mean_mag_b[oo1b],min1=-2,max1=8,min2=0,max2=23,bin1=0.1,bin2=0.25)
grp_g_img2b=hist_2d(g_rp_b[oo2b],phot_g_mean_mag_b[oo2b],min1=-2,max1=8,min2=0,max2=23,bin1=0.1,bin2=0.25)
grp_g_img2z=hist_2d(g_rp_z[oo2z],phot_g_mean_mag_z[oo2z],min1=-2,max1=8,min2=0,max2=23,bin1=0.1,bin2=0.25)
sz=size(grp_g_img1b) & window,3,xsize=sz[1]*8,ysize=sz[2]*8 & tvscl,smooth(1.0*rebin(grp_g_img1b,sz[1]*8,sz[2]*8),5) & write_png,'grp_g_img1b.png',tvrd(/true)
sz=size(grp_g_img2b) & window,3,xsize=sz[1]*8,ysize=sz[2]*8 & tvscl,smooth(1.0*rebin(grp_g_img2b,sz[1]*8,sz[2]*8),5) & write_png,'grp_g_img2b.png',tvrd(/true)
sz=size(grp_g_img2z) & window,3,xsize=sz[1]*8,ysize=sz[2]*8 & tvscl,smooth(1.0*rebin(grp_g_img2z,sz[1]*8,sz[2]*8),5) & write_png,'grp_g_img2z.png',tvrd(/true)

end
