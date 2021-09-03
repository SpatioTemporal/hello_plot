


import numpy
import pystare

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import meshplot as mp
import mpl_toolkits.mplot3d as a3

import cartopy.crs as ccrs
import cartopy.feature as cf

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import xarray
import sys
from datetime import datetime

# Some helper functions for plotting & printing.


def make_stare_htm_info_from_sivs(sivs,dask=None):
    "Make an xarray with all of the STARE htm-ids"
    dask  = ( 'allowed' if dask is None else dask )
    htm = xarray.DataArray(numpy.arange(len(sivs)), dims=("htm"),coords={"htm": sivs})

    data1 = xarray.apply_ufunc(
        pystare.triangulate_indices
        ,htm.htm
        ,input_core_dims=[["htm"]]
        ,dask=dask
    )

    lons = xarray.DataArray(data1.data[()][0].reshape(len(htm),3),dims=("htm","vertex"),coords={"htm": sivs})
    lats = xarray.DataArray(data1.data[()][1].reshape(len(htm),3),dims=("htm","vertex"),coords={"htm": sivs})
    intmat = xarray.DataArray(data1.data[()][2],dims=("htm","vertex"),coords={"htm": sivs})

    tiv = pystare.now()
    return xarray.Dataset(
        {
            "lons"     : lons
            ,"lats"    : lats
            ,"intmat"  : intmat
        }
        ,attrs = {
            "creator"          : "make_stare_htm_info_at_level v1"
            ,"datetime"        : tiv
            ,"datetime_string" : numpy.array(pystare.core._to_utc_approximate([tiv]),dtype='datetime64[ms]')
        }
    )

#

###########################################################################
# https://stackoverflow.com/questions/41596386/tripcolor-using-rgb-values-for-each-vertex
#
def colors_to_cmap(colors):
    '''
    colors_to_cmap(nx3_or_nx4_rgba_array) yields a matplotlib colormap object that, when
    that will reproduce the colors in the given array when passed a list of n evenly
    spaced numbers between 0 and 1 (inclusive), where n is the length of the argument.

    Example:
      cmap = colors_to_cmap(colors)
      zs = numpy.asarray(range(len(colors)), dtype=numpy.float) / (len(colors)-1)
      # cmap(zs) should reproduce colors; cmap[zs[i]] == colors[i]
    '''
    colors = numpy.asarray(colors)
    if colors.shape[1] == 3:
        colors = numpy.hstack((colors, numpy.ones((len(colors),1))))
    steps = (0.5 + numpy.asarray(range(len(colors)-1), dtype=numpy.float))/(len(colors) - 1)
    return mpl.colors.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] + 
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] + 
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors))

# https://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent

def make_transparent(img):
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

#################################

def sgn(x,y):
    dx01=x[1]-x[0]
    dy01=y[1]-y[0]
    dx02=x[2]-x[0]
    dy02=y[2]-y[0]
    cp = dx01*dy02-dx02*dy01
    if cp > 0:
        return 1
    elif cp < 0:
        return -1
    else:
        return 0

class FigAxContainer:
    def __init__(self, figax):
        self.fig = figax[0]  # class 'matplotlib.figure.Figure'
        self.ax = figax[1]   # class 'cartopy.mpl.geoaxes.GeoAxesSubplot'
        return

def make_cyclic(a):
    r = numpy.zeros(len(a)+1,dtype=a.dtype)
    r[:-1]=a; r[-1]=a[0]
    return r

def divert_stderr():
    sys.stderr = open('stderr.out~','w')  
    return

def restore_stderr(_verbose=None):
    _verbose = (True if _verbose is None else _verbose)
    sys.stderr.close()
    if _verbose:
        with open('stderr.out~') as f:
            count = sum(1 for _ in f)
        if count > 0:
            print(f"{count} warnings or errors encountered while stderr diverted. See stderr.out~")
    sys.stderr = sys.__stderr__
    return

def add_coastlines(figax,set_global=False):
    "Add coastlines to the plot."
    ax = figax.ax
    if set_global:
        ax.set_global()
    ax.coastlines()
    return figax


def hello_plot(spatial_index_values=None
               , figax=None, plot_options=None, set_global=None, set_coastlines=None
               , show_fig=None, color=None, lw=None, legend_label=None, use_dash=None
               , bbox_to_anchor=None
               , verbose=None
               , rasterized=None
               , face_zs         = None
#               , face_colors     = None
               , face_edgecolors = None
               , face_lw         = None
               , face_vmin       = None
               , face_vmax       = None
               , face_cmap       = None
               , face_shading    = None
               , face_alpha      = None
               , siv_triang      = None
               , triplot         = None
              ):
    
    spatial_index_values = (None if spatial_index_values is None else spatial_index_values)
    figax = (None if figax is None else figax)
    plot_options = ({'projection': ccrs.PlateCarree()
                     , 'transform': ccrs.Geodetic()} if plot_options is None else plot_options)
    set_global = (False if set_global is None else set_global)
    set_coastlines = (True if set_coastlines is None else set_coastlines)
    show_fig = (True if show_fig is None else show_fig)
    color = (None if color is None else color)
    lw = (1 if lw is None else lw)
    legend_label = (None if legend_label is None else legend_label)
    use_dash = (None if use_dash is None else use_dash)
    bbox_to_anchor = (None if bbox_to_anchor is None else bbox_to_anchor)
    verbose = (True if verbose is None else verbose)
    rasterized = (True if rasterized is None else rasterized)

    face_zs         = None if face_zs is None else face_zs
#    face_colors     = None if face_colors is None else face_colors
    face_edgecolors = None if face_edgecolors is None else face_edgecolors
    face_lw         = 0.5 if face_lw is None else face_lw
    face_vmin       = None if face_vmin is None else face_vmin
    face_vmax       = None if face_vmax is None else face_vmax
    face_cmap       = None if face_cmap is None else face_cmap
    face_shading    = 'gouraud' if face_shading is None else face_shading
    face_alpha      = 0.75 if face_alpha is None else face_alpha
    triplot = True if triplot is None else triplot

    siv_triang = None if siv_triang is None else siv_triang
    

    if figax is None:
        # Initialize the FigAxContainer the first time it is used
        figax = FigAxContainer(plt.subplots(1, subplot_kw=plot_options))
        if set_global:
            figax.ax.set_global()
        if set_coastlines:
            figax.ax.coastlines()

    if spatial_index_values is not None:
        if not isinstance(spatial_index_values,xarray.Dataset):
            # Calculate vertices and interconnection matrix
            spatial_index_values = make_stare_htm_info_from_sivs(spatial_index_values)
            # lons, lats, intmat = pystare.triangulate_indices(spatial_index_values)

        lons = spatial_index_values.lons.data.reshape(3*len(spatial_index_values.htm))
        lats = spatial_index_values.lats.data.reshape(3*len(spatial_index_values.htm))
        intmat = spatial_index_values.intmat.data

        # Make triangulation object & plot
        if siv_triang is None:
            siv_triang = tri.Triangulation(lons, lats, intmat)

        divert_stderr()
        if triplot:
            if use_dash is not None:
                figax.ax.triplot(siv_triang, c=color, transform=plot_options['transform'], lw=lw,
                                 label="Placeholder", dashes=use_dash, rasterized=rasterized)            
            else:
                figax.ax.triplot(siv_triang, c=color, transform=plot_options['transform'], lw=lw, 
                                 label="Placeholder", rasterized=rasterized)
            
        if face_zs is not None:
            if verbose:
                print('face_zs         ',face_zs)
                #            print('face_colors     ',face_colors)
                print('face_edgecolors ',face_edgecolors)
                print('face_lw         ',face_lw)
                print('face_shading    ',face_shading)
                print('face_vmin       ',face_vmin)
                print('face_vmax       ',face_vmax)
                print('face_cmap       ',face_cmap)
                print('face_alpha      ',face_alpha)
                print('po transform    ',plot_options['transform'])
                print('rasterized      ',rasterized)

            if len(face_zs) == len(siv_triang.x):
                if verbose:
                    print('face_zs no change')
                face_zs_input = face_zs
            else:
                if verbose:
                    print('face_zs tripling')
                face_zs_input = numpy.zeros(face_zs.size,dtype=face_zs.dtype)
                for i in range(face_zs.size):
                    face_zs_input[3*i+0] = face_zs[i]
                    face_zs_input[3*i+1] = face_zs[i]
                    face_zs_input[3*i+2] = face_zs[i]

            if verbose:
                print('siv_triang len: ',len(siv_triang.x))
                print('face_zs    len: ',len(face_zs_input))
            
            figax.ax.tripcolor(siv_triang
                               ,face_zs_input
#                               ,face_colors=face_colors
                               ,edgecolors=face_edgecolors
                               ,lw=face_lw
                               ,shading=face_shading
                               ,vmin=face_vmin
                               ,vmax=face_vmax
                               ,cmap=face_cmap
                               ,alpha=face_alpha
                               ,transform=plot_options['transform']
                               ,rasterized=rasterized)

        restore_stderr(_verbose=verbose)
 
        # Add Legend
        if legend_label is not None:
            if isinstance(legend_label, list):
                # plt.triplot produces two legend entries. 
                #  The first of those are the edges
                #  The second contains the points (nodes).
                bbox_to_anchor = ( (1.7,1.0) if bbox_to_anchor is None else bbox_to_anchor )
                the_handels, the_labels = figax.ax.get_legend_handles_labels()
                figax.ax.legend(handles=the_handels[::2], labels=legend_label, bbox_to_anchor=bbox_to_anchor
                                , loc='upper right', frameon=False) 
            elif isinstance(legend_label, str):
                bbox_to_anchor = ( (1.4,1.0) if bbox_to_anchor is None else bbox_to_anchor )
                h, l = figax.ax.get_legend_handles_labels()
                figax.ax.legend(handles=[h[0]], labels=[legend_label], bbox_to_anchor=bbox_to_anchor
                                , loc='upper right', frameon=False)

    if show_fig:
        # Show figure now
        plt.show()

    return figax,spatial_index_values,siv_triang

def hex16(ival):
    return "0x%016x" % ival

def labeled_plot(sivs,figax,plot_labels=True
                 , bbox_to_anchor=None
                 , plot_options=None
                 , rasterized = None
                ):
    bbox_to_anchor = (None if bbox_to_anchor is None else bbox_to_anchor)
    plot_options = ({'projection': ccrs.PlateCarree()
                     , 'transform': ccrs.Geodetic()} if plot_options is None else plot_options)
    rasterized = ( True if rasterized is None else rasterized )
    
    # Plot each increment (overlay on common axes)
    legend_info = []
    n_legend_colors = len(sivs)
    legend_colors = plt.cm.brg(numpy.linspace(0, 1, n_legend_colors))
    
    for siv_idx, siv in enumerate(sivs):
        
        legend_info.append(f"Spatial ID: {hex16(siv)} {siv}")
        the_dash = ((5, 10) if (siv_idx % 2) == 0 else ())
        the_lw = (0.5 if (siv_idx % 2) == 0 else 0.25)
        if len(legend_info[0]) < 32 or not plot_labels:
            hello_plot(spatial_index_values=[siv], figax=figax, plot_options=plot_options
                       , bbox_to_anchor=bbox_to_anchor
                       , color=legend_colors[siv_idx], show_fig=False, lw=the_lw, use_dash=the_dash
                       , verbose=False
                       , rasterized=rasterized
                      )
        else:
            hello_plot(spatial_index_values=[siv], figax=figax, plot_options=plot_options
                       , bbox_to_anchor=bbox_to_anchor
                       , color=legend_colors[siv_idx], legend_label=legend_info, show_fig=False
                       , lw=the_lw, use_dash=the_dash
                       , verbose=False
                       , rasterized=rasterized
                      )
            

## def hello_plot(
##         spatial_index_values=None
##         ,figax=None
##         ,plot_options={'projection':ccrs.PlateCarree(),'transform':ccrs.Geodetic()}
##         ,set_global=False
##         ,set_coastlines=True
##         ,show=True
##         ,color=None
##         ,alpha=1
##         ,lw=1
##         ,verbose = True
##         ,title = None
##         ):
## 
##     if figax is None:
##         figax = FigAxContainer(plt.subplots(1,subplot_kw=plot_options))
##         if set_global:
##             figax.ax.set_global()
##         if set_coastlines:
##             figax.ax.coastlines()
##     else:
##         ax = figax.ax
##     
##     if spatial_index_values is not None:
##         # Calculate vertices and interconnection matrix
##         lons,lats,intmat = pystare.triangulate_indices(spatial_index_values)
##         
##         # Make triangulation object & plot
##         siv_triang = tri.Triangulation(lons,lats,intmat)
##         # print('plot type triang: ',type(siv_triang))
##         divert_stderr()
##         figax.ax.triplot(siv_triang,c=color,alpha=alpha,transform=plot_options['transform'],lw=lw)
##         restore_stderr(verbose=verbose)
##     
##     if title is not None:
##         plt.title(title)
##     if show:
##         plt.show()
##         
##     return figax

## def hex16(i):
##     return "0x%016x"%i

## def lonlat_from_coords(coords):
##     tmp = numpy.array(coords)
##     lat=tmp[:,1]
##     lon=tmp[:,0]
##     return lon,lat

# km  = 1 # Unit of length
# deg = 1 # Unit of angle

# # Set up the projection and transformation
# # proj         = ccrs.PlateCarree()
# # proj        = ccrs.Robinson()
# proj         = ccrs.Mollweide()
# transf       = ccrs.Geodetic()
# plot_options = {'projection':proj,'transform':transf}
# set_global   = True
# 
# # default_dpi = mpl.rcParamsDefault['figure.dpi']
# # mpl.rcParams['figure.dpi'] = 1.5*default_dpi


class stare_prism(object):
    
    def __init__(self
                 ,siv=None
                 ,tiv=None
                 ,tiv_representation=None
                 ,tiv_scale=None
                 ,tiv_offset=None
                 ,color_reverse=None
                 ,color=None
                 ,color_forward=None
                 ,tiv_mock=None
                 ):
        self.siv      = (None if siv is None else siv)
        self.tiv      = None if tiv is None else tiv
        self.tiv_mock = ([0.0,0.33,0.67,1.0] if tiv_mock is True else tiv_mock)

        tiv_scale = 1          if tiv_scale  is None else tiv_scale  # Scale to a day
        tiv_offset= 0          if tiv_offset is None else tiv_offset # In days, by default

        tiv_representation = 'tai' if tiv_representation is None else tiv_representation
        
        if tiv_representation not in ['ms','tai','ms-utc']:
            raise ValueError("tiv_representation not 'ms', 'tai', nor 'ms-utc'.")

        if tiv_mock is not None:
            self.tiv_plot = self.tiv_mock
        else:
            if type(self.tiv) != numpy.ndarray:
                self.tiv = numpy.array([self.tiv],dtype=numpy.int64)
            if tiv_representation == 'tai':
                triple   = numpy.concatenate(pystare.to_temporal_triple_tai(self.tiv))
                # print(triple)
                # print(type(triple))
                t_triple = pystare.to_JulianTAI(triple)
            elif tiv_representation =='ms': 
                triple = numpy.concatenate(pystare.to_temporal_triple_ms(self.tiv))
                t_triple = pystare.to_JulianTAI(triple)
            else: # 'ms-utc'
                triple = numpy.concatenate(pystare.to_temporal_triple_ms(self.tiv))
                t_triple = pystare.to_JulianUTC(triple)
    
            self.t_lo = ( t_triple[0][0] + t_triple[1][0] ) / tiv_scale + tiv_offset
            self.t_mi = ( t_triple[0][1] + t_triple[1][1] ) / tiv_scale + tiv_offset
            self.t_hi = ( t_triple[0][2] + t_triple[1][2] ) / tiv_scale + tiv_offset
    
            alpha = 0.25
            dtlo = self.t_mi - self.t_lo
            dthi = self.t_hi - self.t_mi
            self.tiv_plot=[self.t_lo,self.t_mi-alpha*dtlo,self.t_mi+alpha*dthi,self.t_hi]
            
        self.lons, self.lats, self.intmat = pystare.triangulate_indices([self.siv])
        
        if color_reverse is None and color is None and color_forward is None:
            self.color_reverse = 'red'
            self.color         = 'green'
            self.color_forward = 'blue'
        elif color is not None:
            self.color_reverse = (color if color_reverse is None else color_reverse)
            self.color         =  color
            self.color_forward = (color if color_forward is None else color_forward)
        else:
            self.color_reverse = ('red'   if color_reverse is None else color_reverse)
            self.color         = 'green'
            self.color_forward = ('blue'  if color_forward is None else color_forward)            
        
        return
    
    def plot1_simple(
        self
              ,figax
              ,alpha      = None
              ,edge_alpha = None
              ,edge_color = None
              ,prism_edge_color = None
              ,end_faces_plot = None
              ,rasterized = None
             ):
        
        alpha               = (0.5 if alpha is None else alpha)
        edge_alpha          = (1 if edge_alpha is None else edge_alpha)
        end_faces_plot = ([True]*6 if end_faces_plot is None else end_faces_plot)
        rasterized = False if rasterized is None else rasterized
        
        self.plot0(figax
                   ,color = self.color
                   ,edge_color = edge_color
                   ,prism_edge_color = prism_edge_color
                   ,alpha      = alpha
                   ,edge_alpha = edge_alpha                   
                   ,z=[self.tiv_plot[0],self.tiv_plot[-1]]
                   ,end_faces_plot = [end_faces_plot[0],end_faces_plot[-1]]
                   ,rasterized=rasterized
                  )

        return figax
    
    def plot1(self
              ,figax
              ,alpha      = None
              ,edge_alpha = None
              ,edge_color = None
              ,prism_edge_color = None
              ,end_faces_plot = None
              ,rasterized = None
             ):
        
        alpha               = (0.5 if alpha is None else alpha)
        edge_alpha          = (0.5 if edge_alpha is None else alpha)
        rasterized = False if rasterized is None else rasterized        
        
        end_faces_plot = ([True]*6 if end_faces_plot is None else end_faces_plot)
        
        self.plot0(figax
                   ,color      = self.color_reverse
                   ,edge_color = edge_color
                   ,prism_edge_color = prism_edge_color
                   ,alpha      = alpha
                   ,edge_alpha = edge_alpha
                   ,z=self.tiv_plot[0:2]
                   ,end_faces_plot = end_faces_plot[0:2]
                  )
        self.plot0(figax
                   ,color = self.color
                   ,edge_color = edge_color
                   ,prism_edge_color = prism_edge_color
                   ,alpha      = alpha
                   ,edge_alpha = edge_alpha
                   ,z=self.tiv_plot[1:3]
                   ,end_faces_plot = end_faces_plot[2:4]
                  )
        self.plot0(figax
                   ,color = self.color_forward
                   ,edge_color = edge_color
                   ,prism_edge_color = prism_edge_color
                   ,alpha      = alpha
                   ,edge_alpha = edge_alpha
                   ,z=self.tiv_plot[2:4]
                   ,end_faces_plot = end_faces_plot[4:6]
                  )
        
        return figax
    
    def plot0(self
             ,figax
             ,z=None
             ,color=None
             ,edge_color=None
             ,prism_edge_color=None
             ,alpha=None
             ,edge_alpha=None
             ,end_faces_plot = None
             ,dbg=None
             ,rasterized=None
            ):
        
        z          = ([0.0,1.0] if z is None else z)
        color      = ('grey' if color is None else color)
        edge_color = (None if edge_color is None else edge_color)
        prism_edge_color = (edge_color if prism_edge_color is None else prism_edge_color)
        alpha      = (0.5 if alpha is None else alpha)
        edge_alpha = (1.0 if edge_alpha is None else edge_alpha)
        end_faces_plot = ([True,True] if end_faces_plot is None else end_faces_plot)
        dbg        = (False if dbg is None else dbg)
        rasterized = False if rasterized is None else rasterized        
        
        # z0=self.tiv_plot[0] # bad
        # temporal_scales = [self.tiv_plot[1]]*4
        
        z0_ = z[0]
        z1_ = [z[1]]
        
        for tr in self.intmat:
            
            if dbg:
                print('tr: ',tr)
            if z1_ is None:
                z = [z0_]*4
            else:
                z = [z0_]*4
                z1 = numpy.array([z1_[int(tr[0]/3)]
                                ,z1_[int(tr[1]/3)]
                                ,z1_[int(tr[2]/3)]
                                ,z1_[int(tr[0]/3)]])
            x = make_cyclic(numpy.array([self.lons[tr[0]],self.lons[tr[1]],self.lons[tr[2]]]))
            y = make_cyclic(numpy.array([self.lats[tr[0]],self.lats[tr[1]],self.lats[tr[2]]]))

            if dbg:
                print('==')
                print(sgn(x,y))
        
            thr = 180
        
            if sgn(x,y) < 0:
                if True:
                    if abs(x[0]-x[1]) > thr:
                        if x[0] < x[1]:
                            x[0] += 360
                            x[3] = x[0]
                        else:
                            x[1] += 360
                if True:
                    if abs(x[0]-x[2]) > thr:
                        if x[0] < x[2]:
                            x[0] += 360
                            x[3] = x[0]
                        else:
                            x[2] += 360            
                if True:
                    if abs(x[1]-x[2]) > thr:
                        if x[1] < x[2]:
                            x[1] += 360.0
                        else:
                            x[2] += 360.0
                        
            if dbg:
                print(x)
                print(y)
                print(sgn(x,y))
                print('.')
                
                print('end_faces_plot: ',end_faces_plot)
    
            x=numpy.array(x)
            y=numpy.array(y)
            z=numpy.array(z)
    
            if end_faces_plot[0]:
                figax.ax.plot_trisurf(x,y,z
                                    ,color=color
                                    ,alpha=alpha
                                    ,rasterized=rasterized
                                    )
            else:
                # This is stupid
                 figax.ax.plot_trisurf(x,y,z
                                    ,color=color
                                    ,alpha=0
                                    ,rasterized=rasterized
                                    )               
        
            if z1 is not None:
                
                if dbg:
                    print(100)
                    
                if prism_edge_color is not None:
                    for i in [0,1,2]:
                        figax.ax.plot3D([x[i],x[i]],[y[i],y[i]],[z[i],z1[i]]
                                        ,c=prism_edge_color
                                        ,alpha=edge_alpha
                                        ,rasterized=rasterized
                                       )                
                
                for i in [0,1,2]:
                    
                    fx = numpy.array( [x[i],x[i+1],x[i+1]] )
                    fy = numpy.array( [y[i],y[i+1],y[i+1]] )
                    fz = numpy.array( [z[i],z[i+1],z1[i]] )
                    
                    vtx = numpy.array(
                        [
                            [x[i],x[i+1],x[i+1]]
                            ,[y[i],y[i+1],y[i+1]]
                            ,[z[i],z[i+1],z1[i]]
                        ]
                    )
                    vtx = numpy.transpose(vtx)
                    if dbg:
                        print('vtx: ',vtx)
                    tri = a3.art3d.Poly3DCollection([vtx])
                    tri.set_facecolor(
                        color
                        # self.color_reverse
                        # mpl.colors.rgb2hex([0.5,0.5,0.5])
                    )
                    tri.set_alpha(alpha)
                    tri.set_edgecolor(edge_color)
                    figax.ax.add_collection3d(tri)

                    vtx = numpy.array(
                        [
                            [x[i],x[i+1],x[i]]
                            ,[y[i],y[i+1],y[i]]
                            ,[z[i],z1[i+1],z1[i]]
                        ]
                    )
                    vtx = numpy.transpose(vtx)
                    tri = a3.art3d.Poly3DCollection([vtx])
                    tri.set_color(color)
                    # tri.set_color(mpl.colors.rgb2hex([0.45,0.45,0.45]))
                    tri.set_alpha(alpha)
                    tri.set_edgecolor(edge_color)
                    figax.ax.add_collection3d(tri)
                    
            if end_faces_plot[0]:
                if prism_edge_color is not None:
                    figax.ax.plot3D(x, y, z
                        ,c=prism_edge_color
                        ,alpha=edge_alpha
                        ,rasterized=rasterized
                        ) 
                
            if end_faces_plot[1]:
                if prism_edge_color is not None:                    
                    figax.ax.plot3D(x, y, z1, c=prism_edge_color, alpha=edge_alpha
                                    ,rasterized=rasterized
                                    ) 
                figax.ax.plot_trisurf(x,y,z1,color=color,alpha=alpha
                                      ,rasterized=rasterized
                                      )
        return figax
