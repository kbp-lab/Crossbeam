######### These functions are used in the main code mainly for plotting purposes #########

import numpy as np
import scipy as sp
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.integrate import simps
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar 
from matplotlib.ticker import FormatStrFormatter



# plot the single transducer fields 
def plot_single_trxd_fields(max_pressure, dB_intensity, vx, vy, x, fr, dx, extent, figure_path):

    fig = plt.figure(figsize=(16.5, 3.5), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig) 
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title('(a) Maximum Pressure')
    plt1 = ax1.imshow(max_pressure, cmap='jet', extent=extent)
    fig.colorbar(ax = ax1, mappable = plt1, orientation = 'vertical', 
              ticklocation = 'right', label='$\mathrm{Pa}$', 
               aspect=50, shrink=1)
    ax1.set_ylabel(r'$\mathrm{mm}$', fontsize=16)
    ax1.set_xlabel(r'$\mathrm{mm}$', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('(b) Axial Beam Profile')
    ax2.plot(x, dB_intensity[int(fr/dx), :], color='purple')
    ax2.set_xticks([])
    ax2.set_ylabel(r'$\mathrm{dB}$', fontsize=16)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    plt.axhline(y=-3, color='k', ls=':', lw=1)
    ax2.set_xlim(0, 50)
    plt.grid(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('(c) Lateral Beam Profile')
    ax3.plot(x, dB_intensity[:, int(fr/dx)], color='tab:orange')
    ax3.set_ylabel(r'$\mathrm{dB}$', fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.axhline(y=-3, color='k', ls=':', lw=1)
    ax3.set_xlabel('$\mathrm{mm}$', fontsize=16)
    ax3.set_xlim(0, 50)
    plt.grid(False)

    ax4 = fig.add_subplot(gs[:, 2])
    ax4.set_title(r'(d) Particle Velocity, $v_x(t)$')
    plt4 = ax4.imshow(vx, cmap='RdBu_r', extent=extent)
    fig.colorbar(ax = ax4, mappable = plt4, orientation = 'vertical', 
              ticklocation = 'right', label='$\mathrm{m/s}$', 
               aspect=50, shrink=1)
    ax4.set_ylabel(r'$\mathrm{mm}$', fontsize=16)
    ax4.set_xlabel(r'$\mathrm{mm}$', fontsize=16)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.grid(False)

    ax5 = fig.add_subplot(gs[:, 3])
    ax5.set_title(r'(e) Particle Velocity, $v_y(t)$')
    plt5 = ax5.imshow(vy, cmap='RdBu_r', extent=extent)
    fig.colorbar(ax = ax5, mappable = plt5, orientation = 'vertical', 
              ticklocation = 'right', label='$\mathrm{m/s}$', 
               aspect=50, shrink=1)
    ax5.set_ylabel(r'$\mathrm{mm}$', fontsize=16)
    ax5.set_xlabel(r'$\mathrm{mm}$', fontsize=16)
    ax5.tick_params(axis='both', which='major', labelsize=14)
    ax5.grid(False)

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + 'single_trxd_data', dpi=600)

    plt.show()

# plot the orthgonal transducers fields 
def plot_orthogonal_trxd_fields(max_pressure, dB_intensity, vx, vy, x, fr, dx, extent, figure_path):

    fig = plt.figure(figsize=(16.5, 3.5), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title('(a) Maximum Pressure')
    plt1 = ax1.imshow(max_pressure, cmap='jet', extent=extent)
    fig.colorbar(ax = ax1, mappable = plt1, orientation = 'vertical', 
              ticklocation = 'right', label='$\mathrm{Pa}$', 
               aspect=50, shrink=1)
    ax1.set_ylabel(r'$\mathrm{mm}$', fontsize=16)
    ax1.set_xlabel(r'$\mathrm{mm}$', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('(b) Horizontal Beam Profile')
    ax2.plot(x, dB_intensity[int(fr/dx), :], color='purple')
    ax2.set_xticks([])
    ax2.set_ylabel(r'$\mathrm{dB}$', fontsize=16)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    plt.axhline(y=-3, color='k', ls=':', lw=1)
    ax2.set_xlim(0, 50)
    plt.grid(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('(c) Diagonal Beam Profile')
    ax3.plot(x, np.diag(dB_intensity), color='tab:orange')
    ax3.set_ylabel(r'$\mathrm{dB}$', fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.axhline(y=-3, color='k', ls=':', lw=1)
    ax3.set_xlabel('$\mathrm{mm}$', fontsize=16)
    ax3.set_xlim(0, 50)
    plt.grid(False)

    ax4 = fig.add_subplot(gs[:, 2])
    ax4.set_title(r'(d) Particle Velocity, $v_x(t)$')
    plt4 = ax4.imshow(vx, cmap='RdBu_r', extent=extent)
    fig.colorbar(ax = ax4, mappable = plt4, orientation = 'vertical', 
              ticklocation = 'right', label='$\mathrm{m/s}$', 
               aspect=50, shrink=1)
    ax4.set_ylabel(r'$\mathrm{mm}$', fontsize=16)
    ax4.set_xlabel(r'$\mathrm{mm}$', fontsize=16)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.grid(False)

    ax5 = fig.add_subplot(gs[:, 3])
    ax5.set_title(r'(e) Particle Velocity, $v_y(t)$')
    plt5 = ax5.imshow(vy, cmap='RdBu_r', extent=extent)
    fig.colorbar(ax = ax5, mappable = plt5, orientation = 'vertical', 
              ticklocation = 'right', label='$\mathrm{m/s}$', 
               aspect=50, shrink=1)
    ax5.set_ylabel(r'$\mathrm{mm}$', fontsize=16)
    ax5.set_xlabel(r'$\mathrm{mm}$', fontsize=16)
    ax5.tick_params(axis='both', which='major', labelsize=14)
    ax5.grid(False)

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path +  'orthogonal_trxd_data', dpi=600)

    plt.show()


# functions to plot gradients (single transducer) 
def plot_single_trxd_gradients(vx, vy, dvxdx, dvydx, dvxdy, dvydy, x, fr, dx, t1, figure_path):
  
    plt.figure(figsize=(16.5, 3.5))
    plt.subplot(1, 4, 1)
    plt.title(r'Single Transducer $\nabla_x (v_x)$')
    fig1, = plt.plot(x, vx[t1, int(fr/dx), :], label=r'$v_x$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
        
    plt.twinx()
    
    fig2, = plt.plot(x, dvxdx[0, int(fr/dx), :], 
          label=r'$\frac{\partial v_x}{\partial x}$', color='orangered', lw=1)
    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')
    plt.xlabel('mm')
    plt.xlim(20, 40)
    plt.legend([fig1, fig2], 
               [r'$v_x$', r'$\frac{\partial v_x}{\partial x}$'], 
               facecolor='white', fontsize=18, loc='lower right')
    plt.grid(False)

    plt.subplot(1, 4, 2)
    plt.title(r'Single Transducer $\nabla_x (v_y)$')
    # since on axis the y-component of the velocity is zero, we will plot it 
    # along a non-axial row. So, for example 250 instead of 300
    fig1, plt.plot(x, vy[t1, 250, :], label=r'$v_y$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
    
    plt.twinx()
    
    fig2, = plt.plot(x, dvydx[0, 250, :], 
          label=r'$\frac{\partial v_y}{\partial x}$', color='orangered', lw=1)
    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')
    plt.xlim(20, 40)
    plt.xlabel('mm')
    plt.legend([fig1, fig2], 
               [r'$v_y$', r'$\frac{\partial v_y}{\partial x}$'], 
               facecolor='white', fontsize=18, loc='lower right')    
    plt.grid(False)

    plt.subplot(1, 4, 3)
    plt.title(r'Single Transducer $\nabla_y (v_x)$')
    fig1, = plt.plot(x, vx[t1, :, int(fr/dx)], label=r'$v_x$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
              
    plt.twinx()
    
    fig2, = plt.plot(x, dvxdy[0, :, int(fr/dx)], 
          label=r'$\frac{\partial v_x}{\partial y}$', color='orangered', lw=1)

    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')    
    plt.xlim(20, 40)
    plt.ylabel('au')
    plt.xlabel('mm')
    plt.legend([fig1, fig2], 
               [r'$v_x$', r'$\frac{\partial v_x}{\partial y}$'], 
               facecolor='white', fontsize=18, loc='lower right')        
    plt.grid(False)

    plt.subplot(1, 4, 4)
    plt.title(r'Single Transducer $\nabla_y (v_y)$')
    fig1, = plt.plot(x, vy[t1, :, int(fr/dx)], label=r'$v_y$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
              
    plt.twinx()
        
    fig2, = plt.plot(x, dvydy[0, :, int(fr/dx)], 
          label=r'$\frac{\partial v_y}{\partial y}$', color='orangered', lw=1)
    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')  
    plt.xlim(20, 40)
    plt.ylabel('au')
    plt.xlabel('mm')
    plt.legend([fig1, fig2], 
               [r'$v_y$', r'$\frac{\partial v_y}{\partial y}$'], 
               facecolor='white', fontsize=18, loc='lower right')            
    plt.grid(False)

    plt.tight_layout()

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + 'single_trxd_gradients', dpi=600)
    plt.show()


# functions to plot gradients (orthogonal transducers ) 
def plot_orthogonal_trxd_gradients(vx, vy, dvxdx, dvydx, dvxdy, dvydy, x, fr, dx, t1, figure_path):
    plt.figure(figsize=(16.5, 3.5))
    plt.subplot(1, 4, 1)
    plt.title(r'Orthogonal Transducers $\nabla_x (v_x)$')
    fig1, = plt.plot(x, vx[t1, int(fr/dx), :], label=r'$v_x$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
              
    plt.twinx()
            
    fig2, = plt.plot(x, dvxdx[0, int(fr/dx), :], 
          label=r'$\alpha \frac{\partial v_x}{\partial x}$', color='orangered', lw=1)
    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')     
    plt.xlabel('mm')
    plt.xlim(20, 40)
    plt.legend([fig1, fig2], 
               [r'$v_x$', r'$\frac{\partial v_x}{\partial x}$'], 
               facecolor='white', fontsize=18, loc='lower right')          
    plt.grid(False)

    plt.subplot(1, 4, 2)
    plt.title(r'Orthogonal Transducers $\nabla_x (v_y)$')
    # we will still keep this row at 250, since the vertical and horizontal
    # traces are identical copies in this setup. let's see something more 
    # interesting! 
    fig1, = plt.plot(x, vy[t1, 250, :], label=r'$v_y$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
              
    plt.twinx()
    
    fig2, = plt.plot(x, dvydx[0, 250, :], 
          label=r'$\frac{\partial v_y}{\partial x}$', color='orangered', lw=1)
    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')  
    plt.xlim(20, 40)
    plt.xlabel('mm')
    plt.legend([fig1, fig2], 
               [r'$v_y$', r'$\frac{\partial v_y}{\partial x}$'], 
               facecolor='white', fontsize=18, loc='lower right')        
    plt.grid(False)

    plt.subplot(1, 4, 3)
    plt.title(r'Orthogonal Transducers $\nabla_y (v_x)$')
    fig1, = plt.plot(x, vx[t1, :, int(fr/dx)], label=r'$v_x$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
              
    plt.twinx()
        
    fig2, = plt.plot(x, dvxdy[0, :, int(fr/dx)], 
          label=r'$\frac{\partial v_x}{\partial y}$', color='orangered', lw=1)
    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')     
    plt.xlim(20, 40)
    plt.xlabel('mm')
    plt.legend([fig1, fig2], 
               [r'$v_x$', r'$\frac{\partial v_x}{\partial y}$'], 
               facecolor='white', fontsize=18, loc='lower right')           
    plt.grid(False)

    plt.subplot(1, 4, 4)
    plt.title(r'Orthogonal Transducers $\nabla_y (v_y)$')
    fig1, = plt.plot(x, vy[t1, :, int(fr/dx)], label=r'$v_y$', color='dodgerblue', lw=1)
    plt.ylabel('au', color='dodgerblue')
    plt.yticks(color='dodgerblue')
    plt.grid(False)
              
    plt.twinx()
            
    fig2, = plt.plot(x, dvydy[0, :, int(fr/dx)], 
          label=r'$\frac{\partial v_y}{\partial y}$', color='orangered', lw=1)
    plt.ylabel('au', color='orangered')
    plt.yticks(color='orangered')     
    plt.xlim(20, 40)
    plt.xlabel('mm')
    plt.legend([fig1, fig2], 
               [r'$v_y$', r'$\frac{\partial v_y}{\partial y}$'], 
               facecolor='white', fontsize=18, loc='lower right')               
    plt.grid(False)

    plt.tight_layout()

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + 'orthogonal_trxd_gradients', dpi=600)
    plt.show() 

def plot_Mean_Eulerian_Pressure(pE, x, dx, fr, dpEdx, dpEdy, col, figure_path=None):

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    ax1 = axs[0]
    img = ax1.imshow(pE, extent=[0, 50, 50, 0])
    ax1.set_title(r'(a) Single $\langle P_E \rangle$', fontsize=16)
    ax1.set_xlabel('mm', fontsize=16)
    ax1.set_ylabel('mm', fontsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xlim(10, 50)
    ax1.set_ylim(50, 10)
    ax1.grid(False)
    
    # Add a colorbar
    cbar = fig.colorbar(img, ax=ax1, shrink=1)
    cbar.set_label('Pa', fontsize=20)
    cbar.ax.tick_params(labelsize=14)


    # Select the second subplot for the twin axis plot
    ax1 = axs[1]

    fig1 = ax1.plot(x[col:], pE[int(fr/dx), col:], color='royalblue', lw=1.5)
    ax1.set_title('(b) Axial', fontsize=16)
    ax1.set_xlabel('mm', fontsize=16)
    ax1.set_ylabel('Pa', color='royalblue', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.grid(False)

    ax2 = ax1.twinx()
    fig2 = ax2.plot(x[col:], dpEdx[int(fr/dx), col:] , c='r', lw=1.5)
    ax2.set_ylabel('Pa/m', color='r', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=16)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-1,1))
    ax2.set_xlim(10, 50)
    ax2.grid(False)
    # Add a legend
    ax1.legend([fig1[0], fig2[0]],
               [r'$\langle P_E \rangle$',
                r'$\frac{\partial{\langle P_E \rangle}}{\partial{x}}$'],
               facecolor='white', fontsize=16, loc='lower right', framealpha=1.0)


    # Select the third subplot for the twin axis plot
    ax1 = axs[2]
    fig1 = ax1.plot(x, pE[:, int(fr/dx)], color='royalblue', lw=1.5)
    ax1.set_title('(c) Lateral', fontsize=16)
    ax1.set_xlabel('mm', fontsize=16)
    ax1.set_ylabel('Pa', color='royalblue', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.grid(False)

    ax2 = ax1.twinx()
    fig2 = ax2.plot(x, dpEdy[:, int(fr/dx)], color='r', lw=1.5)
    ax2.set_ylabel('Pa/m', color='r', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=16)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-1,1))
    ax2.set_xlim(10, 50)
    ax2.grid(False)
    # Add a legend
    legend = ax2.legend([fig1[0], fig2[0]],
               [r'$\langle P_E \rangle}$',
                r'$\frac{\partial{\langle P_E \rangle}}{\partial{x}}$'],
               facecolor='white', fontsize=16, loc='lower right', framealpha=1.0)
    legend.set_zorder(100)  # Set a high zorder value
 

    plt.tight_layout()
    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + 'PE_dpEdy_single', dpi=600)

    plt.show()


def plot_single_trxd_ARFx(Fx, filtered_Fx, min_, max_, text_loc, col, 
	x, dx, Nx, dy, Ny, fr, file_name, figure_path):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title('(a) $ARF_x$', fontsize=16)
    # since there are lots of sharp gradients near the surface of the transducer,
    # we end up having some extreme and meaningless values on the surface. To avoid 
    # those, we will slice the x-dimension col onwatds: [:, col:-col]
    plt.imshow(Fx[:, col:-col], cmap='RdBu_r', 
             extent=(col * dx * 1e3, (Nx - col) * dx * 1e3, Ny * dy * 1e3, 0), 
            vmin=min_, vmax=max_)
    plt.colorbar(format='%.2e', label='$N/m^3$', shrink=1)
    plt.clim(min_, max_)
    plt.xlim(10, 40)
    plt.ylim(45, 15)
    plt.ylabel(r'$\mathrm{mm}$', fontsize=16)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.text(text_loc[0], text_loc[1],
           'max: ' + '{:0.2e}'.format(max_), 
           fontsize=14)
    plt.grid(False)

    plt.subplot(1, 3, 2)
    plt.title('(b) $ARF_x$ Trace', fontsize=16)
    plt.plot(x[col:-col], Fx[int(fr/dx), col:-col], color='deepskyblue', label='raw')
    plt.plot(x[col:-col], filtered_Fx[int(fr/dx), col:-col], color='red', lw=2, 
          label='filtered')
    plt.xlim(10, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.ylabel(r'$\mathrm{N/m^3}$', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(facecolor='white', fontsize=14)
    plt.grid(False)

    plt.subplot(1, 3, 3)
    plt.title('(c) Filtered $ARF_x$', fontsize=16)
    plt.imshow(filtered_Fx[:, col:-col], cmap='RdBu_r', 
             extent=(col * dx * 1e3, (Nx - col) * dx * 1e3, Ny * dy * 1e3, 0), 
            vmin=min_, vmax=max_)
    plt.colorbar(format='%.2e', label='$N/m^3$', shrink=1)
    plt.xlim(10, 40)
    plt.ylim(45, 15)
    plt.ylabel(r'$\mathrm{mm}$', fontsize=16)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)

    plt.tight_layout()

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + file_name, dpi=600)
        
    plt.show()


def plot_single_trxd_ARFy(Fy, filtered_Fy, min_, max_, text_loc, col, 
	x, dx, Nx, dy, Ny, fr, file_name, figure_path):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title('(a) $ARF_y$', fontsize=16)
    # since there are lots of sharp gradients near the surface of the transducer,
    # we end up having some extreme and meaningless values on the surface. To avoid 
    # those, we will slice the x-dimension 50 onwatds: [:, 50:]
    plt.imshow(Fy[:, col:-col], cmap='RdBu_r', 
             extent=(col * dx * 1e3, (Nx - col) * dx * 1e3, Ny * dy * 1e3, 0), 
            vmin=min_, vmax=max_)
    plt.colorbar(format='%.2e', label='$N/m^3$', shrink=1)
    plt.clim(min_, max_)
    plt.xlim(10, 40)
    plt.ylim(45, 15)
    plt.ylabel(r'$\mathrm{mm}$', fontsize=16)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.text(text_loc[0], text_loc[1], 'max: ' + 
            '{:0.2e}'.format(max_), fontsize=14)
    plt.grid(False)

    plt.subplot(1, 3, 2)
    plt.title('(b) $ARF_y$ Trace', fontsize=16)
    # note the minus sign before Fy. In subplot (1,3,1), upward/downward directions 
    # correspond to +/- signs. In that regime, the forces are pulling away from 
    # one another. But if we plot a crosssection of this, the +/- signs will translate 
    # to right/left directions, falsely implying that the forces are converging on one 
    # another. That is why we multiply Fy by -1 so to preserve the directionality of the 
    # force. 
    plt.plot(x, -Fy[:,int(fr/dx)], color='deepskyblue', label='raw')
    plt.plot(x, -filtered_Fy[:, int(fr/dx)], color='red', lw=2, label='filtered')
    plt.xlim(10, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.ylabel(r'$\mathrm{N/m^3}$', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(facecolor='white', fontsize=14)
    plt.grid(False)

    plt.subplot(1, 3, 3)
    plt.title('(c) Filtered $ARF_y$', fontsize=16)
    plt.imshow(filtered_Fy[:, col:-col], cmap='RdBu_r', 
             extent=(col * dx * 1e3, (Nx - col) * dx * 1e3, Ny * dy * 1e3, 0), 
            vmin=min_, vmax=max_)
    plt.colorbar(format='%.2e', label='$N/m^3$', shrink=1)
    plt.xlim(10, 40)
    plt.ylim(45, 15)
    plt.ylabel(r'$\mathrm{mm}$', fontsize=16)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)

    plt.tight_layout()

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + file_name, dpi=600)
    
    plt.show()


def plot_single_trxd_ARF_quiver(p_max, Fx, Fy, file_name, scale, col, 
	dx, Nx, dy, Ny, X, Y, qres, figure_path):
    plt.figure(figsize=(12, 8))

    plt.title(r'$P_{max}$ $\mathrm{&}$ $ARF$', fontsize=24)
    plt.imshow(1e-6 * p_max[:, col:-col], cmap='jet', 
            extent=(col * dx * 1e3, (Nx - col) * dx * 1e3, Ny * dy * 1e3, 0))
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label('$\mathrm{MPa}$', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    plt.grid(False)

    plt.quiver(X, Y, Fx[::int(qres/3), col:-col:qres], 
            Fy[::int(qres/3), col:-col:qres], scale=scale, 
            color='white', width=3e-3, alpha=1)
    plt.grid(False)

    plt.ylabel(r'$\mathrm{mm}$', fontsize=22)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(20, 41, 5), fontsize=14)
    plt.xlim(10, 40)
    plt.ylim(40, 20)

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + file_name, dpi=600)

    plt.show()




def plot_orth_trxd_ARF_quiver(p_max, Fx, Fy, x_pos, y_pos, file_name, scale, qres, figure_path):
    qres = qres
    # define a meshgrid over those x and y positions
    X, Y = np.meshgrid(x_pos[50::qres], y_pos[50::qres])

    plt.figure(figsize=(8, 8))
    plt.title('$P_{max}$ & $ARF$', fontsize=24)
    plt.imshow(1e-6 * p_max[50:, 50:], cmap='jet', 
            extent=(5, 50, 50, 5))
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label('$\mathrm{MPa}$', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    plt.grid(False)

    plt.quiver(X, Y, Fx[50::qres, 50::qres], 
            Fy[50::qres, 50::qres], scale=scale, 
            color='k', width=3.5e-3, alpha=1)
    plt.grid(False)

    plt.ylabel(r'mm', fontsize=18)
    plt.xlabel(r'mm', fontsize=18)
    plt.xticks(np.arange(25, 36, 2), fontsize=14)
    plt.yticks(np.arange(25, 36, 2), fontsize=14)
    plt.xlim(25, 35)
    plt.ylim(35, 25)

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + file_name, dpi=600)

    plt.show()




def plot_propagation_snapshot(orthogonal_pressure_trimmed, antidiagonal_pressures,
                              diagonal_pressures,
                              diagonal_axis, file_name, figure_path):


    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title('(a) Pressure')
    plt1 = ax1.imshow(orthogonal_pressure_trimmed[2300] * 1e-6, 
                    cmap='jet', extent=(25, 35, 35, 25))
    fig.colorbar(ax = ax1, mappable = plt1, orientation = 'vertical', 
              ticklocation = 'right', label='MPa', aspect=40, shrink=1)
    ax1.set_ylabel(r'mm', fontsize=16)
    ax1.set_xlabel(r'mm', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(False)

    # these pressure indices are chosen for better visual representation 
    # of the temporal evolution of the waves
    pressure_indices = [14, 20, 25, 30, 35, 39]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('(b) Antidiagonal Beam Profile')
    for i, j in enumerate(pressure_indices):
	    if i < len(pressure_indices) - 1:
	        ax2.plot(diagonal_axis, antidiagonal_pressures[j] * 1e-6,
	              color='purple', alpha=0.1 * (i + 1), label='t-'+
	               str((len(pressure_indices)-1-i)))
	    if i == len(pressure_indices) - 1:
	        ax2.plot(diagonal_axis, antidiagonal_pressures[j] * 1e-6,
	              color='purple', alpha=1, label='t')

    handles, labels = plt.gca().get_legend_handles_labels()
    #specify order of items in legend
    order = [5, 4, 3, 2, 1, 0]
    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
             facecolor='white', loc='right', framealpha=1) 
    plt.xticks([])
    ax2.tick_params(axis='y', which='major', labelsize=14)
    plt.xlim(25, 25 + 10 * np.sqrt(2))
    ax2.set_ylabel(r'Pressure (MPa)', fontsize=14)
    plt.grid(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('(c) Diagonal Beam Profile')
    for i, j in enumerate(pressure_indices):
	    if i < len(pressure_indices) - 1:
	        ax3.plot(diagonal_axis, diagonal_pressures[j] * 1e-6,
	              color='purple', alpha=0.1 * (i + 1), label='t-'+
	               str((len(pressure_indices)-1-i)))
    if i == len(pressure_indices) - 1:
	        ax3.plot(diagonal_axis, diagonal_pressures[j] * 1e-6,
	              color='purple', alpha=1, label='t')

    handles, labels = plt.gca().get_legend_handles_labels()
    #specify order of items in legend
    order = [5, 4, 3, 2, 1, 0]
    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
             facecolor='white', loc='right', framealpha=1) 

    ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim(25, 25 + 10 * np.sqrt(2))
    ax3.set_ylabel(r'Pressure (MPa)', fontsize=14)
    plt.xlabel(r'mm', fontsize=16)
    plt.grid(False)
    #plt.legend()

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + file_name, dpi=600)

    plt.show()




#@title Function to plot FWHM resolutions of the three arrangements (run this cell)

def plot_FWHM_dB(single_trxd, antiparallel_trxd, orthogonal_trxd, orthogonal_pressure_trimmed,
	col, figure_path=None):
  
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.title('(a) Axial FWHM Resolutions', fontsize=16)
    plt.plot(np.linspace(10, 50, 400), 20 * 
          np.log10(single_trxd.max_pressure[300, col:] / 
                    np.max(single_trxd.max_pressure)), 
          color='tab:orange', lw=2, label='single')
    plt.plot(np.linspace(10, 50, 400), 20 * 
          np.log10(antiparallel_trxd[300, :]/ 
                    np.max(antiparallel_trxd)), 
          color='dodgerblue', lw=2, label='antiparallel')
    plt.plot(np.linspace(10, 50, 400), 20 * 
          np.log10(orthogonal_trxd.max_pressure[300, col:] / 
                    np.max(orthogonal_trxd.max_pressure)), 
          color='maroon', lw=2, label='orthogonal')
    plt.axhline(y=-3, color='k', ls=':')
    plt.legend(facecolor='white', fontsize=16)
    plt.xlim(20, 40)
    plt.ylim(-25, 1)
    plt.ylabel(r'Intensity $\mathrm{(dB)}$', fontsize=16)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.xticks(np.arange(20, 41, 5))
    plt.yticks(np.arange(-25, 1, 5))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)

    plt.subplot(1, 3, 2)
    plt.title('(b) Lateral FWHM Resolutions', fontsize=16)
    plt.plot(np.linspace(0, 50, 500), 20 * 
          np.log10(single_trxd.max_pressure[:, 300] / 
                    np.max(single_trxd.max_pressure)), 
          color='tab:orange', lw=2, label='single')
    plt.plot(np.linspace(0, 50, 500), 20 * 
          np.log10(antiparallel_trxd[:, 200]/ 
                    np.max(antiparallel_trxd)), 
          color='dodgerblue', lw=2, label='antiparallel')
    plt.plot(np.linspace(0, 50, 500), 20 * 
          np.log10(orthogonal_trxd.max_pressure[:, 300] / 
                    np.max(orthogonal_trxd.max_pressure)), 
          color='maroon', lw=2, label='orthogonal')
    plt.axhline(y=-3, color='k', ls=':')
    plt.legend(facecolor='white', fontsize=16, loc='lower center')
    plt.xlim(20, 40)
    plt.ylim(-25, 1)
    plt.ylabel(r'Intensity $\mathrm{(dB)}$', fontsize=16)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.xticks(np.arange(20, 41, 5))
    plt.yticks(np.arange(-25, 1, 5))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)

    plt.subplot(1, 3, 3)
    plt.title('(c) Cross FWHM Resolutions', fontsize=16)
    plt.plot(np.linspace(25, 25 + 10 * np.sqrt(2), 100), 
          20 * np.log10(np.diag(np.max(orthogonal_pressure_trimmed, axis=0) / 
                                np.max(orthogonal_pressure_trimmed))),
          label='diagonal', lw=2, color='maroon', ls='--')

    plt.plot(np.linspace(25, 25 + 10 * np.sqrt(2), 100), 
          20 * np.log10(np.diag(np.rot90(np.max(
              orthogonal_pressure_trimmed, axis=0) / 
              np.max(orthogonal_pressure_trimmed)))), label='antidiagonal', 
          lw=2, color='maroon')

    plt.axhline(y=-3, color='k', ls=':')
    plt.axhline(y=-6, color='k', ls=':')
    plt.legend(facecolor='white', fontsize=16)
    plt.xlim(25, 38)
    plt.ylim(-25, 1)
    plt.ylabel(r'Intensity $\mathrm{(dB)}$', fontsize=16)
    plt.xlabel(r'$\mathrm{mm}$', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)


    plt.tight_layout()

    if figure_path == None:
        pass
    else:
        plt.savefig(figure_path + 'FWHM_trace_comparison', dpi=600)
    
    plt.show()






