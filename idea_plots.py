import pickle
import matplotlib.pyplot as plt
import numpy as np
from density2potential.plot.matrix_plot import plot_heat_map, plot_surface_3d
from density2potential.io.input_file import parameters
from density2potential.core.linear_response import sum_rule


def fxc_axis_plot(params, f_xc, x1, x2, x3, w, figname='fxc_axes.pdf'):
    # Plot along various axes of f_xc
    # x1, x2, x3 are three lines along which |x-xi| will be plotted
    # w is the frequency

    plt.close('all')
    fig, ((a, b), (c, d)) = plt.subplots(2,2)

    a.plot(np.diag(f_xc[:,:,w].real))
    a.set_title("Local part (along diag)")

    a.set_xlabel("grid")
    a.set_ylabel("xc kernel")

    b.plot(f_xc[:,x1,w].real)
    b.set_title("|x-x'| along x={}".format(params.space_grid[x1]))

    c.plot(f_xc[:,x2,w].real)
    c.set_title("|x-x'| along x={}".format(params.space_grid[x2]))


    d.plot(f_xc[:,x3,w].real)
    d.set_title("|x-x'| along x={}".format(params.space_grid[x3]))

    fig.tight_layout()
    plt.savefig(f"{figname}")


def get_data(x):
    directory = 'QHO'
    input_file = open(f'{directory}/raw/{x}.db', 'rb')
    return np.array(pickle.load(input_file))

# Load data from an iDEA run
f_hxc = get_data('gs_extlr_fhxc')
f_xc = get_data('gs_extlr_fxc')
v_ext = get_data('gs_ext_vxt')
v_ks = get_data('gs_extre_vks')
v_xc = get_data('gs_extre_vxc')
density = get_data('gs_ext_den')
chi = get_data('gs_extlr_drf')
chi0 = get_data('gs_kslr_extre_drf')
chi_dyson = get_data('gs_extlr_drfdyson')
frequencies = get_data('gs_extlr_frequencies')
oas = get_data('gs_extlr_oas')
oas_ks = get_data('gs_kslr_extre_oas')


# Plot system
plt.xlabel('grid')
plt.plot(v_ext, label='v_ext')
plt.plot(density, label='density')
plt.legend()
plt.savefig('system.pdf')


# Match some relevent parameters for interface w/ d2p
params = parameters(Nspace = len(density),
                    space  = 10)

num_freq = len(frequencies)
                   
# 1D LDA xc kernel
lda_xc_kernel = (-4.437*density**1.61 + 3.381*density**0.61 - 0.7564*density**-0.39) / params.dx

# Error in response functions
error_dyson = np.sum(abs(chi - chi_dyson)) / (params.Nspace**2 * num_freq)

# Plot f_xc along various axes 
fxc_axis_plot(params, f_xc, x1=0, x2=25, x3=50, w=0, figname='fxc_axes.pdf'):

# 3D surface plot of f_xc
plot_surface_3d(params, f_xc[:,:,25])

# Plot zero-force sum rules in cwd
for i in range(num_freq):
    sum_rule(params, f_xc[:,:,i], density, v_xc, i)    

# Generate series of heat maps in a specified dir.
for i in range(num_freq):

    # Error in ||chi-chi_dyson|| on iteration i
    error_dyson = np.sum(abs(chi[:,:,i] - chi_dyson[:,:,i])) / params.Nspace**2
    
    # f_xc heat map
    plot_heat_map(None, f_xc[:,:,i], 
            figtitle='w = {0}, error ||chi - chi_dyson|| = {1}'.format(round(frequencies[i],4), error_dyson), 
            figname=f'exact_fhxc_heat/{i}')
    
    # exact drf heat map
    if i % 10 == 0:
        plot_heat_map(None, chi[:,:,i], figtitle='w = {}'.format(frequencies[i]), figname=f'exact_chi_heat/{i}')

