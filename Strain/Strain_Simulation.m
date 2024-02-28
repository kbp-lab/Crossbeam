clearvars;
clc, clear all;
% =========================================================================
% SIMULATION PARAMETERS
% =========================================================================

% create the computational grid
dim_x = 60e-3;                     % extent along the x-axis [m]
dim_y = 60e-3;                     % extent along the y-axis [m]
dx = 0.2e-3;                       % grid spacing in the x direction [m]
dy = dx;                           % grid spacing in the y direction [m]
Nx = fix(dim_x / dx);              % number of grid points in the x (row) direction 
Ny = fix(dim_y / dy);              % number of grid points in the y (column) direction
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
cp = 1540;                         % compressional wave speed [m/s]
cs = 1.7;                          % shear wave speed [m/s]
rho = 1000;                        % density [kg/m^3]
alpha_p = 0.5;                     % compressional absorption [dB/(MHz^ycm)] was 0.5
alpha_s = 0.75;                    % shear absorption [dB/(MHz^ycm)]         was 0.75

% create the time array
cfl = 0.1;
t_end = 50.0e-6;    % [s]  
kgrid.makeTime(cp, cfl, t_end);

% define the curved transducer
arc_pos = [(dim_x/2)/dx, 1];                  % [grid points] 
focus_pos = [(dim_x/2)/dx, (dim_y/2)/dy];     % [grid points] 
radius = 30e-3 / dx;                         % [grid points] 
diameter = (30e-3) / dx + 1;                 % [grid points] 

% define the source signal
source_freq = 0.25e6;       % [Hz]
source_mag =  5.0e6;        % [Pa]
num_cycles = 5;
pulse_tb = source_mag * toneBurst(1/kgrid.dt, source_freq, ... 
    num_cycles, 'Envelope', 'Gaussian');

% create a sensor mask covering the entire computational domain using the
% opposing corners of a rectangle
sensor.mask = [1, 1, Nx, Ny].';
 
% record the necessary parameters
sensor.record = {'p_max', 'u_non_staggered', 'p'};

% set the input arguments
input_args = {'PMLInside', false, 'DisplayMask', ...
    'off', 'PlotSim', true, 'DataCast', 'single'};

% =========================================================================
% FLUID SIMULATION
% =========================================================================

% assign the medium properties
medium.sound_speed = cp * ones(Nx, Ny);
medium.density = rho * ones(Nx, Ny); 
medium.alpha_coeff = alpha_p * ones(Nx, Ny);
medium.alpha_power = 1.5;
delta_t = kgrid.dt;

% generate the source geometry
source.p_mask = makeArc([Nx, Ny], arc_pos, radius, diameter, focus_pos);
source_mask = source.p_mask;   % we will be reusing this source_mask in line 165
source.p = pulse_tb;
source.p = filterTimeSeries(kgrid, medium, source.p);

% run the fluid simulation
sensor_data_fluid = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

% compute particle displacement
Xp = zeros(Nx, Ny, kgrid.Nt);
Yp = zeros(Nx, Ny, kgrid.Nt);

for i=3:kgrid.Nt
    Yp(:, :, i) = Yp(:, :, i-2) + 2 * kgrid.dt * sensor_data_fluid.uy_non_staggered(:, :, i-1);
end

for i=3:kgrid.Nt
    Xp(:, :, i) = Xp(:, :, i-2) + 2 * kgrid.dt * sensor_data_fluid.ux_non_staggered(:, :, i-1);
end


% =========================================================================
% ELASTIC SIMULATION
% =========================================================================
% create the time array
% get the period from fluid simulation before creating a new time array
period = (1 / source_freq) * (1 / kgrid.dt);                 % [grid points]
cp = 20;            % reduced velocity to be able to simulate for a longer time [m/s]            
cfl = 0.1;
t_end = 50.0e-4;    % [s] this is 2 orders of magnitude longer than the fluid simulation
kgrid.makeTime(cp, cfl, t_end);
N_t = size(sensor_data_fluid.p, 3);


%%%%%%%%%%%% use pressure and velocity terms to comptue stress %%%%%%%%%%%%
% extract the particle velocities and pressure at the source
ux_source = zeros(Nx, Ny, N_t);
uy_source = zeros(Nx, Ny, N_t);
p_source = zeros(Nx, Ny, N_t);

for i = 1:Nx
    for j = 1:Ny
        if source.p_mask(i, j) == 1
            ux_source(i, j, :) = sensor_data_fluid.ux_non_staggered(i, j, :);
            uy_source(i, j, :) = sensor_data_fluid.uy_non_staggered(i, j, :);
            p_source(i, j, :) = sensor_data_fluid.p(i, j, :);
        end
    end
end



% set the lower and upper integration limits over one full period
ll = int16((num_cycles / 2) * period);    % lower integration limit
ul = int16(ll + period);                  % upper integration limit

period = double(ul - ll);

% compute the <p^2> term for stress
p_integrand = p_source(:, :, ll:ul).^2;
p_ave = 1 / period * trapz(delta_t, p_integrand, 3);

% compute the <v_xx> term for stress
v_xx_integrand = ux_source(:, :, ll:ul).^2;
v_xx_ave = 1 / period * trapz(delta_t, v_xx_integrand, 3);

% compute the <v_yy> term for stress
v_yy_integrand = uy_source(:, :, ll:ul).^2;
v_yy_ave = 1 / period * trapz(delta_t, v_yy_integrand, 3);

% compute the <v_xy> term for stress
v_xy_integrand = ux_source(:, :, ll:ul) .* ...
    uy_source(:, :, ll:ul);
v_xy_ave = 1 / period * trapz(delta_t, v_xy_integrand, 3);
            
% compute the <|v|^2> term for stress
v_integrand = v_xx_integrand + v_yy_integrand;
v_sqrd_ave = 1 / period * trapz(delta_t, v_integrand, 3);

% compute the different stress components
% compute sigma_xx
sigma_xx = 1 / (2 * rho * cp^2) * p_ave - ...
    0.5 * rho * v_sqrd_ave + rho * v_xx_ave;
% compute sigma_yy
sigma_yy = 1 / (2 * rho * cp^2) * p_ave - ...
    0.5 * rho * v_sqrd_ave + rho * v_yy_ave;

% compute sigma_xy
sigma_xy = rho * v_xy_ave;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define the medium properties
clear medium
medium.sound_speed_compression = cp .* ones(Nx, Ny);
medium.sound_speed_shear = cs * ones(Nx, Ny);
medium.density = rho * ones(Nx, Ny);
medium.alpha_coeff_compression = alpha_p;
medium.alpha_coeff_shear = alpha_s;

% assign the source
clear source
source.s_mask = source_mask;

% assign the stress tensors to the source
t_stress = (1 / source_freq) * num_cycles * 2;     % stress duration [s] 
signal_duration = int16(t_stress * (1 / kgrid.dt));  % stress duration in time steps

% compute the active source element count
N_elem = length(find(sigma_xx));
% create matrices where each row is the time varying stress for its
% corresponding source elements
Sxx = zeros(N_elem, kgrid.Nt);
Syy = zeros(N_elem, kgrid.Nt);
Sxy = zeros(N_elem, kgrid.Nt);

% find the location of nonzero elements
[rows, cols] = find(sigma_xx);

for i=1:signal_duration
    Sxx(:, i) = sigma_xx(sub2ind(size(sigma_xx), rows, cols));
    Syy(:, i) = sigma_yy(sub2ind(size(sigma_yy), rows, cols));
    Sxy(:, i) = sigma_xy(sub2ind(size(sigma_xy), rows, cols));
end

% assign the fixed stress to the source
source.sxx = - Sxx;
source.syy = - Syy;
source.sxy = - Sxy;

% run the elastic simulation
sensor_data_elastic = pstdElastic2D(kgrid, medium, source, sensor, input_args{:});

% compute the displacements in x and y for the tissue
Xt = zeros(Nx, Ny, kgrid.Nt);
Yt = zeros(Nx, Ny, kgrid.Nt);

for i=3:kgrid.Nt
    Yt(:, :, i) = Yt(:, :, i-2) + 2 * kgrid.dt * sensor_data_elastic.uy_non_staggered(:, :, i-1);
end

for i=3:kgrid.Nt
    Xt(:, :, i) = Xt(:, :, i-2) + 2 * kgrid.dt * sensor_data_elastic.ux_non_staggered(:, :, i-1);
end

peak_disp_yt = max(Yt, [], 3);

Xt_pos = max(Xt, [], 3);
Xt_neg = min(Xt, [], 3);
peak_disp_xt = Xt_pos + Xt_neg;

% rescale the time, since we have a slower velocity here
time_og = linspace(0, t_end, kgrid.Nt);
t_scale = (20 / 1540);
time_scaled = t_scale * time_og;
