clearvars;
clc, clear all;
% =========================================================================
% SIMULATION
% =========================================================================

% create the computational grid
dim_x = 50e-3;           % extent along the x-axis [m]
dim_y = 50e-3;           % extent along the y-axis [m]
dx = 0.1e-3;             % grid spacing in the x direction [m]
dy = dx;                 % grid spacing in the y direction [m]
Nx = fix(dim_x / dx);    % number of grid points in the x (row) direction 
Ny = fix(dim_y / dy);    % number of grid points in the y (column) direction

kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation mediu m
medium.sound_speed = 1500 * ones(Nx, Ny);  % [m/s]
medium.alpha_coeff = 0.75;                 % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;
medium.density = 1000 * ones(Nx, Ny);      %[kg/m^3]

% create the time array
kgrid.makeTime(medium.sound_speed);

% define the first curved transducer element
arc_pos = [30e-3/dx, 1];             % [grid points]    
radius = 30e-3 / dx + 1;             % [grid points]
diameter = (30e-3)/dx + 1;           % [grid points] need to be an odd number
focus_pos = [30e-3/dx, 30e-3/dy];    % [grid points]
source.p_mask = makeArc([Nx, Ny], arc_pos, radius, diameter, focus_pos);
source_mag = 1e6;                      % [Pa]
sampling_freq = 40e6;                  % [Hz]
tone_burst_freq = 1.0e6;               % [Hz]
tone_burst_cycles = 3;
source.p = source_mag * toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles);

source.p = filterTimeSeries(kgrid, medium, source.p);

source.p_mask = source.p_mask;

% create a sensor mask covering the entire computational domain using the
% opposing corners of a rectangle
sensor.mask = [1, 1, Nx, Ny].';

% set the record mode capture the final wave-field and the statistics at
% each sensor point 
sensor.record = {'p'};

% assign the input options
input_args = {'DisplayMask', source.p_mask, 'PMLInside', false, 'PlotPML', false, 'PlotSim', true};
% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:}, 'DataCast', 'single');
