clear

%% load data
system_config = 'SSMF_DP_SC_20_X_50_Km_30_GBd_64_QAM';
Plch = 0;  % dBm
matfile = [system_config '_' num2str(Plch) 'dBm.mat'];
path_to_matfile = [system_config '/' matfile];
load(path_to_matfile)

% TBD - Add field names and turn them all into double.

%% Transfer data to acceptable format
% recv_x = cplx2double(X_in);
% sent_x = cplx2double(X_des);
% recv_y = cplx2double(Y_in);
% sent_y = cplx2double(Y_des);

[recv_x, sent_x, recv_y, sent_y] = cplx2double_batch(X_in, X_des, Y_in, Y_des);

%% 
path_to_h5file = [system_config '/' num2str(Plch) 'dBm.h5'];
% h5create('data.h5', '/recv/xPol', flip(size(recv_x)))
% h5write('data.h5', '/recv/xPol', recv_x')

h5create(path_to_h5file, '/recv/xPol', flip(size(recv_x)))
h5write(path_to_h5file, '/recv/xPol', recv_x')

h5create(path_to_h5file, '/recv/yPol', flip(size(recv_y)))
h5write(path_to_h5file, '/recv/yPol', recv_y')

h5create(path_to_h5file, '/sent/xPol', flip(size(sent_x)))
h5write(path_to_h5file, '/sent/xPol', sent_x')

h5create(path_to_h5file, '/sent/yPol', flip(size(recv_y)))
h5write(path_to_h5file, '/sent/yPol', recv_y')



% Write attributes
% h5writeatt('data.h5','/','alpha', alpha);

% h5disp('data.h5')
%% need to extend this to all attributes










%% Helper functions

% Return double array for data instead of complex numbers
function data_out = cplx2double(data_in)

    data_out = [real(data_in), imag(data_in)];
end

function varargout = cplx2double_batch(varargin)
    for n = 1:nargin
        varargout{n} = cplx2double(varargin{n});
    end
end

