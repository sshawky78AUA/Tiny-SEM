function rs = RandStream(type, varargin)
  % Octave compatibility only (MATLAB has RandStream built in). Seeds Octave's
  % global generator, so random numbers differ from MATLAB but statistics agree.
  seed = varargin{2};
  builtin('rand', 'twister', seed); builtin('randn', 'state', seed);
  rs = struct('isRandStreamShim', true);
end
