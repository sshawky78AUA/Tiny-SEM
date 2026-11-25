function r = rand(varargin)
  if nargin > 0 && isstruct(varargin{1}), varargin(1) = []; end
  r = builtin('rand', varargin{:});
end
