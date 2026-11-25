function r = randi(varargin)
  if nargin > 0 && isstruct(varargin{1}), varargin(1) = []; end
  lim = varargin{1}; rest = varargin(2:end);
  if numel(lim) == 2, lo = lim(1); hi = lim(2); else, lo = 1; hi = lim; end
  if isempty(rest), rest = {1}; end
  r = lo + floor(builtin('rand', rest{:}) * (hi - lo + 1));
end
