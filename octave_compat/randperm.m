function r = randperm(varargin)
  if nargin > 0 && isstruct(varargin{1}), varargin(1) = []; end
  [~, r] = sort(builtin('rand', 1, varargin{1}));
  if numel(varargin) > 1, r = r(1:varargin{2}); end
end
