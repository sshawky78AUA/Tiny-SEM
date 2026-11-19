function [method, opts] = tinysem_method(name)
%TINYSEM_METHOD Map a method label used in the paper to tinysem_run arguments.
switch name
    case 'tinysem',           method = 'tinysem';         opts = {};
    case 'tinysem_hard',      method = 'tinysem';         opts = {'hard', true};
    case 'static',            method = 'static_stackel';  opts = {};
    case 'cmarl',             method = 'cmarl';           opts = {'alpha_mu', 0.02};
    case 'lru_maxacc',        method = 'lru_maxacc';      opts = {};
    case 'agnostic',          method = 'maxacc_agnostic'; opts = {};
    case 'abl_nofb',          method = 'tinysem';         opts = {'feedback', false};
    case 'abl_cum',           method = 'tinysem';         opts = {'estimator', 'cum'};
    case 'abl_lru',           method = 'tinysem';         opts = {'cache', 'lru'};
    case 'abl_maxacc_greedy', method = 'lru_maxacc';      opts = {'cache', 'greedy'};
    otherwise, error('tinysem_method:unknown', 'Unknown method label %s', name);
end
end
