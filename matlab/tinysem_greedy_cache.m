function x = tinysem_greedy_cache(Qhat, b, C)
%TINYSEM_GREEDY_CACHE Density-greedy knapsack: max sum_o Qhat_o b_o x_o s.t. sum b_o x_o <= C.
%   Objects are inserted in decreasing Qhat (= value density Qhat*b/b)
%   while they fit.  Within one item of the fractional optimum.

[~, order] = sort(Qhat(:), 'descend');        % MATLAB's sort is stable
x = zeros(numel(b), 1);
used = 0;
for i = 1:numel(order)
    o = order(i);
    if used + b(o) <= C
        x(o) = 1;
        used = used + b(o);
    end
    if C - used < 1.0                         % smaller than the smallest object
        break
    end
end
end
