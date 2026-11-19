function P = tinysem_params(varargin)
%TINYSEM_PARAMS Default parameters of the Tiny-SEM simulation (ICC 2027 paper).
%   P = tinysem_params() returns the default configuration used in Sec. V.
%   P = tinysem_params('Name', value, ...) overrides any field, e.g.
%       P = tinysem_params('V', 2, 'budget_frac', 0.2, 'drift_every', 1000);
%
%   Units: sizes in MB, energy in mJ, one slot = Delta = 1 s.

P.N           = 50;        % users
P.O           = 1000;      % catalog size (ImageNet-1k classes)
P.G           = 10;        % semantic cluster size
P.zipf        = 0.8;       % Zipf exponent (global and personal profiles)
P.omega       = 0.3;       % weight of the personal intent profile
P.a_lo        = 0.3;       % activity probability a_n ~ U[a_lo, a_hi]
P.a_hi        = 0.7;
P.b_lo        = 1.0;       % object size b_o ~ U[b_lo, b_hi] (MB)
P.b_hi        = 10.0;
P.cache_frac  = 0.10;      % cache capacity C as a fraction of catalog volume

% TinyML models: MobileNetV1-224, width 0.25/0.5/0.75/1.0 (Howard et al. 2017, Tab. 6)
P.acc         = [0.506 0.637 0.684 0.706];   % ImageNet top-1 accuracy A_k
P.mmac        = [41 149 325 569];            % million multiply-accumulates
P.eta_E       = 1e-3;      % mJ per million MAC (= 1 pJ/MAC, NPU class)

P.beta        = 0.6;       % share of errors that stay inside the semantic cluster
P.lam1        = 1.0;       % weight of recognition accuracy in the reward
P.lam2        = 1.0;       % weight of delivery quality in the reward
P.Rbh         = 500.0;     % backhaul capacity, MB per slot (4 Gb/s with Delta = 1 s)
P.W           = 100e6;     % per-user access bandwidth (Hz)
P.snr_lo      = 10.0;      % mean access SNR ~ U[snr_lo, snr_hi] dB, Rayleigh fading
P.snr_hi      = 25.0;
P.budget_frac = 0.4;       % energy budget  Ebar_n = budget_frac * a_n * e_max
P.V           = 10.0;      % Lyapunov parameter (energy measured in units of e_max)
P.tau         = 0.2;       % entropy-regularization temperature
P.margin      = 0.03;      % virtual budget (1 - margin) * Ebar_n   (theta in the paper)
P.gamma       = 0.005;     % EMA forgetting factor of the demand estimate
P.gamma_d     = 0.1;       % EMA factor of the broadcast congestion signal
P.T           = 5000;      % slots per run
P.warm        = 1000;      % warm-up slots excluded from performance metrics
P.drift_every = 0;         % 0 = stationary; otherwise redraw popularity every k slots
P.static_freeze = 300;     % slot at which the static Stackelberg cache freezes
P.uplink_bits = ceil(log2(1000));      % semantic request size
P.raw_bits    = 224*224*3*8;           % raw RGB frame size

for i = 1:2:numel(varargin)
    if ~isfield(P, varargin{i})
        error('tinysem_params:unknown', 'Unknown parameter "%s".', varargin{i});
    end
    P.(varargin{i}) = varargin{i+1};
end

P.energy = P.mmac * P.eta_E;           % e_k in mJ (k = 1..K)
P.eps    = 1 / P.Rbh;                  % congestion sensitivity epsilon = 1/(R_bh*Delta)
P.K      = numel(P.acc);
end
