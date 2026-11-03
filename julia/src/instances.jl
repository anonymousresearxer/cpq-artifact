using Random: Xoshiro, randperm

function add_links!(J, link, src, dst, X)
  idx = findall(permutedims(X))
  add_parts!(J, link, length(idx); NamedTuple{(src, dst)}((getindex.(idx, 2), getindex.(idx, 1)))...)
  nothing
end

function add_measurements!(J, y, qa)
  n = length(y)
  add_parts!(J, :Qs, n; qname=fill(qa, n), domain=fill(:ms, n), value=y, hierarchy=fill(:perf, n))
  add_parts!(J, :QAs, n; qaname=fill(qa, n))
  add_parts!(J, :QAlinks, n; qa=1:n, ql=1:n)
  add_parts!(J, :QMC, n; phi=1:n, psi=1:n)
  nothing
end

"QVM instance with one measured value per row of X."
function population(fnames, X, y; qa=:time)
  I = QVMi()
  add_parts!(I, :Fs, length(fnames); fname=collect(fnames))
  add_parts!(I, :CCs, size(X, 1))
  add_links!(I, :CClinks, :cc, :cf, X)
  add_measurements!(I, collect(y), qa)
  I
end

"""
QVM+ instance: rows `measured` of X become CCs (with values y), the rest VCs. `pred` holds
`rows`, `yhat`, `lo` and `hi` over the rows of X; `eps` has one noise band per measured row.
"""
function population_plus(fnames, X, y, measured; pred=nothing, eps=nothing, qa=:time)
  N = size(X, 1)
  unmeasured = setdiff(1:N, measured)
  nm = length(measured)
  cfg = zeros(Int, N)
  cfg[measured] = 1:nm
  cfg[unmeasured] = nm+1:N
  J = QVMpi()
  add_parts!(J, :Fs, length(fnames); fname=collect(fnames))
  add_parts!(J, :Cfg, N)
  add_parts!(J, :CCs, nm; cc_cfg=1:nm)
  add_parts!(J, :VCs, N - nm; vc_cfg=nm+1:N)
  add_links!(J, :CClinks, :cc, :cf, X[measured, :])
  add_links!(J, :VClinks, :vc, :vf, X[unmeasured, :])
  add_measurements!(J, collect(y), qa)
  if !isnothing(pred)
    add_parts!(J, :Pred, length(pred.rows); pcfg=cfg[pred.rows], pname=fill(qa, length(pred.rows)),
               yhat=pred.yhat, lo=pred.lo, hi=pred.hi)
  end
  isnothing(eps) || add_parts!(J, :Noise, length(eps); nq=1:length(eps), eps=collect(eps))
  J
end

"Random valid space of N distinct configurations over k binary options, with a quadratic response."
function synthetic(N; k, nmeas=200, seed=0)
  rng = Xoshiro(seed)
  codes = randperm(rng, 2^k)[1:N] .- 1
  X = Matrix{Bool}(undef, N, k)
  for j in 1:k
    X[:, j] = isodd.(codes .>> (j - 1))
  end
  w = rand(rng, 1:40, k)
  pairs = [(rand(rng, 1:k), rand(rng, 1:k)) for _ in 1:8]
  y = 1000 .+ X * w .+ [sum(X[i, a] & X[i, b] for (a, b) in pairs) * 25 for i in 1:N] .+
      rand(rng, -5:5, N)
  yhat = 1000.0 .+ X * w
  fnames = Symbol.("f", 1:k)
  measured = sort(randperm(rng, N)[1:nmeas])
  (; fnames, X, y, yhat, measured, width=30.0)
end
