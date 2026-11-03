using QVMPlus, Catlab, Pkg, JSON, Dates

const SIZES = [(2560, 18), (68640, 20), (216000, 22)]
const SEED = 0
const OUT = joinpath(@__DIR__, "..", "results", "raw", "julia_bench.jsonl")

meta() = Dict("time" => string(now()), "julia" => string(VERSION),
              "catlab" => string(Pkg.dependencies()[Base.UUID("134e5e36-593f-5add-ad60-77f754baafbe")].version),
              "acsets" => string(Pkg.dependencies()[Base.UUID("227ef7b5-1206-438b-ac65-934d6da304b8")].version),
              "cpu" => Sys.cpu_info()[1].model, "threads" => Threads.nthreads(),
              "commit" => readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`), "seed" => SEED)

function emit(rec)
  rec = merge(meta(), rec, Dict("maxrss_mb" => round(Sys.maxrss() / 2^20; digits=1)))
  mkpath(dirname(OUT))
  open(io -> println(io, JSON.json(rec)), OUT, "a")
  println(JSON.json(rec))
end

best(f; reps=3) = minimum(@elapsed(f()) for _ in 1:reps)

function population_of(N, k)
  s = synthetic(N; k, seed=SEED)
  population(s.fnames, s.X, s.y)
end

function closed()
  sigma_closed(last(first(fixtures_qvm())))
  for (N, k) in SIZES
    t_gen = @elapsed I = population_of(N, k)
    J = sigma_closed(I)
    emit(Dict("mode" => "closed", "configs" => N, "options" => k, "links" => nparts(I, :CClinks),
              "gen_s" => t_gen, "sigma_closed_s" => best(() -> sigma_closed(I)),
              "delta_F_s" => best(() -> delta_F(J)), "L1_equal" => delta_F(J) == I))
  end
end

function chase_fixtures()
  for (name, I) in fixtures_qvm()
    sigma_chase(I)
    t = best(() -> sigma_chase(I); reps=2)
    emit(Dict("mode" => "chase", "instance" => name, "configs" => nparts(I, :CCs),
              "chase_s" => t, "closed_s" => best(() -> sigma_closed(I)),
              "iso" => is_isomorphic(sigma_closed(I), sigma_chase(I))))
  end
end

function chase_synthetic(N, k)
  I = population_of(N, k)
  t = @elapsed J = sigma_chase(I)
  emit(Dict("mode" => "chase", "instance" => "synthetic", "configs" => N, "chase_s" => t,
            "same_sizes" => all(o -> nparts(J, o) == nparts(sigma_closed(I), o), ob(acset_schema(J)))))
end

function queries_small(N, k)
  s = synthetic(N; k, seed=SEED)
  yhat = s.yhat
  J = population_plus(s.fnames, s.X, s.y[s.measured], s.measured;
                      pred=(rows=1:N, yhat, lo=yhat .- s.width, hi=yhat .+ s.width))
  for (qn, q) in queries()
    spec = to_einsum(qn, q)
    contract(spec, J)
    emit(Dict("mode" => "query", "query" => qn, "configs" => N, "query_s" => best(() -> query(J, q)),
              "contract_s" => best(() -> contract(spec, J); reps=1),
              "equal" => contract(spec, J) == tally(query(J, q))))
  end
end

mode = get(ARGS, 1, "closed")
mode == "closed" && closed()
mode == "chase_fixtures" && chase_fixtures()
mode == "chase" && chase_synthetic(parse(Int, ARGS[2]), parse(Int, ARGS[3]))
mode == "queries" && queries_small(2560, 18)
