struct Axis
  column::String
  index::String
end

struct Tensor
  name::String
  table::String
  axes::Vector{Axis}
end

struct Index
  name::String
  sort::String
  kind::String
end

struct Spec
  query::String
  tensors::Vector{Tensor}
  indices::Vector{Index}
  einsum::String
  output::Vector{String}
end

Base.:(==)(a::T, b::T) where {T<:Union{Axis,Tensor,Index,Spec}} =
  all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))

spec_json(spec::Spec) = JSON.json(spec, 2)

function spec_from_json(s::AbstractString)
  d = JSON.parse(s)
  Spec(d["query"],
       [Tensor(t["name"], t["table"], [Axis(a["column"], a["index"]) for a in t["axes"]])
        for t in d["tensors"]],
       [Index(i["name"], i["sort"], i["kind"]) for i in d["indices"]],
       d["einsum"], String.(d["output"]))
end

"Compile a UWD into one tensor per box, one index per junction and an einsum string."
function to_einsum(name, uwd; S=acset_schema(QVMpi()))
  vars = String.(uwd[:variable])
  sorts = Dict{Int,Tuple{Symbol,Symbol}}()
  tensors = map(parts(uwd, :Box)) do b
    table = uwd[b, :name]
    axes = map(incident(uwd, b, :box)) do p
      col, j = uwd[p, :port_name], uwd[p, :junction]
      sort = col == :_id ? (table, :ob) :
             col in hom(S) ? (codom(S, col), :ob) : (codom(S, col), :attr)
      get!(sorts, j, sort) == sort || error("junction $(vars[j]) has two sorts")
      Axis(String(col), vars[j])
    end
    Tensor("b$b", String(table), axes)
  end
  indices = [Index(vars[j], String(s), String(k)) for (j, (s, k)) in sort(collect(sorts))]
  letters = Dict(i.name => 'a' + k - 1 for (k, i) in enumerate(indices))
  output = vars[uwd[:outer_junction]]
  lhs = join([String([letters[a.index] for a in t.axes]) for t in tensors], ",")
  Spec(String(name), tensors, indices, lhs * "->" * String([letters[o] for o in output]), output)
end

function counts(t::Tensor, X)
  n = nparts(X, Symbol(t.table))
  cols = [a.column == "_id" ? collect(1:n) : X[Symbol(a.column)] for a in t.axes]
  vars = unique(a.index for a in t.axes)
  pos = [findfirst(==(v), [a.index for a in t.axes]) for v in vars]
  tab = Dict{Tuple,Int}()
  for r in 1:n
    all(k -> cols[k][r] == cols[pos[findfirst(==(t.axes[k].index), vars)]][r], eachindex(cols)) ||
      continue
    key = Tuple(cols[p][r] for p in pos)
    tab[key] = get(tab, key, 0) + 1
  end
  vars, tab
end

"Exact contraction of a spec over the bag semiring by hash joins; returns tuple => count."
function contract(spec::Spec, X)
  rel_vars, rel = String[], Dict{Tuple,Int}(() => 1)
  for (k, t) in enumerate(spec.tensors)
    vars, tab = counts(t, X)
    shared = [v for v in vars if v in rel_vars]
    extra = [v for v in vars if !(v in rel_vars)]
    sp = [findfirst(==(v), vars) for v in shared]
    ep = [findfirst(==(v), vars) for v in extra]
    rp = [findfirst(==(v), rel_vars) for v in shared]
    groups = Dict{Tuple,Vector{Pair{Tuple,Int}}}()
    for (key, c) in tab
      push!(get!(groups, Tuple(key[p] for p in sp), Pair{Tuple,Int}[]), Tuple(key[p] for p in ep) => c)
    end
    joined = Dict{Tuple,Int}()
    for (r, c) in rel, (e, c2) in get(groups, Tuple(r[p] for p in rp), Pair{Tuple,Int}[])
      key = (r..., e...)
      joined[key] = get(joined, key, 0) + c * c2
    end
    rel_vars = [rel_vars; extra]
    later = Set(vcat(spec.output, [a.index for u in spec.tensors[k+1:end] for a in u.axes]))
    live = [i for (i, v) in enumerate(rel_vars) if v in later]
    rel = Dict{Tuple,Int}()
    for (key, c) in joined
      kk = Tuple(key[i] for i in live)
      rel[kk] = get(rel, kk, 0) + c
    end
    rel_vars = rel_vars[live]
  end
  order = [findfirst(==(o), rel_vars) for o in spec.output]
  out = Dict{Tuple,Int}()
  for (key, c) in rel
    kk = Tuple(key[i] for i in order)
    out[kk] = get(out, kk, 0) + c
  end
  filter!(p -> p.second != 0, out)
end

"Bag of rows returned by `query`, as tuple => count."
function tally(res)
  res isa AbstractVector && return isempty(res) ? Dict{Tuple,Int}() : Dict{Tuple,Int}(() => length(res))
  out = Dict{Tuple,Int}()
  for row in zip(values(res)...)
    out[row] = get(out, row, 0) + 1
  end
  out
end
