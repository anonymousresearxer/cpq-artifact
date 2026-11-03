const NAME_SORT = Dict(:Name => Symbol, :Z => Int, :R => Float64)

function col_type(S, col)
  col in hom(S) ? Int : NAME_SORT[codom(S, col)]
end

"Columns of every table of X as a dictionary of strings and vectors."
function to_tables(X)
  S = acset_schema(X)
  tab(o) = Dict("n" => nparts(X, o),
                "cols" => Dict(String(c) => (eltype(X[c]) == Symbol ? String.(X[c]) : X[c])
                               for c in [hom(S); attr(S)] if dom(S, c) == o))
  Dict(String(o) => tab(o) for o in ob(S))
end

"Build an instance of type T from the dictionary produced by `to_tables`."
function from_tables(T, tables)
  X = T()
  S = acset_schema(X)
  for o in ob(S)
    add_parts!(X, o, tables[String(o)]["n"])
  end
  for c in [hom(S); attr(S)]
    col = tables[String(dom(S, c))]["cols"][String(c)]
    X[c] = convert(Vector{col_type(S, c)}, col_type(S, c) == Symbol ? Symbol.(col) : col)
  end
  X
end

function check_cfg(J::QVMpi)
  cc, vc = J[:cc_cfg], J[:vc_cfg]
  sort([cc; vc]) == 1:nparts(J, :Cfg) || error("Cfg is not the disjoint union of CCs and VCs")
  J
end

function load_instance end

load_instance(tables::AbstractDict) = check_cfg(from_tables(QVMpi, tables))

dump_instance(J::QVMpi) = to_tables(J)

load_spec(json::AbstractString) = spec_from_json(json)
dump_spec(spec::Spec) = spec_json(spec)

query_spec(name) = dump_spec(to_einsum(name, Dict(queries())[name]))

"Rows of the answer to a named query over X, each a key followed by its multiplicity."
answer(X, name) = [Any[k..., c] for (k, c) in tally(query(X, Dict(queries())[name]))]

fixture(name) = Dict(fixtures_qvmp())[name]
