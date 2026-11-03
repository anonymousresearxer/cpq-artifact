module QVMPlusPythonExt

using PythonCall: Py, pyconvert
using QVMPlus: QVMPlus, QVMpi, acset_schema, from_tables, check_cfg, col_type

"""
Load a QVM+ instance from Python tables.

`tables` maps each object of the schema to `{"n": rows, "cols": {column: values}}`, where
each column is a list or a one-dimensional array: indices (1-based) for arrows, strings for
`Name`, integers for `Z` and floats for `R`. CCs and VCs must partition Cfg.
"""
function QVMPlus.load_instance(tables::Py)
  S = acset_schema(QVMpi())
  conv(c, col) = col_type(S, c) == Symbol ? pyconvert(Vector{String}, col) :
                 pyconvert(Vector{col_type(S, c)}, col)
  t = Dict{String,Any}()
  for (o, tab) in pyconvert(Dict{String,Py}, tables)
    cols = pyconvert(Dict{String,Py}, tab["cols"])
    t[o] = Dict("n" => pyconvert(Int, tab["n"]),
                "cols" => Dict(c => conv(Symbol(c), v) for (c, v) in cols))
  end
  check_cfg(from_tables(QVMpi, t))
end

end
