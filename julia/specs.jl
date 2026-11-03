using QVMPlus

for (qn, q) in queries()
  write(joinpath(@__DIR__, "specs", "$qn.json"), dump_spec(to_einsum(qn, q)) * "\n")
end
