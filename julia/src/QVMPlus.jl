module QVMPlus

using Catlab
using ACSets
import JSON

export SchQVM, SchQVMPlus, SchQVMG, QVM, QVMp, QVMG, QVMi, QVMpi, QVMGi
export F_QVM, G_QVM, delta_F, delta_G, sigma_closed, sigma_chase, chase_cat
export population, population_plus, synthetic, fixtures_qvm, fixtures_qvmp
export q_pairs, q_cfg, q_measured, q_hist, q_pairs_vc, q_pred, q_noise, queries
export to_einsum, contract, tally, Spec, spec_json, spec_from_json
export to_tables, from_tables, load_instance, dump_instance, load_spec, dump_spec
export query_spec, answer, fixture

include("schema.jl")
include("migrations.jl")
include("instances.jl")
include("fixtures.jl")
include("queries.jl")
include("einsum.jl")
include("bridge.jl")

end
