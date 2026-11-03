using Test

@testset "QVMPlus" begin
  include("migrations.jl")
  include("queries.jl")
  include("tables.jl")
  include("listing.jl")
end
