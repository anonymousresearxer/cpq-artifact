using Test

@testset "listing" begin
  listing = read(joinpath(@__DIR__, "..", "..", "paper", "listings", "schema.jl"), String)
  src = join(read(joinpath(@__DIR__, "..", "src", f), String) for f in readdir(joinpath(@__DIR__, "..", "src")))
  @test occursin(listing, src)
  @test all(isascii, listing)
  @test count(==('\n'), listing) <= 12
end
