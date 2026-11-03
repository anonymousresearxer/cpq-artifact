using Catlab, QVMPlus

instances() = vcat(fixtures_qvmp(), ["sigma_$n" => sigma_closed(I) for (n, I) in fixtures_qvm()])

@testset "queries" begin
  @testset "$qn on $name" for (name, X) in instances(), (qn, q) in queries()
    @test contract(to_einsum(qn, q), X) == tally(query(X, q))
  end

  @testset "known counts" begin
    p1 = Dict(fixtures_qvmp())["p1"]
    @test tally(query(p1, q_cfg)) == Dict(() => 5)
    @test tally(query(p1, q_pred)) == Dict(() => 2)
    @test contract(to_einsum("hist", q_hist), p1) == Dict((12,) => 1, (9,) => 1, (11,) => 1)
    pairs = contract(to_einsum("pairs", q_pairs), p1)
    @test pairs[(1, 1)] == 3
    @test pairs[(2, 3)] == 1
    @test pairs[(1, 2)] == pairs[(2, 1)] == 2
    @test sum(values(pairs)) == 1 + 4 + 9
  end

  @testset "saved specs" begin
    for (qn, q) in queries()
      saved = read(joinpath(@__DIR__, "..", "specs", "$qn.json"), String)
      @test load_spec(saved) == to_einsum(qn, q)
    end
  end
end
