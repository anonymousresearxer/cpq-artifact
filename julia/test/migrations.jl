using Catlab, QVMPlus

@testset "migrations" begin
  @testset "L1 on $name" for (name, I) in fixtures_qvm()
    J = sigma_closed(I)
    @test delta_F(J) == I
    @test is_isomorphic(delta_F(J), I)
  end

  @testset "closed form against the chase on $name" for (name, I) in fixtures_qvm()
    @test is_isomorphic(sigma_closed(I), sigma_chase(I))
  end

  @testset "shape of the closed form" begin
    I = last(first(fixtures_qvm()))
    J = sigma_closed(I)
    @test nparts(J, :Cfg) == nparts(I, :CCs)
    @test J[:cc_cfg] == 1:nparts(I, :CCs)
    for o in (:VCs, :VClinks, :Pred, :Noise)
      @test nparts(J, o) == 0
    end
  end

  @testset "views of $name" for (name, J) in fixtures_qvmp()
    @test nparts(delta_F(J), :CCs) == nparts(J, :CCs)
    @test nparts(delta_F(J), :CClinks) == nparts(J, :CClinks)
    G = delta_G(J)
    @test nparts(G, :Pred) == nparts(J, :Pred)
    @test nparts(G, :VClinks) == nparts(J, :VClinks)
    @test sort([J[:cc_cfg]; J[:vc_cfg]]) == 1:nparts(J, :Cfg)
  end
end
