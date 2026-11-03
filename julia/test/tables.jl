using Catlab, QVMPlus

@testset "tables and specs" begin
  @testset "tables round trip on $name" for (name, J) in fixtures_qvmp()
    @test from_tables(QVMpi, dump_instance(J)) == J
    @test load_instance(dump_instance(J)) == J
  end

  @testset "spec round trip for $qn" for (qn, q) in queries()
    spec = to_einsum(qn, q)
    @test load_spec(dump_spec(spec)) == spec
  end

  @testset "unpartitioned Cfg is rejected" begin
    t = dump_instance(last(first(fixtures_qvmp())))
    t["Cfg"]["n"] += 1
    @test_throws ErrorException load_instance(t)
  end
end
