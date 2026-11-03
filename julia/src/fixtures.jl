const P1 = (
  fnames=[:base, :cache, :wal],
  X=Bool[1 0 0; 1 1 0; 1 0 1; 1 1 1; 0 1 1],
  measured=[1, 2, 4],
  y=[12, 9, 11],
  pred=(rows=[1, 2, 3, 4, 5], yhat=[12.0, 9.5, 14.0, 10.5, 10.0]),
  eps=[0.5, 0.5, 0.0],
)

const P2 = (
  fnames=[:base, :a, :b, :c],
  X=Bool[1 0 0 0; 1 0 0 1; 1 0 1 0; 1 0 1 1; 1 1 0 0; 1 1 0 1; 1 1 1 0; 1 1 1 1],
  measured=[2, 3, 5],
  y=[20, 18, 25],
  pred=(rows=[1, 4, 6, 7, 8], yhat=[22.0, 19.5, 24.0, 23.0, 21.0]),
  eps=[1.0, 1.0, 1.0],
)

const P3 = (
  fnames=[:base, :cache, :wal],
  X=Bool[1 0 0; 1 1 0; 1 0 1; 1 1 1],
  measured=[1, 2, 3, 4],
  y=[12, 9, 15, 11],
  pred=(rows=[1, 2, 3, 4], yhat=[12.0, 9.0, 14.5, 11.5]),
  eps=[0.0, 0.0, 0.0, 0.0],
)

const WIDTH = 2.0

function plus(p)
  yhat = p.pred.yhat
  population_plus(p.fnames, p.X, p.y, p.measured;
                  pred=(rows=p.pred.rows, yhat, lo=yhat .- WIDTH, hi=yhat .+ WIDTH), eps=p.eps)
end

# Two quality attributes on one configuration, a featureless configuration and a repeated value.
function multi_qa()
  I = QVMi()
  add_parts!(I, :Fs, 2; fname=[:base, :cache])
  add_parts!(I, :CCs, 3)
  add_parts!(I, :CClinks, 3; cc=[1, 1, 2], cf=[1, 2, 1])
  add_parts!(I, :Qs, 4; qname=[:time, :size, :time, :time], domain=[:ms, :kb, :ms, :ms],
             value=[7, 300, 7, 9], hierarchy=fill(:perf, 4))
  add_parts!(I, :QAs, 4; qaname=[:time, :size, :time, :time])
  add_parts!(I, :QAlinks, 4; qa=1:4, ql=1:4)
  add_parts!(I, :QMC, 4; phi=[1, 1, 2, 3], psi=1:4)
  I
end

function fixtures_qvm()
  ["tiny" => population(P3.fnames, P3.X, P3.y), "multi_qa" => multi_qa(),
   "measured_p1" => delta_F(plus(P1))]
end

fixtures_qvmp() = ["p1" => plus(P1), "p2" => plus(P2), "p3" => plus(P3)]
