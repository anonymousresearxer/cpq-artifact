q_cfg = @relation () where (c::Cfg,) begin
  Cfg(_id=c)
end

q_measured = @relation () where (c::CCs, a::QAs, q::Qs) begin
  QMC(phi=c, psi=a)
  QAlinks(qa=a, ql=q)
end

q_hist = @relation (v=v,) where (c::CCs, a::QAs, q::Qs, v::Z) begin
  QMC(phi=c, psi=a)
  QAlinks(qa=a, ql=q)
  Qs(_id=q, value=v)
end

q_pairs_vc = @relation (f=f, g=g) where (u::VCs, f::Fs, g::Fs) begin
  VClinks(vc=u, vf=f)
  VClinks(vc=u, vf=g)
end

q_pred = @relation () where (g::Cfg, u::VCs) begin
  Pred(pcfg=g)
  VCs(_id=u, vc_cfg=g)
end

q_noise = @relation (v=v, e=e) where (q::Qs, v::Z, e::R) begin
  Noise(nq=q, eps=e)
  Qs(_id=q, value=v)
end

queries() = ["cfg" => q_cfg, "measured" => q_measured, "hist" => q_hist, "pairs" => q_pairs,
             "pairs_vc" => q_pairs_vc, "pred" => q_pred, "noise" => q_noise]
