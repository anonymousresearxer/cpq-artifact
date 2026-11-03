@present SchQVMPlus <: SchQVM begin
  (VCs, VClinks, Cfg, Pred, Noise)::Ob
  R::AttrType
  vc::Hom(VClinks, VCs); vf::Hom(VClinks, Fs)
  cc_cfg::Hom(CCs, Cfg); vc_cfg::Hom(VCs, Cfg); pcfg::Hom(Pred, Cfg)
  pname::Attr(Pred, Name); yhat::Attr(Pred, R); lo::Attr(Pred, R); hi::Attr(Pred, R)
  nq::Hom(Noise, Qs); eps::Attr(Noise, R)
end
q_pairs = @relation (f=f, g=g) where (c::CCs, f::Fs, g::Fs) begin
  CClinks(cc=c, cf=f)
  CClinks(cc=c, cf=g)
end
