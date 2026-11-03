@present SchQVM(FreeSchema) begin
  (Fs, Qs, CCs, CClinks, QAs, QAlinks, QMC)::Ob
  (Name, Z)::AttrType
  fname::Attr(Fs, Name)
  qname::Attr(Qs, Name); domain::Attr(Qs, Name); value::Attr(Qs, Z); hierarchy::Attr(Qs, Name)
  qaname::Attr(QAs, Name)
  cc::Hom(CClinks, CCs); cf::Hom(CClinks, Fs)
  qa::Hom(QAlinks, QAs); ql::Hom(QAlinks, Qs)
  phi::Hom(QMC, CCs); psi::Hom(QMC, QAs)
end

# Reading of Pred: qa is the name of the quality attribute, not an arrow into QAs, so that the
# only arrow from an old object to a new one is cc_cfg.
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

# QVM+ without the measurement side (QMC, QAs, QAlinks, Qs, Noise).
@present SchQVMG(FreeSchema) begin
  (Fs, CCs, CClinks, VCs, VClinks, Cfg, Pred)::Ob
  (Name, R)::AttrType
  fname::Attr(Fs, Name)
  cc::Hom(CClinks, CCs); cf::Hom(CClinks, Fs)
  vc::Hom(VClinks, VCs); vf::Hom(VClinks, Fs)
  cc_cfg::Hom(CCs, Cfg); vc_cfg::Hom(VCs, Cfg); pcfg::Hom(Pred, Cfg)
  pname::Attr(Pred, Name); yhat::Attr(Pred, R); lo::Attr(Pred, R); hi::Attr(Pred, R)
end

@acset_type QVM(SchQVM)
@acset_type QVMp(SchQVMPlus)
@acset_type QVMG(SchQVMG)

const QVMi = QVM{Symbol,Int}
const QVMpi = QVMp{Symbol,Int,Float64}
const QVMGi = QVMG{Symbol,Float64}

function inclusion(dom, cod)
  C, D = FinCat(dom), FinCat(cod)
  FinFunctor(Dict(g => cod[nameof(g)] for g in ob_generators(C)),
             Dict(g => cod[nameof(g)] for g in hom_generators(C)), C, D; homtype=:generator)
end

const F_QVM = inclusion(SchQVM, SchQVMPlus)
const G_QVM = inclusion(SchQVMG, SchQVMPlus)
