chase_cat() = ACSetCategory(VarACSetCat(QVMpi()))

delta_F(J::QVMpi) = migrate(QVMi, J, DeltaMigration(F_QVM))
delta_G(J::QVMpi) = migrate(QVMGi, J, DeltaMigration(G_QVM))

"Closed form of the left Kan extension along F: Cfg is CCs, the other new objects are empty."
function sigma_closed(I::QVMi)
  J = QVMpi()
  S = acset_schema(I)
  for o in ob(S)
    add_parts!(J, o, nparts(I, o))
  end
  for f in [hom(S); attr(S)]
    J[f] = I[f]
  end
  add_parts!(J, :Cfg, nparts(I, :CCs))
  J[:cc_cfg] = 1:nparts(I, :CCs)
  J
end

sigma_chase(I::QVMi; n=1000) =
  SigmaMigrationFunctor(F_QVM, QVMi, QVMpi)(I; n, cat=chase_cat())
