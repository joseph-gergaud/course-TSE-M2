try
    using Suppressor
catch e
    @warn "Il faut installer Suppressor dans l'environnement de base" exception=(e, catch_backtrace())
end
@suppress begin
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
end
nothing