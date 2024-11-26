using LinearAlgebra; Plots
Ti = [ 500, 1000, 2000, 3000, 4000, 5000, 6300];
Ai = [14.5, 13.5, 12.0, 10.8,  9.9,  8.9,  8.0];
n = length(Ti)
Data_C14 = [Ti Ai];
println(Data_C14)

# Estimation a priori des parametres du modele : beta0 = [A0, lambda]
β0 = [10; 0.0001];   # Newton, Gauus-Newton, fminunc et leastsq converge
# beta0 = [15; 0.001];    # Newton, Gauss-Newton, fminunc et leastsq divergent
# beta0 = [15; 0.0005];   # Newton diverge, Gauss-Newton, fminunc et leastsq convergent
# beta0 = [10; 0.0005];   # Gauss-Newton converge

# Calcul et affichage du modele initial ----------------------------------
xmin = 9; xmax = 20;
xx = range(xmin, stop=xmax, length=100);

ymin = -0.0001; ymax = 0.0005;
yy = range(ymin, stop=ymax, length=100);

function r(β,data)
    A₀ = β[1]
    λ = β[2]
    return data[:,2]-A₀*exp.(-λ*data[:,1])
end
# Plot of the function

println("r(beta0,Data_C14) = ", r(beta0,Data_C14))
X = [ones(n) Ti]
f(β) = 0.5*norm(r(β,Data_C14))^2
f_contour(A₀,λ) = f([A₀,λ])
z = @. f_contour(xx', yy)

p1 = plot()
contour!(p1,xx,yy,z,levels=100,cbar=false,color=:turbo)

p2 = scatter(Ti,Ai,title="Data C14")
T = range(0,stop=6500,length=100);
A = beta0[1]*exp.(-beta0[2]*T);
plot!(p2,T,A)

include("../course/assets/julia/MyOptims.jl")
res(β) = r(β,Data_C14)
nb_levels = 5
for nbit in 1:nb_levels
    βsol, flag, fsol, ∇f_xsol , nb_iter  = algo_Gauss_Newton(res,β0,nbit_max=nbit)
 #   A = βsol[1]*exp.(-βsol[2]*T);
    plot!(p2,T,βsol[1]*exp.(-βsol[2]*T))
    scatter!(p1,[βsol[1]],[βsol[2]])
end

plot(p1,p2,legend=false)

