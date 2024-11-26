#x_k=0.8^k(9,(-1)^k)^T



A = [1 0 ; 0 9/2]
b = [0,0]; c=0.

f(x) = 0.5*x'*A*x + b'*x +c

f([0,1])

using LinearAlgebra
using ForwardDiff
"""
   Solve by the steepest descent 
"""
function steepest_descent(f,x0::Vector{<:Real},options)
    # Définition des fonction gardient et hessienne
    ∇f(x) = ForwardDiff.gradient(f, x)

    if options == []
      Tol_Abs  = sqrt(eps())
      Tol_Rel  = Tol_Abs
      nbit_max = 100*length(x0)
      rate = 1.
  else
      Tol_Abs  = options[1]
      Tol_Rel  = options[2]
      nbit_max = options[3]
      rate = option[4]
  end

  norm∇f₀ = norm(∇f(x0))
  continuer = (norm∇f₀ > Tol_Abs)
  xₖ = x0
  k = 0
  
  while continuer == true
      dₖ = - ∇f(xₖ)
      αₖ = 
      k = k+1
      xₖ = xₖ + dₖ
      if (norm(∇f(xₖ)) < max(Tol_Abs, Tol_Rel* norm∇f₀))
          flag = 0
          continuer = false  
      elseif (k == nbit_max)
          flag = 1
          continuer = false
      end
  end
  return xₖ, f(xₖ), ∇f(xₖ), ∇²f(xₖ), k  
end


