using LinearAlgebra
using ForwardDiff

function stop_criteria(∇fₖ,norm∇f₀,xₖ,dₖ,fₖ,f_kp1,k,AbsTol,RelTol,ε,nbit_max)
    continuer = true; flag = 0;
    if (norm(∇fₖ)) < max(AbsTol, RelTol* norm∇f₀)
         flag = 0
        continuer = false 
    elseif norm(dₖ) < ε*max(AbsTol,RelTol*norm(xₖ)) 
         flag = 1
         continuer = false
    elseif abs(f_kp1-fₖ) < ε*max(AbsTol,RelTol*abs(f_kp1))
         flag = 2
         continuer = false
    elseif (k == nbit_max)
        flag = 3
        continuer = false
    end
    return continuer, flag
end

"""
   Solve by Newton's algorithm the optimization problem Min f(x)
   Case where f is a function from R^n to R
"""


function algo_Newton(f,x0::Vector{<:Real};AbsTol= abs(eps()), RelTol = abs(eps()), ε=0.01, nbit_max = 0)
    flag = 0
    if nbit_max==0
        nbit_max = 100*length(x0)
    end
    # Definition of the gradient and Hessian functions
    ∇f(x) = ForwardDiff.gradient(f, x)
    ∇²f(x) = ForwardDiff.hessian(f, x)

    norm∇f₀ = norm(∇f(x0))
    continuer = (norm∇f₀ > AbsTol)
    xₖ = x0
    k = 0
    fₖ = f(xₖ) 
    ∇fₖ = ∇f(xₖ)
    while continuer == true
        ∇²fxₖ = ∇²f(xₖ)
        dₖ = -∇²fxₖ \ ∇fₖ
        k = k+1
        xₖ = xₖ + dₖ
        f_kp1 = f(xₖ)
        ∇fₖ = ∇f(xₖ)
        continuer, flag = stop_criteria(∇fₖ,norm∇f₀,xₖ,dₖ,fₖ,f_kp1,k,AbsTol,RelTol,ε,nbit_max)
        fₖ = f_kp1
    end
    return xₖ, flag, fₖ, ∇fₖ , k  
end

"""
    Gauss-Newton Algorithm
"""
function algo_Gauss_Newton(r,β::Vector{<:Real};AbsTol= abs(eps()), RelTol = abs(eps()), ε=0.01, nbit_max = 0)
  flag = 0
  if nbit_max==0
      nbit_max = 100*length(x0)
  end
  # Definition of the gradient of f(β) = 0.5||r(β)||^2
  Jac_r(β) = ForwardDiff.jacobian(r, β)
  function ∇f(β)
    X = Jac_r(β)
    y = -res(β)
    return X'*X*β - X'*y
  end
  # Initialization
  norm∇f₀ = norm(∇f(β))
  continuer = (norm∇f₀ > AbsTol)
  βₖ = β
  k = 0
  fₖ = f(βₖ) 
  ∇fₖ = ∇f(βₖ)
  # Loop
  while continuer == true
      X = Jac_r(βₖ)
      y = -r(βₖ)
      dₖ = X\y
      k = k+1
      βₖ = βₖ + dₖ
      f_kp1 = f(βₖ)
      ∇fₖ = ∇f(βₖ)
      continuer, flag = stop_criteria(∇fₖ,norm∇f₀,βₖ,dₖ,fₖ,f_kp1,k,AbsTol,RelTol,ε,nbit_max)
      fₖ = f_kp1
  end
  return βₖ, flag, fₖ, ∇fₖ , k  
end

"""
   Solve Min f(x)=0.5x^TAx +b^Tx+c
   by the steepest descent 
"""
function steepest_descent_quad(A::Matrix{<:Real},b::Vector{<:Real},c::Real,x0::Vector{<:Real};AbsTol= abs(eps()), RelTol = abs(eps()), ε=0.01, nbit_max = 0)
  flag = 0
  if nbit_max==0
    nbit_max = 100*length(x0)
  end
  A_sym = (A' + A)/2
  # gardient function
  ∇f(x) = 2*A_sym*x + b
  norm∇f₀ = norm(∇f(x0))
  continuer = (norm∇f₀ > AbsTol)
  xₖ = x0
  k = 0
  fₖ = f(xₖ)
  ∇fₖ = ∇f(xₖ)
  while continuer == true
      dₖ = - ∇fₖ
      a₂ = 2*dₖ'*A_sym*dₖ
      a₁ = 2xₖ'*A_sym*dₖ + b'*dₖ
      αₖ = -a₁/a₂
      k = k+1
      xₖ = xₖ + αₖ*dₖ
      f_kp1 = f(xₖ)
      ∇fₖ = ∇f(xₖ)
      continuer, flag = stop_criteria(∇fₖ,norm∇f₀,xₖ,dₖ,fₖ,f_kp1,k,AbsTol,RelTol,ε,nbit_max)
      fₖ = f_kp1
  end
  return xₖ, flag, fₖ, ∇fₖ, k  
end

"""
   Solve by the descent ...
"""

function my_descent(f,x0::Vector{<:Real};η=0.1,AbsTol= abs(eps()), RelTol = abs(eps()), ε=0.01, nbit_max = 0)
    flag = 0
    if nbit_max==0
      nbit_max = 100*length(x0)
    end
  
    # gardient function
    ∇f(x) = ForwardDiff.gradient(f, x)
    norm∇f₀ = norm(∇f(x0))
    continuer = (norm∇f₀ > AbsTol)
    xₖ = x0
    k = 0
    fₖ = f(xₖ)
    ∇fₖ = ∇f(xₖ)
  
    while continuer == true
        dₖ = - ∇fₖ
        αₖ = η
        k = k+1
        xₖ = xₖ + αₖ*dₖ
        f_kp1 = f(xₖ)
        ∇fₖ = ∇f(xₖ)
        continuer, flag = stop_criteria(∇fₖ,norm∇f₀,xₖ,dₖ,fₖ,f_kp1,k,AbsTol,RelTol,ε,nbit_max)
        fₖ = f_kp1
    end
    return xₖ, flag, fₖ, ∇fₖ, k
  end

"""
   Solve by the descent with backtracking
"""

function backtracking(g,fₖ,∇fₖ,dₖ;α₀=1.,ρ=0.8,c₁=1.e-4)
    αₖ = α₀

    while g(αₖ) > fₖ + c₁*αₖ*∇fₖ'*dₖ
        αₖ = ρ*αₖ
    end
    return αₖ
end

function descent_backtrac(f,x0::Vector{<:Real};AbsTol= abs(eps()), RelTol = abs(eps()), ε=0.01, nbit_max = 0)
  flag = 0
  if nbit_max==0
    nbit_max = 100*length(x0)
  end

  # gardient function
  ∇f(x) = ForwardDiff.gradient(f, x)
  norm∇f₀ = norm(∇f(x0))
  continuer = (norm∇f₀ > AbsTol)
  xₖ = x0
  k = 0
  fₖ = f(xₖ)
  ∇fₖ = ∇f(xₖ)

  while continuer == true
      dₖ = - ∇fₖ
      g(α) = f(xₖ + α*dₖ)
      αₖ = backtracking(g,fₖ,∇fₖ,dₖ)
      k = k+1
      xₖ = xₖ + αₖ*dₖ
      f_kp1 = f(xₖ)
      ∇fₖ = ∇f(xₖ)
      continuer, flag = stop_criteria(∇fₖ,norm∇f₀,xₖ,dₖ,fₖ,f_kp1,k,AbsTol,RelTol,ε,nbit_max)
      fₖ = f_kp1
  end
  return xₖ, flag, fₖ, ∇fₖ, k
end





# Rosenbrock function
# f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2

