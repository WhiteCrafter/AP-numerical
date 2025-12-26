# Notes for Presentation (Personal)

## 1. Big picture: what the code is simulating
- I am modeling how a phone cools down while it is still generating heat.
- The phone is one object, the case is another, and the air is the outside sink.
- The code compares a thin case vs a thick case to see which cools faster.

## 2. Physical meaning of the variables
- `T_p(t)`: phone internal temperature (the thing I care about most).
- `T_c(t)`: case temperature (acts like a buffer between phone and air).
- `T_amb`: ambient air temperature (assumed constant).
- `Q_gen`: constant heat produced by the phone.
- `k_pc`: how easily heat moves from phone to case.
- `k_ca`: how easily heat moves from case to air.
- `C_p`, `C_c`: heat capacity of phone and case (bigger means slower change).

## 3. How the ODE system is represented in code
- In `rhs()`, I calculate the two derivatives based on the equations.
- `T_p'` is heat generation minus heat loss to case.
- `T_c'` is heat gained from phone minus heat lost to air.
- The state vector is `[T_p, T_c]` everywhere.
- `jacobian_f()` is the derivative of the RHS for Newton’s method.

## 4. Why Implicit Euler was chosen
- Implicit Euler is stable even for bigger time steps.
- Heat transfer systems can be stiff (fast changes from large k values).
- If asked why implicit Euler -> mention stability and stiffness handling.

## 5. How the two nonlinear solvers work (idea-level)
- Fixed-Point Iteration:
  - Guess the next temperature, plug into the formula, repeat.
  - I stop when the change is tiny.
  - Simple but can be slow.
- Newton’s Method:
  - Solve the implicit equation by linearizing it.
  - Uses the Jacobian to jump closer to the solution.
  - Usually fewer iterations.
- If asked why compare both -> show convergence speed difference.

## 6. What changes between thin and thick case
- The only change is `k_ca` (case to air transfer).
- Thin case: larger `k_ca` -> heat escapes faster.
- Thick case: smaller `k_ca` -> heat escapes slower.

## 7. What the plots should look like and why
- Each case has a plot of `T_p(t)`.
- Thin case should cool faster and reach lower temperature.
- Thick case should stay warmer for longer.
- The two solver curves should almost overlap (same method, different iterations).

## 8. Typical questions and short answer hints
- If asked why implicit Euler -> mention stability for stiff heat transfer.
- If asked about stiffness -> large heat transfer or larger time step.
- If asked about fixed-point vs Newton -> Newton is faster, fixed-point is simpler.
- If asked about physical meaning -> phone loses heat to case, case loses to air.
- If asked about accuracy -> both solve same implicit step, should match closely.
