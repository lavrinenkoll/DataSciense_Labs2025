from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('GLOP')
x1 = solver.NumVar(0, solver.infinity(), 'x1')
x2 = solver.NumVar(0, solver.infinity(), 'x2')

solver.Add(1.5 * x1 + 2 * x2 <= 12)
solver.Add(x1 + 2 * x2 <= 8)
solver.Add(4 * x1 <= 16)
solver.Add(4 * x2 <= 12)

solver.Minimize(-2 * x1 - 2 * x2)
status = solver.Solve()

x1_opt = x1.solution_value()
x2_opt = x2.solution_value()

if status == pywraplp.Solver.OPTIMAL:
    print("Оптимум: x1 =", x1_opt, ", x2 =", x2_opt)
    print("Мінімум Q =", solver.Objective().Value())
else:
    print("Рішення не знайдено.")