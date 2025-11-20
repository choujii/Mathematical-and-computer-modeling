"""
Лабораторная работа №5. «Модели переноса»
Вариант 16

Уравнение переноса:
    U_t + V(x) * U_x = f(x),   x ∈ [0, 10],  t ∈ [0, 100]

Дано:
    V(x) = x + 1
    f(x) ≡ 14.6  (правая часть — постоянная)
    U(0, x) = max(0, (x - 5)(10 - x))      — начальное условие
    U(t, 0) = t (t - 20)^2 / 200000        — левое граничное условие

Что делает этот файл:
1) Находим аналитическое решение методом характеристик
2) Строим численные решения:
   - явная схема 1-го порядка (явный «уголок», upwind),
   - неявная схема 1-го порядка (неявный «уголок», upwind).
3) Считаем среднеквадратичное отклонение (RMSE) между численным и аналитическим
   решениями на 5 разных сетках по пространству/времени.
4) Строим и сохраняем график сравнения решений при T=100.
5) Создаём текстовый отчёт с таблицей погрешностей.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt

out_dir = Path.home() / "Downloads" / "lab5_var16_results"
out_dir.mkdir(parents=True, exist_ok=True)

L = 10.0
T = 100.0
f_const = 14.6


def V(x):
    """
    Функция скорости переноса V(x).
    По условию варианта: V(x) = x + 1.
    При V(x) > 0 характеристики идут вправо, поэтому граничное
    условие задаётся только слева (x = 0).
    """
    return x + 1.0


def phi_ic(x):
    """
    Начальное условие U(0, x).

    По условию:
        U(0, x) = max(0, (x - 5)(10 - x))

    Это «колокол»-парабола, обрезанная по нулю:
    - в середине (примерно x ∈ (5, 10)) она положительна,
    - вне этого интервала — 0.
    """
    return np.maximum(0.0, (x - 5.0) * (10.0 - x))


def g_bc(t):
    """
    Граничное условие на левой границе x = 0.

    По условию:
        U(t, 0) = t (t - 20)^2 / 200000

    Это некоторый гладкий по времени «вход» на левом краю.
    """
    return t * (t - 20.0) ** 2 / 200000.0


def analytic_U(t, x):
    """
    Аналитическое решение U(t,x).

    Метод характеристик даёт следующую картину (коротко):
    - Характеристики описываются ОДУ: dx/dt = V(x) = x + 1.
    - Решение этого ОДУ даёт первый интеграл: (x + 1) e^{-t} = const.
    - Вдоль каждой характеристики U(t,x) удовлетворяет dU/dt = f(x) = 14.6.
    - В зависимости от того, где характеристика «начинается»,
      мы используем либо начальное условие (t=0), либо граничное (x=0).

    Здесь t и x могут быть:
    - скалярами,
    - либо numpy-массивами одинаковой формы (например, сетка).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    # Разделяющая кривая t = ln(x+1):
    # если t меньше этой величины — характеристика приходит из t=0,
    # если больше — приходит с границы x=0.
    boundary = np.log(x + 1.0)

    # Массив булевых значений: True там, где используем начальные данные
    from_ic = t <= boundary

    U = np.empty_like(t, dtype=float)

    # ---- 1) Точки, для которых характеристика идёт от начальной линии t=0 ----
    if np.any(from_ic):
        # Вычисляем точку x0 на оси t=0, из которой пришла характеристика.
        # Инвариант: (x+1) e^{-t} = (x0+1) e^0 => x0 = (x+1)e^{-t} - 1
        xi0 = (x[from_ic] + 1.0) * np.exp(-t[from_ic]) - 1.0
        # Вдоль характеристики dU/dt = 14.6 ⇒ U(t,x) = 14.6 t + U(0,x0)
        U[from_ic] = f_const * t[from_ic] + phi_ic(xi0)

    # ---- 2) Точки, для которых характеристика пришла с границы x=0 ----
    if np.any(~from_ic):
        # Найдём момент времени tb, когда характеристика пересекает x=0.
        # Инвариант: (x+1)e^{-t} = (0+1)e^{-tb} ⇒ tb = t - ln(x+1)
        tb = t[~from_ic] - np.log(x[~from_ic] + 1.0)
        # На границе U(tb,0) = g(tb). Вдоль характеристики:
        # U(t,x) = U(tb,0) + 14.6 (t - tb) = g(tb) + 14.6 ln(x+1)
        U[~from_ic] = f_const * np.log(x[~from_ic] + 1.0) + g_bc(tb)

    return U



def solve_explicit(Nx, Nt, L=L, T=T, f=f_const):
    h = L / Nx
    tau = T / Nt

    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    Vx = V(x)

    U = np.zeros((Nt + 1, Nx + 1))

    U[0, :] = phi_ic(x)
    U[0, 0] = g_bc(0.0)

    for n in range(Nt):
        # 2.1. Сначала задаём левое граничное условие на новом слое (x=0)
        U[n + 1, 0] = g_bc(t[n + 1])

        # 2.2. Затем считаем значения в остальных узлах по x (i = 1..Nx)
        for i in range(1, Nx + 1):
            a = Vx[i]
            lam = tau * a / h  # число Куранта lambda_i

            U[n + 1, i] = U[n, i] - lam * (U[n, i] - U[n, i - 1]) + tau * f

    return x, t, U


def solve_implicit(Nx, Nt, L=L, T=T, f=f_const):
    """
    Численное решение неявной схемой 1-го порядка (upwind по будущему слою).

    Неявный upwind:
        (U_i^{n+1} - U_i^n) / tau + V_i (U_i^{n+1} - U_{i-1}^{n+1}) / h = f

    Перегруппируем:
        U_i^{n+1} + lambda_i U_i^{n+1} - lambda_i U_{i-1}^{n+1} = U_i^n + tau f
        (1 + lambda_i) U_i^{n+1} - lambda_i U_{i-1}^{n+1} = U_i^n + tau f

    Это нижнетреугольная система относительно U^{n+1}_i по i,
    поэтому можно просто проходить i=1..Nx и выражать U^{n+1}_i:
        U_i^{n+1} = (U_i^n + tau f + lambda_i U_{i-1}^{n+1}) / (1 + lambda_i)

    Граничное условие на новом слое:
        U_0^{n+1} = g(t^{n+1})
    """

    h = L / Nx
    tau = T / Nt

    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    Vx = V(x)

    U = np.zeros((Nt + 1, Nx + 1))

    U[0, :] = phi_ic(x)
    U[0, 0] = g_bc(0.0)

    for n in range(Nt):
        U[n + 1, 0] = g_bc(t[n + 1])

        for i in range(1, Nx + 1):
            a = Vx[i]
            lam = tau * a / h

            U[n + 1, i] = (U[n, i] + tau * f + lam * U[n + 1, i - 1]) / (1.0 + lam)

    return x, t, U


def compute_rmse(U_num, U_exact):
    """
    Формула:
        RMSE = sqrt( (1/N) * Σ (U_num - U_exact)^2 )
    где N — количество узлов (точек сравнения).
    """
    return sqrt(np.mean((U_num - U_exact) ** 2))


if __name__ == "__main__":

    variants = [
        (10, 1110),
        (20, 2220),
        (40, 4440),
        (50, 5550),
        (100, 11100),
    ]

    print("Результаты сравнения (всё посчитано только кодом):")
    print(" Nx   Nt      RMSE_exp      RMSE_imp")
    print("--------------------------------------")

    table_lines = []

    # ----- Цикл по всем вариантам сетки -----
    for Nx, Nt in variants:
        # Численное решение явной схемой
        x, t, U_exp = solve_explicit(Nx, Nt)
        # Численное решение неявной схемой
        _, _, U_imp = solve_implicit(Nx, Nt)

        # Строим сетку (t,x) для вычисления аналитического решения в тех же точках
        TT, XX = np.meshgrid(t, x, indexing="ij")  # TT[n,i] = t^n, XX[n,i] = x_i
        U_an = analytic_U(TT, XX)  # аналитическое U(t^n, x_i)

        # СКО для явной и неявной схем
        rmse_exp = compute_rmse(U_exp, U_an)
        rmse_imp = compute_rmse(U_imp, U_an)

        # Печать в консоль
        print(f"{Nx:3d} {Nt:5d}   {rmse_exp:10.5f}   {rmse_imp:10.5f}")
        # И одновременно копим строку для файла-отчёта
        table_lines.append(f"{Nx:3d} {Nt:5d}   {rmse_exp:10.5f}   {rmse_imp:10.5f}")

    Nx, Nt = 100, 11100
    x, t, U_exp = solve_explicit(Nx, Nt)
    _, _, U_imp = solve_implicit(Nx, Nt)

    TT, XX = np.meshgrid(t, x, indexing="ij")
    U_an = analytic_U(TT, XX)

    U_exp_T = U_exp[-1, :]
    U_imp_T = U_imp[-1, :]
    U_an_T = U_an[-1, :]

    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(x, U_an_T, label="аналитическое решение", linewidth=2)
    plt.plot(x, U_exp_T, "--", label="явная схема", linewidth=1.5)
    plt.plot(x, U_imp_T, ":", label="неявная схема", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("U(T,x),  T = 100")
    plt.title("Сравнение аналитического и численных решений при T=100, Nx=100")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / "compare_T100.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    report_path = out_dir / "report_lab5_var16.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Лабораторная работа №5. Модели переноса. Вариант 16\n")
        f.write("Все численные значения и погрешности получены программно.\n\n")
        f.write("Сравнение среднеквадратических отклонений (RMSE)\n")
        f.write("Nx   Nt      RMSE_exp      RMSE_imp\n")
        f.write("--------------------------------------\n")
        for line in table_lines:
            f.write(line + "\n")
        f.write("\nГрафик U(T,x) при T=100 сохранён в файле: ")
        f.write(str(fig_path.name) + "\n")
        f.write("Папка с результатами: " + str(out_dir) + "\n")

    print("\nВсе файлы сохранены в папку:", out_dir)
    print("  - таблица погрешностей: ", report_path)
    print("  - график сравнения:     ", fig_path)
