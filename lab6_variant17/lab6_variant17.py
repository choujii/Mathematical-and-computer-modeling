"""
Лабораторная работа №6. «Модель диффузии»
Вариант 17

Уравнение:
    U_t = D(x) U_xx - f(x) U + 5

D(x) = 2x + 1
f(x) = x + 1

Область:
    0 <= x <= 10
    0 <= t <= T (в коде T задаём явно, по умолчанию T = 1)

Начальное условие:
    U(0, x) = x (10 - x)^2

Граничные условия (Дирихле):
    U(t, 0)  = 0
    U(t, 10) = 0

Цели:
1) Построить явную и неявную разностные схемы.
2) Исследовать порядок аппроксимации и устойчивость.
3) Выполнить численные эксперименты и визуализировать результаты.


"""

# ========= ИМПОРТ БИБЛИОТЕК =========
import numpy as np  # работа с массивами и векторизацией
import matplotlib.pyplot as plt  # построение графиков
from pathlib import Path  # работа с путями к файлам
from math import sqrt, log2  # sqrt для RMSE, log2 для оценки порядка

# ========= ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ЗАДАЧИ =========

# Пространственный интервал: x ∈ [0, L]
L = 10.0

# Временной интервал: t ∈ [0, T]
# T можно менять по желанию, для экспериментов T=1 достаточно
T = 1.0


# Функции коэффициентов и начально-граничных условий
def D_coef(x):
    """Коэффициент диффузии D(x) = 2x + 1."""
    return 2.0 * x + 1.0


def f_coef(x):
    """Коэффициент при U в реакционном члене: f(x) = x + 1."""
    return x + 1.0


def u_init(x):
    """Начальное условие: U(0,x) = x(10 - x)^2."""
    return x * (10.0 - x) ** 2


def u_left_boundary(t):
    """Левая граница: U(t, 0) = 0."""
    return 0.0


def u_right_boundary(t):
    """Правая граница: U(t, 10) = 0."""
    return 0.0


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========

def compute_stable_tau(h):
    """
    Подбор верхней границы для шага по времени tau для ЯВНОЙ схемы.

    Для уравнения теплопроводности с постоянным D есть условие:
        tau <= h^2 / (2 D_max)

    У нас D(x) = 2x + 1, x ∈ [0, 10] ⇒ D_max = 2 * 10 + 1 = 21.
    Для надёжности берём небольший "запас" (safety_factor < 1).

    Возвращаем рекомендованный tau (можно использовать как tau_max).
    """
    D_max = 2.0 * L + 1.0  # максимум D(x) на [0, L]
    safety_factor = 0.9  # чтобы быть немного консервативными
    return safety_factor * h * h / (2.0 * D_max)


def rmse(U_num, U_ref):
    """
    Среднеквадратическое отклонение (RMSE) между двумя массивами
    одной формы.

    RMSE = sqrt( mean( (U_num - U_ref)^2 ) )
    """
    return float(sqrt(np.mean((U_num - U_ref) ** 2)))


# ========= ЯВНАЯ СХЕМА =========

def solve_explicit(Nx, Nt):
    """
    Численное решение уравнения диффузии явной схемой (FTCS).

    Nx — число интервалов по x (узлов Nx+1),
    Nt — число шагов по времени.

    Возвращает:
        x — массив узлов по x, shape = (Nx+1,)
        t — массив узлов по t, shape = (Nt+1,)
        U — массив решения, shape = (Nt+1, Nx+1),
             U[n, i] ≈ U(t^n, x_i)
    """
    # Шаги сетки
    h = L / Nx  # шаг по x
    tau = T / Nt  # шаг по t

    # Узлы сетки
    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    # Предвычислим коэффициенты D(x) и f(x) в узлах по x
    D_vals = D_coef(x)
    f_vals = f_coef(x)

    # Матрица решения: строки — моменты времени, столбцы — x
    U = np.zeros((Nt + 1, Nx + 1))

    # --- Инициализация: начальное условие и граничные условия ---
    U[0, :] = u_init(x)  # начальное условие при t=0
    U[0, 0] = u_left_boundary(0)  # на всякий случай согласуем точку (0,0)
    U[0, -1] = u_right_boundary(0)

    # --- Цикл по времени ---
    for n in range(Nt):
        # Сначала задаём граничные значения на новом слое (Дирихле)
        U[n + 1, 0] = u_left_boundary(t[n + 1])
        U[n + 1, -1] = u_right_boundary(t[n + 1])

        # Обновляем значения во внутренних узлах (i = 1..Nx-1)
        for i in range(1, Nx):
            # Вторая производная по x на текущем слое (центральная разность)
            U_xx = (U[n, i + 1] - 2.0 * U[n, i] + U[n, i - 1]) / (h * h)

            # Правая часть уравнения на текущем слое:
            # D(x_i) * U_xx - f(x_i) * U + 5
            rhs = D_vals[i] * U_xx - f_vals[i] * U[n, i] + 5.0

            # Явное обновление:
            U[n + 1, i] = U[n, i] + tau * rhs

    return x, t, U


def solve_implicit(Nx, Nt):
    """
    Численное решение уравнения диффузии неявной схемой (Backward Euler).

    На каждом шаге по времени возникает трёхдиагональная СЛАУ
    относительно U^{n+1}, которую решаем методом прогонки.

    Формула в узле i = 1..Nx-1:

        (1 + 2*alpha_i + tau*f_i) * U_i^{n+1}
        - alpha_i * U_{i-1}^{n+1}
        - alpha_i * U_{i+1}^{n+1}
        = U_i^n + tau * 5,

    где alpha_i = tau * D_i / h^2.
    """
    # Шаги сетки
    h = L / Nx
    tau = T / Nt

    # Узлы
    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    # Коэффициенты
    D_vals = D_coef(x)
    f_vals = f_coef(x)

    # Матрица решения
    U = np.zeros((Nt + 1, Nx + 1))

    U[0, :] = u_init(x)
    U[0, 0] = u_left_boundary(0.0)
    U[0, -1] = u_right_boundary(0.0)

    Nint = Nx - 1
    a = np.zeros(Nint)
    b = np.zeros(Nint)
    c = np.zeros(Nint)
    d = np.zeros(Nint)

    for n in range(Nt):
        U[n + 1, 0] = u_left_boundary(t[n + 1])
        U[n + 1, -1] = u_right_boundary(t[n + 1])

        for k in range(Nint):
            i = k + 1
            Di = D_vals[i]
            fi = f_vals[i]
            alpha = tau * Di / (h * h)

            a[k] = -alpha
            b[k] = 1.0 + 2.0 * alpha + tau * fi
            c[k] = -alpha

            d[k] = U[n, i] + tau * 5.0

        # Коррекция правой части с учётом известных граничных значений:
        # на самом деле здесь U[n+1,0]=0, U[n+1,Nx]=0, так что добавки 0,
        # но оставим шаблон для ясности.
        # d[0]   -= a[0]   * U[n+1, 0]
        # d[-1]  -= c[-1]  * U[n+1, Nx]

        # --- Метод прогонки (Thomas algorithm) ---
        # Прямой ход
        for k in range(1, Nint):
            w = a[k] / b[k - 1]
            b[k] -= w * c[k - 1]
            d[k] -= w * d[k - 1]

        y = np.zeros(Nint)
        y[-1] = d[-1] / b[-1]
        for k in range(Nint - 2, -1, -1):
            y[k] = (d[k] - c[k] * y[k + 1]) / b[k]

        U[n + 1, 1:Nx] = y

    return x, t, U


def convergence_experiment():
    """
    Численный эксперимент по оценке порядка аппроксимации по пространству.

    Идея:
    - считаем "почти точное" решение на самой мелкой сетке (Nx_max),
      неявной схемой (как более устойчивой).
    - для более грубых сеток интерполируем "точное" решение на их узлы
      и считаем RMSE.
    - по убыванию ошибки при уменьшении h оцениваем порядок.

    Чтобы не убить время, берём умеренные сетки.
    """

    Nx_list = [20, 40, 80, 160]

    C = 5
    results = []

    # Сначала считаем "точное" решение на самой мелкой сетке (последний Nx)
    Nx_fine = Nx_list[-1]
    h_fine = L / Nx_fine

    Nt_fine = int(C * Nx_fine ** 2)
    x_fine, t_fine, U_fine = solve_implicit(Nx_fine, Nt_fine)
    U_fine_T = U_fine[-1, :]

    print("Convergence experiment (implicit scheme, сравнение с самой мелкой сеткой):")
    print(" Nx    Nt        RMSE")

    for Nx in Nx_list[:-1]:
        h = L / Nx
        Nt = int(C * Nx ** 2)
        x, t, U = solve_implicit(Nx, Nt)
        U_T = U[-1, :]

        # Интерполируем "точное" решение на узлы текущей сетки
        U_ref_on_coarse = np.interp(x, x_fine, U_fine_T)

        err = rmse(U_T, U_ref_on_coarse)
        results.append((Nx, Nt, err))
        print(f"{Nx:3d}  {Nt:6d}   {err:10.6e}")

    print("\nОценка порядка (p ~ log2(e_h / e_{h/2})):")
    for k in range(1, len(results)):
        Nx1, Nt1, e1 = results[k - 1]
        Nx2, Nt2, e2 = results[k]
        p = log2(e1 / e2)
        print(f"сетка {Nx1:3d} -> {Nx2:3d}: p ≈ {p:.3f}")


def plot_slices(x, t, U_exp, U_imp, times_to_plot=(0.0, 0.25, 0.5, 1.0)):
    """
    Строит графики срезов по времени для явной и неявной схем.

    times_to_plot — список значений t, для которых нужно вывести профили U(t,x).
    """
    plt.figure(figsize=(8, 6))

    for tt in times_to_plot:
        n = np.argmin(np.abs(t - tt))
        plt.plot(x, U_exp[n, :], "--", label=f"явная, t≈{t[n]:.2f}")
        plt.plot(x, U_imp[n, :], "-", label=f"неявная, t≈{t[n]:.2f}")

    plt.xlabel("x")
    plt.ylabel("U(t,x)")
    plt.title("Сравнение явной и неявной схем (срезы по времени)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_heatmap(x, t, U, title):
    """
    Строит цветную карту (heatmap) решения U(t,x).
    По горизонтали — x, по вертикали — t.
    """
    plt.figure(figsize=(7, 4))
    # extent=[x_min, x_max, t_min, t_max], origin="lower" — t растёт вверх
    plt.imshow(U, extent=[x[0], x[-1], t[0], t[-1]],
               aspect="auto", origin="lower")
    plt.colorbar(label="U")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    out_dir = Path.home() / "Downloads" / "lab6_var17_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    Nx = 80
    h = L / Nx
    tau_stable = compute_stable_tau(h)
    Nt = int(T / tau_stable) + 1
    print(f"Используем Nx={Nx}, Nt={Nt}, h={h:.4f}, tau≈{T / Nt:.4e}")

    x, t, U_exp = solve_explicit(Nx, Nt)
    _, _, U_imp = solve_implicit(Nx, Nt)

    plot_slices(x, t, U_exp, U_imp, times_to_plot=(0.0, 0.2, 0.5, 1.0))

    plot_heatmap(x, t, U_imp, "Неявная схема (Backward Euler), U(t,x)")

    convergence_experiment()
