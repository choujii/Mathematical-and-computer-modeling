import numpy as np
import matplotlib.pyplot as plt

# ===== Параметры модели =====
r1, r2 = 2.0, 2.0
K1, K2 = 200.0, 200.0

N10, N20 = 100.0, 100.0

# ===== Правая часть системы =====
def dN_dt(t, N, alpha12, alpha21):
    N1, N2 = N
    dN1 = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
    dN2 = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
    return np.array([dN1, dN2])


# ===== Численное решение (метод Рунге–Кутты 4-го порядка) =====
def solve_system(alpha12, alpha21,
                 T=50.0, dt=0.01,
                 N0=(N10, N20)):
    n_steps = int(T / dt) + 1
    t = np.linspace(0, T, n_steps)
    N = np.zeros((n_steps, 2))
    N[0] = N0

    for i in range(n_steps - 1):
        Ni = N[i]
        ti = t[i]

        k1 = dN_dt(ti, Ni, alpha12, alpha21)
        k2 = dN_dt(ti + dt/2, Ni + dt * k1/2, alpha12, alpha21)
        k3 = dN_dt(ti + dt/2, Ni + dt * k2/2, alpha12, alpha21)
        k4 = dN_dt(ti + dt,   Ni + dt * k3,   alpha12, alpha21)

        N[i+1] = Ni + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        # не даём численностям уйти в отрицательные значения из-за численных ошибок
        N[i+1] = np.maximum(N[i+1], 0)

    return t, N[:, 0], N[:, 1]


# ===== Функция для построения графиков =====
def plot_dynamics_and_phase(alpha12, alpha21,
                            T=50.0, dt=0.01,
                            N0=(N10, N20)):
    t, N1, N2 = solve_system(alpha12, alpha21, T=T, dt=dt, N0=N0)

    # ---- Временные зависимости ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(t, N1, label='N1 (вид 1)')
    axs[0].plot(t, N2, label='N2 (вид 2)', linestyle='--')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('Численность')
    axs[0].set_title(f'Динамика численности\nα12={alpha12}, α21={alpha21}')
    axs[0].legend()
    axs[0].grid(True)

    # ---- Фазовая плоскость ----
    ax = axs[1]

    # сетка для поля направлений
    N1_vals = np.linspace(0, K1*1.1, 20)
    N2_vals = np.linspace(0, K2*1.1, 20)
    N1_grid, N2_grid = np.meshgrid(N1_vals, N2_vals)

    dN1_grid, dN2_grid = dN_dt(
        0,
        np.array([N1_grid, N2_grid]),
        alpha12, alpha21
    )

    # нормируем векторы, чтобы стрелки были аккуратными
    speed = np.sqrt(dN1_grid**2 + dN2_grid**2)
    speed[speed == 0] = 1
    dN1n = dN1_grid / speed
    dN2n = dN2_grid / speed

    ax.quiver(N1_grid, N2_grid, dN1n, dN2n, angles='xy')

    # траектория
    ax.plot(N1, N2, color='black', linewidth=2, label='Траектория')

    # изоклины
    # N1-изоклина: N1=0 и N1 + α12 N2 = K1
    N2_line = np.linspace(0, K2*1.2, 100)
    N1_null2 = K1 - alpha12 * N2_line
    ax.plot(np.clip(N1_null2, 0, None), N2_line,
            label='изоклина dN1/dt=0')

    # N2-изоклина: N2=0 и N2 + α21 N1 = K2
    N1_line = np.linspace(0, K1*1.2, 100)
    N2_null2 = K2 - alpha21 * N1_line
    ax.plot(N1_line, np.clip(N2_null2, 0, None),
            label='изоклина dN2/dt=0')

    # особые точки
    ax.scatter([0, K1, 0], [0, 0, K2], color='red', s=40)

    # внутреннее равновесие (если есть)
    denom = 1 - alpha12 * alpha21
    if denom != 0:
        N2_star = (K2 - alpha21 * K1) / denom
        N1_star = K1 - alpha12 * N2_star
        if N1_star > 0 and N2_star > 0:
            ax.scatter([N1_star], [N2_star], color='green', s=60,
                       label='совместное равновесие')

    ax.set_xlabel('N1')
    ax.set_ylabel('N2')
    ax.set_xlim(0, K1*1.1)
    ax.set_ylim(0, K2*1.1)
    ax.set_title('Фазовая диаграмма')
    ax.grid(True)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


# ===== Примеры запусков =====
# 1) Слабая конкуренция — устойчивое сосуществование
plot_dynamics_and_phase(alpha12=0.5, alpha21=0.5)

# 2) Вид 1 сильнее угнетает вид 2
plot_dynamics_and_phase(alpha12=0.5, alpha21=1.5)

# 3) Вид 2 сильнее угнетает вид 1
plot_dynamics_and_phase(alpha12=1.5, alpha21=0.5)

# 4) Сильная конкуренция обоих видов (исход зависит от начальных условий)
plot_dynamics_and_phase(alpha12=1.5, alpha21=1.5)
