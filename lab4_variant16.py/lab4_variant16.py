"""
Лабораторная работа №4. Вариант 16
Тема: Стохастические модели — DLA (диффузионная агрегация) и IFS (итерационные аффинные отображения)
Автор: Львов Максим Олегович

Цель:
- Смоделировать процесс диффузионной агрегации на квадратной решётке.
- Сгенерировать аттрактор IFS по трём заданным аффинным преобразованиям.
- Оценить метрическую (фрактальную) размерность обоих объектов методом боксового счёта.
- Построить графики и сохранить результаты.
"""

# ============================================================
# 1. Импорт необходимых библиотек
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log
from pathlib import Path
import random

# ============================================================
# 2. Настройки среды и сохранения
# ============================================================

# Фиксируем случайность для воспроизводимости результата
random.seed(42)
np.random.seed(42)

# Папка для сохранения всех файлов (в твоей папке Downloads)
out_dir = Path(r"C:\Users\0potter0\Downloads\lab4_variant16")
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3. Метод боксового счёта (Box Counting)
# ============================================================

def box_counting_dimension(points, eps_list=None):
    """
    Оценивает фрактальную размерность множества точек.

    Алгоритм:
    1. Задаём размер бокса ε (квадрата).
    2. Разбиваем пространство на сетку и считаем,
       сколько боксов содержит хотя бы одну точку: N(ε).
    3. Повторяем для разных ε.
    4. Строим зависимость log N(ε) от log(1/ε).
    5. Угловой коэффициент (наклон прямой) — фрактальная размерность D.

    Возвращает:
        D          — оценка размерности
        eps_list   — список масштабов ε
        logs_eps   — список log(1/ε)
        logs_N     — список log N(ε)
    """
    pts = np.asarray(points)
    if pts.size == 0:
        return np.nan, [], [], []

    # Сдвигаем точки, чтобы минимальные координаты начинались с нуля
    mn = pts.min(axis=0)
    pts0 = pts - mn

    # Если список eps не задан, выбираем автоматически на основе размера области
    if eps_list is None:
        span = (pts0.max(axis=0) - pts0.min(axis=0)).max()
        max_pow = int(np.floor(np.log2(span)))
        eps_list = [2**k for k in range(max_pow, 0, -1)]

    logs_eps = []  # log(1/ε)
    logs_N = []    # log N(ε)
    N_values = []  # само N(ε)

    for eps in eps_list:
        # Преобразуем координаты в индексы ячеек (какой бокс занимает точка)
        idx = np.floor(pts0 / eps).astype(int)
        # Считаем количество уникальных ячеек — это N(ε)
        n_boxes = len(np.unique(idx, axis=0))
        if n_boxes > 0:
            logs_eps.append(log(1 / eps))
            logs_N.append(log(n_boxes))
            N_values.append(n_boxes)

    # Если точек мало, оценка невозможна
    if len(logs_eps) < 2:
        return np.nan, eps_list, logs_eps, logs_N

    # Аппроксимация прямой: log N = D * log(1/ε) + const
    A = np.vstack([logs_eps, np.ones(len(logs_eps))]).T
    D, _ = np.linalg.lstsq(A, logs_N, rcond=None)[0]
    return D, eps_list, logs_eps, logs_N


def plot_boxcount(logs_eps, logs_N, title, fname):
    """
    Строит график зависимости log N(ε) от log(1/ε),
    используемый для оценки фрактальной размерности.
    """
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(logs_eps, logs_N, "o-", lw=1)
    plt.xlabel("log(1/ε)")
    plt.ylabel("log N(ε)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 4. DLA — Диффузионная агрегация
# ============================================================

def dla_cluster(n_particles=2000, grid_half=150):
    """
    Генерирует кластер DLA (Diffusion-Limited Aggregation).
    Модель:
      - Частьцы стартуют случайно на внешней окружности.
      - Двигаются случайно по решётке (вверх/вниз/влево/вправо).
      - Когда касаются кластера — прилипают.
      - Процесс повторяется для всех частиц.
    """
    occ = {(0, 0)}  # начальная точка — "затравка"
    radius = 1      # текущий радиус кластера

    def spawn(r):
        """Создаёт частицу на случайной точке окружности чуть дальше кластера"""
        ang = random.random() * 2 * np.pi
        R = max(5, r + 5)
        return [int(R * np.cos(ang)), int(R * np.sin(ang))]

    def near_cluster(x, y):
        """Проверка, есть ли соседняя занятая клетка (прилипание по 4-соседству)"""
        return ((x + 1, y) in occ or (x - 1, y) in occ or
                (x, y + 1) in occ or (x, y - 1) in occ)

    for _ in range(n_particles):
        x, y = spawn(radius)
        while True:
            # Случайное движение
            d = random.randint(0, 3)
            if d == 0: x += 1
            elif d == 1: x -= 1
            elif d == 2: y += 1
            else: y -= 1

            # Проверка прилипания
            if near_cluster(x, y):
                occ.add((x, y))
                # обновляем радиус, если вышли дальше
                r2 = x*x + y*y
                if r2 > radius*radius:
                    radius = int(sqrt(r2)) + 1
                break

            # если частица ушла слишком далеко — перезапускаем
            if abs(x) > grid_half or abs(y) > grid_half:
                x, y = spawn(radius)
    return occ


def plot_cluster(points, title, fname):
    """Отображает и сохраняет кластер DLA"""
    pts = np.array(list(points))
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(pts[:, 0], pts[:, 1], s=1)
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 5. IFS — Итерационные аффинные отображения
# ============================================================

# Матрицы и векторы с листа (вариант 16)
A1 = np.array([[0.426, 1.696], [-1.116, -0.364]], float)
b1 = np.array([32.296, -20.717])
A2 = np.array([[0.169, 0.608], [-1.064, -1.140]], float)
b2 = np.array([16.172, 69.938])
A3 = np.array([[0.926, 0.940], [-1.836, 0.192]], float)
b3 = np.array([-48.903, -26.245])
As = [A1, A2, A3]
bs = [b1, b2, b3]


def chaos_game(A_list, b_list, p=None, n=100_000, burn=500, clip_R=1e6):
    """
    Хаос-игра (Chaos Game) для генерации IFS-аттрактора.

    Алгоритм:
    1. Стартуем из точки (0,0).
    2. Случайно выбираем одно из аффинных преобразований (Ai, bi).
    3. Применяем его: x_{k+1} = Ai * x_k + bi.
    4. Повторяем n раз.
    5. Добавлено ограничение clip_R, чтобы точки не "взрывались" в бесконечность.
    """
    if p is None:
        p = np.ones(len(A_list)) / len(A_list)  # равные вероятности
    cdf = np.cumsum(p)
    x = np.zeros(2)
    pts = []

    # "прожигаем" первые шаги (чтобы уйти от начальных переходных процессов)
    for _ in range(burn):
        r = random.random()
        i = int(np.searchsorted(cdf, r))
        x = A_list[i] @ x + b_list[i]
        if np.linalg.norm(x) > clip_R:
            x[:] = 0

    # Основной цикл генерации точек
    for _ in range(n):
        r = random.random()
        i = int(np.searchsorted(cdf, r))
        x = A_list[i] @ x + b_list[i]
        if np.linalg.norm(x) > clip_R:
            x[:] = 0
        pts.append(x.copy())

    return np.array(pts)


def plot_points(pts, title, fname):
    """Отображает и сохраняет аттрактор IFS"""
    plt.figure(figsize=(7, 6), dpi=150)
    plt.scatter(pts[:, 0], pts[:, 1], s=0.2)
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 6. Основной блок выполнения программы
# ============================================================

if __name__ == "__main__":

    # ---------- ЗАДАНИЕ I: DLA ----------
    print("=== DLA ===")
    occ = dla_cluster(4000)
    pts = np.array(list(occ))
    D, eps_list, logs_eps, logs_N = box_counting_dimension(pts, eps_list=[1, 2, 4, 8, 16, 32, 64])
    print(f"Фрактальная размерность D ≈ {D:.4f}")
    plot_cluster(pts, f"DLA (n=4000) D≈{D:.3f}", "dla_4000.png")
    plot_boxcount(logs_eps, logs_N, "Боксовый счёт для DLA", "dla_boxcount.png")

    # ---------- ЗАДАНИЕ II: IFS ----------
    print("\n=== IFS ===")
    pts2 = chaos_game(As, bs, n=300_000, burn=2000)
    D2, eps_list2, logs_eps2, logs_N2 = box_counting_dimension(pts2)
    print(f"Фрактальная размерность D ≈ {D2:.4f}")
    plot_points(pts2, f"IFS-аттрактор (N=300000) D≈{D2:.3f}", "ifs_300000.png")
    plot_boxcount(logs_eps2, logs_N2, "Боксовый счёт для IFS", "ifs_boxcount.png")

    # ---------- ЗАПИСЬ КРАТКОГО ОТЧЁТА ----------
    report = (
        "Лабораторная работа №4. Вариант 16\n\n"
        "Задание I. DLA (диффузионная агрегация)\n"
        f"  Фрактальная размерность: D ≈ {D:.4f}\n\n"
        "Задание II. IFS (итерационные аффинные преобразования)\n"
        f"  Фрактальная размерность: D ≈ {D2:.4f}\n\n"
        "Все изображения и графики сохранены в папке lab4_variant16."
    )
    with open(out_dir / "report_variant16.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n--- Готово ---")
    print("Все результаты сохранены в:", out_dir)
