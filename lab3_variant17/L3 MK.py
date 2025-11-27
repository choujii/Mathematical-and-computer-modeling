"""
ЛР3: Метод Монте-Карло.
Область D — пересечение пяти окружностей (пересечение).
Интеграл I = ∬_D x^2 y^2 dA.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- 1. Геометрия области ----------

# Пять окружностей: центр (x0, y0) и радиус r
circles = [
    (0.0, 0.0, 2.0),
    (1.0, 0.0, 1.5),
    (-1.0, 0.0, 1.5),
    (0.0, 1.0, 1.5),
    (0.0, -1.0, 1.5),
]

MODE = "intersection"
SEED = 42
Ns = [1000, 3000, 10000, 30000, 100000, 300000]


# ---------- 2. Подынтегральная функция и вспомогательные функции ----------

def f_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Подынтегральная функция f(x,y) = x^2 y^2 (всегда ≥ 0).
    Работает с массивами numpy.
    """
    return (x ** 2) * (y ** 2)


def bounding_box(circles):
    """
    Минимальный охватывающий прямоугольник для всех окружностей.
    Возвращает: xmin, xmax, ymin, ymax.
    """
    xs, ys = [], []
    for (x0, y0, r) in circles:
        xs.extend([x0 - r, x0 + r])
        ys.extend([y0 - r, y0 + r])
    return min(xs), max(xs), min(ys), max(ys)


def point_in_disks(x, y, circles, mode="intersection"):
    """
    Проверяет принадлежность точки (или массива точек) области D.

    mode = "intersection" — пересечение всех дисков:
        точка должна лежать внутри КАЖДОЙ окружности.
    mode = "union" — объединение (в этой ЛР не используется).
    """
    masks = []
    for (x0, y0, r) in circles:
        masks.append((x - x0) ** 2 + (y - y0) ** 2 <= r ** 2)
    masks = np.stack(masks, axis=0)

    if mode == "intersection":
        return np.all(masks, axis=0)
    elif mode == "union":
        return np.any(masks, axis=0)
    else:
        raise ValueError("Unknown mode")


def monte_carlo(circles, N, mode="intersection", rng=None):
    """
    Оценка площади S(D) и интеграла I = ∬_D x^2 y^2 dA методом Монте-Карло.

    Возвращает словарь:
      {
        "N", "Area_estimate", "Area_SE",
        "Integral_estimate", "Integral_SE"
      }
    """
    if rng is None:
        rng = np.random.default_rng()

    # Охватывающий прямоугольник
    xmin, xmax, ymin, ymax = bounding_box(circles)
    S_box = (xmax - xmin) * (ymax - ymin)

    # Сэмплируем N точек в прямоугольнике B
    xs = rng.uniform(xmin, xmax, size=N)
    ys = rng.uniform(ymin, ymax, size=N)

    # Принадлежность области D
    inside = point_in_disks(xs, ys, circles, mode=mode)

    p = inside.mean()  # доля попаданий в область
    area_est = S_box * p  # оценка площади
    area_se = np.sqrt(p * (1 - p) / N) * S_box  # стандартная ошибка площади

    vals = f_xy(xs, ys) * inside
    integral_est = vals.mean() * S_box
    var_hat = vals.var(ddof=1) if N > 1 else 0.0
    integral_se = np.sqrt(var_hat / N) * S_box

    return {
        "N": int(N),
        "Area_estimate": float(area_est),
        "Area_SE": float(area_se),
        "Integral_estimate": float(integral_est),
        "Integral_SE": float(integral_se),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    rows = [monte_carlo(circles, N, mode=MODE, rng=rng) for N in Ns]
    df = pd.DataFrame(rows)

    df_ru = df.rename(columns={
        "N": "N (число точек)",
        "Area_estimate": "Оценка площади",
        "Area_SE": "SE площади",
        "Integral_estimate": "Оценка интеграла",
        "Integral_SE": "SE интеграла"
    })

    pd.set_option("display.float_format", lambda v: f"{v:.6g}")
    print("Результаты Монте-Карло (f = x^2 y^2):")
    print(df_ru)

    df_ru.to_csv("results_monte_carlo_x2y2_ru.csv", index=False, encoding="utf-8-sig")

    xmin, xmax, ymin, ymax = bounding_box(circles)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 400),
                         np.linspace(ymin, ymax, 400))
    mask = point_in_disks(gx, gy, circles, mode=MODE)

    plt.figure()
    plt.imshow(mask.astype(float), origin="lower",
               extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    theta = np.linspace(0, 2 * np.pi, 600)
    for (x0, y0, r) in circles:
        plt.plot(x0 + r * np.cos(theta), y0 + r * np.sin(theta))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Область интегрирования (пересечение 5 окружностей)")
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig("region.png", dpi=300)
    plt.show()

    xN = df["N"].to_numpy(float)
    area = df["Area_estimate"].to_numpy(float)
    se_area = df["Area_SE"].to_numpy(float)
    integ = df["Integral_estimate"].to_numpy(float)
    se_integ = df["Integral_SE"].to_numpy(float)

    # Сходимость площади
    plt.figure()
    plt.plot(xN, area, marker="o")
    plt.fill_between(xN, area - 2 * se_area, area + 2 * se_area, alpha=0.2)
    plt.xscale("log")
    plt.xlabel("N (log)")
    plt.ylabel("S(D)")
    plt.title("Сходимость оценки площади")
    plt.tight_layout()
    plt.savefig("area_convergence.png", dpi=300)
    plt.show()

    # Сходимость интеграла
    plt.figure()
    plt.plot(xN, integ, marker="o")
    plt.fill_between(xN, integ - 2 * se_integ, integ + 2 * se_integ, alpha=0.2)
    plt.xscale("log")
    plt.xlabel("N (log)")
    plt.ylabel("I")
    plt.title("Сходимость оценки интеграла")
    plt.tight_layout()
    plt.savefig("integral_convergence.png", dpi=300)
    plt.show()
