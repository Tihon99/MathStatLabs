import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import os


plt.style.use('ggplot')
np.random.seed(42)

os.makedirs('plots', exist_ok=True)


def save_histogram(dist, params, title, size):
    fig, ax = plt.subplots(figsize=(8, 6))

    sample = dist.rvs(*params, size=size)

    if dist == stats.poisson:
        bins = np.arange(sample.min() - 0.5, sample.max() + 1.5)
        x = np.arange(0, sample.max() + 1)
        pmf = dist.pmf(x, *params)

        ax.hist(
            sample,
            bins=bins,
            density=True,
            alpha=0.5,
            label='Гистограмма'
        )

        ax.plot(x, pmf, 'r-', lw=2, label='Теоретическая PMF')
        ax.set_xticks(x)

    else:
        ax.hist(
            sample,
            bins=30 if size > 50 else 10,
            density=True,
            alpha=0.5,
            label='Гистограмма'
        )

        x = np.linspace(sample.min(), sample.max(), 1000)
        pdf = dist.pdf(x, *params)
        ax.plot(x, pdf, 'r-', lw=2, label='Теоретическая PDF')

    ax.set_title(f'{title}, n={size}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min(sample) - 1 if dist == stats.poisson else None)

    filename = f'plots/{title}_{size}.png'.replace(' ', '_').lower()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def task1_visualization():
    distributions = [
        (stats.norm, (0, 1), 'N'),
        (stats.cauchy, (0, 1), 'C'),
        (stats.poisson, (10,), 'P'),
        (stats.uniform, (-np.sqrt(3), 2 * np.sqrt(3)), 'U')
    ]

    sizes = [10, 50, 1000]

    for dist, params, title in distributions:
        for size in sizes:
            save_histogram(dist, params, title, size)


def task2_statistics():
    distributions = {
        "Коши": stats.cauchy(loc=0, scale=1),
        "Нормальное": stats.norm(loc=0, scale=1),
        "Пуассона": stats.poisson(mu=10),
        "Равномерное": stats.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3))
    }

    sample_sizes = [10, 100, 1000]
    n_iterations = 1000
    stats_names = ["x", "medx", "zR", "zQ", "ztr"]

    for dist_name, dist in distributions.items():
        print(f"\n% === {dist_name.upper()} ===\n")
        print("\\begin{table}[H]")
        print("    \\centering")
        print("    \\begin{tabular}{|l|l|l|l|l|l|}")
        print("    \\hline")
        print(
            "         &  x &   \\textit{med}x   &   z\\textsubscript{R}  &   z\\textsubscript{Q}  &   z\\textsubscript{tr}\\\\ \\hline \\hline")

        for size in sample_sizes:
            stats_data = {name: {"E": [], "D": []} for name in stats_names}

            for _ in range(n_iterations):
                sample = dist.rvs(size=size)
                stats_values = {
                    "x": np.mean(sample),
                    "medx": np.median(sample),
                    "zR": (np.max(sample) + np.min(sample)) / 2,
                    "zQ": (np.quantile(sample, 0.25) + np.quantile(sample, 0.75)) / 2,
                    "ztr": stats.trim_mean(sample, 0.1)
                }

                for name in stats_names:
                    val = stats_values[name]
                    stats_data[name]["E"].append(val)
                    stats_data[name]["D"].append(val ** 2)

            print(f"         n = {size}& & & & & \\\\ \\hline")

            e_line = "         \\textit{E(z)} "
            for name in stats_names:
                E = np.mean(stats_data[name]["E"])
                e_line += f"&  {E:.6f} "
            print(e_line + "\\\\ \\hline")

            d_line = "         \\textit{D(z)} "
            for name in stats_names:
                E = np.mean(stats_data[name]["E"])
                D = np.mean(stats_data[name]["D"]) - E ** 2
                d_line += f"&  {D:.6f} "
            print(d_line + "\\\\ \\hline\\hline")

        print("    \\end{tabular}")
        print(f"    \\caption{{Таблица характеристик для {dist_name} распределения}}")
        print("    \\label{tab:" + dist_name.lower() + "}")
        print("\\end{table}")


if __name__ == "__main__":
    print("Создание графиков...")
    task1_visualization()

    print("\nСоздание таблиц...")
    task2_statistics()