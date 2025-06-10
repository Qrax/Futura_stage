import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


# --- Hoofdfuncties ---

def generate_synthetic_data(num_traces, num_points, base_signal_func, noise_std):
    """
    Genereert een set van 'traces' met een basis-signaal en willekeurige ruis.

    Args:
        num_traces (int): Het aantal metingen (N).
        num_points (int): Het aantal datapunten per meting (bv. lengte van de window).
        base_signal_func (function): Een functie die het 'perfecte' signaal definieert.
        noise_std (float): De standaarddeviatie (sigma) van de ruis. Meer is meer spreiding.

    Returns:
        tuple: Een tuple met (x_axis, data_traces).
    """
    x_axis = np.linspace(0, 100, num_points)
    true_signal = base_signal_func(x_axis)

    # Maak een 2D-array voor alle traces
    data_traces = np.zeros((num_traces, num_points))

    # Genereer elke trace door ruis toe te voegen aan het perfecte signaal
    for i in range(num_traces):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=num_points)
        data_traces[i, :] = true_signal + noise

    return x_axis, data_traces


def calculate_summary_stats(data_traces):
    """
    Berekent het gemiddelde, std, sem en N van een set traces.
    Vergelijkbaar met de logica in je project.
    """
    n = data_traces.shape[0]
    mean_trace = np.mean(data_traces, axis=0)
    std_trace = np.std(data_traces, axis=0, ddof=1)  # ddof=1 voor steekproef std
    sem_trace = std_trace / np.sqrt(n)  # Standard Error of the Mean

    return {
        'n': n,
        'mean': mean_trace,
        'std': std_trace,
        'sem': sem_trace
    }


def plot_intervals(ax, x, stats, title):
    """
    Maakt een plot op een gegeven 'ax' met de gemiddelde lijn, CI en PI.
    """
    n = stats['n']
    mean_trace = stats['mean']
    std_trace = stats['std']
    sem_trace = stats['sem']

    # Kritieke t-waarde voor 95% intervallen
    # Voor grote N (bv > 30) is dit ~1.96. Voor kleine N is het groter.
    tcrit = t.ppf(0.975, df=n - 1)

    # --- Plot de data ---
    # Plot de gemiddelde lijn
    ax.plot(x, mean_trace, color='black', lw=2.5, label='Gemiddelde Lijn')

    # Bereken en plot het 95% Confidence Interval (CI)
    ci_margin = tcrit * sem_trace
    ax.fill_between(x, mean_trace - ci_margin, mean_trace + ci_margin,
                    color='blue', alpha=0.5, label=f'95% CI (Zekerheid Gemiddelde)')

    # Bereken en plot het 95% Prediction Interval (PI)
    pi_margin = tcrit * std_trace * np.sqrt(1 + 1 / n)
    ax.fill_between(x, mean_trace - pi_margin, mean_trace + pi_margin,
                    color='cyan', alpha=0.3, label=f'95% PI (Spreiding Data)')

    # Opmaak van de plot
    avg_std_dev = np.mean(std_trace)  # Gemiddelde standaarddeviatie voor titel
    ax.set_title(f"{title}\n(N={n}, Gem. Standaarddev. σ ≈ {avg_std_dev:.2f})", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_xlabel("Tijd (sample index)")
    ax.set_ylabel("Meting (Voltage/ADC)")


# --- Hoofdscript ---
if __name__ == "__main__":
    # Definieer een basis-signaal. We gebruiken een sinusgolf met een piek.
    base_signal = lambda x: 2 * np.sin(x / 15) + 5 * np.exp(-((x - 60) ** 2) / 200)

    # Maak een figuur met 3 subplots onder elkaar
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, sharey=True)
    fig.suptitle("Het Verschil tussen Confidence (CI) en Prediction (PI) Intervallen", fontsize=16, y=0.96)

    # --- SCENARIO 1: Ideale situatie (lage spreiding, veel data) ---
    # Je "small difference" case.
    N_1 = 100
    NOISE_STD_1 = 0.75
    x1, data1 = generate_synthetic_data(N_1, 200, base_signal, NOISE_STD_1)
    stats1 = calculate_summary_stats(data1)
    plot_intervals(axes[0], x1, stats1, "Scenario 1: Veel Data, Lage Spreiding")

    # --- SCENARIO 2: Hoge spreiding (veel ruis) ---
    # Je "big difference" case. De data is "more out of order".
    N_2 = 100
    NOISE_STD_2 = 3.0  # Vier keer zoveel ruis als in scenario 1!
    x2, data2 = generate_synthetic_data(N_2, 200, base_signal, NOISE_STD_2)
    stats2 = calculate_summary_stats(data2)
    plot_intervals(axes[1], x2, stats2, "Scenario 2: Veel Data, Hoge Spreiding")

    # --- SCENARIO 3: Weinig data ---
    # Dit toont een ander effect dat data "out of order" kan laten lijken: onzekerheid door weinig metingen.
    N_3 = 5  # Heel weinig metingen!
    NOISE_STD_3 = 0.75  # Zelfde lage ruis als in scenario 1
    x3, data3 = generate_synthetic_data(N_3, 200, base_signal, NOISE_STD_3)
    stats3 = calculate_summary_stats(data3)
    plot_intervals(axes[2], x3, stats3, "Scenario 3: Weinig Data, Lage Spreiding")

    # Zorg voor een nette layout en toon de plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Maak ruimte voor de suptitle
    plt.show()