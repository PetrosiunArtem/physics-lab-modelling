import numpy as np
import matplotlib.pyplot as plt


def init_spins(N):
    return np.random.choice([-1, 1], size=N)


def calculate_energy(spins, B):
    M = np.sum(spins)
    return -B * M


def calculate_shannon_entropy(spins):
    N = len(spins)
    N_up = np.sum(spins == 1)
    N_down = N - N_up

    p_up = N_up / N
    p_down = N_down / N

    term_up = p_up * np.log(p_up) if p_up > 0 else 0
    term_down = p_down * np.log(p_down) if p_down > 0 else 0

    S = - (term_up + term_down)
    return S


def monte_carlo_step(spins, T, B):
    N = len(spins)
    for _ in range(N):
        i = np.random.randint(0, N)
        s = spins[i]
        dE = -B * (-s) - (-B * s)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i] = -s
    return spins


N_spins = 100
B = 1.0
T_range = np.linspace(0.1, 100.0, 50)
steps_per_temp = 600
samples_per_temp = 500

mean_magnetizations = []
mean_entropies = []
theoretical_entropies = []

print("Начинаем моделирование...")

for T in T_range:
    spins = init_spins(N_spins)

    for _ in range(steps_per_temp):
        monte_carlo_step(spins, T, B)

    m_accum = 0
    s_accum = 0

    for _ in range(samples_per_temp):
        monte_carlo_step(spins, T, B)
        m_accum += np.sum(spins) / N_spins
        s_accum += calculate_shannon_entropy(spins)

    mean_magnetizations.append(m_accum / samples_per_temp)
    mean_entropies.append(s_accum / samples_per_temp)

    x = B / T
    Z = 2 * np.cosh(x)
    p_u_theo = np.exp(x) / Z
    p_d_theo = np.exp(-x) / Z
    S_theo = - (p_u_theo * np.log(p_u_theo) + p_d_theo * np.log(p_d_theo))
    theoretical_entropies.append(S_theo)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(T_range, mean_magnetizations, 'o-', label='Монте-Карло', markersize=4)
plt.plot(T_range, np.tanh(B / T_range), 'r--', label='Теория: tanh(B/T)')
plt.xlabel('Температура (T)')
plt.ylabel('Средняя намагниченность <M>/N')
plt.title('Намагниченность парамагнетика')
plt.legend()
plt.grid(True)

# График 2: Энтропия Шеннона
plt.subplot(1, 2, 2)
plt.plot(T_range, mean_entropies, 'o-', label='МК: Энтропия Шеннона', markersize=4, color='green')
plt.plot(T_range, theoretical_entropies, 'k--', label='Теория')
plt.axhline(y=np.log(2), color='gray', linestyle=':', label='Max (ln 2)')
plt.xlabel('Температура (T)')
plt.ylabel('Энтропия S')
plt.title('Энтропия Шеннона')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
