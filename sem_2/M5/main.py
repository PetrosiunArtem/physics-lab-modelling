import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


class TunnelDiode:
    def __init__(self, a=4e-3, b=-16e-3, c=17e-3):
        self.a = a
        self.b = b
        self.c = c

    def current(self, U):
        if U <= 0:
            return 0.0
        return self.a * U ** 3 + self.b * U ** 2 + self.c * U

    def conductance(self, U):
        if U <= 0:
            return 0.0
        return 3 * self.a * U ** 2 + 2 * self.b * U + self.c

    def stationary_voltage(self, E, R, guess=1.3):
        def eq(U0):
            return self.current(U0) - (E - U0) / R

        U0 = fsolve(eq, guess)[0]
        return U0


class TunnelDiodeOscillator:
    def __init__(self, diode, R, L, C, E):
        self.diode = diode
        self.R = R
        self.L = L
        self.C = C
        self.E = E
        self.U0 = diode.stationary_voltage(E, R)
        self.I0 = diode.current(self.U0)
        self.g0 = diode.conductance(self.U0)

    def system(self, t, y):
        I, U = y
        dI = (self.E - I * self.R - U) / self.L
        dU = (I - self.diode.current(U)) / self.C
        return [dI, dU]

    def stability_analysis(self):
        tr = self.R / self.L + self.g0 / self.C
        det = (1 + self.R * self.g0) / (self.L * self.C)
        return {'trace': tr, 'det': det, 'unstable': (tr < 0 < det)}

    def linearized_solution(self, init_current_offset=1e-3, init_voltage_offset=0.0):
        jacobian_matrix = np.array([
            [-self.R / self.L, -1.0 / self.L],
            [1.0 / self.C, -self.g0 / self.C]
        ])

        eigenvalues, eigenvectors = np.linalg.eig(jacobian_matrix)

        current_offset = init_current_offset
        voltage_offset = init_voltage_offset

        try:
            coefficients = np.linalg.solve(eigenvectors, [current_offset, voltage_offset])
        except np.linalg.LinAlgError:
            print("Ошибка: собственные векторы вырождены")
            return None, None, False

        def analytical_solution(t):
            t = np.asarray(t)
            current_deviation_complex = np.zeros_like(t, dtype=complex)
            voltage_deviation_complex = np.zeros_like(t, dtype=complex)

            for idx, eigenvalue in enumerate(eigenvalues):
                exp_factor = np.exp(eigenvalue * t)
                current_deviation_complex += coefficients[idx] * exp_factor * eigenvectors[0, idx]
                voltage_deviation_complex += coefficients[idx] * exp_factor * eigenvectors[1, idx]

            current_deviation = np.real(current_deviation_complex)
            voltage_deviation = np.real(voltage_deviation_complex)

            current_analytical = self.I0 + current_deviation
            voltage_analytical = self.U0 + voltage_deviation
            return current_analytical, voltage_analytical

        return analytical_solution, eigenvalues, True


class OscillatorSimulator:
    def __init__(self, oscillator, t_span=(0, 0.01), t_eval_points=5000):
        self.osc = oscillator
        self.t_span = t_span
        self.t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)

    def run(self, I_init_offset=0.001):
        y0 = [self.osc.I0 + I_init_offset, self.osc.U0]
        sol = solve_ivp(self.osc.system, self.t_span, y0,
                        t_eval=self.t_eval, method='RK45')
        self.t = sol.t
        self.I = sol.y[0]
        self.U = sol.y[1]
        return self

    def compare_with_linear_analysis(self, init_current_offset=1e-3,
                                     init_voltage_offset=0.0,
                                     short_time_span=(0, 0.0002)):
        lin_solution, eigvals, ok = self.osc.linearized_solution(
            init_current_offset, init_voltage_offset
        )
        if not ok:
            print("Не удалось построить аналитическое решение")
            return
        sim_short = OscillatorSimulator(self.osc, t_span=short_time_span, t_eval_points=2000)
        sim_short.run(I_init_offset=init_current_offset)
        t_num = sim_short.t
        I_num = sim_short.I
        U_num = sim_short.U
        I_anal, U_anal = lin_solution(t_num)

        xi_num = I_num - self.osc.I0
        eta_num = U_num - self.osc.U0
        xi_anal = I_anal - self.osc.I0
        eta_anal = U_anal - self.osc.U0

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(t_num * 1000, xi_num * 1000, 'b-', label='Численное')
        axes[0].plot(t_num * 1000, xi_anal * 1000, 'r--', label='Аналитическое (линеар.)')
        axes[0].set_xlabel('Время, мс')
        axes[0].set_ylabel('ΔI, мА')
        axes[0].set_title('Отклонение тока')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(t_num * 1000, eta_num, 'b-', label='Численное')
        axes[1].plot(t_num * 1000, eta_anal, 'r--', label='Аналитическое (линеар.)')
        axes[1].set_xlabel('Время, мс')
        axes[1].set_ylabel('ΔU, В')
        axes[1].set_title('Отклонение напряжения')
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle(f'Сравнение линеаризованного и численного решений\n')
        plt.tight_layout()
        plt.show()
        return


def compare_with_analytics(diode, R, L, C, E, t_span=(0, 0.01)):
    osc = TunnelDiodeOscillator(diode, R, L, C, E)
    sim = OscillatorSimulator(osc, t_span=t_span)
    sim.run(I_init_offset=0.001)

    print(f"Параметры: R={R} Ом, L={L * 1000:.1f} мГн, C={C * 1e6:.1f} мкФ, E={E:.3f} В")
    print(f"Стационарная точка: U0 = {osc.U0:.4f} В, I0 = {osc.I0 * 1000:.3f} мА")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Автогенератор: R={R} Ом, L={L * 1000:.1f} мГн, C={C * 1e6:.1f} мкФ")

    # Ток катушки
    axes[0].plot(sim.t * 1000, sim.I * 1000, 'b-', lw=1)
    axes[0].set_xlabel('Время, мс')
    axes[0].set_ylabel('Ток I, мА')
    axes[0].grid(True)
    axes[0].set_title('Ток катушки')

    # Напряжение на диоде
    axes[1].plot(sim.t * 1000, sim.U, 'r-', lw=1)
    axes[1].set_xlabel('Время, мс')
    axes[1].set_ylabel('Напряжение U, В')
    axes[1].grid(True)
    axes[1].set_title('Напряжение на диоде')

    # Фазовый портрет
    axes[2].plot(sim.I * 1000, sim.U, 'g-', lw=1)
    axes[2].set_xlabel('Ток I, мА')
    axes[2].set_ylabel('Напряжение U, В')
    axes[2].grid(True)
    axes[2].set_title('Фазовый портрет')

    plt.tight_layout()
    plt.show()
    return sim, osc


if __name__ == "__main__":
    diode = TunnelDiode(a=4e-3, b=-16e-3, c=17e-3)

    R_opt = 10.0
    L_opt = 10e-3
    C_opt = 1e-6
    U_target = 1.333
    I_target = diode.current(U_target)
    E_opt = U_target + R_opt * I_target

    R_nogen = 200.0
    E_nogen = U_target + R_nogen * I_target

    R_border = 30.0
    E_border = U_target + R_border * I_target

    compare_with_analytics(diode, R_opt, L_opt, C_opt, E_opt, t_span=(0, 0.01))
    compare_with_analytics(diode, R_nogen, L_opt, C_opt, E_nogen, t_span=(0, 0.005))
    compare_with_analytics(diode, R_border, L_opt, C_opt, E_border, t_span=(0, 0.01))

    osc_test = TunnelDiodeOscillator(diode, R_opt, L_opt, C_opt, E_opt)
    sim_test = OscillatorSimulator(osc_test, t_span=(0, 0.01))
    sim_test.compare_with_linear_analysis(init_current_offset=0.001,
                                          short_time_span=(0, 0.003))
