import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class MagneticTrap:

    def __init__(self, ring_radius=1.0, distance=2.0, current=100.0, mu0=1.0, num_phi=200):
        self.ring_radius = ring_radius
        self.distance = distance
        self.current = current
        self.mu0 = mu0
        self.num_phi = num_phi

        self.phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
        self.dphi = self.phi[1] - self.phi[0]
        self.cos_phi = np.cos(self.phi)
        self.sin_phi = np.sin(self.phi)

    def magnetic_field_single_loop(self, x, y, z, z_center):
        element_x = self.ring_radius * self.cos_phi
        element_y = self.ring_radius * self.sin_phi
        element_z = z_center

        Rx = x - element_x
        Ry = y - element_y
        Rz = z - element_z
        R_norm = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)

        dl_x = -self.ring_radius * self.sin_phi
        dl_y = self.ring_radius * self.cos_phi
        dl_z = 0.0

        cross_x = dl_y * Rz - dl_z * Ry
        cross_y = dl_z * Rx - dl_x * Rz
        cross_z = dl_x * Ry - dl_y * Rx

        integrand_x = cross_x / R_norm ** 3
        integrand_y = cross_y / R_norm ** 3
        integrand_z = cross_z / R_norm ** 3

        prefactor = self.mu0 * self.current / (4 * np.pi)
        Bx = prefactor * np.trapezoid(integrand_x, dx=self.dphi)
        By = prefactor * np.trapezoid(integrand_y, dx=self.dphi)
        Bz = prefactor * np.trapezoid(integrand_z, dx=self.dphi)

        return Bx, By, Bz

    def total_field(self, x, y, z):
        Bx1, By1, Bz1 = self.magnetic_field_single_loop(x, y, z, +self.distance)
        Bx2, By2, Bz2 = self.magnetic_field_single_loop(x, y, z, -self.distance)
        return Bx1 + Bx2, By1 + By2, Bz1 + Bz2


class Particle:

    def __init__(self, charge=1.0, mass=1.0, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)):
        self.charge = charge
        self.mass = mass
        self.x0, self.y0, self.z0 = position
        self.vx0, self.vy0, self.vz0 = velocity
        self.initial_state = [self.x0, self.y0, self.z0, self.vx0, self.vy0, self.vz0]
        self.trajectory = None
        self.time_points = None


class Simulation:
    def __init__(self, magnetic_trap, particle, max_time=200.0, num_points=5000):
        self.magnetic_trap = magnetic_trap
        self.particle = particle
        self.max_time = max_time
        self.time_eval = np.linspace(0, max_time, num_points)
        self.solution = None

    def equations_of_motion(self, t, state):
        x, y, z, vx, vy, vz = state
        Bx, By, Bz = self.magnetic_trap.total_field(x, y, z)
        ax = (self.particle.charge / self.particle.mass) * (vy * Bz - vz * By)
        ay = (self.particle.charge / self.particle.mass) * (vz * Bx - vx * Bz)
        az = (self.particle.charge / self.particle.mass) * (vx * By - vy * Bx)
        return [vx, vy, vz, ax, ay, az]

    def run(self):
        print("Интегрирование...")
        sol = solve_ivp(
            self.equations_of_motion,
            [0, self.max_time],
            self.particle.initial_state,
            t_eval=self.time_eval,
            method='RK45',
            rtol=1e-8, atol=1e-10
        )
        print("Готово.")
        self.solution = sol
        self.particle.trajectory = sol.y[:3]
        self.particle.time_points = sol.t


class Animator:

    def __init__(self, magnetic_trap, particle, frame_step=10, frame_interval=20):
        self.magnetic_trap = magnetic_trap
        self.particle = particle
        self.frame_step = frame_step
        self.frame_interval = frame_interval
        self.figure = None
        self.axes = None
        self.trajectory_line = None
        self.particle_point = None

    def setup_plot(self):
        self.figure = plt.figure(figsize=(10, 7))
        self.axes = self.figure.add_subplot(111, projection='3d')

        theta_ring = np.linspace(0, 2 * np.pi, 100)
        x_ring = self.magnetic_trap.ring_radius * np.cos(theta_ring)
        y_ring = self.magnetic_trap.ring_radius * np.sin(theta_ring)
        self.axes.plot(
            x_ring,
            y_ring,
            np.full_like(theta_ring, self.magnetic_trap.distance),
            color='red',
            linewidth=2,
            label='кольца'
        )
        self.axes.plot(
            x_ring,
            y_ring,
            np.full_like(theta_ring, -self.magnetic_trap.distance),
            color='red',
            linewidth=2
        )

        self.axes.set_xlim([-1.2 * self.magnetic_trap.ring_radius, 1.2 * self.magnetic_trap.ring_radius])
        self.axes.set_ylim([-1.2 * self.magnetic_trap.ring_radius, 1.2 * self.magnetic_trap.ring_radius])
        self.axes.set_zlim([-1.2 * self.magnetic_trap.distance, 1.2 * self.magnetic_trap.distance])

        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_zlabel('z')
        self.axes.set_title('Траектория частицы в магнитной пробке')
        self.axes.legend()

        self.trajectory_line, = self.axes.plot([], [], [], 'b-', linewidth=0.8, alpha=0.7)
        self.particle_point, = self.axes.plot([], [], [], 'ro', markersize=3)

    def init_animation(self):
        self.trajectory_line.set_data([], [])
        self.trajectory_line.set_3d_properties([])
        self.particle_point.set_data([], [])
        self.particle_point.set_3d_properties([])
        return self.trajectory_line, self.particle_point

    def update(self, frame):
        idx = frame * self.frame_step
        if idx >= self.particle.trajectory.shape[1]:
            idx = self.particle.trajectory.shape[1] - 1

        x_traj, y_traj, z_traj = self.particle.trajectory
        self.trajectory_line.set_data(x_traj[:idx + 1], y_traj[:idx + 1])
        self.trajectory_line.set_3d_properties(z_traj[:idx + 1])
        self.particle_point.set_data([x_traj[idx]], [y_traj[idx]])
        self.particle_point.set_3d_properties([z_traj[idx]])

        return self.trajectory_line, self.particle_point

    def animate(self):
        self.setup_plot()
        num_frames = len(self.particle.trajectory[0]) // self.frame_step
        frames = range(num_frames)

        ani = animation.FuncAnimation(
            self.figure,
            self.update,
            frames=frames,
            init_func=self.init_animation,
            interval=self.frame_interval,
            blit=True,
            repeat=True
        )
        plt.show()
        return ani


if __name__ == "__main__":
    magnetic_trap = MagneticTrap(ring_radius=1.0, distance=3.0, current=100.0, num_phi=150)

    particle = Particle(charge=1.0, mass=1.0, position=(0.2, 0.0, 0.0), velocity=(0.0, 0.098, 0.02))

    simulation = Simulation(magnetic_trap, particle, max_time=200.0, num_points=5000)
    simulation.run()

    animator = Animator(magnetic_trap, particle, frame_step=10, frame_interval=20)
    animator.animate()
