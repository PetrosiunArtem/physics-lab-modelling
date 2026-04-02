import pyvista as pv
import numpy as np
from numpy import linalg
from scipy.integrate import ode
import functools

g = 9.8
EPSILON = 1e-3


def compute_gravity_proj(m: float) -> np.ndarray:
    basis_change_matrix = np.array([get_basis_1(), get_basis_2(), get_surface_normal()]).T 

    full_gravity = np.array([0, 0, -m * g])
    projected_gravity = (linalg.inv(basis_change_matrix) @ full_gravity)
    return projected_gravity[:2]


def compute_friction(v: np.ndarray, w: np.ndarray, m: float, r: float, I: float, 
                     friction_coefficient: float, gravity: np.ndarray) -> np.ndarray:
    v_contact = v - np.array([w[1], -w[0]]) * r

    max_friction = -np.dot(friction_coefficient * np.array([0, 0, -m * g]), get_surface_normal())

    stable_friction = gravity / (I / m / r ** 2 + 1)
    if linalg.norm(stable_friction) > max_friction:
        stable_friction = gravity / linalg.norm(gravity)

    if linalg.norm(v_contact) > EPSILON:
        return max_friction * v_contact / linalg.norm(v_contact)
    else:
        return stable_friction


def derivatives(state: np.ndarray, 
                m: float, r: float, I: float, friction_coefficient: float) -> np.ndarray:
    v = state[2:4]
    w = state[4:]

    gravity = compute_gravity_proj(m)
    friction = compute_friction(v, w, m, r, I, friction_coefficient, gravity)

    derivatives = np.array([
        *v,
        *(gravity - friction),
        -friction[1] * r / I,
        friction[0] * r / I,
    ])
    return derivatives
    

surface_angles = np.zeros(2)


def get_basis_1():
    return np.array([np.cos(surface_angles[0]), 0,  np.sin(surface_angles[0])])


def get_basis_2():
    return np.array([0, np.cos(surface_angles[1]), np.sin(surface_angles[1])])
    

def get_surface_normal():
    cross_product = np.cross(get_basis_1(), get_basis_2())
    return cross_product / linalg.norm(cross_product)


M = 1
R = 1
I = 2 / 5 * M * R ** 2
FRICTION_COEFFICIENT = 0.1

state = np.zeros(6)


plotter = pv.Plotter()

sphere = pv.Sphere(theta_resolution=200, phi_resolution=200, radius=R)

array = np.array([[[255, 255, 255], [0, 0, 0]], [[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

texture = pv.Texture(array)
sphere.texture_map_to_sphere(inplace=True)

plotter.add_mesh(sphere, texture=texture)


PLANE_SIZE = 20
plane = pv.Plane(center=(0, 0, 0), direction=get_surface_normal(), 
                 i_size=PLANE_SIZE, j_size=PLANE_SIZE)
plotter.add_mesh(plane)


movement_keys = [False] * 4

def movement_key_press(movement_key_index):
    movement_keys[movement_key_index] = True

MOVEMENT_KEYS = ['5', '8', '4', '6']

for i, key in enumerate(MOVEMENT_KEYS):
    plotter.add_key_event(key, functools.partial(movement_key_press, i))

def key_release(obj, _):
    key = obj.GetKeySym()
    if key in MOVEMENT_KEYS:
        movement_keys[MOVEMENT_KEYS.index(key)] = False


def update(*args):
    ANGLE_SPEED = 0.01

    global surface_angles
    old_angles = surface_angles.copy()

    if movement_keys[0]:
        surface_angles[0] -= ANGLE_SPEED
    if movement_keys[1]:
        surface_angles[0] += ANGLE_SPEED
    if movement_keys[2]:
        surface_angles[1] += ANGLE_SPEED
    if movement_keys[3]:
        surface_angles[1] -= ANGLE_SPEED

    surface_angles = surface_angles.clip(-np.pi / 4, np.pi / 4)

    plane.rotate_y(-np.degrees(surface_angles[0] - old_angles[0]), inplace=True)
    plane.rotate_x(np.degrees(surface_angles[1] - old_angles[1]), inplace=True)
    

    DT = 0.01

    global state
    state += derivatives(state, M, R, I, FRICTION_COEFFICIENT) * DT

    max_x, max_y = np.abs(PLANE_SIZE / 2 * np.cos(surface_angles))

    # if state[0] > max_x:
    #     state[0] = max_x
    #     state[2] = min(state[2], 0)
    # elif state[0] < -max_x:
    #     state[0] = -max_x
    #     state[2] = max(state[2], 0)

    # if state[1] > max_y:
    #     state[1] = max_y
    #     state[3] = min(state[3], 0)
    # elif state[1] < -max_y:
    #     state[1] = -max_y
        # state[3] = max(state[3], 0)

    if np.abs(state[0]) > max_x or np.abs(state[1]) > max_y:
        state = np.zeros_like(state)


    x, y, *_, w_x, w_y = state

    surface_normal = get_surface_normal()
    z = R - x * surface_normal[0] - y * surface_normal[1]

    sphere.translate(np.array([x, y, z]) - np.array(sphere.center), 
                     inplace=True)

    w = np.array([w_x, w_y, 0])
    w_norm = linalg.norm(w)
    if w_norm > 0:
        sphere.rotate_vector(w / w_norm, np.degrees(w_norm * DT), inplace=True)

    plotter.render()


plotter.iren.initialize()

plotter.iren.add_observer('KeyReleaseEvent', key_release)

plotter.iren.create_timer(10)
plotter.iren.add_observer('TimerEvent', update)

plotter.show()
