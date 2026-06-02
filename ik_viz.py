import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Forward Kinematics allows us to find the end-effector pose given the joint configurations.
Inverse Kinematics allows us to find the joint configurations required to reach a particular end-effector pose.

q_matrix = [[theta_1] , [theta_2] , [theta_3]]
end_effector = [[x],[y],[z]]

The Jacobian is defined as J(q) belonging to R^3x3 in this case
We assume the following structure regarding a 3 Degree of Freedom case:
Joint_1 - Yaw about the vertical Z-axis
Joint_2 - Shoulder pitch with length l1
Joint_3 - elbow pitch with length l2

We can now calculate the end-effector position. We will first see the position with no z-axis rotation, so 
the x-z plane only. 
r = l1*cos(theta_2) + l2*cos(theta_2+theta_3) - distance from z-axis 
z = l1*sin(theta_2) + l2*sin(theta_2+theta_3) - height, distance along z-axis
Now when we rotate about the z-axis, height isnt going to change, but the position of the end effector will change as 
x = r*cos(theta_1), y = r*sin(theta_1), z = z

The jacobian is the matrix for first order differentials of multi-variables. 
For a single variable, f(x + delta_x) = f(x) + f'(x)*delta_x
for multi-variable, f(q+delta_q) = f(q) + J(q)*delta_q. This is useful as we can now solve for the joint configurations for a desired pose.
Current pose = q_c Desired Pose = q_p. We know that f(qc + error) = f(q_p), so f(q_p) = f(q_c) + J(q)*delta_q, delta_q = J^-1(q)*error

Forward kinematics tells us where our end-effector will be given the joint configuration, simply put:
end_effector = f(q), when we move the joint configuration by some delta, we can approximate the position to be:
end_effector_new = f(q+delta_q) = f(q) + J(q)*delta_q
Inverse kinematics works in the opposite way. Suppose we have the initial end effector pose and the final end effector pose, 
we want to find the joint configurations for it to reach there. 
let desired pose be x_d, and current be x. we know x = f(q), error between desired and current = x_d - x
x_d = f(q+delta_q) = f(q) + J(q)*delta_q. We want delta_q. error = x_d - x or x_d = x + error, so error = J(q)*delta_q
which means that delta_q = J(q)^-1*error
This process is iterative till we reach error = 0 or very low. 
We will now try to build our own IK solver for the 3-DoF case
'''

l1 = l2 = 1.0 # link lengths

def get_position(prompt):
    x, y, z = map(float, input(prompt).split())
    return np.array([[x],[y],[z]])

theta_1 = 0.0
theta_2 = np.pi / 2
theta_3 = np.pi / 2

def joint_to_cartesian(theta_1, theta_2, theta_3):
    r = l1*np.cos(theta_2) + l2*np.cos(theta_2 + theta_3)
    z = l1*np.sin(theta_2) + l2*np.sin(theta_2 + theta_3)
    x = r*np.cos(theta_1)
    y = r*np.sin(theta_1)
    return np.array([[x], [y], [z]])


def get_joint_positions(theta_1, theta_2, theta_3):
    base = np.array([0, 0, 0])

    # Shoulder is at base in this simplified model
    shoulder = np.array([0, 0, 0])

    # Elbow position
    r1 = l1 * np.cos(theta_2)
    z1 = l1 * np.sin(theta_2)

    elbow = np.array([
        r1 * np.cos(theta_1),
        r1 * np.sin(theta_1),
        z1
    ])

    # End-effector position
    ee = joint_to_cartesian(theta_1, theta_2, theta_3).flatten()
    return np.array([base, elbow, ee])


def jacobian(theta_1, theta_2, theta_3, delta=1e-6):
    f_base = joint_to_cartesian(theta_1, theta_2, theta_3)

    f_delta1 = joint_to_cartesian(theta_1 + delta, theta_2, theta_3)
    f_delta2 = joint_to_cartesian(theta_1, theta_2 + delta, theta_3)
    f_delta3 = joint_to_cartesian(theta_1, theta_2, theta_3 + delta)

    column_1 = (f_delta1 - f_base) / delta
    column_2 = (f_delta2 - f_base) / delta
    column_3 = (f_delta3 - f_base) / delta

    return np.hstack([column_1, column_2, column_3])


def inverse_kinematics(theta_1, theta_2, theta_3, desired_position):
    history = []
    alpha = 0.2

    for i in range(200):
        current_position = joint_to_cartesian(theta_1, theta_2, theta_3)
        error = desired_position - current_position
        error_mag = np.linalg.norm(error)

        history.append((theta_1, theta_2, theta_3, error_mag))

        if error_mag < 0.01:
            break

        J = jacobian(theta_1, theta_2, theta_3)
        delta_theta = np.linalg.pinv(J) @ error

        theta_1 += alpha * delta_theta[0, 0]
        theta_2 += alpha * delta_theta[1, 0]
        theta_3 += alpha * delta_theta[2, 0]

    return history, theta_1, theta_2, theta_3

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 2.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

line, = ax.plot([], [], [], "o-", linewidth=3)
target_plot = ax.scatter([], [], [], s=80, marker="x")
text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

plt.show(block=False)


def animate_live(history, desired_position):
    global target_plot

    target_plot.remove()

    target_plot = ax.scatter(
        desired_position[0, 0],
        desired_position[1, 0],
        desired_position[2, 0],
        s=80,
        marker="x"
    )

    for frame in range(len(history)):
        theta_1, theta_2, theta_3, error_mag = history[frame]

        points = get_joint_positions(theta_1, theta_2, theta_3)

        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

        text.set_text(f"Step: {frame}, error: {error_mag:.4f}")

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.pause(0.05)

while True:
    user_input = input("\nEnter goal position as x y z, or q to quit: ")

    if user_input.lower() == "q":
        print("Exiting.")
        break

    x, y, z = map(float, user_input.split())

    desired_position = np.array([
        [x],
        [y],
        [z]
    ])

    history, theta_1, theta_2, theta_3 = inverse_kinematics(
        theta_1,
        theta_2,
        theta_3,
        desired_position
    )

    print("Final angles:")
    print(theta_1, theta_2, theta_3)

    print("Final angles in degrees:")
    print((180 / np.pi) * np.array([theta_1, theta_2, theta_3]))

    print("Final error:")
    print(history[-1][3])

    animate_live(history, desired_position)