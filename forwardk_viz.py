import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Forward kinematics is simpler considered to IK, and is used in IK to an extent as well
Using a 3 joint 2 link setup as in the IK system, we define the joint-space as the angles of the joints.
q_matrix = [[theta_1],[theta_2],[theta_3]] and position as position = [[x],[y],[z]]

The mapping f(q) gives us the end effector position as position = f(q)
We can now calculate the end-effector position. We will first see the position with no z-axis rotation, so 
the x-z plane only. 
r = l1*cos(theta_2) + l2*cos(theta_2+theta_3) - distance from z-axis 
z = l1*sin(theta_2) + l2*sin(theta_2+theta_3) - height, distance along z-axis
Now when we rotate about the z-axis, height isnt going to change, but the position of the end effector will change as 
x = r*cos(theta_1), y = r*sin(theta_1), z = z 

So for any joint configuration value, we get the end effector position:
end_effector_new = f(q_new), where f is simply the function described by the equations above'''

l1 = l2 = 1.0 # link lengths

def get_angles(prompt):
    theta_1, theta_2, theta_3 = map(float, input(prompt).split())
    return np.array([[theta_1],[theta_2],[theta_3]])

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
text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

plt.show(block=False)


while True:
    user_input = input("\nEnter joint angles in degrees as theta1 theta2 theta3, or q to quit: ")

    if user_input.lower() == "q":
        print("Exiting.")
        break

    theta_1_deg, theta_2_deg, theta_3_deg = map(float, user_input.split())

    theta_1 = np.deg2rad(theta_1_deg)
    theta_2 = np.deg2rad(theta_2_deg)
    theta_3 = np.deg2rad(theta_3_deg)

    ee_position = joint_to_cartesian(theta_1, theta_2, theta_3)
    points = get_joint_positions(theta_1, theta_2, theta_3)

    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])

    text.set_text(
        f"Angles: ({theta_1_deg:.1f}, {theta_2_deg:.1f}, {theta_3_deg:.1f}) deg\n"
        f"EE: x={ee_position[0,0]:.3f}, y={ee_position[1,0]:.3f}, z={ee_position[2,0]:.3f}"
    )

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)