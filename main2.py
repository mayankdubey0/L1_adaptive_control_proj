import numpy as np
import pybullet as p
import pybullet_data
import time
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

class QuadcopterLQR:
    def __init__(self, wind_strength=2.0):
        self.m = 0.5
        self.g = 9.81
        self.L = 0.25
        self.I = np.diag([0.01, 0.01, 0.02])
        self.wind_strength = wind_strength
        
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.drone_id = self.create_drone()
        
        Q = np.diag([10, 10, 10, 1, 1, 1, 100, 100, 50, 1, 1, 1])
        R = np.diag([0.1, 0.5, 0.5, 0.5])
        
        A = self.get_A_matrix()
        B = self.get_B_matrix()
        
        try:
            P = solve_continuous_are(A, B, Q, R)
            self.K = np.linalg.inv(R) @ B.T @ P
            print("LQR Controller initialized")
            print(f"Gain matrix shape: {self.K.shape}")
        except Exception as e:
            print(f"Error computing LQR gains: {e}")
            self.K = np.zeros((4, 12))
        
        # Initialize error tracking
        self.time_history = []
        self.position_error_history = []
        self.velocity_error_history = []
        self.attitude_error_history = []
        self.total_error_history = []
    
    def create_drone(self):
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], 
                                       rgbaColor=[1, 0, 0, 1])
        
        rotor_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=0.02)
        rotor_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.02,
                                       rgbaColor=[0.2, 0.2, 0.2, 1])
        
        rotor_positions = [
            [0.25, 0.25, 0.0],
            [-0.25, 0.25, 0.0],
            [-0.25, -0.25, 0.0],
            [0.25, -0.25, 0.0]
        ]
        
        drone_id = p.createMultiBody(
            baseMass=self.m,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 2],
            linkMasses=[0.01] * 4,
            linkCollisionShapeIndices=[rotor_col] * 4,
            linkVisualShapeIndices=[rotor_vis] * 4,
            linkPositions=rotor_positions,
            linkOrientations=[[0, 0, 0, 1]] * 4,
            linkInertialFramePositions=[[0, 0, 0]] * 4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED] * 4,
            linkJointAxis=[[0, 0, 1]] * 4
        )
        
        return drone_id
    
    def get_A_matrix(self):
        A = np.zeros((12, 12))
        A[0:3, 3:6] = np.eye(3)
        A[3, 7] = self.g
        A[4, 6] = -self.g
        A[6:9, 9:12] = np.eye(3)
        return A
    
    def get_B_matrix(self):
        B = np.zeros((12, 4))
        B[5, 0] = 1.0 / self.m
        B[9, 1] = 1.0 / self.I[0, 0]
        B[10, 2] = 1.0 / self.I[1, 1]
        B[11, 3] = 1.0 / self.I[2, 2]
        return B
    
    def get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        euler = p.getEulerFromQuaternion(quat)
        
        return np.array([
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            euler[0], euler[1], euler[2],
            ang_vel[0], ang_vel[1], ang_vel[2]
        ])
    
    def compute_control(self, state, target):
        error = state - target
        u_hover = np.array([self.m * self.g, 0, 0, 0])
        u_feedback = -self.K @ error
        u = u_hover + u_feedback
        
        u[0] = np.clip(u[0], 0, 2 * self.m * self.g)
        u[1:4] = np.clip(u[1:4], -0.5, 0.5)
        
        return u
    
    def apply_control(self, u):
        thrust, tau_roll, tau_pitch, tau_yaw = u
        
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        thrust_world = rotation_matrix @ np.array([0, 0, thrust])
        
        p.applyExternalForce(self.drone_id, -1, thrust_world, pos, p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1, [tau_roll, tau_pitch, tau_yaw], p.WORLD_FRAME)
    
    def apply_wind(self, time_step):
        wind_base = np.array([
            self.wind_strength * np.sin(0.5 * time_step),
            self.wind_strength * np.cos(0.3 * time_step),
            0.5 * self.wind_strength * np.sin(0.2 * time_step)
        ])
        
        noise = np.random.randn(3) * 0.3 * self.wind_strength
        wind_force = wind_base + noise
        
        pos = p.getBasePositionAndOrientation(self.drone_id)[0]
        p.applyExternalForce(self.drone_id, -1, wind_force, pos, p.WORLD_FRAME)
    
    def track_error(self, state, target, current_time):
        """Track different components of error"""
        error = state - target
        
        # Position error (x, y, z)
        pos_error = np.linalg.norm(error[0:3])
        
        # Velocity error
        vel_error = np.linalg.norm(error[3:6])
        
        # Attitude error (roll, pitch, yaw)
        att_error = np.linalg.norm(error[6:9])
        
        # Total state error
        total_error = np.linalg.norm(error)
        
        self.time_history.append(current_time)
        self.position_error_history.append(pos_error)
        self.velocity_error_history.append(vel_error)
        self.attitude_error_history.append(att_error)
        self.total_error_history.append(total_error)
    
    def plot_errors(self):
        """Plot the tracking errors"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('LQR Controller Tracking Error with Wind Disturbance', fontsize=14, fontweight='bold')
        
        # Position error
        axes[0, 0].plot(self.time_history, self.position_error_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position Error (m)')
        axes[0, 0].set_title('Position Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity error
        axes[0, 1].plot(self.time_history, self.velocity_error_history, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity Error (m/s)')
        axes[0, 1].set_title('Velocity Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Attitude error
        axes[1, 0].plot(self.time_history, self.attitude_error_history, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Attitude Error (rad)')
        axes[1, 0].set_title('Attitude Error (Roll, Pitch, Yaw)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total error
        axes[1, 1].plot(self.time_history, self.total_error_history, 'k-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Total State Error')
        axes[1, 1].set_title('Total State Error (L2 Norm)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_simulation(self, target_pos, duration=15):
        target = np.array([
            target_pos[0], target_pos[1], target_pos[2],
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ])
        
        target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, 
                                           rgbaColor=[0, 1, 0, 0.5])
        target_id = p.createMultiBody(baseVisualShapeIndex=target_visual,
                                     basePosition=target_pos)
        
        trail_points = []
        
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=45, 
                                     cameraPitch=-30, cameraTargetPosition=target_pos)
        
        dt = 1./240.
        steps = int(duration / dt)
        
        print("\nStarting simulation...")
        print(f"Target: {target_pos}")
        
        for step in range(steps):
            current_time = step * dt
            state = self.get_state()
            
            # Track error every step
            self.track_error(state, target, current_time)
            
            u = self.compute_control(state, target)
            self.apply_control(u)
            self.apply_wind(current_time)
            p.stepSimulation()
            
            if step % 10 == 0:
                trail_points.append(state[0:3])
                if len(trail_points) > 1:
                    p.addUserDebugLine(trail_points[-2], trail_points[-1],
                                      [0, 0, 1], 2, lifeTime=0)
            
            if step % 240 == 0:
                pos_error = np.linalg.norm(state[0:3] - target[0:3])
                print(f"t={current_time:.1f}s | Position: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] | Error: {pos_error:.3f}m")
            
            time.sleep(dt)
        
        final_state = self.get_state()
        final_error = np.linalg.norm(final_state[0:3] - target[0:3])
        print(f"\nSimulation complete!")
        print(f"Final position error: {final_error:.4f} m")
        
        # Plot the errors
        print("\nGenerating error plots...")
        self.plot_errors()
        
        print("Press Ctrl+C to exit...")
        
        try:
            while True:
                p.stepSimulation()
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
    
    def close(self):
        p.disconnect()


if __name__ == "__main__":
    controller = QuadcopterLQR(wind_strength=2.0)
    
    p.resetBasePositionAndOrientation(controller.drone_id, 
                                     [2, 2, 1],
                                     [0, 0, 0, 1])
    
    try:
        controller.run_simulation(target_pos=[0, 0, 2.5], duration=15)
    finally:
        controller.close()