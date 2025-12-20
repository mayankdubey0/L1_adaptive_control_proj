import numpy as np
import pybullet as p
import pybullet_data
import time
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


class L1AdaptiveController:

    def __init__(self, A, B, K, dt, gamma=200.0, bandwidth=10.0):
        self.A = A
        self.B = B
        self.dt = dt
        self.gamma = gamma
        self.bandwidth = bandwidth

        self.n = A.shape[0]
        self.m = B.shape[1]

        self.Am = A - B @ K

        Qm = np.eye(self.n)
        self.P = solve_continuous_are(
            self.Am,
            self.B,
            Qm,
            np.eye(self.m)
        )

        self.x_hat = np.zeros(self.n)
        self.theta_hat = np.zeros(self.m)
        self.u_ad = np.zeros(self.m)

        self.theta_max = 10.0

    def project(self, theta):
        norm = np.linalg.norm(theta)
        if norm > self.theta_max:
            return theta * (self.theta_max / norm)
        return theta

    def update(self, x, u_baseline):
        e = x - self.x_hat

        theta_dot = -self.gamma * (self.B.T @ self.P @ e)
        self.theta_hat = self.project(self.theta_hat + theta_dot * self.dt)

        xhat_dot = self.Am @ self.x_hat + self.B @ (u_baseline + self.theta_hat)
        self.x_hat += xhat_dot * self.dt

        return np.linalg.norm(e)

    def get_control(self):
       
        u_desired = -self.theta_hat

        alpha = self.bandwidth * self.dt / (1.0 + self.bandwidth * self.dt)
        self.u_ad = (1 - alpha) * self.u_ad + alpha * u_desired

        return self.u_ad

    def reset(self):
        self.x_hat[:] = 0.0
        self.theta_hat[:] = 0.0
        self.u_ad[:] = 0.0



class QuadcopterLQR:
    def __init__(self, wind_strength=2.0, use_l1=True):
        self.m = 0.5
        self.g = 9.81
        self.L = 0.25
        self.I = np.diag([0.01, 0.01, 0.02])
        self.wind_strength = wind_strength
        self.use_l1 = use_l1
        
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.drone_id = self.create_drone()
        
        Q = np.diag([10, 10, 10, 1, 1, 1, 100, 100, 50, 1, 1, 1])
        R = np.diag([0.1, 0.5, 0.5, 0.5])
        
        self.A = self.get_A_matrix()
        self.B = self.get_B_matrix()
        
    
        P = solve_continuous_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R) @ self.B.T @ P
            
       
        self.dt = 1./240.
        self.l1_controller = L1AdaptiveController(
            A=self.A,
            B=self.B,
            K=self.K,
            dt=self.dt,
            gamma=200.0,
            bandwidth=10.0
        )
        print(" L1 Adaptive Controller initialized")

        
        self.time_history = []
        self.position_error_history = []
        self.velocity_error_history = []
        self.attitude_error_history = []
        self.total_error_history = []
        self.adaptive_signal_history = []
        self.prediction_error_history = []
        self.theta_hat_history = []
    
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
        u_baseline = u_hover + u_feedback

        if self.use_l1:
            prediction_error = self.l1_controller.update(state, u_baseline)
            u_adaptive = self.l1_controller.get_control()
            theta_hat_norm = np.linalg.norm(self.l1_controller.theta_hat)
            u = u_baseline + u_adaptive
        else:
            prediction_error = 0.0
            u_adaptive = np.zeros(4)
            theta_hat_norm = 0.0
            u = u_baseline

        u[0] = np.clip(u[0], 0, 2.5 * self.m * self.g)
        u[1:4] = np.clip(u[1:4], -1.0, 1.0)

        return u, u_adaptive, prediction_error, theta_hat_norm

    
    def apply_control(self, u):
        thrust, tau_roll, tau_pitch, tau_yaw = u
        
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        thrust_world = rotation_matrix @ np.array([0, 0, thrust])
        
        p.applyExternalForce(self.drone_id, -1, thrust_world, pos, p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1, [tau_roll, tau_pitch, tau_yaw], p.WORLD_FRAME)
    
    def apply_wind(self, time_step):
        wind_direction = np.pi/4
        wind_base = np.array([
            self.wind_strength * np.sin(wind_direction),
            self.wind_strength * np.cos(wind_direction),
            0
        ])
        
        noise = np.random.randn(3) * 0.3 * self.wind_strength
        wind_force = wind_base + noise
        
        pos = p.getBasePositionAndOrientation(self.drone_id)[0]
        p.applyExternalForce(self.drone_id, -1, wind_force, pos, p.WORLD_FRAME)
    
    def track_error(self, state, target, current_time, u_adaptive, pred_error, theta_norm):
        error = state - target
        
        # Position error (x, y, z)
        pos_error = np.linalg.norm(error[0:3])
        
        # Velocity error
        vel_error = np.linalg.norm(error[3:6])
        
        # Attitude error (roll, pitch, yaw)
        att_error = np.linalg.norm(error[6:9])
        
        # Total state error
        total_error = np.linalg.norm(error)
        
        # Adaptive signal
        adaptive_magnitude = np.linalg.norm(u_adaptive)
        
        self.time_history.append(current_time)
        self.position_error_history.append(pos_error)
        self.velocity_error_history.append(vel_error)
        self.attitude_error_history.append(att_error)
        self.total_error_history.append(total_error)
        self.adaptive_signal_history.append(adaptive_magnitude)
        self.prediction_error_history.append(pred_error)
        self.theta_hat_history.append(theta_norm)
    
    def plot_errors(self):
        if self.use_l1:
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        controller_name = "LQR + L1 Adaptive" if self.use_l1 else "LQR Only"
        fig.suptitle(f'{controller_name} Controller - Tracking Error (No Wind, No Payload)', 
                     fontsize=14, fontweight='bold')
        
        if self.use_l1:
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
            
            # Adaptive signal (filtered)
            axes[2, 0].plot(self.time_history, self.adaptive_signal_history, 'm-', linewidth=2, label='Filtered output')
            axes[2, 0].plot(self.time_history, self.theta_hat_history, 'c--', linewidth=1, alpha=0.7, label='Theta estimate')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_ylabel('Signal Magnitude')
            axes[2, 0].set_title('L1 Adaptive Signals')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Prediction error
            axes[2, 1].plot(self.time_history, self.prediction_error_history, 'orange', linewidth=2)
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Prediction Error')
            axes[2, 1].set_title('L1 State Prediction Error')
            axes[2, 1].grid(True, alpha=0.3)
            
        else:
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
        
        # Print statistics
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Mean Position Error:  {np.mean(self.position_error_history):.4f} m")
        print(f"Std Position Error:   {np.std(self.position_error_history):.4f} m")
        print(f"Max Position Error:   {np.max(self.position_error_history):.4f} m")
        print(f"Final Position Error: {self.position_error_history[-1]:.4f} m")
        if self.use_l1:
            print(f"\nMean Adaptive Signal: {np.mean(self.adaptive_signal_history):.4f}")
            print(f"Max Adaptive Signal:  {np.max(self.adaptive_signal_history):.4f}")
            print(f"Mean Pred Error:      {np.mean(self.prediction_error_history):.4f}")
            
            # Compute settling time (when error stays below threshold)
            threshold = 0.1  # 10cm
            settled_indices = [i for i, err in enumerate(self.position_error_history) if err < threshold]
            if settled_indices and settled_indices[0] < len(self.time_history):
                settling_time = self.time_history[settled_indices[0]]
                print(f"Settling time (<10cm): {settling_time:.2f} s")
        print("="*60)
        
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
        
        steps = int(duration / self.dt)
        
        controller_name = "LQR + L1 Adaptive" if self.use_l1 else "LQR Only"
        print(f"\nStarting simulation with {controller_name} controller...")
        print(f"Target: {target_pos}")
        print(f"Wind strength: {self.wind_strength} N")
        
        for step in range(steps):
            current_time = step * self.dt
            state = self.get_state()
            
            u, u_adaptive, pred_error, theta_norm = self.compute_control(state, target)
            

            self.track_error(state, target, current_time, u_adaptive, pred_error, theta_norm)
            
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
                if self.use_l1:
                    print(f"t={current_time:.1f}s | Pos: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] | "
                          f"Error: {pos_error:.3f}m | u_ad: {np.linalg.norm(u_adaptive):.3f} | θ: {theta_norm:.3f}")
                else:
                    print(f"t={current_time:.1f}s | Pos: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] | "
                          f"Error: {pos_error:.3f}m")
            
            time.sleep(self.dt)
        
        final_state = self.get_state()
        final_error = np.linalg.norm(final_state[0:3] - target[0:3])
        mean_error = np.mean(self.position_error_history)
        
        print(f"\nSimulation complete!")
        print(f"Final position error: {final_error:.4f} m")
        print(f"Mean position error: {mean_error:.4f} m")
        
        # Plot the errors
        print("\nGenerating error plots...")
        self.plot_errors()
        
        print("\nPress Ctrl+C to exit...")
        
        try:
            while True:
                p.stepSimulation()
                time.sleep(self.dt)
        except KeyboardInterrupt:
            pass
    
    def close(self):
        p.disconnect()


if __name__ == "__main__":
    USE_L1_ADAPTIVE = False
    
    controller = QuadcopterLQR(wind_strength=0.0, use_l1=USE_L1_ADAPTIVE)
    
    p.resetBasePositionAndOrientation(controller.drone_id, 
                                     [2, 2, 1],
                                     [0, 0, 0, 1])
    
    try:
        controller.run_simulation(target_pos=[0, 0, 2.5], duration=15)
    finally:
        controller.close()