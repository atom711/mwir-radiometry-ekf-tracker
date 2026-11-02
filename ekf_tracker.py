import numpy as np
import matplotlib.pyplot as plt

class TargetEKF:
    def __init__(self, initial_state, initial_covariance, process_noise_q=1.0, measurement_noise_r=10.0):
        """
        3D Extended Kalman Filter for OPIR Kinematic Tracking.
        State vector: [x, y, z, vx, vy, vz]^T
        """
        self.x = np.array(initial_state, dtype=float)  # 6x1 state vector
        self.P = np.array(initial_covariance, dtype=float)  # 6x6 error covariance matrix
        
        # Process noise covariance (Q)
        self.Q = np.eye(6) * process_noise_q
        
        # Measurement noise covariance (R) - e.g., range and angular tracking uncertainty
        self.R = np.eye(3) * measurement_noise_r

    def predict(self, dt):
        """
        Predict step: Propagates state and covariance forward by time dt using constant velocity model.
        State transition matrix F:
        [ 1  0  0  dt  0  0 ]
        [ 0  1  0   0 dt  0 ]
        [ 0  0  1   0  0 dt ]
        [ 0  0  0   1  0  0 ]
        [ 0  0  0   0  1  0 ]
        [ 0  0  0   0  0  1 ]
        """
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Predict state
        self.x = F @ self.x
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_measurement):
        """
        Update step: Incorporates direct 3D position measurements (or converted LOS + range)
        Measurement matrix H (mapping 6 state variables to 3 measured positions):
        [ 1 0 0 0 0 0 ]
        [ 0 1 0 0 0 0 ]
        [ 0 0 1 0 0 0 ]
        """
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        
        # Innovation (residual) y = z - H * x
        y = z_measurement - (H @ self.x)
        
        # Innovation covariance S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain K = P * H^T * S^(-1)
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.x = self.x + (K @ y)
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(6)
        self.P = (I - (K @ H)) @ self.P

def simulate_tracking_run():
    """Simulates a target trajectory and runs the EKF tracker, generating diagnostic plots."""
    dt = 1.0  # 1 second time step
    steps = 50
    
    # True initial state: Position [1000, 2000, 35000] km/m, Velocity [50, -20, 200] m/s
    true_state = np.array([1000.0, 2000.0, 350000.0, 50.0, -20.0, 200.0])
    
    # Initialize EKF with slight offset and uncertainty
    ekf_initial_state = true_state + np.array([50.0, -50.0, 200.0, 5.0, 5.0, -10.0])
    initial_cov = np.diag([100.0, 100.0, 500.0, 10.0, 10.0, 50.0])
    
    tracker = TargetEKF(ekf_initial_state, initial_cov)
    
    true_trajectory = []
    estimated_trajectory = []
    
    np.random.seed(42)
    
    for _ in range(steps):
        # Update true position (constant velocity motion)
        true_state[0] += true_state[3] * dt
        true_state[1] += true_state[4] * dt
        true_state[2] += true_state[5] * dt
        true_trajectory.append(true_state[:3].copy())
        
        # Simulate noisy measurement (adding Gaussian sensor noise)
        measurement_noise = np.random.normal(0.0, 25.0, size=3)
        z = true_state[:3] + measurement_noise
        
        # EKF Predict & Update Cycle
        tracker.predict(dt)
        tracker.update(z)
        estimated_trajectory.append(tracker.x[:3].copy())
        
    return np.array(true_trajectory), np.array(estimated_trajectory)

def plot_results(true_traj, est_traj):
    """Generates professional, clean 3D trajectory and error plots for portfolio documentation."""
    # Convert meters to kilometers for cleaner visualization
    true_traj_km = true_traj / 1000.0
    est_traj_km = est_traj / 1000.0
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(true_traj_km[:, 0], true_traj_km[:, 1], true_traj_km[:, 2], label='True Trajectory', color='blue', lw=2)
    ax1.plot(est_traj_km[:, 0], est_traj_km[:, 1], est_traj_km[:, 2], label='EKF Estimate', color='orange', linestyle='--', lw=2)
    
    ax1.set_title("3D Target Kinematic Tracking (GEO OPIR)", fontsize=11, pad=10)
    ax1.set_xlabel("X Position (km)", labelpad=8)
    ax1.set_ylabel("Y Position (km)", labelpad=8)
    ax1.set_zlabel("Z Position (km)", labelpad=8)
    ax1.legend(loc='upper left')
    
    # Position Error Magnitude (in meters for precise tracking error view)
    ax2 = fig.add_subplot(122)
    position_errors = np.linalg.norm(true_traj - est_traj, axis=1)
    ax2.plot(position_errors, color='red', lw=1.5)
    ax2.set_title("EKF 3D Position Error Magnitude", fontsize=11, pad=10)
    ax2.set_xlabel("Time Step (s)")
    ax2.set_ylabel("Error Euclidean Distance (m)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("ekf_tracking_performance.png", dpi=300)
    print("Successfully generated and saved updated tracking performance figure: ekf_tracking_performance.png")
    plt.show()

if __name__ == "__main__":
    print("--- Running 3D EKF Kinematic Tracking Simulation ---")
    true_t, est_t = simulate_tracking_run()
    plot_results(true_t, est_t)