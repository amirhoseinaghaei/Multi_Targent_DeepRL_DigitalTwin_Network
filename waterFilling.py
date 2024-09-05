import numpy as np

def calculate_interference(P, h, selected_channels):
    M, N = h.shape  # M systems, N channel conditions per system
    I = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            interference_sum = 0
            for m in range(M):
                if m != i:
                    interference_sum += P[m][selected_channels[m]] * h[m][j]
            I[i][j] = interference_sum
    return I

def water_filling_with_dynamic_interference(h, N0, P_total, num_iterations=100):
    M, N = h.shape  # M systems, N channel conditions per system
    P = np.zeros((M, N))  # Power allocation
    lambda_opt = np.zeros(M)
    selected_channels = np.zeros(M, dtype=int)  # To track which channel is selected

    for _ in range(num_iterations):
        # Determine channel selection for each system
        for i in range(M):
            # Calculate interference for each channel option
            interference = np.zeros(N)
            for j in range(N):
                selected_channels[i] = j
                I = calculate_interference(P, h, selected_channels)
                interference[j] = I[i][j]

            # Choose the channel that minimizes the interference
            selected_channel = np.argmin(interference)
            selected_channels[i] = selected_channel

        # Calculate interference based on selected channels
        I = calculate_interference(P, h, selected_channels)
        
        for i in range(M):
            lambda_min, lambda_max = 0, P_total[i] + (N0 + np.max(I[i])) / np.min(h[i])
            
            def power_allocation(lambda_val):
                return np.maximum(lambda_val - (N0 + I[i][selected_channels[i]]) / h[i][selected_channels[i]], 0)
            
            def total_power(lambda_val):
                return np.sum(power_allocation(lambda_val))
            
            # Binary search for water level lambda
            for _ in range(100):
                lambda_mid = (lambda_min + lambda_max) / 2
                if total_power(lambda_mid) > P_total[i]:
                    lambda_max = lambda_mid
                else:
                    lambda_min = lambda_mid
            
            lambda_opt[i] = (lambda_min + lambda_max) / 2
            P[i] = power_allocation(lambda_opt[i])
    
    return P, lambda_opt

# Example channel gains and parameters for 5 systems with 2 channel gain values each
h = np.array([
    [1.0, 0.8],
    [0.9, 0.7],
    [0.8, 0.6],
    [0.7, 0.5],
    [0.6, 0.4]
])
N0 = 1.0
P_total = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

P, lambda_opt = water_filling_with_dynamic_interference(h, N0, P_total)
print("Optimal Power Allocation: \n", P)
print("Water Levels (lambda): \n", lambda_opt)
