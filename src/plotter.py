
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Plotting Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'sans-serif' # Use a generic font family

# --- Data Processing Functions ---

def plot_static_contact(file_path, output_dir):
    """Plots data from the 'Static contact_Thermal' experiment."""
    filename = os.path.basename(file_path)
    df = pd.read_csv(file_path, header=None)
    sampling_rate = 100  # Hz
    duration = 60  # seconds
    time = np.linspace(0, duration, len(df))

    plt.figure()
    if filename == 'forces.csv':
        df.columns = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        plt.subplot(2, 1, 1)
        plt.plot(time, df[['Fx', 'Fy', 'Fz']])
        plt.title('Static Contact: Forces')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend(['Fx', 'Fy', 'Fz'])
        
        plt.subplot(2, 1, 2)
        plt.plot(time, df[['Tx', 'Ty', 'Tz']])
        plt.title('Static Contact: Torques')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nmm)')
        plt.legend(['Tx', 'Ty', 'Tz'])
        
    elif filename == 'temperature.csv':
        df.columns = ['Temperature']
        plt.plot(time, df['Temperature'])
        plt.title('Static Contact: Finger Temperature')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        
    elif filename == 'heatflux.csv':
        df.columns = ['Heat Flux']
        plt.plot(time, df['Heat Flux'])
        plt.title('Static Contact: Heat Flux')
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flux (W/m^2)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Static_Contact_{os.path.splitext(filename)[0]}.png'))
    plt.close()

def plot_pressing(file_path, output_dir):
    """Plots data from the 'Pressing_Compliance' experiment."""
    filename = os.path.basename(file_path)
    df = pd.read_csv(file_path, header=None)
    
    plt.figure()
    if filename == 'forces.csv':
        sampling_rate = 10000  # Hz
        duration = 12 # seconds
        time = np.linspace(0, duration, len(df))
        df.columns = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        
        plt.subplot(2, 1, 1)
        plt.plot(time, df[['Fx', 'Fy', 'Fz']])
        plt.title('Pressing: Forces')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend(['Fx', 'Fy', 'Fz'])
        
        plt.subplot(2, 1, 2)
        plt.plot(time, df[['Tx', 'Ty', 'Tz']])
        plt.title('Pressing: Torques')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nmm)')
        plt.legend(['Tx', 'Ty', 'Tz'])

    elif filename == 'stageposition.csv':
        sampling_rate = 10 # Hz
        duration = 12 # seconds
        time = np.linspace(0, duration, len(df))
        df.columns = ['Position']
        plt.plot(time, df['Position'])
        plt.title('Pressing: Stage Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (mm)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Pressing_{os.path.splitext(filename)[0]}.png'))
    plt.close()

def plot_tapping(file_path, output_dir):
    """Plots data from the 'Tapping_Hardness' experiment."""
    filename = os.path.basename(file_path)
    df = pd.read_csv(file_path, header=None)
    sampling_rate = 10000  # Hz
    duration = 5  # seconds
    time = np.linspace(0, duration, len(df))

    plt.figure()
    if filename == 'forces.csv':
        df.columns = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        plt.subplot(2, 1, 1)
        plt.plot(time, df[['Fx', 'Fy', 'Fz']])
        plt.title('Tapping: Forces')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend(['Fx', 'Fy', 'Fz'])
        
        plt.subplot(2, 1, 2)
        plt.plot(time, df[['Tx', 'Ty', 'Tz']])
        plt.title('Tapping: Torques')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nmm)')
        plt.legend(['Tx', 'Ty', 'Tz'])

    elif filename == 'accelerations.csv':
        df.columns = ['Ax', 'Ay', 'Az']
        plt.plot(time, df)
        plt.title('Tapping: Accelerations')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        plt.legend(['Ax', 'Ay', 'Az'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Tapping_{os.path.splitext(filename)[0]}.png'))
    plt.close()

def plot_sliding(file_path, output_dir):
    """Plots data from the 'Sliding_Friction-roughness' experiment."""
    filename = os.path.basename(file_path)
    df = pd.read_csv(file_path, header=None)
    
    plt.figure()
    if 'Material_sensor' in filename:
        sampling_rate = 10000  # Hz
        time = np.arange(len(df)) / sampling_rate
        df.columns = ['Ax', 'Ay', 'Az', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        
        # Convert acceleration from g to m/s^2
        accel_df = df[['Ax', 'Ay', 'Az']] * 9.8
        
        plt.subplot(3, 1, 1)
        plt.plot(time, accel_df)
        plt.title('Sliding: Accelerations')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s^2)')
        plt.legend(['Ax', 'Ay', 'Az'])
        
        plt.subplot(3, 1, 2)
        plt.plot(time, df[['Fx', 'Fy', 'Fz']])
        plt.title('Sliding: Forces')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend(['Fx', 'Fy', 'Fz'])
        
        plt.subplot(3, 1, 3)
        plt.plot(time, df[['Tx', 'Ty', 'Tz']])
        plt.title('Sliding: Torques')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nmm)')
        plt.legend(['Tx', 'Ty', 'Tz'])

    elif 'Material_IR_pos' in filename:
        df.columns = ['Elapsed Time', 'Speed', 'Position X', 'Position Y']
        time = df['Elapsed Time']
        
        plt.subplot(2, 1, 1)
        plt.plot(time, df['Speed'])
        plt.title('Sliding: Speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (mm/s)')
        
        plt.subplot(2, 1, 2)
        plt.plot(df['Position X'], df['Position Y'])
        plt.title('Sliding: Trajectory')
        plt.xlabel('Position X (mm)')
        plt.ylabel('Position Y (mm)')
        plt.axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Sliding_{os.path.splitext(filename)[0]}.png'))
    plt.close()

# --- Main Execution Logic ---

def main(material_name):
    """
    Main function to find CSV files and generate plots for a given material.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', material_name)
    output_dir = os.path.join(base_dir, 'output', 'plot', material_name)

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # Map experiment types to plotting functions
    plotter_map = {
        'Static contact_Thermal': plot_static_contact,
        'Pressing_Compliance': plot_pressing,
        'Tapping_Hardness': plot_tapping,
        'Sliding_Friction-roughness': plot_sliding
    }

    # Walk through the data directory and process CSV files
    for root, dirs, files in os.walk(data_dir):
        # Determine the experiment type from the directory path
        experiment_type = os.path.basename(os.path.dirname(root))
        
        plot_func = plotter_map.get(experiment_type)
        
        if plot_func:
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Processing '{file_path}' with '{experiment_type}' plotter...")
                    try:
                        # Create a specific output directory for each participant
                        participant_name = os.path.basename(root)
                        participant_output_dir = os.path.join(output_dir, participant_name)
                        os.makedirs(participant_output_dir, exist_ok=True)
                        plot_func(file_path, participant_output_dir)
                    except Exception as e:
                        print(f"  -> Failed to plot {file}: {e}")

    print("Plotting complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots from material sensation data.")
    parser.add_argument('material', type=str, help="Name of the material folder in the 'data' directory (e.g., 'Material40').")
    args = parser.parse_args()
    
    main(args.material)
