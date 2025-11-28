import pandas as pd
import matplotlib.pyplot as plt

def load_data(exp_path, model_path):
	exp_df = pd.read_csv(exp_path)
	model_df = pd.read_csv(model_path)
	return exp_df, model_df

def get_fault_time(df):
	fault_idx = df[df['Signal'] != 0].index[0] if not df[df['Signal'] != 0].empty else None
	fault_time = df.loc[fault_idx, 'Time'] if fault_idx is not None else None
	return fault_time

def plot_deltaT(exp_df, fault_time, label, out_path):
	plt.figure(figsize=(10,6))
	plt.plot(exp_df['Time'], exp_df['Tout'] - exp_df['Tin'], label=label)
	if fault_time is not None:
		plt.axvline(x=fault_time, color='red', linestyle='dotted', label='Fault Start')
	plt.xlabel('Time')
	plt.ylabel('Delta T (Tout - Tin)')
	plt.title(f'Delta T vs Time for {label} (40C)')
	plt.legend()
	plt.grid(True)
	plt.savefig(out_path)
	plt.close()

def plot_exp_vs_model(exp_df, model_df, fault_time, label, out_path, conf_band=False):
	plt.figure(figsize=(10,6))
	exp_deltaT = exp_df['Tout'] - exp_df['Tin']
	model_deltaT = model_df['DeltaT']
	time_exp = exp_df['Time']
	time_model = model_df['Time']
	if conf_band:
		offset = 0.5
		upper = model_deltaT + offset
		lower = model_deltaT - offset
		plt.fill_between(time_model, lower, upper, color='green', alpha=0.3, label=f'{label} Model ±0.5')
	plt.plot(time_model, model_deltaT, 'k-', linewidth=2, label=f'{label} Model')
	plt.plot(time_exp, exp_deltaT, label=f'{label} Experimental')
	if fault_time is not None:
		plt.axvline(x=fault_time, color='red', linestyle='dotted', label='Fault Start')
	plt.xlabel('Time')
	plt.ylabel('Delta T (Tout - Tin)')
	plt.title(f'{label} Experimental vs Model Delta T vs Time (40C)' + (' with Confidence Band' if conf_band else ''))
	plt.legend()
	plt.grid(True)
	plt.savefig(out_path)
	plt.close()

# File paths
ce100_exp_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\40C\RCS_CE100_40C_Topo1.csv'
sc05_exp_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\40C\RCS_SC05_40C_Topo1.csv'
ce100_model_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\ThermalOutputCSV\40C\RCS_CE100_40C_Topo1ThermalModel.csv'
sc05_model_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\ThermalOutputCSV\40C\RCS_SC05_40C_Topo1ThermalModel.csv'

# Load data
ce100_df, ce100_model_df = load_data(ce100_exp_path, ce100_model_path)
sc05_df, sc05_model_df = load_data(sc05_exp_path, sc05_model_path)

# Get fault times
ce100_fault_time = get_fault_time(ce100_df)
sc05_fault_time = get_fault_time(sc05_df)

# Plot and save CE100 and SC05 experimental delta T
plot_deltaT(ce100_df, ce100_fault_time, 'CE100', r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\CE100_deltaT_vs_Time.png')
plot_deltaT(sc05_df, sc05_fault_time, 'SC05', r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\SC05_deltaT_vs_Time.png')

# Plot experimental vs model delta T
plot_exp_vs_model(ce100_df, ce100_model_df, ce100_fault_time, 'CE100', r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\CE100_Exp_vs_Model_deltaT_vs_Time.png')
plot_exp_vs_model(sc05_df, sc05_model_df, sc05_fault_time, 'SC05', r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\SC05_Exp_vs_Model_deltaT_vs_Time.png')

# Plot experimental vs model delta T with confidence band
plot_exp_vs_model(ce100_df, ce100_model_df, ce100_fault_time, 'CE100', r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\CE100_Exp_vs_Model_ConfidenceBand.png', conf_band=True)
plot_exp_vs_model(sc05_df, sc05_model_df, sc05_fault_time, 'SC05', r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\SC05_Exp_vs_Model_ConfidenceBand.png', conf_band=True)
import pandas as pd
import matplotlib.pyplot as plt

# File paths
ce100_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\40C\RCS_CE100_40C_Topo1.csv'
sc05_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\40C\RCS_SC05_40C_Topo1.csv'

# Read CSV files
ce100_df = pd.read_csv(ce100_path)
sc05_df = pd.read_csv(sc05_path)

# Find fault start index for CE100
ce100_fault_idx = ce100_df[ce100_df['Signal'] != 0].index[0] if not ce100_df[ce100_df['Signal'] != 0].empty else None
ce100_fault_time = ce100_df.loc[ce100_fault_idx, 'Time'] if ce100_fault_idx is not None else None

# Find fault start index for SC05
sc05_fault_idx = sc05_df[sc05_df['Signal'] != 0].index[0] if not sc05_df[sc05_df['Signal'] != 0].empty else None
sc05_fault_time = sc05_df.loc[sc05_fault_idx, 'Time'] if sc05_fault_idx is not None else None

# Plot and save CE100 graph
plt.figure(figsize=(10,6))
plt.plot(ce100_df['Time'], ce100_df['Tout'] - ce100_df['Tin'], label='CE100')
if ce100_fault_time is not None:
	plt.axvline(x=ce100_fault_time, color='red', linestyle='dotted', label='Fault Start')
plt.xlabel('Time')
plt.ylabel('Delta T (Tout - Tin)')
plt.title('Delta T vs Time for CE100 (40C)')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\CE100_deltaT_vs_Time.png')
plt.close()

# Plot and save SC05 graph
plt.figure(figsize=(10,6))
plt.plot(sc05_df['Time'], sc05_df['Tout'] - sc05_df['Tin'], label='SC05')
if sc05_fault_time is not None:
	plt.axvline(x=sc05_fault_time, color='red', linestyle='dotted', label='Fault Start')
plt.xlabel('Time')
plt.ylabel('Delta T (Tout - Tin)')
plt.title('Delta T vs Time for SC05 (40C)')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\SC05_deltaT_vs_Time.png')
plt.close()

# Load model data
ce100_model_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\ThermalOutputCSV\40C\RCS_CE100_40C_Topo1ThermalModel.csv'
sc05_model_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\ThermalOutputCSV\40C\RCS_SC05_40C_Topo1ThermalModel.csv'
ce100_model_df = pd.read_csv(ce100_model_path)
sc05_model_df = pd.read_csv(sc05_model_path)

# Plot CE100 experimental vs model delta T
plt.figure(figsize=(10,6))
plt.plot(ce100_df['Time'], ce100_df['Tout'] - ce100_df['Tin'], label='CE100 Experimental')
plt.plot(ce100_model_df['Time'], ce100_model_df['DeltaT'], label='CE100 Model')
if ce100_fault_time is not None:
	plt.axvline(x=ce100_fault_time, color='red', linestyle='dotted', label='Fault Start')
plt.xlabel('Time')
plt.ylabel('Delta T (Tout - Tin)')
plt.title('CE100 Experimental vs Model Delta T vs Time (40C)')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\CE100_Exp_vs_Model_deltaT_vs_Time.png')
plt.close()

# Plot SC05 experimental vs model delta T
plt.figure(figsize=(10,6))
plt.plot(sc05_df['Time'], sc05_df['Tout'] - sc05_df['Tin'], label='SC05 Experimental')
plt.plot(sc05_model_df['Time'], sc05_model_df['DeltaT'], label='SC05 Model')
if sc05_fault_time is not None:
	plt.axvline(x=sc05_fault_time, color='red', linestyle='dotted', label='Fault Start')
plt.xlabel('Time')
plt.ylabel('Delta T (Tout - Tin)')
plt.title('SC05 Experimental vs Model Delta T vs Time (40C)')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\SC05_Exp_vs_Model_deltaT_vs_Time.png')
plt.close()
import pandas as pd
import matplotlib.pyplot as plt

# File paths
ce100_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\40C\RCS_CE100_40C_Topo1.csv'
sc05_path = r'C:\Users\320252531\Documents\thesis\cc_new\FaultMatrix\40C\RCS_SC05_40C_Topo1.csv'

# Read CSV files
ce100_df = pd.read_csv(ce100_path)
sc05_df = pd.read_csv(sc05_path)



# Find fault start index for CE100
ce100_fault_idx = ce100_df[ce100_df['Signal'] != 0].index[0] if not ce100_df[ce100_df['Signal'] != 0].empty else None
ce100_fault_time = ce100_df.loc[ce100_fault_idx, 'Time'] if ce100_fault_idx is not None else None

# Find fault start index for SC05
sc05_fault_idx = sc05_df[sc05_df['Signal'] != 0].index[0] if not sc05_df[sc05_df['Signal'] != 0].empty else None
sc05_fault_time = sc05_df.loc[sc05_fault_idx, 'Time'] if sc05_fault_idx is not None else None


# Plot and save CE100 graph
plt.figure(figsize=(10,6))
plt.plot(ce100_df['Time'], ce100_df['Tout'] - ce100_df['Tin'], label='CE100')
if ce100_fault_time is not None:
	plt.axvline(x=ce100_fault_time, color='red', linestyle='dotted', label='Fault Start')
plt.xlabel('Time')
plt.ylabel('Delta T (Tout - Tin)')
plt.title('Delta T vs Time for CE100 (40C)')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\CE100_deltaT_vs_Time.png')
plt.close()

# Plot and save SC05 graph
plt.figure(figsize=(10,6))
plt.plot(sc05_df['Time'], sc05_df['Tout'] - sc05_df['Tin'], label='SC05')
if sc05_fault_time is not None:
	plt.axvline(x=sc05_fault_time, color='red', linestyle='dotted', label='Fault Start')
plt.xlabel('Time')
plt.ylabel('Delta T (Tout - Tin)')
plt.title('Delta T vs Time for SC05 (40C)')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\320252531\Documents\thesis\cc_new\output_ta\SC05_deltaT_vs_Time.png')
plt.close()