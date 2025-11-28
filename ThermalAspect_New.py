import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dt_values():
	csv_path = os.path.join(os.path.dirname(__file__), 'FaultMatrix', 'DT_Values.csv')
	output_dir = os.path.join(os.path.dirname(__file__), '..', 'output_tadt')
	os.makedirs(output_dir, exist_ok=True)
	df = pd.read_csv(csv_path)
	plt.figure(figsize=(14, 8))
	# Only plot columns with operational temp 25
	cols_25 = [col for col in df.columns if col != 'Time' and col.split('_')[-1] == '25' and 'sc20' not in col.lower()]
	for col in cols_25:
		plt.plot(df['Time'], df[col], label=col)
	plt.xlabel('Time')
	plt.ylabel('Values')
	plt.title('Delta Time Values Over Time')
	plt.legend(loc='upper right', fontsize='small', ncol=2)
	plt.tight_layout()
	output_path = os.path.join(output_dir, 'DT_Values_plot.png')
	plt.savefig(output_path)
	plt.close()
	print(f"Plot saved to {output_path}")

# Uncomment to run directly
plot_dt_values()
