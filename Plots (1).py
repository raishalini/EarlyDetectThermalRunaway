import plotly.graph_objects as go
import os
from collections import defaultdict
from plotly.subplots import make_subplots
import math
import numpy as np

class MultiRunCorrelationPlotter:
    def __init__(self, df_dict, time_column="Time", desired_window_in_seconds=60, step_fraction=0.5):
        self.df_dict = df_dict
        self.V_cols = ["V1", "V2", "V3", "V4", "V5"]  # Voltage columns
        self.DV_cols = ["DV1", "DV2", "DV3", "DV4", "DV5"]  # Delta Voltage columns
        self.IApp_col = "IApp"  # Current application column
        self.DT_col = "DT"  # Delta time column
        self.FaultIN_col = "FaultIN"  # Fault indicator column
        self.time_column = time_column
        self.desired_window_in_seconds = desired_window_in_seconds
        self.step_fraction = step_fraction

    def sliding_correlation(self, df, var1, var2):
        """
        Perform sliding window correlation for the given variables.
        """
        # Calculate the window size based on the desired time in seconds
        seconds_per_row = df[self.time_column].diff().median()
        window_size = int(self.desired_window_in_seconds / seconds_per_row)
        step_size = max(1, int(window_size * self.step_fraction))

        correlations = []
        times = []

        for start in range(0, len(df) - window_size + 1, step_size):
            end = start + window_size
            window_df = df.iloc[start:end]

            corr = window_df[[var1, var2]].corr().iloc[0, 1]
            correlations.append(corr)
            times.append(df[self.time_column].iloc[start])

        return times, correlations

    def plot_with_subplots(self, fault_type, plot_title, var_pairs, fault_group, secondary_y_trace_func=None):
        """
        Generates a single plot with subplots for each run, for the given fault type.
        """
        num_runs = len(fault_group)
        cols = 2
        rows = math.ceil(num_runs / cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=[f"{run_name}" for run_name in sorted(fault_group.keys())]
        )

        for i, (run_name, df) in enumerate(sorted(fault_group.items())):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            for var_pair in var_pairs:
                var1, var2 = var_pair
                if var1 in df.columns and var2 in df.columns:
                    times, correlations = self.sliding_correlation(df, var1, var2)
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=correlations,
                            mode="lines",
                            name=f"{var1} vs {var2}",
                            line=dict(color=self.get_combination_color(var_pair))
                        ),
                        row=row,
                        col=col
                    )
            if secondary_y_trace_func:
                secondary_y_trace_func(fig, df, row, col)

            # Add FaultIN markers
            if self.FaultIN_col in df.columns:
                fault_times = df[df[self.FaultIN_col] == 1][self.time_column].unique()
                x_vals = []
                y_vals = []
                for ft in fault_times:
                    x_vals.extend([ft, ft, None])
                    y_vals.extend([1, -1, None])  # spans the whole vertical range

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line=dict(color="red", width=1, dash="dash"),
                    name="FaultIN",
                    showlegend=True  # show legend only once
                ), row=row, col=col)

            # Add axis titles only where appropriate
            if row == rows:
                fig.update_xaxes(title_text="Time (s)", row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text="Correlation", row=row, col=col)

        fig.update_layout(
            title=plot_title,
            height=300 * rows,
            showlegend=True,
        )

        output_dir = f"/content/drive/MyDrive/Thesis/Plots/{fault_type}"
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(f"{output_dir}/{plot_title}.html")


    def analyze_all(self):
        """
        Analyzes and plots data for all fault types, generating a total of 5 plots for each fault type.
        """
        print("Grouping dataframes by fault type...")
        fault_groups = self.group_dataframes_by_fault()

        # Loop through the fault types and generate the required 5 plots for each one
        for fault_type, fault_group in fault_groups.items():
            print(f"****** Running analysis for {fault_type} ******")
            
            # 1. V vs V (Voltage vs Voltage)
            v_v_pairs = [(self.V_cols[i], self.V_cols[j]) for i in range(len(self.V_cols)) for j in range(i + 1, len(self.V_cols))]
            self.plot_with_subplots(fault_type, f"Correlation Between Voltages (V1-V5) - {fault_type}", v_v_pairs, fault_group, secondary_y_trace_func=self.add_iapp_secondary_y)
            
            # 2. DV vs IApp (Delta Voltage vs Current Application)
            dv_iapp_pairs = [(self.DV_cols[i], self.IApp_col) for i in range(len(self.DV_cols))]
            self.plot_with_subplots(fault_type, f"Correlation Between Delta Voltages and IApp - {fault_type}", dv_iapp_pairs, fault_group)

            # 3. DT vs V (Delta Time vs Voltage)
            dt_v_pairs = [(self.DT_col, self.V_cols[i]) for i in range(len(self.V_cols))]
            self.plot_with_subplots(fault_type, f"Correlation Between Delta Time and Voltages (V1-V5) - {fault_type}", dt_v_pairs, fault_group)

            # 4. DT vs DV (Delta Time vs Delta Voltage)
            dt_dv_pairs = [(self.DT_col, self.DV_cols[i]) for i in range(len(self.DV_cols))]
            self.plot_with_subplots(fault_type, f"Correlation Between Delta Time and Delta Voltages (DV1-DV5) - {fault_type}", dt_dv_pairs, fault_group)

            # 5. DT vs IApp (Current Application vs Voltage)
            dt_iapp_pairs = [(self.DT_col, self.IApp_col)]
            self.plot_with_subplots(fault_type, f"Correlation Between Delta Time and IApp - {fault_type}", dt_iapp_pairs, fault_group)

            # 6. Shannon Entropy for DT
            self.plot_entropy_dt(fault_type, fault_group)

    def group_dataframes_by_fault(self):
        """
        Groups the dataframes in the df_dict by their fault type.
        """
        fault_groups = defaultdict(dict)

        for name, df in self.df_dict.items():
            # Extract the fault type from the dataframe name (assuming format df_faulttype_runX)
            parts = name.split('_')
            fault_type = parts[1]  # Get the fault type (e.g., "50CE")
            fault_groups[fault_type][name] = df

        return dict(fault_groups)

    def get_combination_color(self, var_pair):
        """
        Generates a unique color for each variable combination.
        """
        # A simple way to generate a unique color based on variable pair
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        hash_value = hash(f"{var_pair[0]}_{var_pair[1]}") % len(colors)
        return colors[hash_value]

    def add_iapp_secondary_y(self, fig, df, row, col):
        # Calculate yaxis index based on the row and column of the subplot
        yaxis_index = (row - 1) * 2 + col  # Ensure correct mapping for yaxis1, yaxis2, etc.
        
        # Add the IApp trace with the secondary y-axis (y2, y3, etc.)
        fig.add_trace(
            go.Scatter(
                x=df[self.time_column],
                y=df[self.IApp_col],
                name="IApp",
                mode="lines",
                line=dict(color="black", dash="dot"),
                yaxis=f"y{yaxis_index}2",  # Correctly reference the secondary y-axis
                showlegend=True  # Show IApp in the legend for each subplot
            ),
            row=row, col=col,
            secondary_y=True
        )
        
        # Update the secondary y-axis title (IApp)
        fig.update_yaxes(title_text="IApp", secondary_y=True, row=row, col=col)

    def calculate_shannon_entropy(self, df, dt_column):
        # Assuming time_column is already sorted
        time_column = self.time_column  # Your time column
        # Calculate the probability distribution for Delta Time (DT)
        dt_values, counts = np.unique(df[dt_column], return_counts=True)
        probabilities = counts / len(df)
        
        # Calculate the Shannon entropy for the entire column
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy

    def plot_entropy_dt(self, fault_type, fault_group, window_size=60, step_size=10):
        import scipy.stats

        num_runs = len(fault_group)
        cols = 2
        rows = (num_runs + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[run for run in sorted(fault_group.keys())],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        for i, (run_name, df) in enumerate(sorted(fault_group.items())):
            row = i // cols + 1
            col = i % cols + 1

            entropy_vals = []
            time_vals = []

            dt_values = df[self.DT_col].dropna().values
            time_series = df[self.time_column].values

            for start in range(0, len(dt_values) - window_size + 1, step_size):
                window = dt_values[start:start + window_size]
                binned = np.floor(window).astype(int)
                value_counts = np.bincount(binned)
                probs = value_counts[value_counts > 0] / value_counts.sum()
                entropy = scipy.stats.entropy(probs, base=2)
                entropy_vals.append(entropy)
                time_vals.append(time_series[start + window_size // 2])

            fig.add_trace(
                go.Scatter(
                    x=time_vals,
                    y=entropy_vals,
                    name="Shannon Entropy",
                    mode="lines",
                    line=dict(color="purple"),
                    showlegend=True
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            height=300 * rows,
            width=1000,
            title_text=f"Shannon Entropy of Delta T (Sliding Window) - {fault_type}",
        )
        fig.update_yaxes(title_text="Entropy (bits)")
        fig.update_xaxes(title_text="Time (s)")

        # Save the plot
        output_dir = f"/content/drive/MyDrive/Thesis/Plots/{fault_type}"
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(f"{output_dir}/Shannon_Entropy_{fault_type}.html")




