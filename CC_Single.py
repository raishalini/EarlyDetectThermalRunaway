import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from pathlib import Path
import warnings

def CorLACoef(A, B):
    """
    Calculate correlation coefficient between two arrays.
    Replace this with your actual correlation function implementation.
    """
    return np.corrcoef(A, B)[0, 1]

def process_battery_data():
    pwd = 'C:/Users/320252531/Documents/thesis/cc_new'

    # Get all .csv files in FaultMatrix subfolders
    fault_matrix_path = os.path.join(pwd, 'FaultMatrix')
    csv_files = []
    
    for root, dirs, files in os.walk(fault_matrix_path):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_CCvalues.csv'):
                csv_files.append({
                    'folder': root,
                    'name': file
                })
    
    for csv_file in csv_files:
        # Extract file info
        csv_file_path = os.path.join(csv_file['folder'], csv_file['name'])
        file_name = os.path.splitext(csv_file['name'])[0]
        
        # Parse Test_T and Test_case from filename
        pattern = r'RCS_(\w+)_([0-9]+)C_Topo\d+'
        match = re.search(pattern, file_name)
        
        if not match:
            warnings.warn(f"Filename {file_name} does not match expected pattern.")
            continue
        
        Test_case = match.group(1)
        Test_T = int(match.group(2))
        topology = 1  # or parse if needed
        
        # Load CSV file
        comb_table = pd.read_csv(csv_file_path)
        
        # Debug: print column names
        print(f"Processing: {file_name}")
        print(f"Columns in CSV: {comb_table.columns.tolist()}")
        
        ## CC application steps
        ds_steps = 1
        new_time = np.arange(1, comb_table['Time'].max() + 1, ds_steps)
        
        # Create downsampled table
        ds_table = pd.DataFrame({'Time': new_time})
        
        vars_list = [col for col in comb_table.columns if col != 'Time']
        
        for var in vars_list:
            f = interpolate.interp1d(comb_table['Time'], comb_table[var], 
                                    kind='linear', fill_value='extrapolate')
            ds_table[var] = f(new_time)
        
        # Calculate Vtot - find V and C columns dynamically
        v_cols = [col for col in ds_table.columns if col.startswith('V') and col[1:].isdigit()]
        c_cols = [col for col in ds_table.columns if col.startswith('C') and col[1:].isdigit()]
        varsAll = v_cols + c_cols
        
        if len(varsAll) > 0:
            ds_table['Vtot'] = ds_table[varsAll].sum(axis=1)
        else:
            print(f"Warning: No V or C columns found in {file_name}")
            continue
        
        # Noise parameters
        AmpNoiseV = 0.05
        AmpNoiseC = 0.005
        AmpNoiseI = 0.05
        AmpNoiseString = 0.05
        
        t = ds_table['Time'].values
        WinSize = 60
        maxrow_t = np.where(t >= WinSize)[0][0]
        endrow_t = len(t)
        diff1 = endrow_t - maxrow_t + 1
        calculation_range = range(diff1)
        
        # Create base noise
        baseNois1 = np.zeros(maxrow_t)
        for i in range(int((maxrow_t - 4) / 4) + 1):
            a1 = i * 4
            a2 = 2 + i * 4
            b1 = 3 + i * 4
            b2 = 4 + i * 4
            if a2 <= maxrow_t:
                baseNois1[a1:a2] = 1
            if b2 <= maxrow_t:
                baseNois1[b1:b2] = -1
        
        NoisV = AmpNoiseV * baseNois1
        NoisC = AmpNoiseC * baseNois1
        NoisI = AmpNoiseI * baseNois1
        NoisString = AmpNoiseString * baseNois1
        
        # Create pair names
        a = 0
        b = 0
        PairNames = []
        
        # V-V pairs (45 pairs)
        for i in range(1, 10):
            b += 1
            nam1 = f'V{i}'
            nam3 = f'C{i}'
            for j in range(i + 1, 11):
                nam2 = f'V{j}'
                PairNames.append([f'R{a + 1}', nam1, nam2, f'r_{{{nam1},{nam2}}}'])
                a += 1
        
        # IApp-C pairs (9 pairs)
        for i in range(1, 10):
            nam3 = f'C{i}'
            PairNames.append([f'R{46 + i - 1}', 'IApp', nam3, f'r_{{IApp,{nam3}}}'])
        
        # C-C pairs (36 pairs)
        for i in range(1, 9):
            nam1 = f'C{i}'
            for j in range(i + 1, 10):
                b += 1
                nam2 = f'C{j}'
                PairNames.append([f'R{45 + b}', nam1, nam2, f'r_{{{nam1},{nam2}}}'])
        
        # Vstring-Vtot pair
        extendno = len(PairNames)
        PairNames.append([f'R{extendno + 1}', 'Vstring', 'Vtot', 'r_{Vstr,Vtot}'])
        
        # Get data matrix
        dataMatrix = ds_table.values
        VarNames = ds_table.columns.tolist()
        
        # Find column indices for pairs
        pairCols = []
        for pair in PairNames:
            idx2 = VarNames.index(pair[1]) if pair[1] in VarNames else -1
            idx3 = VarNames.index(pair[2]) if pair[2] in VarNames else -1
            pairCols.append([idx2, idx3])
        
        # Calculate correlation coefficients
        CCvalues = {'Time': np.zeros(len(calculation_range))}
        for pair in PairNames:
            CCvalues[pair[0]] = np.zeros(len(calculation_range))
        
        for i in calculation_range:
            range1 = slice(i, i + maxrow_t)
            CCvalues['Time'][i] = t[i] + WinSize
            
            for j, pair in enumerate(PairNames):
                col1, col2 = pairCols[j]
                A = dataMatrix[range1, col1]
                B = dataMatrix[range1, col2]
                
                if j < 45:
                    noisA = NoisV
                    noisB = NoisV
                elif 45 <= j < 54:
                    noisA = NoisI
                    noisB = NoisC
                elif 54 < j <= 90:
                    noisA = NoisC
                    noisB = NoisC
                else:
                    noisA = NoisString
                    noisB = NoisString
                
                A = A + noisA
                B = B + noisB
                CCvalues[pair[0]][i] = CorLACoef(A, B)
        
        # Create output folder
        out_folder = os.path.join(pwd, 'output', f'T{Test_T}_TestCase{Test_case}')
        os.makedirs(out_folder, exist_ok=True)
        
        # Convert CCvalues to DataFrame and save
        AA = pd.DataFrame(CCvalues)
        
        # Determine subfolder based on temperature
        if Test_T == 5:
            subfolder = '5C'
        elif Test_T == 25:
            subfolder = '25C'
        elif Test_T == 40:
            subfolder = '40C'
        else:
            subfolder = 'Other'
        
        out_sub_folder = os.path.join(pwd, 'output', subfolder)
        os.makedirs(out_sub_folder, exist_ok=True)
        
        # Save CSV files
        cor_names = os.path.join(out_sub_folder, f'{file_name}_CCvalues.csv')
        AA.to_csv(cor_names, index=False)
        
        com_name = os.path.join(out_sub_folder, f'{file_name}.csv')
        comb_table.to_csv(com_name, index=False)
        
        # Find fault injection time from signal column
        fault_time = None
        print(f"Available columns: {comb_table.columns.tolist()}")
        if 'Signal' in comb_table.columns:
            print(f"Signal column found. Unique values: {comb_table['Signal'].unique()}")
            # Find first time when signal changes from 0 (or becomes non-zero)
            signal_change = comb_table[comb_table['Signal'] != 0]
            if len(signal_change) > 0:
                fault_time = signal_change['Time'].iloc[0]
                print(f"Fault injection starts at Time = {fault_time}")
                print(f"CCvalues Time range: {CCvalues['Time'][0]} to {CCvalues['Time'][-1]}")
            else:
                print("No non-zero signal values found")
        else:
            print("'Signal' column not found in data")
        
        # Plotting
        G1 = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 9, 17, 18, 19, 20, 21, 22, 23],
            [2, 10, 17, 24, 25, 26, 27, 28, 29],
            [3, 11, 18, 24, 30, 31, 32, 33, 34],
            [4, 12, 19, 25, 30, 35, 36, 37, 38],
            [5, 13, 20, 26, 31, 35, 39, 40, 41],
            [6, 14, 21, 27, 32, 36, 39, 42, 43],
            [7, 15, 22, 28, 33, 37, 40, 42, 44],
            [8, 16, 23, 29, 34, 38, 41, 43, 44]
        ]
        
        # Figure 1 - Only Cell 1 vs. Others
        fig1, ax = plt.subplots(figsize=(10, 6))

        titlestr = 'Cell 1 vs. Others'
        for a in range(9):
            idx_plot = G1[0][a]  # G1[0] for Cell 1
            ax.plot(CCvalues['Time'], CCvalues[PairNames[idx_plot][0]], 
                   label=PairNames[idx_plot][3])

        # Add vertical line at fault injection time
        if fault_time is not None:
            ax.axvline(x=fault_time, color='red', linestyle='--', linewidth=2, label='Fault Injection')

        # Limit x-axis for specific file
        if csv_file_path.replace('\\', '/').endswith('cc_new/FaultMatrix/25C/RCS_NF_25C_Topo1.csv'):
            ax.set_xlim([0, 800])

        ax.set_title(titlestr)
        ax.set_xlabel('Time')
        ax.set_ylabel('Correlation Coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, 'CC_Group1.png'), dpi=150)
        plt.close(fig1)
        
        # Figure 2
        # Group 2
        # fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig2, ax1 = plt.subplots(figsize=(10, 6))
        fig4, ax2 = plt.subplots(figsize=(10, 6))

        ax1.set_title('CC group 2 (IApp vs. Contact)')
        for i in range(45, 54):
            ax1.plot(CCvalues['Time'], CCvalues[PairNames[i][0]], 
                    label=PairNames[i][3])
        if fault_time is not None:
            ax1.axvline(x=fault_time, color='red', linestyle='--', linewidth=2, label='Fault Injection')
        
        # Limit x-axis for specific file
        if csv_file_path.replace('\\', '/').endswith('cc_new/FaultMatrix/25C/RCS_NF_25C_Topo1.csv'):
            ax1.set_xlim([0, 800])

        ax1.legend(fontsize=8)
        # ax1.set_title(titlestr)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.legend()

        fig2.tight_layout()
        fig2.savefig(os.path.join(out_folder, 'CC_Group2.png'))
        plt.close(fig2)
        
        # Group 3
        ax2.set_title('CC group 3 (Contact vs. Contact)')
        for i in range(54, 90):
            ax2.plot(CCvalues['Time'], CCvalues[PairNames[i][0]], 
                    label=PairNames[i][3])
        if fault_time is not None:
            ax2.axvline(x=fault_time, color='red', linestyle='--', linewidth=2, label='Fault Injection')

        # Limit x-axis for specific file
        if csv_file_path.replace('\\', '/').endswith('cc_new/FaultMatrix/25C/RCS_NF_25C_Topo1.csv'):
            ax2.set_xlim([0, 800])

        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, prop={'size': 6})
        # ax2.set_title(titlestr)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Correlation Coefficient')

        fig4.tight_layout()
        fig4.savefig(os.path.join(out_folder, 'CC_Group3.png'))
        plt.close(fig4)
        
        # Figure 3
        fig3, ax = plt.subplots(figsize=(10, 6))
        ax.plot(CCvalues['Time'], CCvalues[PairNames[90][0]], 
               label=PairNames[90][3])
        if fault_time is not None:
            ax.axvline(x=fault_time, color='red', linestyle='--', linewidth=2, label='Fault Injection')
        ax.set_title('V_{total} vs. V_{string}')
        ax.legend()
        
        plt.savefig(os.path.join(out_folder, 'CC_Vtot_vs_Vstring.png'))
        plt.close(fig3)
        
        print(f"Processed: {file_name}")

if __name__ == "__main__":
    process_battery_data()