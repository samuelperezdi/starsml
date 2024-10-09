import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.plot import plot_histograms_numerical

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'out_data': os.path.join(base_path, 'out_data')
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot.py histograms <path_to_csv_file or 'all'>")
        sys.exit(1)
    
    plot_type = sys.argv[1]
    input_option = sys.argv[2]
    
    if plot_type != 'histograms':
        print("Currently only 'histograms' plot type is supported.")
        sys.exit(1)
    
    if input_option.lower() == 'all':
        paths = setup_paths()
    
        # read full parquet file
        df = pd.read_parquet(os.path.join(paths['out_data'], 'nway_csc21_gaia3_full.parquet'))
    else:
        df = pd.read_csv(input_option)
    
    plot_histograms_numerical(df)
    print(f"Histograms have been saved to 'histograms.pdf'")

if __name__ == "__main__":
    main()