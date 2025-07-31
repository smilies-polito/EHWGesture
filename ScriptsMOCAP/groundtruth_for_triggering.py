import argparse

from TrackingData import FPS, FT, OC, PS, NOSE, TR
import pandas as pd
import os
import re

# ----- Utility Functions for Data Parsing -----
def parse_marker_mapping(header_df):
    """
    Parse a header DataFrame (one or more rows) to detect marker names and their corresponding X, Y, Z columns.
    This example assumes that after the first two columns ("Frame" and "Time"), every three columns share the same marker name,
    which is given in a header row (here we use row index 2).
    Returns a dictionary mapping marker names to a tuple of (X, Y, Z) column names.
    """
    marker_mapping = {}
    col_names = header_df.columns.tolist()
    start_idx = 2
    header_row = header_df.iloc[2]
    for i in range(start_idx, len(col_names), 3):
        marker_label = str(header_row[col_names[i]]).strip()
        if marker_label.lower() not in ["position", "nan", ""]:
            marker_mapping[marker_label] = (col_names[i], col_names[i+1], col_names[i+2])
    return marker_mapping

def expected_cadence_from_filename(file_path):
    """Determine the expected cadence (in frames) from the filename based on BPM (120 FPS)."""
    file_path = file_path.upper()
    if re.search(r'F[12]', file_path):
        return int(round(FPS / (140 / 60)))  # ~51 frames
    elif re.search(r'S[12]', file_path):
        return int(round(FPS / (75 / 60)))   # ~96 frames
    elif re.search(r'N[12]', file_path):
        return int(round(FPS / (115 / 60)))  # ~61 frames
    return None

# ----- Main Processing -----
def extract_gt_triggers(file_path, out_path):
    base_name = os.path.basename(file_path)
    if 'LEFT' in file_path:
        handedness='LEFT'
    else:
        handedness='RIGHT'
    task = None
    for m in ["FT", "OC", "NOSE", "PS", "TR"]:
        if m in base_name.upper():
            task = m
            break
    if not task:
        raise ValueError("No known metric found in the filename. Expecting one of: FT, OC, NOSE, PS, TR.")

    # For tasks other than TR, get expected cadence in frames.
    exp_cadence = None
    if task not in ["TR", "NOSE"]:
        exp_cadence = expected_cadence_from_filename(base_name)
    else:
        exp_cadence=FPS/2

    # Read the whole file without skipping header rows.
    try:
        full_df = pd.read_csv(file_path, header=None)
    except:
        print(f"File {file_path} is missing, moving to the next..")
        return
    header_rows = 6
    header_df = full_df.iloc[:header_rows]
    # Data starts after header rows + one extra row (adjust as needed).
    data_df = pd.read_csv(file_path, skiprows=header_rows+1, header=None)
    # Rename first two columns
    data_df.rename(columns={data_df.columns[0]: "Frame", data_df.columns[1]: "Time"}, inplace=True)
    data_df["Frame"] = pd.to_numeric(data_df["Frame"], errors="coerce")
    data_df["Time"] = pd.to_numeric(data_df["Time"], errors="coerce")
    data_df = data_df.dropna(subset=["Frame", "Time"]).reset_index(drop=True)

    # Use the header from data_df for marker mapping.
    marker_mapping = parse_marker_mapping(header_df)
    # Reassign new column names: first two are Frame and Time, then three columns per marker in the order of marker_mapping keys.
    new_col_names = ['Frame', 'Time']
    for marker in marker_mapping:
        new_col_names.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])
    data_df.columns = new_col_names

    # Reorganize the DataFrame to a dictionary where each marker name maps to its (X, Y, Z) as numpy arrays.
    tracking_data = {}
    tracking_data['Frame'] = data_df['Frame']
    tracking_data['Time'] = data_df['Time']
    for marker in marker_mapping:
        tracking_data[marker] = data_df[[f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].values

    # Create and run the task based on filename.
    if task == 'FT':
        obj = FT(tracking_data, base_name.split(".")[0], exp_cadence, handedness)
    elif task == 'OC':
        obj = OC(tracking_data, base_name.split(".")[0], exp_cadence, handedness)
    elif task == 'NOSE':
        # For NOSE, you might still want to use cadence if desired.
        obj = NOSE(tracking_data, base_name.split(".")[0], exp_cadence if exp_cadence else 60, handedness)
    elif task == 'PS':
        obj = PS(tracking_data, base_name.split(".")[0], exp_cadence, handedness)

    obj.compute_ref_signal()
    obj.compute_triggers(exp_cadence)
    obj.plot_reference()
    save=input("Save triggers? (Y/N)")
    if str.upper(save)=='Y':
        obj.save_triggers(out_path)
    else:
        if task!='FT':
            obj.compute_triggers_with_grd(exp_cadence)
            obj.plot_reference()
            save=input("If this version with gradient is better, write Y to save")
            if str.upper(save) == 'Y':
                obj.save_triggers(out_path)
                return
        print(f"{file_path}: triggers were not saved")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract ground truth triggers for trigger detection task."
    )
    parser.add_argument("--src", type=str, default="F:\EHWGesture",
                        help="Source dataset folder.")

    args = parser.parse_args()
    src=args.src
    input_dir=os.path.join(src, "DataMOCAP")
    dst=os.path.join(src, 'ScriptsMOCAP', 'output')

    folder_path=os.path.join(src, 'DataMocap')
    os.makedirs(dst, exist_ok=True)
    sbj_list = subj_list = [name for name in os.listdir(folder_path)
             if os.path.isdir(os.path.join(folder_path, name))]

    for sbj in sbj_list:
        for side in ['Left', 'Right']:
            for task in ['NOSE', 'OC', 'PS', 'FT']:
                if task=='NOSE':
                    file_path=os.path.join(input_dir,sbj,side,task+'1.csv')
                    out_path = os.path.join(dst, sbj, side, task + "1")
                    try:
                        extract_gt_triggers(file_path, out_path)
                    except Exception as e:
                        print(f"Error:{e} while processing {file_path}, moving to next...")

                    file_path = os.path.join(input_dir, sbj, side, task + '2.csv')
                    out_path = os.path.join(dst, sbj, side, task + "2")
                    try:
                        extract_gt_triggers(file_path, out_path)
                    except Exception as e:
                        print(f"Error:{e} while processing {file_path}, moving to next...")
                else:
                    for cad in ['S1', 'S2', 'N1', 'N2', 'F1', 'F2']:
                        file_path = os.path.join(input_dir, sbj, side, task + cad + '.csv')
                        out_path = os.path.join(dst, sbj, side, task + cad)
                        try:
                            extract_gt_triggers(file_path, out_path)
                        except Exception as e:
                            print(f"Error:{e} while processing {file_path}, moving to next...")


