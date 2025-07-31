import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_bounding_boxes(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            try:
                left, right, top, bottom = map(float, parts)
                data.append((left, right, top, bottom))
            except ValueError:
                continue
    if data:
        return np.array(data)
    else:
        return np.empty((0, 4))

def plot_scatter(files, group_name):
    plt.figure(figsize=(8, 6))
    
    colors = cm.get_cmap('hsv', len(files))
    
    for idx, file in enumerate(files):
        data = read_bounding_boxes(file)
        if data.shape[0] == 0:
            continue
        centers_x = (data[:, 0] + data[:, 1]) / 2.0
        centers_y = (data[:, 2] + data[:, 3]) / 2.0
        # Plot the center points. In cv2 image coordinate systems, y=0 is at the top.
        plt.scatter(centers_x, 1080-centers_y, s=10, color=colors(idx),
                    label=os.path.basename(file))
    
    plt.title(f"{group_name.capitalize()} Videos: Bounding Box Centers")
    plt.xlabel("Center X")
    plt.ylabel("Center Y")
    # plt.legend(fontsize='small', loc='best', markerscale=2)
    plt.grid(True)
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)
    plt.tight_layout()
    plt.show()

def plot_histograms(files, group_name):
    all_data = []
    for file in files:
        data = read_bounding_boxes(file)
        if data.shape[0] > 0:
            all_data.append(data)
    if not all_data:
        print(f"No data to plot for {group_name} files.")
        return
    all_data = np.concatenate(all_data, axis=0)
    
    lefts   = all_data[:, 0]
    rights  = all_data[:, 1]
    tops    = all_data[:, 2]
    bottoms = all_data[:, 3]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    
    axs[0].hist(lefts, bins=30, color='skyblue', edgecolor='black')
    axs[0].set_title("Left Coordinates")
    axs[0].set_xlabel("Left")
    axs[0].set_ylabel("Frequency")
    
    axs[1].hist(rights, bins=30, color='salmon', edgecolor='black')
    axs[1].set_title("Right Coordinates")
    axs[1].set_xlabel("Right")
    axs[1].set_ylabel("Frequency")
    
    # again, if you are confused, y=0 is the top
    axs[2].hist(tops, bins=30, color='lightgreen', edgecolor='black')
    axs[2].set_title("Top Coordinates")
    axs[2].set_xlabel("Top")
    axs[2].set_ylabel("Frequency")
    
    axs[3].hist(bottoms, bins=30, color='plum', edgecolor='black')
    axs[3].set_title("Bottom Coordinates")
    axs[3].set_xlabel("Bottom")
    axs[3].set_ylabel("Frequency")
    
    plt.suptitle(f"{group_name.capitalize()} Videos: Coordinate Histograms", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    output_dir = "traj_output"
    all_txt_files = glob.glob(os.path.join(output_dir, "**", "*.txt"), recursive=True)
    
    master_files = [f for f in all_txt_files if "master" in os.path.basename(f).lower()]
    sub_files    = [f for f in all_txt_files if "sub" in os.path.basename(f).lower()]
    
    if master_files:
        print(f"Found {len(master_files)} master files. Plotting scatter plot...")
        plot_scatter(master_files, group_name="master")
        print(f"Plotting histograms for master files...")
        plot_histograms(master_files, group_name="master")
    else:
        print("No master files found.")
    
    if sub_files:
        print(f"Found {len(sub_files)} sub files. Plotting scatter plot...")
        plot_scatter(sub_files, group_name="sub")
        print(f"Plotting histograms for sub files...")
        plot_histograms(sub_files, group_name="sub")
    else:
        print("No sub files found.")

if __name__ == '__main__':
    main()
