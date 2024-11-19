import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_path = "squeezenet1_1_output.csv"
logging.info(f"Loading data from {file_path}")
data_frame = pd.read_csv(file_path)

if data_frame.empty:
    logging.error("The DataFrame is empty. Please check the input file.")
    raise ValueError("The DataFrame is empty.")

logging.info("Converting execution time to numeric.")
data_frame["gpu_time_ns"] = pd.to_numeric(data_frame["gpu__time_duration.sum"], errors='coerce')

logging.info("Calculating additional metrics.")
data_frame["Memory Traffic Total"] = data_frame["dram__bytes_read.sum"] + data_frame["dram__bytes_write.sum"]
data_frame["Total FLOPs"] = (
    data_frame["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"]
    + data_frame["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"] * 2
    + data_frame["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"]
)
data_frame["Execution Time (Seconds)"] = data_frame["gpu_time_ns"] / 1e9  # Convert nanoseconds to seconds

# Convert metrics to numeric, coercing errors to NaN
logging.info("Converting FLOPs and Memory Traffic to numeric.")
data_frame["Total FLOPs"] = pd.to_numeric(data_frame["Total FLOPs"], errors='coerce')
data_frame["Memory Traffic Total"] = pd.to_numeric(data_frame["Memory Traffic Total"], errors='coerce')

logging.info("Calculating operational intensity.")
data_frame["Operational Intensity"] = data_frame["Total FLOPs"] / data_frame["Memory Traffic Total"]

logging.info("Calculating performance in GFLOPs.")
data_frame["Performance (GFLOPs)"] = data_frame["Total FLOPs"] / data_frame["Execution Time (Seconds)"] / 1e9

memory_bandwidth_limit = 1500  
computational_performance_limit = 20 

fig, ax = plt.subplots(figsize=(7, 7)) 


logging.info("Plotting performance boundaries.")
operational_intensity_range = np.logspace(-2, 2, 100) 
memory_limit_curve = operational_intensity_range * memory_bandwidth_limit
compute_limit_curve = [computational_performance_limit] * len(operational_intensity_range)


ax.plot(operational_intensity_range, memory_limit_curve, label="Memory Bandwidth (1500 GB/s)", linestyle="-.", color="lightblue")
ax.plot(operational_intensity_range, compute_limit_curve, label="Compute Peak (20 TFLOPs)", linestyle=":", color="lightcoral")


logging.info("Plotting data points.")
ax.scatter(data_frame["Operational Intensity"], data_frame["Performance (GFLOPs)"], color="orange", marker="o", s=100, label="Kernels")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Operational Intensity (FLOPs/Byte)", fontsize=11)
ax.set_ylabel("Performance (GFLOPs)", fontsize=11)
ax.set_title("Roofline (Squeezenet1_1)", fontsize=14)  # Changed title to "Roofline"
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)  # Position legend at the bottom
ax.grid(False)
plt.tight_layout()
logging.info("Displaying the plot.")
plt.show()

output_file = "roofline_squeezenet1_1.png"
fig.savefig(output_file, bbox_inches='tight')
logging.info(f"Plot saved as {output_file}.")

# Save the plot to a file
output_file = "roofline_squeezenet1_1.png"
fig.savefig(output_file, bbox_inches='tight')
logging.info(f"Plot saved as {output_file}.")
