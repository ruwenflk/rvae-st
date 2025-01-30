# h# purpose: required for automatic gpu selection
import re, subprocess
import os

from shared import logger


def _run_command(cmd):
    # Run command, return output as string.
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def _list_available_gpus():
    # Returns list of available GPU ids. # TODO: remove unnecessary comments?
    output = _run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []

    for line in output.strip().split("\n"):
        regex_result = gpu_regex.match(line)
        assert regex_result, "Couldnt parse " + line
        result.append(int(regex_result.group("gpu_id")))

    return result


def _gpu_memory_map():
    # Returns map of GPU id to memory allocated on that GPU. # TODO: remove unnecessary comments?

    output = _run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory") :]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(
        r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB"  # TODO: remove unnecessary pid
    )
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in _list_available_gpus()}

    for row in rows:
        regex_result = memory_regex.search(row)

        if not regex_result:
            continue

        gpu_id = int(regex_result.group("gpu_id"))
        gpu_memory = int(regex_result.group("gpu_memory"))
        result[gpu_id] += gpu_memory

    return result


# use this function only if os.environ["CUDA_VISIBLE_DEVICES"] has exactly one device
def _enable_memory_growth():
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# change OS environment variable to force TF to only use a specific gpu, else it would use memory from all gpu devices
# this method shows orders by the memory used rather than the available memory.
def select_gpu_with_lowest_memory():
    if os.getenv("MAKE_GPU_SELECTION", "True") == "True":
        # Returns GPU ID (pci_bus order) with the least allocated memory # TODO: remove unnecessary comments?

        memory_gpu_map = [
            (memory, gpu_id) for (gpu_id, memory) in _gpu_memory_map().items()
        ]
        best_memory, best_gpu = sorted(memory_gpu_map)[0]

        logger.log_separator()
        logger.log(f"Chosing GPU: {best_gpu}, used memory: {best_memory} MiB")
        logger.log(
            f"Setting environment Variable CUDA_VISIBLE_DEVICES to GPU ID {best_gpu}"
        )
        logger.log_separator()

        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        _enable_memory_growth()


def autoselect_gpu():
    if os.getenv("MAKE_GPU_SELECTION", "True") == "True":
        _ = select_gpu_with_lowest_memory()
