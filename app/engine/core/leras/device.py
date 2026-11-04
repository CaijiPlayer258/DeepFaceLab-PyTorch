import sys
import os
import multiprocessing
import json
import time
from pathlib import Path
from core.interact import interact as io


class Device(object):
    def __init__(self, index, device_type, name, total_mem, free_mem):
        self.index = index
        self.device_type = device_type
        self.name = name
        
        self.total_mem = total_mem
        self.total_mem_gb = total_mem / 1024**3
        self.free_mem = free_mem
        self.free_mem_gb = free_mem / 1024**3

    def __str__(self):
        return f"[{self.index}]:[{self.name}][{self.free_mem_gb:.3}/{self.total_mem_gb :.3}]"

class Devices(object):
    all_devices = None

    @staticmethod
    def _enumerate_torch_devices_dict():
        """Return dict[index] = (type, name, total_mem). Safe to call in-process."""
        if sys.platform[0:3] == 'win':
            compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('ComputeCache_ALL')
            os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
            if not compute_cache_path.exists():
                io.log_info("正在缓存 GPU 内核...")
                compute_cache_path.mkdir(parents=True, exist_ok=True)

        import torch

        devices = []

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    'index': i,
                    'type': 'GPU',
                    'name': props.name,
                    'total_mem': props.total_memory,
                    'free_mem': props.total_memory,
                })

        try:
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            mps_available = False

        if mps_available and len(devices) == 0:
            total_mem = 0
            try:
                page_size = os.sysconf('SC_PAGE_SIZE')
                phys_pages = os.sysconf('SC_PHYS_PAGES')
                total_mem = int(page_size) * int(phys_pages)
            except Exception:
                total_mem = 0

            devices.append({
                'index': 0,
                'type': 'MPS',
                'name': 'Apple MPS',
                'total_mem': total_mem,
                'free_mem': total_mem,
            })

        out = {}
        for dev in devices:
            out[dev['index']] = (dev['type'], dev['name'], dev['total_mem'])
        return out

    def __init__(self, devices):
        self.devices = devices

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, key):
        result = self.devices[key]
        if isinstance(key, slice):
            return Devices(result)
        return result

    def __iter__(self):
        for device in self.devices:
            yield device

    def get_best_device(self):
        result = None
        idx_mem = 0
        for device in self.devices:
            mem = device.total_mem
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_worst_device(self):
        result = None
        idx_mem = sys.maxsize
        for device in self.devices:
            mem = device.total_mem
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_device_by_index(self, idx):
        for device in self.devices:
            if device.index == idx:
                return device
        return None

    def get_devices_from_index_list(self, idx_list):
        result = []
        for device in self.devices:
            if device.index in idx_list:
                result += [device]
        return Devices(result)

    def get_equal_devices(self, device):
        device_name = device.name
        result = []
        for device in self.devices:
            if device.name == device_name:
                result.append (device)
        return Devices(result)

    def get_devices_at_least_mem(self, totalmemsize_gb):
        result = []
        for device in self.devices:
            if device.total_mem >= totalmemsize_gb*(1024**3):
                result.append (device)
        return Devices(result)

    @staticmethod
    def _get_torch_devices_proc(q : multiprocessing.Queue):
        
        physical_devices_f = Devices._enumerate_torch_devices_dict()
                        
        q.put(physical_devices_f)
        time.sleep(0.1)
        
        
    @staticmethod
    def initialize_main_env():
        if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 0:
            return
            
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        
        os.environ['CUDA_​CACHE_​MAXSIZE'] = '2147483647'
        
        # On macOS (and Python 3.14+), spawn can fail when __main__ has no real file
        # (e.g. python -c / stdin). Fall back to in-process enumeration in that case.
        use_subprocess = True
        try:
            import __main__
            main_file = getattr(__main__, '__file__', None)
            use_subprocess = bool(main_file) and Path(main_file).exists()
        except Exception:
            use_subprocess = False

        if use_subprocess:
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=Devices._get_torch_devices_proc, args=(q,), daemon=True)
            p.start()
            p.join()
            visible_devices = q.get()
        else:
            visible_devices = Devices._enumerate_torch_devices_dict()

        os.environ['NN_DEVICES_INITIALIZED'] = '1'
        os.environ['NN_DEVICES_COUNT'] = str(len(visible_devices))
        
        for i in visible_devices:
            dev_type, name, total_mem = visible_devices[i]

            os.environ[f'NN_DEVICE_{i}_DEV_TYPE'] = dev_type
            os.environ[f'NN_DEVICE_{i}_NAME'] = name
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(total_mem)
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(total_mem)
            
        

    @staticmethod
    def getDevices():
        if Devices.all_devices is None:
            if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 1:
                raise Exception("nn devices are not initialized. Run initialize_main_env() in main process.")
            devices = []
            for i in range ( int(os.environ['NN_DEVICES_COUNT']) ):
                devices.append ( Device(index=i,
                                        device_type=os.environ[f'NN_DEVICE_{i}_DEV_TYPE'],
                                        name=os.environ[f'NN_DEVICE_{i}_NAME'],
                                        total_mem=int(os.environ[f'NN_DEVICE_{i}_TOTAL_MEM']),
                                        free_mem=int(os.environ[f'NN_DEVICE_{i}_FREE_MEM']), )
                                )
            Devices.all_devices = Devices(devices)

        return Devices.all_devices

"""

        
        # {'name'      : name.split(b'\0', 1)[0].decode(),
        #     'total_mem' : totalMem.value
        # }

        
        
        
        
        return

        
        
        
        min_cc = int(os.environ.get("TF_MIN_REQ_CAP", 35))
        libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll')
        for libname in libnames:
            try:
                cuda = ctypes.CDLL(libname)
            except:
                continue
            else:
                break
        else:
            return Devices([])

        nGpus = ctypes.c_int()
        name = b' ' * 200
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        freeMem = ctypes.c_size_t()
        totalMem = ctypes.c_size_t()

        result = ctypes.c_int()
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        error_str = ctypes.c_char_p()

        devices = []

        if cuda.cuInit(0) == 0 and \
            cuda.cuDeviceGetCount(ctypes.byref(nGpus)) == 0:
            for i in range(nGpus.value):
                if cuda.cuDeviceGet(ctypes.byref(device), i) != 0 or \
                    cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) != 0 or \
                    cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != 0:
                    continue

                if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device) == 0:
                    if cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem)) == 0:
                        cc = cc_major.value * 10 + cc_minor.value
                        if cc >= min_cc:
                            devices.append ( {'name'      : name.split(b'\0', 1)[0].decode(),
                                              'total_mem' : totalMem.value,
                                              'free_mem'  : freeMem.value,
                                              'cc'        : cc
                                              })
                    cuda.cuCtxDetach(context)

        os.environ['NN_DEVICES_COUNT'] = str(len(devices))
        for i, device in enumerate(devices):
            os.environ[f'NN_DEVICE_{i}_NAME'] = device['name']
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(device['total_mem'])
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(device['free_mem'])
            os.environ[f'NN_DEVICE_{i}_CC'] = str(device['cc'])
"""