import multiprocessing
import queue as Queue
import threading
import time
import traceback
import os


class SubprocessGenerator(object):
    
    @staticmethod
    def launch_thread(generator): 
        generator._start()
        
    @staticmethod
    def start_in_parallel( generator_list ):
        """
        Start list of generators in parallel
        """
        for generator in generator_list:
            thread = threading.Thread(target=SubprocessGenerator.launch_thread, args=(generator,) )
            thread.daemon = True
            thread.start()

        while not all ([generator._is_started() for generator in generator_list]):
            time.sleep(0.005)
    
    def __init__(self, generator_func, user_param=None, prefetch=2, start_now=True):
        super().__init__()
        self.prefetch = prefetch
        self.generator_func = generator_func
        self.user_param = user_param
        self.sc_queue = multiprocessing.Queue()
        self.cs_queue = multiprocessing.Queue()
        self.p = None
        self._closed = False
        if start_now:
            self._start()

    def close(self):
        """Terminate worker process and release IPC resources.

        This generator is infinite in normal training, so relying on StopIteration
        never triggers. Explicit close prevents multiprocessing semaphore leaks.
        """
        if self._closed:
            return
        self._closed = True

        p = self.p
        self.p = None
        if p is not None:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
            try:
                p.join(timeout=2.0)
            except Exception:
                pass

        for q in (getattr(self, 'sc_queue', None), getattr(self, 'cs_queue', None)):
            if q is None:
                continue
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

    def _start(self):
        if self.p == None:
            user_param = self.user_param
            self.user_param = None
            p = multiprocessing.Process(target=self.process_func, args=(user_param,) )
            p.daemon = True
            p.start()
            self.p = p
            
    def _is_started(self):
        return self.p is not None
        
    def process_func(self, user_param):
        try:
            seed_env = os.environ.get('DFL_SUBPROC_SEED', None)
            if seed_env is not None and str(seed_env).strip() != '':
                try:
                    base_seed = int(seed_env)
                except Exception:
                    base_seed = None

                if base_seed is not None:
                    # 尽量稳定：用进程名中的序号（Process-1/Process-2...）做偏移。
                    proc_name = multiprocessing.current_process().name
                    proc_idx = 0
                    try:
                        tail = proc_name.split('-')[-1]
                        proc_idx = int(tail)
                    except Exception:
                        proc_idx = 0

                    seed = int(base_seed + proc_idx)

                    try:
                        import random

                        random.seed(seed)
                    except Exception:
                        pass

                    try:
                        import numpy as np

                        np.random.seed(seed % (2**32 - 1))
                    except Exception:
                        pass

                    try:
                        import torch

                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                    except Exception:
                        pass

            self.generator_func = self.generator_func(user_param)
            while True:
                while self.prefetch > -1:
                    try:
                        gen_data = next(self.generator_func)
                    except StopIteration:
                        self.cs_queue.put(None)
                        return
                    self.cs_queue.put(gen_data)
                    self.prefetch -= 1
                self.sc_queue.get()
                self.prefetch += 1
        except Exception:
            # Propagate worker exception to parent to avoid deadlocks.
            try:
                self.cs_queue.put({'__error__': traceback.format_exc()})
            except Exception:
                pass
            return

    def __iter__(self):
        return self

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict

    def __next__(self):
        self._start()
        # Never block forever: periodically check worker liveness.
        while True:
            try:
                gen_data = self.cs_queue.get(timeout=0.2)
                break
            except Exception:
                p = self.p
                if p is not None:
                    try:
                        if not p.is_alive():
                            raise RuntimeError('SubprocessGenerator worker exited unexpectedly.')
                    except RuntimeError:
                        self.close()
                        raise
                    except Exception:
                        pass
                continue

        if isinstance(gen_data, dict) and gen_data.get('__error__') is not None:
            err = gen_data.get('__error__')
            self.close()
            raise RuntimeError(f'SubprocessGenerator worker error:\n{err}')
        if gen_data is None:
            try:
                if self.p is not None:
                    self.p.terminate()
                    self.p.join()
            except Exception:
                pass
            self.close()
            raise StopIteration()
        self.sc_queue.put (1)
        return gen_data

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
