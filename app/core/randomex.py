import numpy as np

def random_normal(size=(1,), trunc_val=2.5, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
    n = int(np.array(size).prod())
    result = rnd_state.normal(size=n).astype(np.float32)
    np.clip(result, -trunc_val, trunc_val, out=result)
    result /= trunc_val
    return result.reshape(size)