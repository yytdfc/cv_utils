import time
import os
import inspect


class timer():
    def __init__(self, name=''):
        if name:
            self.name = name
        else:
            frame = inspect.stack()[1]
            filename = frame[0].f_code.co_filename
            lineno = frame[0].f_lineno
            self.name = f'[{os.path.basename(filename)}:{lineno}]'

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, a, b, c):
        print(self.name, f'using {time.time() - self.tic:.6f} seconds.')
