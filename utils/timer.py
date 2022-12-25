import time

class Timer:

    def __init__(self):
        self.st = -1
        self.nt = -1


    def tic(self):
        self.st = time.perf_counter()

    def toc(self, **kwargs):
        self.nt = time.perf_counter()
        self.print(**kwargs)

    def print(self, **kwargs):
        print(f"time taken is {self.nt - self.st} (seconds)")
        if kwargs:
            print(f"   {kwargs}")