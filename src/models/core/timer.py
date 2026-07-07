import time
import os
import psutil


class Timer:

    def __init__(self):

        self.start_time = None
        self.start_memory = None

    def tic(self):

        self.start_time = time.time()

        process = psutil.Process(os.getpid())

        self.start_memory = process.memory_info().rss

    def toc(self):

        process = psutil.Process(os.getpid())

        end_memory = process.memory_info().rss

        elapsed_time = time.time() - self.start_time

        memory_used = (end_memory - self.start_memory) / (1024 ** 2)

        return elapsed_time, memory_used


if __name__ =="__main__":
    timer = Timer()

    model = ...

    timer.tic()

    model.fit(...)

    training_time = timer.toc()