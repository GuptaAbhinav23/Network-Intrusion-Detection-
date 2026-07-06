import time

class Timer:

    def __init__(self):
        self.start = None

    def tic(self):
        self.start =time.time()
    
    def toc(self):
        return  time.time()-self.start


if __name__ =="__main__":
    timer = Timer()

    model = ...

    timer.tic()

    model.fit(...)

    training_time = timer.toc()