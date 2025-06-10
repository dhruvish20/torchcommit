import time 
# import torch 

# in this class, we want to take parameters like model, optimizer, loss function, schedular, checkpoints etc etc
# and create a function which has .start(), .end() and .log() methods to keep track of the model parameters and training process.
class ExperimentTracker:
    def init(self, model, optimizer, loss_function, scheduler=None, checkpoints=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.checkpoints = checkpoints
        self.start_time = None
        self.end_time = None
        self.logs = []


    def start(self):
        self.start_time = time.time()
        self.logs.append(f"Experiment started at {time.ctime(self.start_time)}")
        # print("Experiment started.")
    
    def end(self):
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        self.logs.append(f"Experiment ended at {time.ctime(self.end_time)}")
        self.logs.append(f"Total time taken: {total_time:.2f} seconds")
        # print("Experiment ended.")

    def log(self, message):
        timestamp = time.time()
        self.logs.append(f"{time.ctime(timestamp)}: {message}")
