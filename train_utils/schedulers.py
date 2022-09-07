
class no_schedule: 
    def __init__(self) -> None:
        pass 
    def step(self): 
        pass

class decay_schedule: 
    def __init__(self, optimizer, config) -> None:
        schedule_params = config['lr_scheduling']['decay']
        self.optimizer = optimizer
        self.interval = schedule_params['interval']
        self.decay_rate = schedule_params['decay_rate']
        # if the model is competitive
        self.optim_params = config['lr_scheduling']['params']
        self.count = 0
        
    def step(self): 
        self.count += 1
        if self.count == self.interval: 
            self.count = 0
            if 'lr_min' in self.optim_params: 
                self.optimizer.state['lr_min'] /= self.decay_rate
            if 'lr_max' in self.optim_params: 
                self.optimizer.state['lr_max'] /= self.decay_rate
                if self.optimizer.state['lr_max'] < self.optimizer.state['lr_min']: 
                    self.optimizer.state['lr_max'] = self.optimizer.state['lr_min']  
    