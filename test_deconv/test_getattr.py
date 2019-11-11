class ProcessWrapper(object):
    def __init__(self):
        pass

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            self.send_environment(name, *args, **kwargs)
        return wrapper

    def send_environment(self, name, *args, a=8, **kwargs):
        


p=ProcessWrapper()
p.senf_if(34)