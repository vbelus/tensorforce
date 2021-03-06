from tensorforce.environments import Environment
from tensorforce import TensorforceError, util

from multiprocessing import Process
from multiprocessing import Pipe

import importlib
import json
import os
from threading import Thread

def worker(environment, conn2):
    while True:
        # Receive the environment method's name, and (optional) arguments
        (name, *args, kwargs) = conn2.recv()

        attr = object.__getattribute__(environment, name)
        if hasattr(attr, '__call__'):
            # Get what is returned by the method
            result = attr(*args, **kwargs)
        else:
            # Get the attribute
            result = attr

        # Send this to the Wrapper
        conn2.send(result)

class ProcessWrapper(Environment):
    def __init__(self, environment):
        super(ProcessWrapper, self).__init__()

        # Instanciate a bidirectional Pipe
        self.conn1, conn2 = Pipe(duplex=True)

        # Start the worker process, which interacts directly with the environment
        self.p = Process(target=worker, args=(environment, conn2))
        self.p.start()

    # To call a custom method of your environment
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            return self.send_environment(name, *args, **kwargs)
        return wrapper

    def send_environment(self, name, *args, **kwargs):
        """Send a request through the pipe, and wait for the answer message.
        """
        to_send = (name, *args, kwargs)

        # Send request
        self.conn1.send(to_send)

        # Wait for the result
        result = self.conn1.recv()

        return result
    
    def close(self):
        self.p.join()
        self.send_environment('close')

    def states(self):
        return self.send_environment('states')

    def actions(self):
        return self.send_environment('actions')

    def reset(self):
        return self.send_environment('reset')

    def execute(self, actions):
        return self.send_environment('execute', actions)
