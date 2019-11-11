class Foo(object):
    def __init__(self, bar):
        self.bar = bar
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            print("{} was called".format(name))
        return wrapper

    def hey(self):
        print('hey')

class Bar():
    def to_print(self):
        print("hey")

bar = Bar()
f = Foo(bar)
f.to_print()
f.hey()