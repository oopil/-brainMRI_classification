import datetime

class DatetimeDecorator:
    def __init__(self, f):
        self.func = f
    def __call__(self, *args, **kwargs):
        print
        datetime.datetime.now()
        self.func(*args, **kwargs)
        print
        datetime.datetime.now()

def datetime_decorator(func):
    def decorated():
            print(datetime.datetime.now())
            func()
            print(datetime.datetime.now())
    return decorated

def print_decorator(func):
    def decorated():
            print(datetime.datetime.now())
            func()
            print(datetime.datetime.now())
    return decorated