import datetime

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