import time
import math


def pretty_sleep(duration, dot_length=5, interval=0.1):
    iterations = math.ceil(duration / interval)
    for i in range(iterations):
        num_dots = i % dot_length + 1
        print(f"\33[2K\r{'.' * num_dots}", end="")
        time.sleep(interval)
    print()


if __name__ == "__main__":
    pretty_sleep(3)
