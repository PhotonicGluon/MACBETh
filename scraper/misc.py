import time
import math


def pretty_sleep(duration: float, dot_length: int = 5, interval: float = 0.1):
    """
    A nicer version of the `sleep` function.

    :param duration: sleep duration in seconds
    :param dot_length: number of dots to display while sleeping, defaults to 5
    :param interval: interval between updating the number of dots in seconds, defaults to 0.1
    """
    iterations = math.ceil(duration / interval)
    for i in range(iterations):
        num_dots = i % dot_length + 1
        print(f"\33[2K\r{'.' * num_dots}", end="")
        time.sleep(interval)
    print()


if __name__ == "__main__":
    pretty_sleep(3)
