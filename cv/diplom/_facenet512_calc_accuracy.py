from .recognition_lib.accuracy_calc import calc_accuracy
from .config import DB_PATH

if __name__ == '__main__':
    print(calc_accuracy(DB_PATH))
