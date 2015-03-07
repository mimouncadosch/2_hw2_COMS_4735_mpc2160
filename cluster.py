import numpy as np

def cluster():

    D = np.array([[0.0, 0.5, 0.1, 0.2], [0.5, 0.0, 0.4, 0.6], [0.1, 0.4, 0.0, 0.3], [0.2, 0.6, 0.3, 0.0]])



    D[D == 0.0] = 2 # pick np.max

    # D = np.diff(1, S)




    return True


if __name__ == "__main__":
    cluster()