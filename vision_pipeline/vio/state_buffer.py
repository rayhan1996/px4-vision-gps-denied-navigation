from collections import deque


class StateBuffer:
    """
    Sliding window state storage.
    """

    def __init__(self, max_size=20):

        self.buffer = deque(maxlen=max_size)

    def append(self, state):

        self.buffer.append(state)

    def get_all(self):

        return list(self.buffer)

    def latest(self):

        if len(self.buffer) == 0:
            return None

        return self.buffer[-1]
