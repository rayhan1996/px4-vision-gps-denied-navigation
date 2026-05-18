class KeyframeManager:
    """
    Manage SLAM keyframes.
    """

    def __init__(self):

        self.keyframes = []

    def add_keyframe(self, keyframe):

        self.keyframes.append(keyframe)

    def get_latest(self):

        if len(self.keyframes) == 0:
            return None

        return self.keyframes[-1]

    def get_all(self):

        return self.keyframes
