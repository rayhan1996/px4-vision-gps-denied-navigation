from vision_pipeline.mapping.landmark import Landmark


class LandmarkManager:
    """
    Manage SLAM landmarks.
    """

    def __init__(self):

        self.landmarks = {}

        self.next_landmark_id = 0

    def create_landmark(self, position):

        landmark = Landmark(
            self.next_landmark_id,
            position
        )

        self.landmarks[
            self.next_landmark_id
        ] = landmark

        self.next_landmark_id += 1

        return landmark

    def get_landmark(self, landmark_id):

        return self.landmarks.get(
            landmark_id,
            None
        )

    def get_all_landmarks(self):

        return list(self.landmarks.values())
