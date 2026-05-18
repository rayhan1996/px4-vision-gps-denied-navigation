import numpy as np

from vision_pipeline.mapping.landmark_manager import (
    LandmarkManager
)

from vision_pipeline.mapping.keyframe_manager import (
    KeyframeManager
)

from vision_pipeline.mapping.keyframe import (
    Keyframe
)

from vision_pipeline.mapping.pose_graph import (
    PoseGraph
)

from vision_pipeline.mapping.loop_closure import (
    LoopClosureDetector
)


class MapBuilder:
    """
    Main SLAM map builder.
    """

    def __init__(self):

        self.landmarks = LandmarkManager()

        self.keyframes = KeyframeManager()

        self.pose_graph = PoseGraph()

        self.loop_closure = (
            LoopClosureDetector()
        )

    def add_keyframe(
        self,
        frame_id,
        image,
        position,
        rotation
    ):

        keyframe = Keyframe(
            frame_id,
            image,
            position,
            rotation
        )

        self.keyframes.add_keyframe(
            keyframe
        )

        self.pose_graph.add_node({
            "position": position,
            "rotation": rotation
        })

        return keyframe

    def add_landmark(
        self,
        position
    ):

        landmark = (
            self.landmarks.create_landmark(
                position
            )
        )

        return landmark

    def check_loop_closure(self):

        keyframes = self.keyframes.get_all()

        if len(keyframes) < 5:
            return False

        current = keyframes[-1]

        previous_positions = [
            k.position
            for k in keyframes[:-1]
        ]

        detected, idx = (
            self.loop_closure.detect(
                current.position,
                previous_positions
            )
        )

        if detected:

            self.pose_graph.add_edge(
                idx,
                len(keyframes) - 1,
                current.position
            )

            print(
                f"Loop closure detected with keyframe {idx}"
            )

            return True

        return False
