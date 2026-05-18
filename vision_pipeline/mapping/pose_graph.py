import numpy as np


class PoseGraph:
    """
    Simple pose graph for SLAM.
    """

    def __init__(self):

        self.nodes = []

        self.edges = []

    def add_node(self, pose):

        self.nodes.append(pose)

    def add_edge(
        self,
        from_id,
        to_id,
        relative_pose
    ):

        self.edges.append({
            "from": from_id,
            "to": to_id,
            "relative_pose": relative_pose
        })

    def optimize(self):
        """
        Placeholder for graph optimization.
        """

        print("Pose graph optimization not implemented yet")
