from nuscenes.utils.data_classes import PointCloud

class LidarPointCloud(PointCloud):
    @classmethod
    def from_points(cls, points) -> 'LidarPointCloud':
        cls.points = points
        return cls(points)

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 3

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        return None