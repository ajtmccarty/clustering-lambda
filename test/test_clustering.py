import typing as t
from collections import defaultdict

from src.clusters import do_clustering


class Point:
    def __init__(self, _id, coords, extra):
        self._id = _id
        self.coords = coords
        self.extra = extra

    def serialize(self):
        return {"coords": self.coords, "_id": self._id, "extra_data": self.extra}


class TestClustering:
    def setup_method(self):
        pass

    def __call_system_under_test(
        self, serial_points: t.List[t.Dict], max_size: t.Optional[int] = 5
    ) -> t.List[t.Dict]:
        return do_clustering(serial_points, max_size)

    def __verify_clusters(
        self, expected_clusters: t.List[t.List[t.Any]], returned_points: t.List[t.List]
    ):
        points_by_cluster = defaultdict(list)
        for p in returned_points:
            points_by_cluster[p["cluster"]].append(p["_id"])
        cluster_ids = list(points_by_cluster.values())
        assert cluster_ids == expected_clusters

    def __verify_unordered_clusters(
        self, expected_clusters: t.List[t.List[t.Any]], returned_points: t.List[t.List]
    ):
        points_by_cluster = defaultdict(list)
        for p in returned_points:
            points_by_cluster[p["cluster"]].append(p["_id"])
        cluster_ids = list(points_by_cluster.values())
        assert len(cluster_ids) == len(expected_clusters)
        for expected_c in expected_clusters:
            assert expected_c in cluster_ids

    def test_trivial_clustering(self):
        p1 = (1.0, 1.0)
        p2 = (5.0, 5.0)
        points = [
            Point(1, p1, {}).serialize(),
            Point(2, p1, {}).serialize(),
            Point(3, p1, {}).serialize(),
            Point(4, p1, {}).serialize(),
            Point(5, p2, {}).serialize(),
            Point(6, p2, {}).serialize(),
            Point(7, p2, {}).serialize(),
            Point(8, p2, {}).serialize(),
        ]
        expected_clusters = [[1, 2, 3, 4], [5, 6, 7, 8]]

        returned_points = self.__call_system_under_test(points)

        self.__verify_unordered_clusters(expected_clusters, returned_points)

    def test_max_size(self):
        p1 = (1.0, 1.0)
        p2 = (2.0, 2.0)
        points = [
            Point(1, p1, {}).serialize(),
            Point(2, p1, {}).serialize(),
            Point(3, p2, {}).serialize(),
            Point(4, p2, {}).serialize(),
        ]
        expected_clusters = [[1, 2], [3, 4]]

        returned_points = self.__call_system_under_test(points, max_size=2)

        self.__verify_unordered_clusters(expected_clusters, returned_points)

    def test_urgency_ordering(self):
        p1 = (1.0, 1.0)
        p2 = (2.0, 2.0)
        p3 = (3.0, 3.0)
        points = [
            Point(1, p1, {"is_urgent": True}).serialize(),
            Point(2, p1, {"is_urgent": True}).serialize(),
            Point(3, p1, {"is_urgent": True}).serialize(),
            Point(4, p2, {"is_urgent": True}).serialize(),
            Point(5, p2, {"is_urgent": True}).serialize(),
            Point(6, p2, {"is_urgent": False}).serialize(),
            Point(7, p3, {"is_urgent": True}).serialize(),
            Point(8, p3, {"is_urgent": False}).serialize(),
            Point(9, p3, {"is_urgent": False}).serialize(),
        ]
        expected_clusters = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        returned_points = self.__call_system_under_test(points, max_size=3)

        self.__verify_clusters(expected_clusters, returned_points)
