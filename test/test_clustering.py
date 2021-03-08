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

    def __call_system_under_test(self, serial_points: t.List[t.Dict]) -> t.List[t.Dict]:
        return do_clustering(serial_points)

    def __verify_clusters(
        self, expected_clusters: t.List[t.List[t.Any]], returned_points: t.List[t.List]
    ):
        points_by_cluster = defaultdict(list)
        for p in returned_points:
            points_by_cluster[p["cluster"]].append(p["_id"])
        cluster_ids = list(points_by_cluster.values())
        assert cluster_ids == expected_clusters

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

        self.__verify_clusters(expected_clusters, returned_points)
