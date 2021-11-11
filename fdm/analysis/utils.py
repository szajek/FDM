

def create_weights_distributor(close_point_finder):
    def distribute(point, value):
        close_points = close_point_finder(point)
        distance_sum = sum(close_points.values())
        return dict(
            {p: (1. - distance/distance_sum)*value for p, distance in close_points.items()},
        )
    return distribute
