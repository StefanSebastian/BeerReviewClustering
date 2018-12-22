from random import randint


class RandomSeeds:
    def __init__(self, points, k):
        self.points = points
        self.k = k

    def get_seeds(self):
        means = []
        chosen = []
        while len(chosen) != self.k:
            random_idx = randint(0, len(self.points) - 1)
            if random_idx not in chosen:
                means.append(self.points[random_idx])
                chosen.append(random_idx)
        return means
