import numpy as np
import cv2
from random import choice, randint, sample
from dataclasses import dataclass
from enum import Enum
import math

class SnakeState(Enum):
    OK = 1
    ATE = 2
    DED = 3
    WON = 4


def _rotate_image(cv_image, _rotation_angle):
    axes_order = (1, 0, 2) if len(cv_image.shape) == 3 else (1, 0)
    if _rotation_angle == -90:
        return np.transpose(cv_image, axes_order)[:, ::-1]

    if _rotation_angle == 90:
        return np.transpose(cv_image, axes_order)[::-1, :]

    if _rotation_angle in [-180, 180]:
        return cv_image[::-1, ::-1]

    return cv_image


@dataclass(eq=True, frozen=True)
class Point:
    x: int
    y: int

    def copy(self, xincr, yincr):
        return Point(self.x + xincr, self.y + yincr)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['x'], d['y'])

    def __repr__(self):
        return f"(x: {self.x}, y: {self.y})"

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y)

    def dist(self, other):
        m = Point(self.x-other.x, self.y-other.y)
        return abs(m.x) + abs(m.y)

action_dir_map = {
    'up': Point(0, -1),
    'down': Point(0, 1),
    'left': Point(-1, 0),
    'right': Point(1, 0),
}

dir_map_to_angle = {
    Point(0, -1): 0,
    Point(0, 1): 180,
    Point(-1, 0): 90,
    Point(1, 0): -90,
}

sprites = {
    'head': cv2.imread('./sprites/head.png', 0),
    'body': cv2.imread('./sprites/body.png', 0),
    'turn': cv2.imread('./sprites/turn.png', 0),
    'fruit': cv2.imread('./sprites/fruit.png', 0),
    'tail': cv2.imread('./sprites/tail.png', 0),
}

action_dir_order = ['right', 'up', 'left', 'down']


class Env:
    def __init__(self, grid_size=10, main_gs=10, num_fruits=10):
        self.gs = grid_size
        self.subgrid_loc = None
        self.main_gs = main_gs
        self.num_fruits = num_fruits
        self.reset()

        self.update()

    def reset(self):
        self.step = 0
        self.last_ate = 0
        grid_size = self.gs
        # self.subgrid_loc = Point(randint(0, self.main_gs - self.gs), randint(0, self.main_gs - self.gs))
        self.subgrid_loc = Point(0, 0)
        if grid_size in [10, 20, 38]:
            self.subgrid_loc = Point(1, 1)
        self.snake = Snake()
        self.snake.head = Point(self.gs//2, self.gs//2)

        pos_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                pos_list.append(Point(i, j))

        self.pos_set = set(pos_list)
        self.fruit_locations = []
        self.set_fruits()

    @property
    def stamina(self):
        a = self.gs ** 2
        stamina = a + len(self.snake.tail) + 1
        stamina = min(a * 2, stamina)
        return stamina

    def to_dict(self):
        return {
            'snake': self.snake.to_dict(),
            'fruit': self.fruit_loc.to_dict()
        }


    def from_dict(self, d):
        self.snake = Snake.from_dict(d['snake'])
        self.fruit_location = Point.from_dict(d['fruit'])


    def update(self, direction=None):
        self.last_ate += 1
        snake = self.snake
        self.snake.apply_direction(direction)
        self.snake.update()
        out_enum = SnakeState.OK

        if snake.head in self.fruit_locations:
            self.fruit_locations.pop(self.fruit_locations.index(snake.head))
            self.last_ate = 0
            try:
                self.set_fruits()
                self.snake.tail_size += 1
                out_enum = SnakeState.ATE
            except IndexError:
                out_enum = SnakeState.WON
            if len(self.fruit_locations) == 0:
                out_enum = SnakeState.WON
        self.snake.shed()
        if not self._bounds_check(snake.head) or self.snake.self_collision():
            out_enum = SnakeState.DED
        elif self.last_ate > self.stamina:
            out_enum = SnakeState.DED

        return out_enum

    @property
    def fruit_loc(self):
        return self.fruit_locations

    def set_fruits(self):
        snake = self.snake
        snake_locs = set([snake.head] + snake.tail + self.fruit_locations)
        possible_positions = self.pos_set.difference(snake_locs)
        diff = self.num_fruits - len(self.fruit_locations)
        new_locs = sample(list(possible_positions), k=min(diff, len(possible_positions)))
        self.fruit_locations.extend(new_locs)

    def _bounds_check(self, pos):
        return pos.x >= 0 and pos.x < self.gs and pos.y >= 0 and pos.y < self.gs

    def to_image(self, gradation=True):
        snake = self.snake
        fl = self.fruit_loc
        scale = 8

        full_canvas = np.zeros((self.main_gs*scale, self.main_gs*scale), 'uint8')
        h, w = self.gs*8, self.gs*8
        canvas = full_canvas[self.subgrid_loc.y*scale:self.subgrid_loc.y*scale+h,
                             self.subgrid_loc.x*scale:self.subgrid_loc.x*scale+w]
        canvas += 64

        def apply_rotation(im, angle):
            return _rotate_image(im, angle)

        def draw_sprite(canvas, y, x, stype, scale=8, rotation=0):
            s = scale
            canvas[y*s:(y+1)*s, x*s:(x+1)*s] = apply_rotation(sprites[stype], rotation)

        for f in fl:
            draw_sprite(canvas, f.y, f.x, 'fruit')

        if self._bounds_check(snake.head):
            draw_sprite(canvas, snake.head.y, snake.head.x, 'head',
                        rotation=dir_map_to_angle[self.snake.direction])

        limbs = [snake.head] + list(reversed(snake.tail))
        for nxt, curr, prev in zip(limbs, limbs[1:], limbs[2:]):
            d2 = curr - prev
            d1 = nxt - curr
            if d1 == d2:
                draw_sprite(canvas, curr.y, curr.x, 'body',
                            rotation=dir_map_to_angle[d2])
                continue

            rotation = None

            d2 = curr - prev
            d1 = nxt - curr

            if (d1.x > 0 and d2.y < 0) or (d1.y > 0 and d2.x < 0):
                rotation = 0
            elif (d1.y > 0 and d2.x > 0) or (d1.x < 0 and d2.y < 0):
                rotation = -90
            elif (d1.x > 0 and d2.y > 0) or (d1.y < 0 and d2.x < 0):
                rotation = 90
            elif (d1.y < 0 and d2.x > 0) or (d1.x < 0 and d2.y > 0):
                rotation = 180

            if rotation is not None:
                draw_sprite(canvas, curr.y, curr.x, 'turn',
                            rotation=rotation)

        if len(limbs) > 1:
            draw_sprite(canvas, limbs[-1].y, limbs[-1].x, 'tail', rotation=dir_map_to_angle[limbs[-2]-limbs[-1]])

        return full_canvas

class Snake:
    def __init__(self, x: int = 0, y: int = 0):
        self.head = Point(x, y)
        self.tail = []
        self.tail_size = 1
        self.direction = Point(1, 0)  # Need to add validation later
        self.dir_idx = 0

    def self_collision(self):
        for t in self.tail:
            if self.head.x == t.x and self.head.y == t.y:
                return True
        return False

    def to_dict(self):
        return {
            'head': self.head.to_dict(),
            'tail': [t.to_dict() for t in self.tail],
            'tail_size': self.tail_size,
            'direction': self.direction.to_dict()
        }

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.head = Point.from_dict(d['head'])
        s.tail = [Point.from_dict(t) for t in d['tail']]
        s.tail_size = d['tail_size']
        s.direction = Point.from_dict(d['direction'])
        return s

    def update(self):
        new_head = self.head.copy(self.direction.x, self.direction.y)

        self.tail.append(self.head)  # OK direction? or do I need to add this to the top?
        self.head = new_head

    def shed(self):
        if self.tail_size > 0:
            self.tail = self.tail[-self.tail_size:]
        else:
            self.tail = []

    def __repr__(self):
        return f"""Head: {self.head}
        Tail: {self.tail}
        Dir: {self.direction}
        """

    def apply_turn(self, turn_dir):
        if not turn_dir:
            return
        assert turn_dir in ['left', 'right']
        shift = 1 if turn_dir == 'left' else -1
        self.dir_idx = (self.dir_idx + shift) % 4
        action = action_dir_order[self.dir_idx]
        self.apply_direction(new_dir=action)

    def apply_direction(self, new_dir=None):
        if not new_dir:
            return
        assert new_dir in action_dir_map, f"Unknown direction {new_dir}"

        self.direction = action_dir_map[new_dir]


if __name__ == '__main__':
    import cv2
    # s = Snake()
    env = Env(4)

    cv2.imwrite('/home/jack/test.png', cv2.resize(env.to_image(), (640, 640), interpolation=cv2.INTER_NEAREST))

    while True:
        n = input()
        print(env.update(n))
        cv2.imwrite('/home/jack/test.png', cv2.resize(env.to_image(), (640, 640), interpolation=cv2.INTER_NEAREST))


    # env controls the snake now

    # env.set_fruits(s)

    # for i in range(50):
    #     s.update()
    #     assert env.bounds_check(s.head)

    # s.apply_direction('down')
    # s.update()

    # import cv2
