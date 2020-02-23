import numpy as np
import cv2
from random import choice
from dataclasses import dataclass
from enum import Enum

class SnakeState(Enum):
    OK = 1
    ATE = 2
    DED = 3
    WON = 4

@dataclass(eq=True, frozen=True)
class Point:
    x: int
    y: int

    def copy(self, xincr, yincr):
        return Point(self.x + xincr, self.y + yincr)

    def __repr__(self):
        return f"(x: {self.x}, y: {self.y})"

action_dir_map = {
    'up': Point(0, -1),
    'down': Point(0, 1),
    'left': Point(-1, 0),
    'right': Point(1, 0)
}

class Env:
    def __init__(self, grid_size=40):
        self.gs = grid_size
        self.reset()

    def reset(self):
        grid_size = self.gs
        self.snake = Snake()

        pos_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                pos_list.append(Point(i, j))

        self.pos_set = set(pos_list)
        self.fruit_location = None
        self.set_fruit()


    def update(self, direction=None):
        snake = self.snake
        self.snake.apply_direction(direction)
        self.snake.update()
        out_enum = SnakeState.OK

        if not self._bounds_check(snake.head) or self.snake.self_collision():
            out_enum = SnakeState.DED
        elif snake.head == self.fruit_location:
            try:
                self.set_fruit()
                self.snake.tail_size += 1
                out_enum = SnakeState.ATE
            except IndexError:
                out_enum = SnakeState.WON

        self.snake.shed()

        return out_enum

    @property
    def fruit_loc(self):
        assert self.fruit_location is not None, "Fruit hasn't been initialized"
        return self.fruit_location

    def set_fruit(self):
        snake = self.snake
        snake_locs = set([snake.head] + snake.tail)
        possible_positions = self.pos_set.difference(snake_locs)
        self.fruit_location = choice(list(possible_positions))

    def _bounds_check(self, pos):
        return pos.x >= 0 and pos.x < self.gs and pos.y >= 0 and pos.y < self.gs

    def to_image(self, gradation=False):
        snake = self.snake
        out = np.zeros((self.gs, self.gs, 3), 'uint8')
        fl = self.fruit_loc
        out[fl.y, fl.x] = 255

        l = ([snake.head] + snake.tail[::-1])[::-1]

        for i, s in enumerate(l):
            if self._bounds_check(s):
                if gradation:
                    out[s.y, s.x] = 100 + 100.0/len(l)*i
                else:
                    out[s.y, s.x] = 128

        return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


class Snake:
    def __init__(self):
        self.head = Point(0, 0)
        self.tail = []
        self.tail_size = 2
        self.direction = Point(1, 0)  # Need to add validation later

    def self_collision(self):
        for t in self.tail:
            if self.head.x == t.x and self.head.y == t.y:
                return True
        return False

    def update(self):
        new_head = self.head.copy(self.direction.x, self.direction.y)

        self.tail.append(self.head)  # OK direction? or do I need to add this to the top?
        self.head = new_head

    def shed(self):
        self.tail = self.tail[-self.tail_size:]

    def __repr__(self):
        return f"""Head: {self.head}
        Tail: {self.tail}
        Dir: {self.direction}
        """

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

    # env.set_fruit(s)

    # for i in range(50):
    #     s.update()
    #     assert env.bounds_check(s.head)

    # s.apply_direction('down')
    # s.update()

    # import cv2
