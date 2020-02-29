import numpy as np
import cv2
from random import choice, randint
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

action_dir_order = ['right', 'up', 'left', 'down']


def draw_boxes(canvas, j, i, color, padding=2, scale=4):
    canvas[j*scale+padding:j*scale+(scale-padding), i*scale+padding:i*scale+(scale-padding)] = color

def connect_boxes(canvas, j1, i1, j2, i2, color, padding=2, scale=4):
    J1 = min(j1*scale+padding, j1*scale+(scale-padding))
    J2 = max(j2*scale+padding, j2*scale+(scale-padding))
    I1 = min(i1*scale+padding, i1*scale+(scale-padding))
    I2 = max(i2*scale+padding, i2*scale+(scale-padding))
    canvas[J1:J2, I1:I2] = color

class Env:
    def __init__(self, grid_size=10, main_gs=10):
        self.gs = grid_size
        # self.subgrid_loc = None
        self.main_gs = main_gs
        self.reset()

    def reset(self):
        self.step = 0
        self.last_ate = 0
        # if (not self.subgrid_loc): # or (self.subgrid_loc and self.rand_grid_loc_always):
        # self.gs = randint(5, 40)
        grid_size = self.gs
        # self.subgrid_loc = Point(randint(0, self.main_gs - self.gs), randint(0, self.main_gs - self.gs))
        self.snake = Snake()

        pos_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                pos_list.append(Point(i, j))

        self.pos_set = set(pos_list)
        self.fruit_location = None
        self.set_fruit()

    @property
    def stamina(self):
        a = self.gs ** 2
        stamina = a + len(self.snake.tail) + 1
        stamina = min(a * 2, stamina)
        return stamina

    def update(self, direction=None):
        self.last_ate += 1
        snake = self.snake
        self.snake.apply_turn(direction)
        self.snake.update()
        out_enum = SnakeState.OK


        if snake.head == self.fruit_location:
            self.last_ate = 0
            try:
                self.set_fruit()
                self.snake.tail_size += 1
                out_enum = SnakeState.ATE
            except IndexError:
                out_enum = SnakeState.WON
        self.snake.shed()
        if not self._bounds_check(snake.head) or self.snake.self_collision():
            out_enum = SnakeState.DED
        elif self.last_ate > self.stamina:
            out_enum = SnakeState.DED

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

    def to_image(self):
        snake = self.snake
        # out_main = np.zeros((self.main_gs, self.main_gs, 3), 'uint8')
        # l = self.subgrid_loc
        # out = out_main[l.y:l.y+self.gs, l.x:l.x+self.gs]
        # out[:, :] = 32
        fl = self.fruit_loc
        # out[fl.y, fl.x] = 255
        # if self._bounds_check(snake.head):
        #     out[snake.head.y, snake.head.x] = 128 + 32

        # for i, s in enumerate(reversed(snake.tail)):
        #     if self._bounds_check(s):
        #         out[s.y, s.x] = 128

        # return cv2.resize(out_main, (128, 128), interpolation=0)[:, :, 0:1]

        scale = 8

        canvas = np.zeros((self.gs*scale, self.gs*scale), 'uint8') + 32

        draw_boxes(canvas, fl.y, fl.x, 255, scale=scale, padding=1)

        if self._bounds_check(snake.head):
            draw_boxes(canvas, snake.head.y, snake.head.x, 128, 1, scale=scale)

        last_el = snake.head

        for i, s in enumerate(reversed(snake.tail)):
            if self._bounds_check(s) and self._bounds_check(last_el):
                # draw_boxes(canvas, s.y, s.x, 128, scale=scale)
                connect_boxes(canvas, last_el.y, last_el.x, s.y, s.x,128, scale=scale, padding=2)
                connect_boxes(canvas, s.y, s.x,last_el.y, last_el.x, 128, scale=scale, padding=2)
                last_el = s

        return np.expand_dims(cv2.resize(canvas, (128, 128), interpolation=0), -1)

class Snake:
    def __init__(self):
        self.head = Point(0, 0)
        self.tail = []
        self.tail_size = 2
        self.direction = Point(1, 0)  # Need to add validation later
        self.dir_idx = 0

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

    # env.set_fruit(s)

    # for i in range(50):
    #     s.update()
    #     assert env.bounds_check(s.head)

    # s.apply_direction('down')
    # s.update()

    # import cv2
