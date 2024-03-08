"""
Creates a two-dimensional maze where each cell in the maze is defined by the sum of existing exits
(North = 1, East = 2, South = 4, West = 8), with each cell corresponding to a value stored in a two-dimensional array.
"""
import numpy as np
import pygame as pg

NORTH = 1
EAST  = 2
SOUTH = 4
WEST  = 8


class Maze:
    """
    Builds a maze of given dimensions by building the NumPy array maze describing the maze.

    Inputs:
        dimensions: Tuple containing two integers describing the height and length of the maze.
        seed: The random seed used to generate the maze. The same seed produces the same maze.
    """
    def __init__(self, dimensions, seed):
        self.cases_img = []
        self.maze  = np.zeros(dimensions, dtype=np.int8)
        is_visited = np.zeros(dimensions, dtype=np.int8)
        historic = []

        # We choose the central cell as the initial cell.
        cur_ind = (dimensions[0]//2, dimensions[1]//2)
        historic.append(cur_ind)
        while (len(historic) > 0):
            cur_ind = historic[-1]
            is_visited[cur_ind] = 1
            # First, we check if there is at least one unvisited neighboring cell of the current cell:
            #   1. Calculating the neighbors of the current cell:
            neighbours         = []
            neighbours_visited = []
            direction          = []
            if cur_ind[1] > 0 and is_visited[cur_ind[0], cur_ind[1]-1] == 0:  # West cell no visited
                neighbours.append((cur_ind[0], cur_ind[1]-1))
                direction.append((WEST, EAST))
            if cur_ind[1] < dimensions[1]-1 and is_visited[cur_ind[0], cur_ind[1]+1] == 0:  # East cell
                neighbours.append((cur_ind[0], cur_ind[1]+1))
                direction.append((EAST, WEST))
            if cur_ind[0] < dimensions[0]-1 and is_visited[cur_ind[0]+1, cur_ind[1]] == 0:  # South cell
                neighbours.append((cur_ind[0]+1, cur_ind[1]))
                direction.append((SOUTH, NORTH))
            if cur_ind[0] > 0 and is_visited[cur_ind[0]-1, cur_ind[1]] == 0:  # North cell
                neighbours.append((cur_ind[0]-1, cur_ind[1]))
                direction.append((NORTH, SOUTH))
            if len(neighbours) > 0:  # In this case, a cell is non visited
                neighbours = np.array(neighbours)
                direction  = np.array(direction)
                seed = (16807*seed) % 2147483647
                chosen_dir = seed % len(neighbours)
                dir        = direction[chosen_dir]
                historic.append((neighbours[chosen_dir, 0], neighbours[chosen_dir, 1]))
                self.maze[cur_ind] |= dir[0]
                self.maze[neighbours[chosen_dir, 0], neighbours[chosen_dir, 1]] |= dir[1]
                is_visited[cur_ind] = 1
            else:
                historic.pop()
       

    def display(self):
        """
        Create a picture of the maze :
        """
         #  Load patterns for maze display :
        img = pg.image.load("cases.png").convert_alpha()
        for i in range(0, 128, 8):
            self.cases_img.append(pg.Surface.subsurface(img, i, 0, 8, 8))
            
        maze_img = pg.Surface((8*self.maze.shape[1], 8*self.maze.shape[0]), flags=pg.SRCALPHA)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                maze_img.blit(self.cases_img[self.maze[i, j]], (j*8, i*8))

        return maze_img


if __name__  == "__main__":
    import time
    dimensions = (50, 80)
    pg.init()
    resolution = dimensions[1]*8, dimensions[0]*8
    print(f"resolution : {resolution}")
    screen = pg.display.set_mode(resolution)

    t1 = time.time()
    maze = Maze(dimensions, 12345)
    t2 = time.time()
    print(f"Temps construction labyrinthe : {t2-t1} secondes")

    screen.fill((255, 255, 255))
    mazeImg = maze.display()
    screen.blit(mazeImg, (0, 0))
    pg.display.update()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
