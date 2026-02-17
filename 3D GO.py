import numpy as np
import pygame
import sys
import time
import math
from math import sin, cos, radians

class Go3D:
    # Initialize the 3D Go game board and state variables
    def __init__(self, size=10):
        self.size = size  # Board size (size x size x size)
        self.board = np.zeros((size, size, size), dtype=int)  # 3D array representing the board: 0 empty, 1 black, 2 white
        self.current_player = 1  # 1: Black, 2: White
        self.consecutive_passes = 0  # Counter for consecutive passes to detect game end
        self.ko = None  # Position for ko rule (prevents immediate recapture)
        self.last_move = None  # Last played move for highlighting
        self.history = []  # List to store game states for undo
        self.komi = 0  # Komi compensation for white (typically for 2D Go, but included here)
        self.dead_positions = set()  # Set of dead stones marked at game end
        self.group_reps = {}  # Dictionary mapping positions to group representatives (for union-find like structure)
        self.group_members = {}  # Dictionary mapping group reps to sets of member positions
        self.group_liberties = {}  # Dictionary mapping group reps to sets of liberty positions

    # Save the current game state to history for undo
    def save_state(self):
        self.history.append((self.board.copy(), self.current_player, self.ko, self.consecutive_passes, self.last_move, self.group_reps.copy(), self.group_members.copy(), self.group_liberties.copy()))

    # Undo the last move by restoring the previous state
    def undo(self):
        if self.history:
            self.board, self.current_player, self.ko, self.consecutive_passes, self.last_move, self.group_reps, self.group_members, self.group_liberties = self.history.pop()
            return True
        return False

    # Return the six possible directions in 3D (up, down, left, right, forward, backward)
    def directions(self):
        return [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    # Check if a position is within the board boundaries
    def is_valid_pos(self, x, y, z):
        return 0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size

    # Get neighboring positions for a given point
    def get_neighbors(self, x, y, z):
        return [(x+dx, y+dy, z+dz) for dx,dy,dz in self.directions() if self.is_valid_pos(x+dx, y+dy, z+dz)]

    # Find all connected stones of the same color forming a group using DFS
    def find_group(self, x, y, z):
        color = self.board[x, y, z]
        if color == 0:
            return []
        visited = set()
        group = []
        stack = [(x, y, z)]
        while stack:
            px, py, pz = stack.pop()
            p = (px, py, pz)
            if p in visited:
                continue
            visited.add(p)
            if self.board[px, py, pz] == color:
                group.append(p)
                for nx, ny, nz in self.get_neighbors(px, py, pz):
                    if (nx, ny, nz) not in visited:
                        stack.append((nx, ny, nz))
        return group

    # Get the set of liberty positions for a group (empty adjacent positions)
    def get_liberties_pos(self, group):
        liberties = set()
        for p in group:
            x, y, z = p
            for nx, ny, nz in self.get_neighbors(x, y, z):
                if self.board[nx, ny, nz] == 0:
                    liberties.add((nx, ny, nz))
        return liberties

    # Get the number of liberties for a group
    def get_liberties(self, group):
        return len(self.get_liberties_pos(group))

    # Remove a group from the board by setting positions to 0
    def remove_group(self, group):
        for p in group:
            x, y, z = p
            self.board[x, y, z] = 0

    # Play a move at position (x, y, z), handling captures, ko, suicide, and group merging
    def play_move(self, x, y, z):
        self.save_state()
        if not self.is_valid_pos(x, y, z) or self.board[x, y, z] != 0:
            self.undo()
            return False, "Invalid position"
        opponent = 3 - self.current_player
        p = (x, y, z)
        self.board[x, y, z] = self.current_player
        adjacent_opponent_reps = set()
        adjacent_own_reps = set()
        possible_ko = None
        captured = []
        # Handle captures: check adjacent opponent groups and remove if no liberties
        for nx, ny, nz in self.get_neighbors(x, y, z):
            np = (nx, ny, nz)
            if self.board[nx, ny, nz] == opponent:
                rep = self.group_reps.get(np)
                if rep:
                    self.group_liberties[rep].discard(p)
                    if len(self.group_liberties[rep]) == 0:
                        group = self.group_members[rep]
                        captured.extend(group)
                        if len(group) == 1:
                            possible_ko = list(group)[0]
                        for gp in group:
                            del self.group_reps[gp]
                            self.board[gp[0], gp[1], gp[2]] = 0
                        del self.group_members[rep]
                        del self.group_liberties[rep]
                    else:
                        adjacent_opponent_reps.add(rep)
            elif self.board[nx, ny, nz] == self.current_player:
                rep = self.group_reps.get(np)
                if rep:
                    adjacent_own_reps.add(rep)
        # Add back liberties to adjacent own groups from captured positions
        for cp in captured:
            for cx, cy, cz in self.get_neighbors(*cp):
                cnp = (cx, cy, cz)
                if self.board[cx, cy, cz] == self.current_player:
                    rep = self.group_reps.get(cnp)
                    if rep:
                        self.group_liberties[rep].add(cp)
        # Check ko rule: prevent immediate recapture of a single stone
        if possible_ko and len(captured) == 1 and possible_ko == self.ko:
            self.undo()
            return False, "Ko rule violation"
        self.ko = possible_ko if len(captured) == 1 else None
        # Initialize new stone as its own group
        self.group_reps[p] = p
        self.group_members[p] = {p}
        self.group_liberties[p] = set([np for np in self.get_neighbors(x, y, z) if self.board[np[0], np[1], np[2]] == 0])
        # Merge with adjacent own groups using a union-find like approach
        if adjacent_own_reps:
            main_rep = min(adjacent_own_reps | {p}, key=lambda r: (r[0], r[1], r[2]))
            all_members = {p}
            all_lib = self.group_liberties[p].copy()
            for rep in adjacent_own_reps:
                if rep == main_rep:
                    continue
                members = self.group_members.pop(rep)
                all_members.update(members)
                all_lib.update(self.group_liberties.pop(rep))
                for mp in members:
                    self.group_reps[mp] = main_rep
            if main_rep != p:
                self.group_members.pop(p)
                del self.group_liberties[p]
                del self.group_reps[p]
            self.group_members[main_rep] = self.group_members.get(main_rep, set()) | all_members
            for mp in all_members:
                self.group_reps[mp] = main_rep
            # Recalculate liberties for the merged group
            self.group_liberties[main_rep] = self.get_liberties_pos(self.group_members[main_rep])
        # Check for suicide: undo if the move leaves the group with no liberties
        final_rep = self.group_reps[p]
        if len(self.group_liberties[final_rep]) == 0:
            self.undo()
            return False, "Suicide move"
        self.consecutive_passes = 0
        self.last_move = (x, y, z)
        self.current_player = opponent
        return True, "Move played"

    # Handle a pass move by the current player
    def pass_move(self):
        self.save_state()
        self.consecutive_passes += 1
        self.current_player = 3 - self.current_player

    # Check if the game is over (two consecutive passes)
    def is_game_over(self):
        return self.consecutive_passes >= 2

    # Find an empty region (connected empty spaces) and its bordering stones using DFS
    def find_empty_region(self, x, y, z, visited, board):
        region = []
        borders = set()
        stack = [(x, y, z)]
        while stack:
            px, py, pz = stack.pop()
            p = (px, py, pz)
            if p in visited:
                continue
            visited.add(p)
            if board[px, py, pz] == 0:
                region.append(p)
                for nx, ny, nz in self.get_neighbors(px, py, pz):
                    if board[nx, ny, nz] == 0:
                        if (nx, ny, nz) not in visited:
                            stack.append((nx, ny, nz))
                    else:
                        borders.add((nx, ny, nz))
        return region, borders

    # Calculate scores including territory and stones, accounting for dead positions
    def get_score(self):
        board_copy = self.board.copy()
        for px, py, pz in self.dead_positions:
            board_copy[px, py, pz] = 0
        visited = set()
        black_territory = 0
        white_territory = 0
        black_terr_pos = []
        white_terr_pos = []
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    p = (x, y, z)
                    if p in visited or board_copy[x, y, z] != 0:
                        continue
                    region, borders = self.find_empty_region(x, y, z, visited, board_copy)
                    border_colors = {board_copy[nx, ny, nz] for nx, ny, nz in borders}
                    if len(border_colors) == 1:
                        color = next(iter(border_colors))
                        if color == 1:
                            black_territory += len(region)
                            black_terr_pos.extend(region)
                        elif color == 2:
                            white_territory += len(region)
                            white_terr_pos.extend(region)
        # Include live stones as part of territory score
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    if board_copy[x, y, z] == 1:
                        black_territory += 1
                        black_terr_pos.append((x, y, z))
                    elif board_copy[x, y, z] == 2:
                        white_territory += 1
                        white_terr_pos.append((x, y, z))
        black_score = black_territory
        white_score = white_territory + self.komi
        return black_score, white_score, black_terr_pos, white_terr_pos

# Update layout parameters for the UI based on screen size
def update_layout(screen_width, screen_height, size, slices_panel_width):
    margin = max(10, screen_width // 100)  # Margin size scaled to screen
    control_height = 150  # Height for control panel
    layers_per_row = math.ceil(math.sqrt(size))  # Arrange layers in a grid
    num_rows = math.ceil(size / layers_per_row)
    avail_height = screen_height - margin * 3 - control_height
    if slices_panel_width < 0:
        avail_width_for_slices = screen_width // 2
    else:
        avail_width_for_slices = slices_panel_width - margin
    cell_size = min(
        (avail_width_for_slices - margin * (layers_per_row + 1)) // (layers_per_row * (size - 1)),
        (avail_height - margin * (num_rows + 1)) // (num_rows * (size - 1))
    )
    if cell_size < 10:
        cell_size = 10
    board_width = (size - 1) * cell_size
    slices_width = layers_per_row * (board_width + margin) + margin
    slices_height = num_rows * (board_width + margin) + margin
    avail_width_for_cube = screen_width - slices_width - margin * 2 if slices_panel_width < 0 else screen_width - slices_panel_width - margin
    cube_display_size = min(avail_width_for_cube, avail_height)
    if cube_display_size < 200:
        cube_display_size = 200
    return cell_size, board_width, slices_width, slices_height, cube_display_size, margin, control_height, layers_per_row, num_rows

# Main function to run the Pygame UI and game loop
def main():
    pygame.init()  # Initialize Pygame
    size = 10  # Default board size
    screen_width = 1600  # Initial screen width
    screen_height = 900  # Initial screen height
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)  # Create resizable window
    pygame.display.set_caption("3D Go Game (10x10x10)")

    game = Go3D(size)  # Create game instance
    clock = pygame.time.Clock()  # Clock for frame rate control
    start_time = time.time()  # Start timer
    font = pygame.font.SysFont(None, 24)  # Small font
    large_font = pygame.font.SysFont(None, 36)  # Large font

    message = ""  # Message to display (e.g., errors or status)

    # 3D view parameters
    rot_x = 0  # Rotation around x-axis
    rot_y = 0  # Rotation around y-axis
    rot_z = 0  # Rotation around z-axis
    rotating = False  # Flag for mouse rotation
    last_mouse = (0, 0)  # Last mouse position
    fov = 500  # Field of view for projection
    eye_z = 500  # Eye distance for projection

    light_brown = (210, 180, 140)  # Color for board background

    slices_panel_width = -1  # Width of slices panel (initialized to auto)
    dragging_separator = False  # Flag for dragging panel separator
    input_mode = None  # Input mode for new game (size or komi)
    input_text = ""  # Text input buffer
    input_prompt = ""  # Input prompt text

    running = True
    while running:
        # Update layout based on current screen size
        cell_size, board_width, slices_width, slices_height, cube_display_size, margin, control_height, layers_per_row, num_rows = update_layout(screen_width, screen_height, size, slices_panel_width)
        scale = cube_display_size / (size * 1.5)  # Scale for 3D points
        fov = cube_display_size * 1.2  # Adjust FOV
        eye_z = cube_display_size * 1.5  # Adjust eye distance

        if slices_panel_width < 0:
            slices_panel_width = slices_width + margin

        # Define rectangles for UI elements
        separator_rect = pygame.Rect(slices_panel_width - 5, 0, 10, screen_height)
        cube_rect = pygame.Rect(slices_panel_width, margin, cube_display_size, cube_display_size)

        center_x = screen_width // 2
        bottom_y = screen_height - margin - 40
        pass_button_rect = pygame.Rect(center_x - 200, bottom_y, 100, 40)
        undo_button_rect = pygame.Rect(center_x - 100, bottom_y, 100, 40)
        new_game_rect = pygame.Rect(center_x + 100, bottom_y, 100, 40)

        # Get current time and scores
        current_time = time.time() - start_time
        black_score, white_score, black_terr, white_terr = game.get_score()

        mx, my = pygame.mouse.get_pos()  # Mouse position
        hover_pos = None  # Hovered position for placement
        hovered_stone = None  # Hovered stone for group info
        radius = cell_size // 2 - 2  # Stone radius
        hovered_group_libs = set()  # Liberties of hovered group

        # Event handling loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen_width, screen_height = event.w, event.h
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if separator_rect.collidepoint(mx, my):
                    dragging_separator = True
                elif game.is_game_over():
                    # Toggle dead groups at game end by clicking
                    toggled = False
                    for z in range(size):
                        row = z // layers_per_row
                        col = z % layers_per_row
                        board_x = col * (board_width + margin) + margin
                        board_y = row * (board_width + margin) + margin
                        if board_x - radius <= mx <= board_x + board_width + radius and board_y - radius <= my <= board_y + board_width + radius:
                            dx = mx - board_x
                            dy = my - board_y
                            x = round(dx / cell_size)
                            y = round(dy / cell_size)
                            if 0 <= x < size and 0 <= y < size and game.board[x, y, z] != 0:
                                group = game.find_group(x, y, z)
                                dead_count = sum(1 for p in group if p in game.dead_positions)
                                if dead_count == len(group):
                                    for p in group:
                                        game.dead_positions.remove(p)
                                else:
                                    for p in group:
                                        game.dead_positions.add(p)
                                toggled = True
                                break
                    if toggled:
                        continue
                if pass_button_rect.collidepoint(mx, my) and not game.is_game_over() and input_mode is None:
                    game.pass_move()
                    message = f"Player {3 - game.current_player} passed"
                    if game.is_game_over():
                        message = f"Game Over! Final Scores: Black {black_score} - White {white_score}"
                elif undo_button_rect.collidepoint(mx, my) and input_mode is None:
                    if game.undo():
                        message = "Move undone"
                    else:
                        message = "No more moves to undo"
                elif new_game_rect.collidepoint(mx, my) and input_mode is None:
                    input_mode = 'size'
                    input_text = str(size)
                    input_prompt = "Enter board size (3-16):"
                elif cube_rect.collidepoint(mx, my):
                    rotating = True
                    last_mouse = (mx, my)
                elif not game.is_game_over() and input_mode is None:
                    # Handle clicks on layer slices to play moves
                    for z in range(size):
                        row = z // layers_per_row
                        col = z % layers_per_row
                        board_x = col * (board_width + margin) + margin
                        board_y = row * (board_width + margin) + margin
                        if board_x - radius <= mx <= board_x + board_width + radius and board_y - radius <= my <= board_y + board_width + radius:
                            dx = mx - board_x
                            dy = my - board_y
                            x = round(dx / cell_size)
                            y = round(dy / cell_size)
                            if 0 <= x < size and 0 <= y < size:
                                success, msg = game.play_move(x, y, z)
                                message = msg
                                if not success:
                                    print(msg) # For debugging
                                break
            elif event.type == pygame.MOUSEMOTION:
                if dragging_separator:
                    slices_panel_width = max(200, min(mx, screen_width - 200))
                elif rotating:
                    mx, my = event.pos
                    dx = mx - last_mouse[0]
                    dy = my - last_mouse[1]
                    rot_y += dx * 0.5
                    rot_x += dy * 0.5
                    last_mouse = (mx, my)
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_separator = False
                rotating = False
            elif event.type == pygame.KEYDOWN and input_mode:
                if event.key == pygame.K_RETURN:
                    try:
                        if input_mode == 'size':
                            new_size = int(input_text)
                            if 3 <= new_size <= 16:
                                input_mode = 'komi'
                                input_text = str(game.komi)
                                input_prompt = "Enter komi:"
                            else:
                                message = "Size must be between 3 and 16"
                        elif input_mode == 'komi':
                            new_komi = float(input_text)
                            game = Go3D(new_size)
                            game.komi = new_komi
                            size = new_size
                            start_time = time.time()
                            message = ""
                            input_mode = None
                    except ValueError:
                        message = "Invalid input"
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode

        # Handle keyboard rotation inputs
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            rot_y -= 2
        if keys[pygame.K_RIGHT]:
            rot_y += 2
        if keys[pygame.K_UP]:
            rot_x -= 2
        if keys[pygame.K_DOWN]:
            rot_x += 2
        if keys[pygame.K_a]:
            rot_y -= 2
        if keys[pygame.K_d]:
            rot_y += 2
        if keys[pygame.K_w]:
            rot_x -= 2
        if keys[pygame.K_s]:
            rot_x += 2
        if keys[pygame.K_q]:
            rot_z -= 2
        if keys[pygame.K_e]:
            rot_z += 2

        # Detect hover position on layers for placement or group liberties
        hovered_stone = None
        for z in range(size):
            row = z // layers_per_row
            col = z % layers_per_row
            board_x = col * (board_width + margin) + margin
            board_y = row * (board_width + margin) + margin
            if board_x - radius <= mx <= board_x + board_width + radius and board_y - radius <= my <= board_y + board_width + radius:
                dx = mx - board_x
                dy = my - board_y
                x = round(dx / cell_size)
                y = round(dy / cell_size)
                if 0 <= x < size and 0 <= y < size:
                    if game.board[x, y, z] == 0 and not game.is_game_over():
                        hover_pos = (x, y, z, board_x, board_y)
                    if game.board[x, y, z] != 0:
                        hovered_stone = (x, y, z)
                    break

        hovered_group_libs = set()
        if hovered_stone:
            rep = game.group_reps.get(hovered_stone)
            if rep:
                hovered_group_libs = game.group_liberties[rep]

        # Clear screen
        screen.fill((255, 255, 255))

        # Draw separator between slices and cube view
        pygame.draw.rect(screen, (150, 150, 150), separator_rect)

        # Draw drag cues on separator
        arrow_size = 10
        y_pos = screen_height - 72  # ~1 inch from bottom
        drag_text = font.render("drag", True, (0, 0, 0))
        text_width = drag_text.get_width()
        spacing = 5
        total_width = arrow_size * 2 + text_width + spacing * 2
        start_x = slices_panel_width - total_width // 2
        mid_y = y_pos
        top_y = mid_y - arrow_size // 2
        bottom_y = mid_y + arrow_size // 2

        # Left arrow
        left_tip_x = start_x
        left_base_x = start_x + arrow_size
        points_left = [(left_tip_x, mid_y), (left_base_x, top_y), (left_base_x, bottom_y)]
        pygame.draw.polygon(screen, (0, 0, 0), points_left)

        # Drag text
        text_x = left_base_x + spacing
        screen.blit(drag_text, (text_x, y_pos - drag_text.get_height() // 2))

        # Right arrow
        right_base_x = text_x + text_width + spacing
        right_tip_x = right_base_x + arrow_size
        points_right = [(right_tip_x, mid_y), (right_base_x, top_y), (right_base_x, bottom_y)]
        pygame.draw.polygon(screen, (0, 0, 0), points_right)

        # Draw each layer slice
        for z in range(size):
            row = z // layers_per_row
            col = z % layers_per_row
            board_x = col * (board_width + margin) + margin
            board_y = row * (board_width + margin) + margin

            # Draw board background
            pygame.draw.rect(screen, light_brown, (board_x, board_y, board_width + 1, board_width + 1))

            # Draw grid lines
            for i in range(size):
                pygame.draw.line(screen, (0, 0, 0), (board_x, board_y + i * cell_size), (board_x + board_width, board_y + i * cell_size))
                pygame.draw.line(screen, (0, 0, 0), (board_x + i * cell_size, board_y), (board_x + i * cell_size, board_y + board_width))

            # Draw liberties if group hovered
            for lx, ly, lz in hovered_group_libs:
                if lz == z:
                    pos_x = board_x + lx * cell_size
                    pos_y = board_y + ly * cell_size
                    red_radius = (cell_size // 2 - 2) // 2
                    pygame.draw.circle(screen, (255, 0, 0), (pos_x, pos_y), red_radius)

            # Draw stones
            for x in range(size):
                for y in range(size):
                    if game.board[x, y, z] != 0:
                        pos_x = board_x + x * cell_size
                        pos_y = board_y + y * cell_size
                        radius = cell_size // 2 - 2
                        if game.board[x, y, z] == 1:
                            pygame.draw.circle(screen, (0, 0, 0), (pos_x, pos_y), radius)
                        elif game.board[x, y, z] == 2:
                            pygame.draw.circle(screen, (255, 255, 255), (pos_x, pos_y), radius)
                            pygame.draw.circle(screen, (0, 0, 0), (pos_x, pos_y), radius, 1) # Outline
                        if game.last_move == (x, y, z):
                            purple_radius = int(radius * 0.45)
                            pygame.draw.circle(screen, (128, 0, 128), (pos_x, pos_y), purple_radius)

            # Draw territory markers if game over
            if game.is_game_over():
                for tx, ty, tz in black_terr:
                    if tz == z:
                        pos_x = board_x + tx * cell_size
                        pos_y = board_y + ty * cell_size
                        terr_radius = int((cell_size // 2 - 2) * 0.45)
                        pygame.draw.circle(screen, (0, 0, 255), (pos_x, pos_y), terr_radius)
                for tx, ty, tz in white_terr:
                    if tz == z:
                        pos_x = board_x + tx * cell_size
                        pos_y = board_y + ty * cell_size
                        terr_radius = int((cell_size // 2 - 2) * 0.45)
                        pygame.draw.circle(screen, (0, 255, 0), (pos_x, pos_y), terr_radius)

            # Label the layer
            label = font.render(f"Layer {z}", True, (0, 0, 0))
            screen.blit(label, (board_x + board_width // 2 - 30, board_y + board_width + 5))

        # Draw hover preview for next stone
        if hover_pos:
            x, y, z, board_x, board_y = hover_pos
            pos_x = board_x + x * cell_size
            pos_y = board_y + y * cell_size
            radius = cell_size // 2 - 2
            hover_surf = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
            col = (0, 0, 0, 100) if game.current_player == 1 else (255, 255, 255, 100)
            pygame.draw.circle(hover_surf, col, (radius, radius), radius)
            if game.current_player == 2:
                pygame.draw.circle(hover_surf, (0, 0, 0, 100), (radius, radius), radius, 1)
            screen.blit(hover_surf, (pos_x - radius, pos_y - radius))

        # Draw 3D cube view background
        pygame.draw.rect(screen, (200, 200, 200), cube_rect) # Background for cube
        cube_center_x = cube_rect.x + cube_rect.width // 2
        cube_center_y = cube_rect.y + cube_rect.height // 2

        # Collect stones, last move, territory, and hover for 3D projection
        stones = []
        purples = []
        half = (size - 1) / 2
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    p = (x, y, z)
                    if game.board[x, y, z] != 0:
                        point = np.array([x - half, y - half, z - half]) * scale
                        color = game.board[x, y, z]
                        stones.append((point, color))
                    if game.last_move == (x, y, z):
                        purples.append((point, 5))
        if game.is_game_over():
            for x, y, z in black_terr:
                point = np.array([x - half, y - half, z - half]) * scale
                stones.append((point, 3))
            for x, y, z in white_terr:
                point = np.array([x - half, y - half, z - half]) * scale
                stones.append((point, 4))
        if hover_pos:
            hx, hy, hz = hover_pos[:3]
            point = np.array([hx - half, hy - half, hz - half]) * scale
            stones.append((point, 7 if game.current_player == 1 else 8))

        # Create rotation matrices
        rx = radians(rot_x)
        ry = radians(rot_y)
        rz = radians(rot_z)
        rot_mat_x = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
        rot_mat_y = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
        rot_mat_z = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
        rot_mat = np.dot(rot_mat_z, np.dot(rot_mat_y, rot_mat_x))

        # Project stones to 2D
        projected_stones = []
        for point, color in stones:
            rotated = np.dot(rot_mat, point)
            dist = eye_z + rotated[2]
            if dist > 0:
                factor = fov / dist
                proj_x = rotated[0] * factor + cube_center_x
                proj_y = -rotated[1] * factor + cube_center_y # Invert y if needed
                radius = scale / 2 * factor * 0.75
                if color in (3,4):
                    radius *= 0.45
                projected_stones.append((proj_x, proj_y, rotated[2], radius, color))

        projected_purples = []
        for point, color in purples:
            rotated = np.dot(rot_mat, point)
            dist = eye_z + rotated[2]
            if dist > 0:
                factor = fov / dist
                proj_x = rotated[0] * factor + cube_center_x
                proj_y = -rotated[1] * factor + cube_center_y
                radius = (scale / 2 * factor * 0.75) * 0.45
                projected_purples.append((proj_x, proj_y, rotated[2], radius, color))

        # Sort projections by depth for correct rendering order
        projected_stones.sort(key=lambda s: s[2])
        projected_purples.sort(key=lambda s: s[2])

        # Draw 3D grid lines
        lines = []
        for dim in range(3):
            for i in range(size):
                for j in range(size):
                    start = [0] * 3
                    start[(dim + 1) % 3] = i
                    start[(dim + 2) % 3] = j
                    end = start[:]
                    end[dim] = size - 1
                    start = np.array(start) - half
                    end = np.array(end) - half
                    lines.append((start * scale, end * scale))

        for start, end in lines:
            rot_start = np.dot(rot_mat, start)
            rot_end = np.dot(rot_mat, end)
            dist_s = eye_z + rot_start[2]
            dist_e = eye_z + rot_end[2]
            if dist_s > 0 and dist_e > 0:
                factor_s = fov / dist_s
                proj_xs = rot_start[0] * factor_s + cube_center_x
                proj_ys = -rot_start[1] * factor_s + cube_center_y
                factor_e = fov / dist_e
                proj_xe = rot_end[0] * factor_e + cube_center_x
                proj_ye = -rot_end[1] * factor_e + cube_center_y
                pygame.draw.line(screen, light_brown, (proj_xs, proj_ys), (proj_xe, proj_ye), 1)

        # Draw projected stones
        for proj_x, proj_y, depth, radius, color in projected_stones:
            r = int(radius)
            if r < 1:
                continue
            stone_surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            if color == 1:
                col = (0, 0, 0, 128)
            elif color == 2:
                col = (255, 255, 255, 128)
            elif color == 3:
                col = (0, 0, 255, 128)
            elif color == 4:
                col = (0, 255, 0, 128)
            elif color == 7:
                col = (0, 0, 0, 100)
            elif color == 8:
                col = (255, 255, 255, 100)
            pygame.draw.circle(stone_surf, col, (r, r), r)
            if color in (2,8):
                pygame.draw.circle(stone_surf, (0, 0, 0, 100 if color==8 else 255), (r, r), r, 1)
            screen.blit(stone_surf, (int(proj_x - r), int(proj_y - r)))

        # Draw projected last move markers
        for proj_x, proj_y, depth, radius, color in projected_purples:
            r = int(radius)
            if r < 1:
                continue
            stone_surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            col = (128, 0, 128, 255)
            pygame.draw.circle(stone_surf, col, (r, r), r)
            screen.blit(stone_surf, (int(proj_x - r), int(proj_y - r)))

        # Draw UI text and buttons
        time_text = font.render(f"Time: {int(current_time // 60)}:{int(current_time % 60):02d}", True, (0, 0, 0))
        screen.blit(time_text, (screen_width - margin - time_text.get_width(), margin))

        player_text = large_font.render(f"Current Player: {'Black' if game.current_player == 1 else 'White'}", True, (0, 0, 0))
        screen.blit(player_text, (margin, screen_height - margin - 120))

        score_text = font.render(f"Scores - Black: {black_score} | White: {white_score}", True, (0, 0, 0))
        screen.blit(score_text , (margin, screen_height - margin - 90))

        message_text = font.render(message, True, (255, 0, 0))
        screen.blit(message_text, (margin, screen_height - margin - 60))

        # Pass button
        pygame.draw.rect(screen, (200, 200, 200), pass_button_rect)
        pass_text = font.render("Pass", True, (0, 0, 0))
        screen.blit(pass_text, (pass_button_rect.x + 25, pass_button_rect.y + 10))

        # Undo button
        pygame.draw.rect(screen, (200, 200, 200), undo_button_rect)
        undo_text = font.render("Undo", True, (0, 0, 0))
        screen.blit(undo_text, (undo_button_rect.x + 20, undo_button_rect.y + 10))

        # New game button
        pygame.draw.rect(screen, (200, 200, 200), new_game_rect)
        new_game_text = font.render("New Game", True, (0, 0, 0))
        screen.blit(new_game_text, (new_game_rect.x + 5, new_game_rect.y + 10))

        # Display game over message with winner
        if game.is_game_over():
            if black_score > white_score:
                winner = "Black"
                score_diff = black_score - white_score
                final_text = large_font.render(f"Game Over! {winner} wins by {score_diff}", True, (0, 0, 255))
            elif black_score < white_score:
                winner = "White"
                score_diff = white_score - black_score
                final_text = large_font.render(f"Game Over! {winner} wins by {score_diff}", True, (0, 0, 255))
            else:
                final_text = large_font.render("Game Over! It's a tie!", True, (0, 0, 255))
            screen.blit(final_text, (margin, screen_height - margin - 150))

        # Draw input box if in input mode
        if input_mode:
            input_box_rect = pygame.Rect(center_x - 150, screen_height // 2 - 50, 300, 100)
            pygame.draw.rect(screen, (200, 200, 200), input_box_rect)
            prompt_text = font.render(input_prompt, True, (0, 0, 0))
            screen.blit(prompt_text, (input_box_rect.x + 20, input_box_rect.y + 20))
            input_disp = font.render(input_text, True, (0, 0, 0))
            screen.blit(input_disp, (input_box_rect.x + 20, input_box_rect.y + 50))

        # Update display and limit frame rate
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()  # Quit Pygame
    sys.exit()  # Exit program

if __name__ == "__main__":
    main()
