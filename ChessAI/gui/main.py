"""PyGame interface: play either color against the local neural model."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import chess
import pygame

from agent.mcts import MCTS
from agent.policy import ChessPolicy
from config import DEFAULT_MCTS_SIMULATIONS, ROOT

SQUARE = 96
BOARD_PIXELS = 8 * SQUARE
PANEL_WIDTH = 340
WIDTH = BOARD_PIXELS + PANEL_WIDTH
HEIGHT = BOARD_PIXELS
FPS = 60

LIGHT = (238, 218, 181)
DARK = (181, 136, 99)
PANEL = (30, 34, 42)
TEXT = (235, 238, 242)
MUTED = (160, 168, 178)
ACCENT = (92, 166, 112)
HIGHLIGHT = (235, 204, 75)


class ChessApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI | Neural Policy + PUCT")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.small = pygame.font.SysFont("Arial", 17)
        self.title_font = pygame.font.SysFont("Arial", 27, bold=True)
        self.board = chess.Board()
        self.human_color = chess.WHITE
        self.simulations = DEFAULT_MCTS_SIMULATIONS
        self.selected_square = None
        self.last_move = None
        self.pending_promotion = []
        self.move_rows = []
        self.ai_statistics = []
        self.ai_value = None
        self.ai_thinking = False
        self.ai_error = None
        self.generation = 0
        self.ai_results = queue.Queue()
        self.policy = ChessPolicy()
        self.images = self._load_pieces()
        self.buttons = self._make_buttons()
        self.status_override = "Choose a piece, then its destination."
        self._maybe_start_ai()

    def _load_pieces(self):
        images = {}
        directory = ROOT / "assets" / "pieces"
        for symbol in "PNBRQKpnbrqk":
            prefix = "w" if symbol.isupper() else "b"
            path = directory / f"{prefix}{symbol.upper()}.svg"
            if not path.is_file():
                raise FileNotFoundError(f"Missing chess piece SVG: {path}")
            images[symbol] = pygame.image.load_sized_svg(str(path), (SQUARE, SQUARE)).convert_alpha()
        return images

    def _make_buttons(self):
        x = BOARD_PIXELS + 20
        return {
            "white": pygame.Rect(x, 105, 140, 42),
            "black": pygame.Rect(x + 150, 105, 150, 42),
            "restart": pygame.Rect(x, 162, 280, 42),
            "minus": pygame.Rect(x, 258, 54, 38),
            "zero": pygame.Rect(x + 64, 258, 82, 38),
            "plus": pygame.Rect(x + 156, 258, 54, 38),
        }

    def _square_rect(self, square):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.human_color == chess.WHITE:
            col, row = file_index, 7 - rank_index
        else:
            col, row = 7 - file_index, rank_index
        return pygame.Rect(col * SQUARE, row * SQUARE, SQUARE, SQUARE)

    def _screen_square(self, position):
        x, y = position
        if x < 0 or y < 0 or x >= BOARD_PIXELS or y >= BOARD_PIXELS:
            return None
        col, row = x // SQUARE, y // SQUARE
        if self.human_color == chess.WHITE:
            return chess.square(col, 7 - row)
        return chess.square(7 - col, row)

    def _record_and_push(self, move):
        san = self.board.san(move)
        move_number = self.board.fullmove_number
        if self.board.turn == chess.WHITE:
            self.move_rows.append([move_number, san, ""])
        elif self.move_rows:
            self.move_rows[-1][2] = san
        else:
            self.move_rows.append([move_number, "…", san])
        self.board.push(move)
        self.last_move = move
        self.selected_square = None
        self.pending_promotion = []
        self.status_override = ""

    def _maybe_start_ai(self):
        if self.board.is_game_over(claim_draw=True) or self.ai_thinking:
            return
        if self.board.turn == self.human_color:
            return
        self.ai_thinking = True
        self.ai_error = None
        token = self.generation
        snapshot = self.board.copy(stack=True)
        simulations = self.simulations

        def worker():
            try:
                if simulations > 0:
                    search = MCTS(policy=self.policy, simulations=simulations)
                    move, statistics = search.search(snapshot, simulations=simulations, temperature=0.0)
                    _move, value, _probabilities = self.policy.predict(snapshot)
                else:
                    move, value = self.policy.select_move(snapshot)
                    statistics = []
                self.ai_results.put((token, move, statistics, value, None))
            except Exception as exc:  # forwarded safely to the GUI thread
                self.ai_results.put((token, None, [], None, str(exc)))

        threading.Thread(target=worker, daemon=True, name="chess-ai-worker").start()

    def _poll_ai(self):
        while True:
            try:
                token, move, statistics, value, error = self.ai_results.get_nowait()
            except queue.Empty:
                break
            if token != self.generation:
                continue
            self.ai_thinking = False
            if error:
                self.ai_error = error
                self.status_override = f"AI error: {error[:45]}"
                continue
            if move not in self.board.legal_moves:
                self.ai_error = f"The model produced an illegal move: {move}"
                self.status_override = self.ai_error
                continue
            self._record_and_push(move)
            self.ai_statistics = statistics[:5]
            self.ai_value = value
            if not self.board.is_game_over(claim_draw=True):
                self.status_override = "Your turn."

    def _new_game(self, human_color=None):
        self.generation += 1
        if human_color is not None:
            self.human_color = human_color
        self.board = chess.Board()
        self.selected_square = None
        self.last_move = None
        self.pending_promotion = []
        self.move_rows = []
        self.ai_statistics = []
        self.ai_value = None
        self.ai_thinking = False
        self.ai_error = None
        self.status_override = "New game."
        self._maybe_start_ai()

    def _handle_board_click(self, position):
        if self.ai_thinking or self.board.is_game_over(claim_draw=True) or self.board.turn != self.human_color:
            return
        square = self._screen_square(position)
        if square is None:
            return
        piece = self.board.piece_at(square)
        if self.selected_square is None:
            if piece is not None and piece.color == self.human_color:
                self.selected_square = square
            return

        candidates = [m for m in self.board.legal_moves if m.from_square == self.selected_square and m.to_square == square]
        promotions = [m for m in candidates if m.promotion]
        if promotions:
            self.pending_promotion = promotions
            self.status_override = "Promote: press Q, R, B or N."
            return
        if candidates:
            self._record_and_push(candidates[0])
            self._maybe_start_ai()
            return
        if piece is not None and piece.color == self.human_color:
            self.selected_square = square
        else:
            self.selected_square = None

    def _choose_promotion(self, key):
        key_map = {pygame.K_q: chess.QUEEN, pygame.K_r: chess.ROOK, pygame.K_b: chess.BISHOP, pygame.K_n: chess.KNIGHT}
        piece_type = key_map.get(key)
        if piece_type is None:
            return
        move = next((candidate for candidate in self.pending_promotion if candidate.promotion == piece_type), None)
        if move is not None:
            self._record_and_push(move)
            self._maybe_start_ai()

    def _handle_click(self, position):
        for color, button_key in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            if self.buttons[button_key].collidepoint(position):
                self._new_game(color)
                return
        if self.buttons["restart"].collidepoint(position):
            self._new_game()
            return
        if self.buttons["minus"].collidepoint(position):
            self.simulations = max(0, self.simulations - 10)
            return
        if self.buttons["zero"].collidepoint(position):
            self.simulations = 0
            return
        if self.buttons["plus"].collidepoint(position):
            self.simulations = min(300, self.simulations + 10)
            return
        self._handle_board_click(position)

    def _draw_board(self):
        self.screen.fill((20, 23, 28))
        for row in range(8):
            for col in range(8):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                pygame.draw.rect(self.screen, color, (col * SQUARE, row * SQUARE, SQUARE, SQUARE))

        if self.last_move:
            overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
            overlay.fill((*HIGHLIGHT, 80))
            for square in (self.last_move.from_square, self.last_move.to_square):
                self.screen.blit(overlay, self._square_rect(square).topleft)

        if self.selected_square is not None:
            rect = self._square_rect(self.selected_square)
            pygame.draw.rect(self.screen, ACCENT, rect, 4)
            for move in self.board.legal_moves:
                if move.from_square != self.selected_square:
                    continue
                target = self._square_rect(move.to_square)
                center = target.center
                if self.board.is_capture(move):
                    pygame.draw.circle(self.screen, (35, 125, 75), center, 29, 5)
                else:
                    pygame.draw.circle(self.screen, (35, 125, 75), center, 9)

        for square, piece in self.board.piece_map().items():
            self.screen.blit(self.images[piece.symbol()], self._square_rect(square).topleft)

        if self.board.is_check():
            king_square = self.board.king(self.board.turn)
            if king_square is not None:
                pygame.draw.rect(self.screen, (215, 60, 60), self._square_rect(king_square), 4)

    def _draw_button(self, key, label, selected=False):
        rect = self.buttons[key]
        fill = (68, 112, 82) if selected else (57, 65, 78)
        pygame.draw.rect(self.screen, fill, rect, border_radius=7)
        rendered = self.small.render(label, True, TEXT)
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def _status(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return f"Checkmate — {winner} wins", (233, 191, 93)
        if self.board.is_stalemate():
            return "Draw — stalemate", (233, 191, 93)
        if self.board.is_game_over(claim_draw=True):
            outcome = self.board.outcome(claim_draw=True)
            reason = outcome.termination.name.replace("_", " ").title() if outcome else "Draw"
            return f"Draw — {reason}", (233, 191, 93)
        if self.ai_thinking:
            return "AI is thinking…", (233, 191, 93)
        return ("Your turn" if self.board.turn == self.human_color else "AI to move"), ACCENT

    def _draw_panel(self):
        x = BOARD_PIXELS
        pygame.draw.rect(self.screen, PANEL, (x, 0, PANEL_WIDTH, HEIGHT))
        self.screen.blit(self.title_font.render("CHESS AI", True, TEXT), (x + 20, 22))
        self.screen.blit(self.small.render("Stockfish-supervised ResNet", True, MUTED), (x + 20, 57))
        pygame.draw.line(self.screen, (70, 77, 89), (x + 18, 86), (WIDTH - 18, 86), 1)

        self.screen.blit(self.small.render("Play as", True, MUTED), (x + 20, 88))
        self._draw_button("white", "White", self.human_color == chess.WHITE)
        self._draw_button("black", "Black", self.human_color == chess.BLACK)
        self._draw_button("restart", "New game / restart")

        self.screen.blit(self.font.render("Search", True, TEXT), (x + 20, 220))
        self.screen.blit(self.small.render(f"PUCT simulations: {self.simulations}  (0 = direct network)", True, MUTED), (x + 20, 240))
        self._draw_button("minus", "−10")
        self._draw_button("zero", "Direct", self.simulations == 0)
        self._draw_button("plus", "+10")

        status, color = self._status()
        self.screen.blit(self.font.render("Status", True, TEXT), (x + 20, 318))
        self.screen.blit(self.small.render(status[:36], True, color), (x + 20, 350))
        if self.pending_promotion:
            self.screen.blit(self.small.render("Press Q / R / B / N to promote", True, HIGHLIGHT), (x + 20, 378))
        elif self.ai_value is not None:
            self.screen.blit(self.small.render(f"Network value (side-to-move): {self.ai_value:+.3f}", True, MUTED), (x + 20, 378))
        elif self.status_override and not self.ai_error:
            self.screen.blit(self.small.render(self.status_override[:38], True, MUTED), (x + 20, 378))

        pygame.draw.line(self.screen, (70, 77, 89), (x + 18, 411), (WIDTH - 18, 411), 1)
        self.screen.blit(self.font.render("Move history", True, TEXT), (x + 20, 425))
        tail = self.move_rows[-8:]
        for row_index, (number, white, black) in enumerate(tail):
            text = f"{number:>2}.  {white:<9} {black}"
            self.screen.blit(self.small.render(text[:31], True, MUTED), (x + 22, 461 + row_index * 23))

        if self.ai_statistics:
            y = min(760, 755 - len(self.ai_statistics) * 20)
            self.screen.blit(self.small.render("Top PUCT moves", True, TEXT), (x + 20, y))
            for i, item in enumerate(self.ai_statistics[:4]):
                line = f"{item['move']}  visits {item['visits']}  Q {item['q_value']:+.2f}"
                self.screen.blit(self.small.render(line, True, MUTED), (x + 20, y + 22 + i * 19))

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            self._poll_ai()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif self.pending_promotion:
                        self._choose_promotion(event.key)
                    elif event.key == pygame.K_r:
                        self._new_game()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            self._draw_board()
            self._draw_panel()
            pygame.display.flip()
        pygame.quit()


def main():
    ChessApp().run()


if __name__ == "__main__":
    main()
