import pygame
import random
import os
import re
import itertools
from datetime import datetime
import time


from config import LLM_PROVIDER, OPENAI_LLM_MODEL, OLLAMA_LLM_MODEL, WIDTH, HEIGHT
from llm_agent import get_llm_advice

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

if LLM_PROVIDER.upper() == "OPENAI":
    llm_model = OPENAI_LLM_MODEL
elif LLM_PROVIDER.upper() == "OLLAMA":
    llm_model = OLLAMA_LLM_MODEL


# === Configuration & UI Constants ===
CONSTANT_SPACING = 2

# Advice widget dimensions (left side)
ADVICE_WIDGET_X = 0
ADVICE_WIDGET_Y = 0
ADVICE_WIDGET_WIDTH = 600
ADVICE_WIDGET_HEIGHT = HEIGHT

# Game area (right side)
GAME_AREA_X_OFFSET = ADVICE_WIDGET_WIDTH
GAME_AREA_WIDTH = WIDTH - ADVICE_WIDGET_WIDTH

# Button dimensions and positions (bottom-right)
BUTTON_MARGIN = 20
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50

button_advice_rect = pygame.Rect(
    GAME_AREA_X_OFFSET + GAME_AREA_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN,
    HEIGHT - 3 * (BUTTON_HEIGHT + BUTTON_MARGIN),
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
button_play_rect = pygame.Rect(
    GAME_AREA_X_OFFSET + GAME_AREA_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN,
    HEIGHT - 2 * (BUTTON_HEIGHT + BUTTON_MARGIN),
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
button_pass_rect = pygame.Rect(
    GAME_AREA_X_OFFSET + GAME_AREA_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN,
    HEIGHT - BUTTON_HEIGHT - BUTTON_MARGIN,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)

# AI players' names
ai_names = ["賭神高進", "賭俠", "賭聖"]

# Fonts (using SimHei.ttf for Chinese characters)
font = pygame.font.Font("SimHei.ttf", 20)
advice_font = pygame.font.Font("SimHei.ttf", 30)
font.set_bold(True)
advice_font.set_bold(True)

# Card dimensions
card_width, card_height = 110, 165


# === Utility Functions for Cards and Moves ===

def card_value(card):
    """Returns a tuple representing the value of a card."""
    ranks = ['3','4','5','6','7','8','9','10','J','Q','K','A','2']
    suits = ['D','C','H','S']
    return (ranks.index(card[:-1]), suits.index(card[-1]))

def get_combo_type(cards):
    """
    Returns a tuple (combo_type, key) for the given cards.
    For singles, pairs, and triples, key is the card_value of one card.
    For five-card combos, it returns:
      - "straight": key is the card_value of the highest card.
      - "flush": key is a tuple of the sorted card_value tuples (in descending order).
      - "full_house": key is a tuple (triplet_index, pair_index) based on the rank order.
      - "four_plus_one": key is a tuple (quad_index, kicker_index) based on the rank order.
      - "straight_flush": key is the card_value of the highest card.
    Returns (None, None) if the cards do not form a valid combo.
    """
    rank_order = ['3','4','5','6','7','8','9','10','J','Q','K','A','2']
    if len(cards) == 1:
        return ("single", card_value(cards[0]))
    elif len(cards) == 2:
        if cards[0][:-1] == cards[1][:-1]:
            return ("pair", card_value(cards[0]))
        else:
            return (None, None)
    elif len(cards) == 3:
        if cards[0][:-1] == cards[1][:-1] == cards[2][:-1]:
            return ("triple", card_value(cards[0]))
        else:
            return (None, None)
    elif len(cards) == 5:
        is_flush = len(set(c[-1] for c in cards)) == 1
        sorted_cards = sorted(cards, key=lambda c: card_value(c))
        ranks = [card_value(c)[0] for c in sorted_cards]
        is_straight = all(ranks[i] + 1 == ranks[i+1] for i in range(4))
        if is_straight and is_flush:
            return ("straight_flush", card_value(sorted_cards[-1]))
        elif is_flush:
            # For flush, compare all cards (highest first)
            sorted_desc = tuple(sorted((card_value(c) for c in cards), reverse=True))
            return ("flush", sorted_desc)
        elif is_straight:
            return ("straight", card_value(sorted_cards[-1]))
        else:
            counts = {}
            for c in cards:
                r = c[:-1]
                counts[r] = counts.get(r, 0) + 1
            sorted_counts = sorted(counts.values())
            # Full House: counts [2, 3]
            if sorted_counts == [2, 3]:
                triplet_rank = None
                pair_rank = None
                for r, cnt in counts.items():
                    if cnt == 3:
                        triplet_rank = r
                    elif cnt == 2:
                        pair_rank = r
                triplet_index = rank_order.index(triplet_rank)
                pair_index = rank_order.index(pair_rank)
                return ("full_house", (triplet_index, pair_index))
            # Four Plus One: counts [1, 4]
            if sorted_counts == [1, 4]:
                quad_rank = None
                kicker_rank = None
                for r, cnt in counts.items():
                    if cnt == 4:
                        quad_rank = r
                    elif cnt == 1:
                        kicker_rank = r
                quad_index = rank_order.index(quad_rank)
                kicker_index = rank_order.index(kicker_rank)
                return ("four_plus_one", (quad_index, kicker_index))
            return (None, None)
    else:
        return (None, None)

def compare_combos(combo1, combo2):
    """
    Returns True if combo1 beats combo2.
    For singles, pairs, and triples, it compares their key (a card_value tuple).
    For five-card combos, the ranking order is:
      straight < flush < full_house < four_plus_one < straight_flush.
    If the combos are of different types, this order is used.
    If they are the same type:
      - For straights/straight flushes: compare highest card (i.e. key).
      - For flush: compare the sorted descending tuples lexicographically.
      - For full house: compare first the triplet rank then the pair rank.
      - For four_plus_one: compare first the quadruple rank then the kicker rank.
    """
    t1, key1 = combo1
    t2, key2 = combo2
    if t1 in ["single", "pair", "triple"]:
        return key1 > key2
    else:
        order = {
            "straight": 1,
            "flush": 2,
            "full_house": 3,
            "four_plus_one": 4,
            "straight_flush": 5
        }
        if t1 != t2:
            return order.get(t1, 0) > order.get(t2, 0)
        else:
            # Same type: perform detailed comparison.
            if t1 in ["straight", "straight_flush"]:
                return key1 > key2  # key is highest card value tuple
            elif t1 == "flush":
                return key1 > key2  # key is a descending tuple of card values
            elif t1 == "full_house":
                return key1 > key2  # key is (triplet_index, pair_index)
            elif t1 == "four_plus_one":
                return key1 > key2  # key is (quad_index, kicker_index)
            else:
                return key1 > key2

def is_valid_play_move(new_move, current_move):
    """Determines whether new_move is valid against the current_move."""
    new_type, new_key = get_combo_type(new_move)
    if new_type is None:
        return False
    if not current_move:
        return True
    if len(new_move) != len(current_move):
        return False
    curr_type, curr_key = get_combo_type(current_move)
    if curr_type != new_type:
        return False
    return compare_combos((new_type, new_key), (curr_type, curr_key))

def get_valid_moves(hand, required_count=None):
    """
    Returns all valid moves from the given hand.
    If required_count is specified, only moves with that many cards are considered.
    """
    valid_moves = []
    counts = [required_count] if required_count else [1, 2, 3, 5]
    for count in counts:
        for combo in itertools.combinations(hand, count):
            move = list(combo)
            combo_type, _ = get_combo_type(move)
            if combo_type is not None:
                if required_count is None or len(move) == required_count:
                    valid_moves.append(move)
    return valid_moves

def ai_fallback_move(ai_hand, current_table_move):
    """
    Generates a fallback move for the AI from its hand.
    If three passes have occurred, it ignores table restrictions.
    """
    global game_state
    if game_state.consecutive_passes >= 3:
        valid_moves = get_valid_moves(ai_hand)
    elif current_table_move:
        required = len(current_table_move)
        valid_moves = get_valid_moves(ai_hand, required)
        valid_moves = [move for move in valid_moves if is_valid_play_move(move, current_table_move)]
    else:
        valid_moves = get_valid_moves(ai_hand)
    if valid_moves:
        chosen = min(valid_moves, key=lambda move: card_value(move[0]))
        for card in chosen:
            ai_hand.remove(card)
        return chosen
    else:
        return None

# === Image Loading & Card Translation ===

def load_card_image(card):
    """Loads an image for a given card."""
    return pygame.image.load(os.path.join('assets','cards',f'{card}.png'))

def load_all_card_images(deck):
    """Loads and scales images for all cards in the deck."""
    images = {}
    for card in deck:
        img = load_card_image(card)
        img = pygame.transform.scale(img, (card_width, card_height))
        images[card] = img
    return images

def generate_card_translation_dict():
    """Returns a dictionary mapping card codes to their display names."""
    suits = ['D','C','H','S']
    ranks = ['3','4','5','6','7','8','9','10','J','Q','K','A','2']
    suit_names = {'D':'階磚','C':'梅花','H':'紅心','S':'葵扇'}
    rank_names = {'3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'10','J':'Jack','Q':'Queen','K':'King','A':'煙(A)','2':'DEE'}
    d = {}
    for s in suits:
        for r in ranks:
            card = r+s
            d[card] = suit_names[s] + rank_names[r]
    return d

def translate_card_names(text):
    """Replaces card codes in text with their translated names."""
    d = generate_card_translation_dict()
    for key in sorted(d.keys(), key=lambda x: len(x), reverse=True):
        text = text.replace(key, d[key])
    return text

# === GameState Class ===

class GameState:
    def __init__(self):
        # Initialize deck and shuffle
        self.suits = ['D','C','H','S']
        self.ranks = ['3','4','5','6','7','8','9','10','J','Q','K','A','2']
        self.deck = [f"{rank}{suit}" for rank in self.ranks for suit in self.suits]
        random.shuffle(self.deck)
        
        # Initialize hands
        self.player_hand = sorted(self.deck[:13],
                                  key=lambda c: (self.ranks.index(c[:-1]), self.suits.index(c[-1])))
        self.ai_hands = [self.deck[13:26], self.deck[26:39], self.deck[39:52]]
        
        # Turn and move data
        self.table_history = []
        self.last_table_move = []
        self.turn_order = []  # List of tuples, e.g. ("human", None) or ("ai", index)
        self.current_turn_index = 0
        self.consecutive_passes = 0
        self.free_play_mode = False
        self.last_played = None
        self.selected_cards = []
        
        # Determine starting player (3D rule)
        if "3D" in self.player_hand:
            starting_player = ("human", None)
        else:
            starting_player = None
            for i, hand in enumerate(self.ai_hands):
                if "3D" in hand:
                    starting_player = ("ai", i)
                    break
        # Define base order and rotate to start with starting_player
        base_order = [("ai", 0), ("ai", 2), ("human", None), ("ai", 1)]
        if starting_player is None:
            starting_player = ("human", None)
        start_index = base_order.index(starting_player)
        self.turn_order = base_order[start_index:] + base_order[:start_index]

# Initialize game state
game_state = GameState()

# Load card images now that the deck is set up
card_images = load_all_card_images(game_state.deck)
card_back_image = pygame.image.load(os.path.join('assets','cards','back.png'))
card_back_image = pygame.transform.scale(card_back_image, (card_width, card_height))
card_back_image_left = pygame.transform.rotate(card_back_image, 90)
card_back_image_right = pygame.transform.rotate(card_back_image, -90)

# Create background surface
background = pygame.Surface((WIDTH, HEIGHT))
background.fill((34,139,34))

# === Advice Logging & Scrolling ===

advice_log = []
cached_advice_lines = []
cache_dirty = True
scroll_offset = 0
dragging_scrollbar = False
drag_offset_y = 0
human_turn_notified = False

def wrap_text(text, font, max_width):
    lines = []
    current_line = ""
    for char in text:
        if char == "\n":
            lines.append(current_line)
            current_line = ""
            continue
        test_line = current_line + char
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = char
    if current_line:
        lines.append(current_line)
    return lines

def update_advice_cache_wrapper():
    global cached_advice_lines, cache_dirty
    all_text = "\n".join(advice_log)
    cached_advice_lines[:] = wrap_text(all_text, advice_font, ADVICE_WIDGET_WIDTH - 30)
    cache_dirty = False

def append_advice_wrapper(message):
    global advice_log, scroll_offset, cache_dirty
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    advice_log.append(f"{timestamp} {message}")
    cache_dirty = True
    update_advice_cache_wrapper()
    visible = (ADVICE_WIDGET_HEIGHT - 20) // advice_font.get_linesize()
    global scroll_offset
    scroll_offset = max(0, len(cached_advice_lines) - visible)

append_advice = append_advice_wrapper

# === Restart Game Function ===
def restart_game():
    """Resets the game state for a new game and clears the advice widget. ### NEW"""
    global game_state, human_turn_notified, scroll_offset, cache_dirty, game_over_flag, advice_log
    game_state = GameState()
    human_turn_notified = False
    scroll_offset = 0
    cache_dirty = True
    game_over_flag = False
    advice_log.clear()  # Clear the advice widget's log
    append_advice("～～～鋤大DEE牌局開始～～～")

# === Global Game Over Flag ===
game_over_flag = False  # ### Added global flag to indicate game over

# === AI Move Functions ===

def ai_make_move(ai_hand, current_move, ai_name="AI"):
    """
    Uses LLM advice to determine the AI move.
    If LLM responds with 'pass', the AI passes.
    If LLM output is invalid, fallback move is used.
    """
    forced = False
    if not game_state.last_table_move and "3D" in ai_hand:
        forced = True

    # Build opponents' info string
    opponent_counts = []
    opponent_counts.append(f"Human: {len(game_state.player_hand)}")
    for i, name in enumerate(ai_names):
        if name != ai_name:
            opponent_counts.append(f"{name}: {len(game_state.ai_hands[i])}")
    opponents_str = ", ".join(opponent_counts)

    # Format table history string
    if game_state.table_history:
        history_str = "; ".join([", ".join(move) for move in game_state.table_history])
    else:
        history_str = "None"

    free_mode_str = ("FREE-PLAY MODE ACTIVE: Ignore table restrictions." 
                     if game_state.free_play_mode 
                     else "NORMAL MODE: Your move must beat the current table move and match its card count.")

    # Updated prompt focusing on smart strategic play.
    prompt = f"""
        You are a highly skilled expert in Hong Kong Big Two (鋤大DEE). Your goal is to maximize your chances of winning by choosing the optimal move.

        Current Game State:
        - Your hand: {', '.join(ai_hand)}
        - Current table: {', '.join(game_state.last_table_move) if game_state.last_table_move else 'None'}
        - Table history: {history_str}
        - Opponents' card counts: {opponents_str}
        - {free_mode_str}

        Game Rules:
        - Valid moves: a single card, a pair (two cards of the same rank, e.g. 9C 9H; JS JD; AH AD. Combinations like AS QS or 8D 10D are not considered a pair.), , three of a kind (three cards of the same rank), or a valid five-card combination (which may be a straight, flush, full house, or straight flush). (Four-card moves are not allowed.)
        - Card ranking from lowest to highest: 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2. Card Suits from lowest to highest: D (Diamond), C (Club), H (Heart), S (Spade). (So, example, 2S > 2C; 2D > AH; KC > KD; JS > 10S etc.). So, example, 2S > 2C; 2D > AH; KC > KD; JS > 10S etc.
        - Your move must be able to beat {game_state.last_table_move}.
        - The number of cards you play must be the same as that of {game_state.last_table_move}.
        

        Strategic Guidelines:
        - Choose the move that gives you the best chance of winning in the long run.
        - Consider the opponents' card counts to assess risk. Usually you will need to play the card with highest rank when one of the opponent has only one card on hand.
        - Think ahead about future rounds rather than only focusing on the current trick.

        Based on the above, choose your optimal move. Do not put your reasons for the move in the response. Simply return only the card codes separated by spaces (e.g. "3D 4D 5D") or "pass" if the chance of winning is higher to keep the cards at the moment or no valid move is possible.
        """
    move = get_llm_advice(prompt, llm_model).strip()
    print(f"[DEBUG] LLM raw output for {ai_name}: '{move}'")
    # Remove commas and extra punctuation, then split.
    # The pattern matches a rank (3,4,5,6,7,8,9,10,J,Q,K,A,2) followed by one of the suits (D, C, H, S)
    card_pattern = r'\b(?:3|4|5|6|7|8|9|10|[JQKA2])[DCHS]\b'
    extracted_cards = re.findall(card_pattern, move)
    if extracted_cards:
        move_cards = extracted_cards
    else:
        move_cards = move.replace(",", " ").split()  # fallback to splitting by spaces if no valid codes found

    if forced and "3D" not in move_cards:
        print(f"[DEBUG] Forced move check failed for {ai_name}, move: {move_cards}")
        # Instead of passing, force the move by playing 3D if available.
        if "3D" in ai_hand:
            print(f"[DEBUG] Forcing move: 3D")
            ai_hand.remove("3D")
            return ["3D"]
        else:
            # If somehow 3D is no longer in hand, fallback.
            return ai_fallback_move(ai_hand, game_state.last_table_move)

    if move.lower() == "pass":
        print(f"AI {ai_name}: LLM advised 'pass'. Following that advice.")
        return None

    if all(card in ai_hand for card in move_cards):
        if (not game_state.last_table_move) or (len(move_cards) == len(game_state.last_table_move) and is_valid_play_move(move_cards, game_state.last_table_move)):
            if forced and "3D" not in move_cards:
                print(f"AI {ai_name}: LLM move invalid (forced 3D missing), using fallback.")
                return ai_fallback_move(ai_hand, game_state.last_table_move)
            print(f"AI {ai_name}: Using LLM move: {move_cards}")
            for card in move_cards:
                ai_hand.remove(card)
            return move_cards

    print(f"AI {ai_name}: FALLBACK - LLM move invalid, using fallback. Move was: {game_state.last_table_move}")
    return ai_fallback_move(ai_hand, game_state.last_table_move)

# === Turn Advancement ===

def advance_turn():
    """
    Advances the turn.
    If three consecutive passes occur, activates free-play mode and clears the active table.
    """
    if game_state.consecutive_passes >= 3 and game_state.last_played is not None:
        append_advice("所有其他人都Pass，可以自由出牌。")
        game_state.consecutive_passes = 0
        game_state.free_play_mode = True
        game_state.last_table_move = []
        game_state.current_turn_index = game_state.turn_order.index(game_state.last_played)
    else:
        game_state.current_turn_index = (game_state.current_turn_index + 1) % len(game_state.turn_order)

# === Event Handling ===

def handle_events(events):
    global scroll_offset, dragging_scrollbar, drag_offset_y
    for event in events:
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                scroll_offset = max(0, scroll_offset - 3)
            elif event.key == pygame.K_DOWN:
                scroll_offset += 3
        elif event.type == pygame.MOUSEWHEEL:
            scroll_offset = max(0, scroll_offset - event.y * 3)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if cache_dirty:
                update_advice_cache_wrapper()
            vis = (ADVICE_WIDGET_HEIGHT - 20) // advice_font.get_linesize()
            sb_width = 10
            sb_x = ADVICE_WIDGET_X + ADVICE_WIDGET_WIDTH - sb_width - 5
            sb_y = ADVICE_WIDGET_Y
            sb_height = ADVICE_WIDGET_HEIGHT
            if len(cached_advice_lines) > 0:
                slider_h = max(20, (vis / len(cached_advice_lines)) * sb_height)
                slider_y = sb_y + (scroll_offset / len(cached_advice_lines)) * sb_height
            else:
                slider_h = sb_height
                slider_y = sb_y
            slider_rect = pygame.Rect(sb_x, slider_y, sb_width, slider_h)
            if slider_rect.collidepoint(event.pos):
                dragging_scrollbar = True
                drag_offset_y = event.pos[1] - slider_y
            # Handle card selection in player's hand.
            x, y = event.pos
            if HEIGHT - 150 < y < HEIGHT - 30:
                tot = len(game_state.player_hand) * card_width + (len(game_state.player_hand) - 1) * CONSTANT_SPACING
                start_x = GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - tot) // 2
                for idx, card in enumerate(game_state.player_hand):
                    card_x = start_x + idx * (card_width + CONSTANT_SPACING)
                    if card_x <= x <= card_x + card_width:
                        if card in game_state.selected_cards:
                            game_state.selected_cards.remove(card)
                        else:
                            game_state.selected_cards.append(card)
                        break
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_scrollbar = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging_scrollbar:
                vis = (ADVICE_WIDGET_HEIGHT - 20) // advice_font.get_linesize()
                sb_y = ADVICE_WIDGET_Y
                sb_height = ADVICE_WIDGET_HEIGHT
                if len(cached_advice_lines) > 0:
                    slider_h = max(20, (vis / len(cached_advice_lines)) * sb_height)
                else:
                    slider_h = sb_height
                new_slider_y = event.pos[1] - drag_offset_y
                new_slider_y = max(sb_y, min(new_slider_y, sb_y + sb_height - slider_h))
                if len(cached_advice_lines) > 0:
                    scroll_offset = int((new_slider_y - sb_y) / sb_height * len(cached_advice_lines))
                else:
                    scroll_offset = 0
    return True

# === Drawing Function ===

def draw_screen():
    """Draws the game screen: background, hands, table moves, AI opponents, and UI."""
    screen.blit(background, (0, 0))
    
    # Draw player's hand
    tot = len(game_state.player_hand) * card_width + (len(game_state.player_hand) - 1) * CONSTANT_SPACING
    start_x = GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - tot) // 2
    for idx, card in enumerate(game_state.player_hand):
        x = start_x + idx * (card_width + CONSTANT_SPACING)
        y = HEIGHT - card_height - 20
        if card in game_state.selected_cards:
            y -= 20
        screen.blit(card_images[card], (x, y))
    
    # Draw table moves from history
    all_cards = [card for move in game_state.table_history for card in move]
    row_gap = 2
    max_cards_per_row = 10
    rows = [all_cards[i:i+max_cards_per_row] for i in range(0, len(all_cards), max_cards_per_row)]
    total_rows = len(rows)
    grid_height = total_rows * card_height + (total_rows - 1) * row_gap
    available_height = HEIGHT - (card_height + 20)
    table_start_y = (available_height - grid_height) // 2
    current_y = table_start_y
    for row in rows:
        row_width = len(row) * card_width + (len(row) - 1) * CONSTANT_SPACING
        row_start_x = GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - row_width) // 2
        for card in row:
            screen.blit(card_images[card], (row_start_x, current_y))
            row_start_x += card_width + CONSTANT_SPACING
        current_y += card_height + row_gap
    
    # Draw AI opponents
    # Top AI:
    ai_top = game_state.ai_hands[0]
    if ai_top:
        tot = len(ai_top) * card_width + (len(ai_top) - 1) * CONSTANT_SPACING
        start_x = GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - tot) // 2
        y = 20
        for i in range(len(ai_top)):
            x = start_x + i * (card_width + CONSTANT_SPACING)
            screen.blit(card_back_image, (x, y))
        top_label = font.render(ai_names[0], True, (255,255,255))
        screen.blit(top_label, (GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - top_label.get_width()) // 2, y + card_height + 5))
    # Left AI:
    ai_left = game_state.ai_hands[1]
    if ai_left:
        x = GAME_AREA_X_OFFSET + 20
        tot = len(ai_left) * card_width + (len(ai_left) - 1) * CONSTANT_SPACING
        start_y = (HEIGHT - tot) // 2
        for i in range(len(ai_left)):
            y = start_y + i * (card_width + CONSTANT_SPACING)
            screen.blit(card_back_image_left, (x, y))
        left_label = font.render(ai_names[1], True, (255,255,255))
        screen.blit(left_label, (x + card_height + 30, start_y - 25))
    # Right AI:
    ai_right = game_state.ai_hands[2]
    if ai_right:
        x = GAME_AREA_X_OFFSET + GAME_AREA_WIDTH - 20 - card_height
        tot = len(ai_right) * card_width + (len(ai_right) - 1) * CONSTANT_SPACING
        start_y = (HEIGHT - tot) // 2
        for i in range(len(ai_right)):
            y = start_y + i * (card_width + CONSTANT_SPACING)
            screen.blit(card_back_image_right, (x, y))
        right_label = font.render(ai_names[2], True, (255,255,255))
        screen.blit(right_label, (x - 60, start_y - 25))
    
    # Draw advice widget and scrollbar
    pygame.draw.rect(screen, (50,50,50), (ADVICE_WIDGET_X, ADVICE_WIDGET_Y, ADVICE_WIDGET_WIDTH, ADVICE_WIDGET_HEIGHT))
    pygame.draw.rect(screen, (255,255,255), (ADVICE_WIDGET_X, ADVICE_WIDGET_Y, ADVICE_WIDGET_WIDTH, ADVICE_WIDGET_HEIGHT), 2)
    if cache_dirty:
        update_advice_cache_wrapper()
    vis = (ADVICE_WIDGET_HEIGHT - 20) // advice_font.get_linesize()
    global scroll_offset
    if scroll_offset > max(0, len(cached_advice_lines) - vis):
        scroll_offset = max(0, len(cached_advice_lines) - vis)
    y_offset = ADVICE_WIDGET_Y + 10
    for line in cached_advice_lines[scroll_offset: scroll_offset + vis]:
        rend = advice_font.render(line, True, (255,255,0))
        screen.blit(rend, (ADVICE_WIDGET_X + 15, y_offset))
        y_offset += advice_font.get_linesize()
    sb_width = 10
    sb_x = ADVICE_WIDGET_X + ADVICE_WIDGET_WIDTH - sb_width - 5
    sb_y = ADVICE_WIDGET_Y
    sb_height = ADVICE_WIDGET_HEIGHT
    pygame.draw.rect(screen, (100,100,100), (sb_x, sb_y, sb_width, sb_height))
    if len(cached_advice_lines) > 0:
        slider_h = max(20, (vis / len(cached_advice_lines)) * sb_height)
        slider_y = sb_y + (scroll_offset / len(cached_advice_lines)) * sb_height
    else:
        slider_h = sb_height
        slider_y = sb_y
    pygame.draw.rect(screen, (200,200,200), (sb_x, slider_y, sb_width, slider_h))
    
    # Draw buttons
    pygame.draw.rect(screen, (0,128,0), button_advice_rect)
    pygame.draw.rect(screen, (0,0,128), button_play_rect)
    pygame.draw.rect(screen, (128,0,0), button_pass_rect)
    adv_text = font.render("求建議", True, (255,255,255))
    ply_text = font.render("出牌", True, (255,255,255))
    pas_text = font.render("Pass", True, (255,255,255))
    screen.blit(adv_text, adv_text.get_rect(center=button_advice_rect.center))
    screen.blit(ply_text, ply_text.get_rect(center=button_play_rect.center))
    screen.blit(pas_text, pas_text.get_rect(center=button_pass_rect.center))
    
    # ### If game over, overlay a restart button at the center of the table area.
    if game_over_flag:
        restart_button_rect = pygame.Rect(
            GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - 500) // 2,
            (HEIGHT - 150) // 2,
            500,
            150
        )
        pygame.draw.rect(screen, (128,0,0), restart_button_rect)
        restart_text = font.render("再玩一局", True, (255,255,255))
        screen.blit(restart_text, restart_text.get_rect(center=restart_button_rect.center))


    # Draw instructions.
    instr = font.render("點擊牌以選擇/取消。按「出牌」出選中牌，或按「求建議」/「Pass」。可拖動滾動條或用滑鼠滾輪捲動建議。", True, (255,255,255))
    instr_x = GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - instr.get_width()) // 2
    screen.blit(instr, (instr_x, HEIGHT - card_height - 75))
    
    pygame.display.flip()

# === Main Game Loop ===

def main_loop():
    global game_state, scroll_offset, cache_dirty, human_turn_notified, game_over_flag
    clock = pygame.time.Clock()
    current_time = time.time()
    last_weird_message_time = current_time
    while True:
        clock.tick(30)
        current_time = time.time()
        if current_time - last_weird_message_time >= 180:  # At least 3 minutes passed
            print(current_time, last_weird_message_time, current_time - last_weird_message_time)
            if random.random() < 0.05:  # 5% chance to print the message on this frame
                random_response_pb = random.random()
                random_interval = 1 / 5
                if random_response_pb < random_interval:
                    append_advice("又會有條例投降輸一半咁怪都有既")
                    append_advice("係呀係呀，呢度興架！")
                elif random_response_pb >= random_interval and random_response_pb < random_interval * 2:
                    append_advice("電腦分析，30%係7，70%係Jack ～～")
                    append_advice("唔係7就係Jack ～～")
                elif random_response_pb >= random_interval * 2 and random_response_pb < random_interval * 3:
                    append_advice("你試下行出去窗口望下睇下望唔望到大嶼山！")
                elif random_response_pb >= random_interval * 3 and random_response_pb < random_interval * 4:
                    append_advice("各位觀眾，五條煙！")    
                else:
                    append_advice("今次重唔捉到你呢隻老狐狸！")
                last_weird_message_time = current_time
        events = pygame.event.get()  # Get events once per frame
        if not handle_events(events):
            break

        if not game_over_flag:
            # Determine current player
            current_player = game_state.turn_order[game_state.current_turn_index]
            if current_player[0] == "human":
                if not human_turn_notified:
                    append_advice("到你出牌")
                    human_turn_notified = True
                human_moved = False
                for event in events:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if button_advice_rect.collidepoint(event.pos):
                            append_advice("等等，AI會教你出牌")
                            draw_screen()  # Update display to show message
                            pygame.display.flip()
                            # Prepare context strings for the advisor prompt:
                            if game_state.table_history:
                                table_history_str = "; ".join([", ".join(move) for move in game_state.table_history])
                            else:
                                table_history_str = "None"

                            opponent_counts = []
                            opponent_counts.append(f"Human: {len(game_state.player_hand)}")
                            for i, name in enumerate(ai_names):
                                opponent_counts.append(f"{name}: {len(game_state.ai_hands[i])}")
                            opponents_str = ", ".join(opponent_counts)

                            mode_str = ("FREE-PLAY MODE ACTIVE: Table restrictions are lifted." 
                                        if game_state.free_play_mode 
                                        else "NORMAL MODE: Your move must beat the current table move and match its count.")

                            advisor_prompt = f"""
                            You are a top expert advisor in Hong Kong Big Two (鋤大DEE) with deep strategic insight.
                            Your goal is to help the human player maximize their chances of winning the game.
                            Please consider the following information:

                            - My cards: {game_state.player_hand}
                            - Current table cards: {game_state.last_table_move if game_state.last_table_move else 'None'}
                            - Table history: {table_history_str}
                            - Opponents' card counts: {opponents_str}
                            - {mode_str}

                            Game Rules:
                            - Valid moves: a single card, a pair (two cards of the same rank, e.g. 9C 9H; JS JD; AH AD. Combinations like AS QS or 8D 10D are not considered a pair.), , three of a kind (three cards of the same rank), or a valid five-card combination (which may be a straight, flush, full house, or straight flush). (Four-card moves are not allowed.)
                            - Card ranking from lowest to highest: 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2. Card Suits from lowest to highest: D (Diamond), C (Club), H (Heart), S (Spade). (So, example, 2S > 2C; 2D > AH; KC > KD; JS > 10S etc.). So, example, 2S > 2C; 2D > AH; KC > KD; JS > 10S etc.
                            - Your move must be able to beat {game_state.last_table_move}.
                            - The number of cards you play must be the same as that of {game_state.last_table_move}.

                            Strategic Guidelines:
                            - Choose the move that gives you the best chance of winning in the long run.
                            - Consider the opponents' card counts to assess risk. Usually you will need to play the card with highest rank when one of the opponent has only one card on hand.
                            - Think ahead about future rounds rather than only focusing on the current trick.

                            Please provide a recommendation for my next move and briefly explain your reasoning in Hong Kong Cantonese.
                            Return your response in the following format:
                            - First, a brief explanation.
                            - On the next line, only the card codes (e.g. "3D 4D 5D") or "pass" if no valid move exists.
                            """

                            # Then, for the advice-seeking branch (when the advice button is clicked), use:
                            resp = get_llm_advice(advisor_prompt, llm_model).strip()
                            resp = translate_card_names(resp)
                            append_advice(resp)

                            break
                        elif button_play_rect.collidepoint(event.pos):
                            if not game_state.selected_cards:
                                append_advice("請選擇要出牌嘅牌")
                            else:
                                if (not game_state.last_table_move) and ("3D" in game_state.player_hand) and ("3D" not in game_state.selected_cards):
                                    append_advice("你必須首先出階磚3")
                                else:
                                    if not game_state.free_play_mode:
                                        if game_state.last_table_move and len(game_state.selected_cards) != len(game_state.last_table_move):
                                            append_advice("出牌數量必須同桌面上一樣")
                                        elif game_state.last_table_move and not is_valid_play_move(game_state.selected_cards, game_state.last_table_move):
                                            append_advice("唔可以咁出牌喎，而家唔係打美國例麻雀喎！")
                                        else:
                                            for card in game_state.selected_cards:
                                                game_state.player_hand.remove(card)
                                            game_state.last_table_move = game_state.selected_cards.copy()
                                            game_state.table_history.append(game_state.selected_cards.copy())
                                            append_advice(f"你出左 {translate_card_names(' '.join(game_state.selected_cards))}")
                                            game_state.last_played = current_player
                                            game_state.consecutive_passes = 0
                                            game_state.selected_cards.clear()
                                            human_moved = True
                                    else:
                                        if get_combo_type(game_state.selected_cards)[0] is None:
                                            append_advice("唔可以咁出牌喎，而家唔係打美國例麻雀喎！")
                                        else:
                                            for card in game_state.selected_cards:
                                                game_state.player_hand.remove(card)
                                            game_state.last_table_move = game_state.selected_cards.copy()
                                            game_state.table_history.append(game_state.selected_cards.copy())
                                            append_advice(f"你出左 {translate_card_names(' '.join(game_state.selected_cards))}")
                                            game_state.last_played = current_player
                                            game_state.consecutive_passes = 0
                                            game_state.free_play_mode = False
                                            game_state.selected_cards.clear()
                                            human_moved = True
                            break
                        elif button_pass_rect.collidepoint(event.pos):
                            if game_state.last_played == ("human", None):
                                append_advice("你係依家最大，唔可以Pass，必須出牌！")
                            else:
                                append_advice("你 passed.")
                                human_moved = True
                                game_state.consecutive_passes += 1
                            game_state.selected_cards.clear()
                            break
                if human_moved:
                    advance_turn()
                    human_turn_notified = False
            else:
                # AI turn
                ai_index = current_player[1]
                append_advice(f"{ai_names[ai_index]} 諗緊出咩牌")  ### NEW: AI thinking message
                draw_screen()
                pygame.display.flip()
                pygame.time.wait(100)
                draw_screen()
                thinking_text = font.render("AI thinking...", True, (255,255,255))
                screen.blit(thinking_text, (GAME_AREA_X_OFFSET + 20, 20))
                pygame.display.flip()
                pygame.time.wait(500)
                ai_move = ai_make_move(game_state.ai_hands[ai_index], game_state.last_table_move, ai_names[ai_index])
                if ai_move and len(ai_move) > 0:
                    game_state.last_table_move = ai_move.copy()
                    game_state.table_history.append(ai_move.copy())
                    append_advice(f"{ai_names[ai_index]} 出左 {translate_card_names(' '.join(ai_move))}.")
                    game_state.last_played = current_player
                    game_state.consecutive_passes = 0
                else:
                    if game_state.last_played == current_player:
                        append_advice(f"{ai_names[ai_index]} 必須出牌，唔可以Pass！")
                        continue
                    else:
                        append_advice(f"{ai_names[ai_index]} passed.")
                        game_state.consecutive_passes += 1
                pygame.display.flip()
                pygame.time.wait(1000)
                advance_turn()
            
            # Check win conditions
            if game_state.player_hand == []:
                append_advice("你贏咗！遊戲結束。")
                game_over_flag = True  ### NEW: Set game over flag
            else:
                for i, hand in enumerate(game_state.ai_hands):
                    if hand == []:
                        append_advice(f"{ai_names[i]} 贏咗！遊戲結束。")
                        game_over_flag = True  ### NEW: Set game over flag
                        break
        else:
            # ### UPDATED: Game is over. Overlay Restart button and check for clicks.
            draw_screen()
            restart_button_rect = pygame.Rect(
                GAME_AREA_X_OFFSET + (GAME_AREA_WIDTH - 200) // 2,
                (HEIGHT - 50) // 2,
                200,
                50
            )
            pygame.draw.rect(screen, (128,0,0), restart_button_rect)
            restart_text = font.render("再玩一局", True, (255,255,255))
            screen.blit(restart_text, restart_text.get_rect(center=restart_button_rect.center))
            pygame.display.flip()
            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if restart_button_rect.collidepoint(event.pos):
                        restart_game()  ### NEW: Restart the game
                        game_over_flag = False
                        break
        
        draw_screen()
    pygame.quit()

if __name__ == "__main__":
    # Print start message in the widget  ### NEW
    append_advice("～～～鋤大DEE牌局開始～～～")
    main_loop()
