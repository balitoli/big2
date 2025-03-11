# Hong Kong Big Two (鋤大DEE) - An AI-Driven Card Game

Welcome to my *Hong Kong Big Two* card game, designed with ChatGPT in a 15-hour coding sprint! This project blends *AI coding for beginners* with game design, using Python and Pygame to bring a local classic to life. I leaned on ChatGPT to write most of the code, aiming to see if an LLM could build a playable game—and even play it smartly as opponents. Spoiler: it’s a quirky, fun ride with some flaws! Check out the full story in my blog post: [AI Coding for Beginners: Designing a Hong Kong Big Two Game with ChatGPT in 15 Hours](https://tmleung.xyz/ai-coding-for-beginners-my-15-hour-quest-to-design-a-hong-kong-big-two-game-with-chatgpt/).

## Features

- **Classic Big Two Gameplay**: Play singles, pairs, triples, or five-card combos (straights, flushes, etc.), following Hong Kong rules like the “3D” start and free-play after three passes.
- **AI Opponents**: Three LLM-driven opponents (賭神高進, 賭俠, 賭聖) powered by ChatGPT or Ollama—functional but not genius-level smart.
- **Advice Feature**: Hit '求建議' (Ask for Advice) for Cantonese tips from the LLM on what to play—helpful, though sometimes off the mark.
- **Movie Quotes**: Random quips from Hong Kong gambling films (*God of Gamblers*, *Gambling Hero*) like "五條煙！" (Five aces!) or "又會有條例投降輸一半咁怪都有既！" pop up for nostalgic flair.
- **Fallback Function**: `ai_fallback_move` kicks in when the LLM tries illegal moves (e.g., a pair vs. a triple), keeping the game unstuck with a rule-based backup.
- **Pygame UI**: Clean design with centered card stacks, hidden AI hands, and an advice widget—all tweaked by hand when ChatGPT’s prompts fell short.

## Prerequisites

- **Python 3.x**: Ensure it’s installed ([Download here](https://www.python.org/downloads/)).
- **Pygame**: For the game interface.
- **OpenAI API** or **Ollama**: For LLM-driven AI (opponents and advice). Pick one based on your `config.py` setup.
- **SimHei Font**: For Cantonese text rendering—place `SimHei.ttf` in your project folder (download from a free font site if needed).

## Installation

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/[your-username]/[repo-name].git
   cd [repo-name]
   ```

2. **Install Dependencies**:
   ```bash
   pip install pygame
   pip install openai  # If using OpenAI API
   ```

3. **Set Up LLM Config**:
   - Edit `config.py`:
     - For OpenAI: Add your `OPENAI_API_KEY` and set `LLM_PROVIDER = "OPENAI"`.
     - For Ollama: Adjust `OLLAMA_BASE_URL` (e.g., `http://localhost:11434/v1`) and set `LLM_PROVIDER = "OLLAMA"`.

4. **Add Font**:
   - Drop `SimHei.ttf` into the project root or adjust the font path in the code.

## Running the Game

```bash
python main.py
```

- Click cards to select, then use “出牌” (Play), “Pass,” or “求建議” (Advice).
- Scroll the advice widget with your mouse wheel or drag the scrollbar.
- Enjoy the gambling movie quotes popping up every few minutes!

## How It Came Together

This game was my experiment in *ChatGPT game development*—could an LLM design a full card game with minimal coding from me? I prompted ChatGPT for rules, UI, and AI logic, but it wasn’t perfect. I had to:
- Add rules step-by-step (it started with singles only!).
- Fix the table layout myself in `draw_screen` after prompt fails.
- Solve game stalls with `ai_fallback_move` when ChatGPT suggested it.

The AI opponents and advice aren’t brilliant (despite my hard prompting!), but the fallback keeps it playable. For the full scoop—quirks, fixes, and all—read my blog post: [AI Coding for Beginners: Designing a Hong Kong Big Two Game with ChatGPT in 15 Hours](https://tmleung.xyz/ai-coding-for-beginners-my-15-hour-quest-to-design-a-hong-kong-big-two-game-with-chatgpt/).

## Known Quirks

- **Dumb AI**: Opponents and advice can miss winning moves—LLM smarts hit a ceiling.
- **Sloppy Code**: `current_move` in `ai_make_move` is unused (oops!).
- **Visual Tweaks**: Table layout needed manual coding—ChatGPT struggled with prompts.

## Contributing

Got ideas to make the AI smarter or the design slicker? Fork this repo, tweak away, and send a pull request! I’d love to see your *AI+Design* twists.

## License

This project is open-source under the [MIT License](LICENSE)—feel free to use, modify, or share it.

## Acknowledgments

- ChatGPT for coding most of this (with my nudges).
- Hong Kong gambling movies for the fun quotes.
- You, for checking it out!
