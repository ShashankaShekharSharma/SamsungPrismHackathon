# 🎮 APEX AI SURVIVAL ARENA

<div align="center">

**Multi-Agent Reinforcement Learning Survival Simulation with Betting Mechanics**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Where AI agents fight for survival, and you bet on the champion*

**[▶️ Watch Demo on YouTube](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)**

</div>

---

## 🎯 Overview

**APEX AI SURVIVAL ARENA** combines deep reinforcement learning with survival mechanics and betting gameplay. AI agents powered by Deep Q-Networks compete in a dynamic environment with seasons, resources, and temperature regulation. Bet on your favorite agent and watch them battle for survival!

**Key Features:**
- 🤖 8+ unique AI agent personalities with DQN architecture
- 🌍 Dynamic 12×12 grid world with seasons, day/night cycles, and resource management
- 💰 Betting system with persistent credit economy (start with 1000 credits)
- 🎨 Modern UI with particle effects, animations, and smooth transitions
- 📊 Real-time leaderboard and comprehensive statistics

---

## 🚀 Quick Start

### Windows (.exe)
1. Download `APEX-AI-Survival-Arena.exe` from [Releases](https://github.com/yourusername/apex-ai-survival-arena/releases)
2. Double-click to run
3. If Windows Defender blocks: Click "More info" → "Run anyway"

### macOS (.dmg)
```bash
# Install dependencies
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0

# Mount and install
hdiutil attach ./APEX-AI-Survival-Arena.dmg
cp -R "/Volumes/APEX AI Survival Arena/APEX AI Survival Arena.app" /Applications/
xattr -rd com.apple.quarantine "/Applications/APEX AI Survival Arena.app"
open "/Applications/APEX AI Survival Arena.app"
```

### CLI Docker (Cross-Platform)
```bash
git clone https://github.com/yourusername/apex-ai-survival-arena.git
cd apex-ai-survival-arena/CLIMode
docker-compose run --rm --build survival-cli
```

### From Source
```bash
git clone https://github.com/yourusername/apex-ai-survival-arena.git
cd apex-ai-survival-arena
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0
python enhanced_aesthetic_survival_betting.py
```

---

## 🎮 Gameplay

**Controls:**
- `SPACE` - Play/Pause
- `↑/↓` - Adjust speed (1-100×)
- `F` - Focus mode (highlight bet agent)
- `F11` - Fullscreen
- `Mouse Wheel` - Scroll leaderboard

**How to Play:**
1. Select an agent from 8+ trained models
2. Choose bet amount (50-1000 credits)
3. Watch agents compete for survival
4. Win doubles your bet, loss forfeits it

---

## 🧠 Technical Details

### Agent System
- **Architecture:** 5-layer DQN (512→256→128→64 neurons)
- **State Space:** 25 dimensions (position, health, resources, temperature, exploration)
- **Action Space:** 11 actions (move, eat, drink, explore, start fire, swim, rest)
- **Training:** Deep Q-Learning with experience replay

### Environment
- **World:** 12×12 procedurally generated grid
- **Seasons:** Summer, Autumn, Winter, Spring (2 days each)
- **Mechanics:** Hunger, thirst, temperature regulation, exploration, fire management
- **Death Causes:** Starvation, dehydration, hypothermia, hyperthermia

### Betting System
- Start with 1000 credits
- Bet tiers: 50, 100, 250, 500, 1000
- 2× payout on victory
- Credits persist between sessions

---

## 🏗️ Architecture

```
Game State Manager (Credits & Persistence)
    │
    ├── Home Screen (Agent Selection & Betting)
    │
    ├── Main Visualizer
    │   ├── Multi-Agent Environment (Grid World, Seasons, Resources)
    │   ├── RL Agent System (DQN Inference)
    │   └── Particle Effects & UI Rendering
    │
    └── Results Screen (Statistics & Outcomes)
```

---

## 🔧 Troubleshooting

**Windows:** If SmartScreen blocks, click "More info" → "Run anyway"

**macOS:** If "app is damaged" error appears:
```bash
xattr -rd com.apple.quarantine "/Applications/APEX AI Survival Arena.app"
```

**Docker:** If permission denied:
```bash
sudo usermod -aG docker $USER  # Then logout/login
```

**Performance:** Reduce agents or lower speed (↓ key) for better FPS

---

## 📦 System Requirements

**Minimum:**
- CPU: Intel i3 or equivalent
- RAM: 4GB
- OS: Windows 10/11, macOS 10.14+, Linux

**Recommended:**
- CPU: Intel i5+
- RAM: 8GB
- GPU: GTX 1050+ (optional, for faster inference)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional agent architectures (CNNs, Transformers)
- Enhanced environment mechanics
- Multiplayer betting system
- Pre-trained model zoo

---

## 📜 License

MIT License - See `LICENSE` file for details

---

## 👥 Credits

**Technologies:** PyTorch, Pygame, NumPy

**Inspired by:** DeepMind's AlphaGo, OpenAI Multi-Agent Systems, Don't Starve

---

<div align="center">

**⭐ Star this repo if you enjoyed it! ⭐**

**Contact:** [Your GitHub] | [Your Email]

Made with ❤️ for AI & Gaming

</div>
