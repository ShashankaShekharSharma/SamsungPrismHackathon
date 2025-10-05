# 🎮 APEX AI SURVIVAL ARENA

<div align="center">

**A Multi-Agent Reinforcement Learning Survival Simulation with Betting Mechanics**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Where AI agents fight for survival, and you bet on the champion*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Game Mechanics](#game-mechanics) • [Screenshots](#screenshots)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [🎬 Video Demo](#-video-demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Game Architecture](#game-architecture)
- [Reinforcement Learning Implementation](#reinforcement-learning-implementation)
- [Environment Dynamics](#environment-dynamics)
- [Betting System](#betting-system)
- [Visual System](#visual-system)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [Technical Specifications](#technical-specifications)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## 🎬 Video Demo

[![APEX AI Survival Arena - Full Gameplay](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

**[▶️ Watch Full Gameplay Demo on YouTube](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)**

See the arena in action! This video showcases:
- 🎮 Complete gameplay walkthrough from betting to victory
- 🤖 AI agents competing with different strategies
- 🌍 Dynamic environment and seasonal changes
- 📊 Real-time statistics and leaderboard updates
- 🎨 Modern UI with particle effects and animations

**Duration:** ~10 minutes | **Recommended Quality:** 1080p60

---

## 🎯 Overview

**APEX AI SURVIVAL ARENA** is an advanced multi-agent reinforcement learning simulation where AI agents compete to survive in a dynamically changing environment. The project combines:

- **Deep Q-Network (DQN)** agents with adaptive neural architectures
- **Complex survival mechanics** including hunger, thirst, temperature regulation, and exploration
- **Dynamic environmental systems** with seasons, day/night cycles, and resource management
- **Betting mechanics** that create a competitive gambling-style experience
- **Modern UI/UX** with particle effects, animations, and responsive design

Players select an AI agent to bet on, then watch as multiple agents compete for survival in a procedurally generated wilderness. The last agent standing wins, and successful bets double your credits!

---

## ✨ Features

### 🤖 Multi-Agent AI System
- **8+ Unique Agent Personalities**: Explorer, Survivor, Strategist, Gatherer, Nomad, Guardian, Hunter, Monk
- **Deep Q-Network Architecture**: 512→256→128→64 neurons with dropout regularization
- **Adaptive Model Loading**: Automatically handles different checkpoint formats and state dict variations
- **Greedy Policy Execution**: Real-time decision making based on trained policies

### 🌍 Advanced Environment Simulation
- **Grid-Based World**: 12x12 procedurally generated survival arena
- **Dynamic Resources**: Trees (food) and water sources with randomized placement
- **Fire Mechanics**: Agents can start fires for warmth with realistic duration and particle effects
- **Seasonal System**: 4 seasons (Summer, Autumn, Winter, Spring) with different survival challenges
- **Day/Night Cycle**: Morning and night phases affecting temperature and strategy

### 💰 Betting & Progression System
- **Credit Economy**: Start with 1000 credits, bet on agents to win
- **Multiple Bet Tiers**: 50, 100, 250, 500, 1000 credit options
- **Persistent Progress**: Credits saved between sessions via JSON
- **Risk/Reward Mechanics**: Double your bet on victory, lose it on defeat

### 🎨 Modern Visual Design
- **Particle System**: Discovery particles, death effects, fire animations, water ripples
- **Gradient UI**: Modern color schemes with glassmorphic surfaces
- **Smooth Animations**: Pulsing effects, smooth transitions, dynamic scaling
- **Responsive Layout**: Adapts to different window sizes with scrollable components
- **Focus Mode**: Press 'F' to highlight only your bet agent

### 📊 Comprehensive Statistics
- **Real-Time Leaderboard**: Scrollable ranking with health, hunger, thirst bars
- **Exploration Tracking**: Map completion percentage per agent
- **Resource Discovery**: Trees and water sources found counters
- **Survival Analytics**: Death causes, time survived, health metrics

### ⚙️ Advanced Game Mechanics
- **25-Dimensional State Space**: Position, health, resources, temperature, time, exploration data
- **11 Agent Actions**: Idle, Rest, Movement (4-dir), Eat, Drink, Explore, Swim, Start Fire
- **Mandatory Rest System**: Agents must rest after 240 time steps to prevent burnout
- **Temperature Regulation**: Hypothermia and hyperthermia with emergency cooling/heating
- **Resource Management**: Hunger and thirst increase dynamically based on season and activity

---

## 🚀 Installation

We provide **three different ways** to run APEX AI Survival Arena based on your platform and preferences:

1. **Windows Executable (.exe)** - Ready-to-run application
2. **macOS Application (.dmg)** - Native Mac app bundle
3. **CLI Docker Version** - Terminal-based, cross-platform

---

### 🪟 Option 1: Windows Executable

**Easiest option for Windows users - no installation required!**

#### Steps:

1. Download `APEX-AI-Survival-Arena.exe` from the [Releases](https://github.com/yourusername/apex-ai-survival-arena/releases) page

2. Double-click the `.exe` file to launch

3. Windows Defender may show a warning. Click **"More info"** → **"Run anyway"**
   - This is normal for unsigned applications

**System Requirements:**
- Windows 10 or 11 (64-bit)
- 4GB RAM minimum
- No Python installation needed

**Troubleshooting:**
- If antivirus blocks the file, add it to your exceptions
- Run as administrator if you encounter permission issues

---

### 🍎 Option 2: macOS Application (.dmg)

**Native Mac experience with full GUI support**

#### Prerequisites:

```bash
# Install Python dependencies (required)
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0
```

#### Installation Steps:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/apex-ai-survival-arena.git
cd apex-ai-survival-arena

# 2. Mount the DMG file
hdiutil attach ./APEX-AI-Survival-Arena.dmg

# 3. Copy app to Applications folder
cp -R "/Volumes/APEX AI Survival Arena/APEX AI Survival Arena.app" /Applications/

# 4. Remove quarantine flag (bypass Gatekeeper)
xattr -rd com.apple.quarantine "/Applications/APEX AI Survival Arena.app"

# 5. Launch the application
open "/Applications/APEX AI Survival Arena.app"
```

#### Alternative: Direct Download Installation

```bash
# If you downloaded the DMG separately:

# 1. Install Python dependencies
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0

# 2. Mount the DMG (adjust path if in Downloads)
hdiutil attach ~/Downloads/APEX-AI-Survival-Arena.dmg

# 3. Copy to Applications
cp -R "/Volumes/APEX AI Survival Arena/APEX AI Survival Arena.app" /Applications/

# 4. Remove quarantine flag
xattr -rd com.apple.quarantine "/Applications/APEX AI Survival Arena.app"

# 5. Launch
open "/Applications/APEX AI Survival Arena.app"
```

**System Requirements:**
- macOS 10.14 (Mojave) or later
- 4GB RAM minimum
- Python 3.8+ with dependencies installed

**Troubleshooting:**
- If "App is damaged" error appears, the quarantine removal step is required
- For M1/M2 Macs: Ensure you're using native ARM64 Python packages
- If app won't open: Check System Preferences → Security & Privacy

---

### 🐳 Option 3: CLI Docker Version (Cross-Platform)

**Terminal-based version - works on Windows, Mac, and Linux**

Perfect for:
- Headless servers
- Cloud deployments
- Debugging and development
- Systems without GUI support

#### Prerequisites:

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed

#### Steps:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/apex-ai-survival-arena.git
cd apex-ai-survival-arena

# 2. Navigate to CLI mode directory
cd CLIMode

# 3. Build and run with Docker Compose
docker-compose run --rm --build survival-cli
```

#### Manual Docker Commands:

```bash
# Build the image
docker build -t apex-survival-cli ./CLIMode

# Run the container
docker run --rm -it apex-survival-cli
```

#### CLI Output Example:

```
╔════════════════════════════════════════════════════════════╗
║         APEX AI SURVIVAL ARENA - CLI MODE                  ║
╚════════════════════════════════════════════════════════════╝

Loading models...
✓ Agent 0: explorer_model.pth
✓ Agent 1: survivor_model.pth
✓ Agent 2: strategist_model.pth

Starting simulation...
Day 1 | Summer | Morning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 0 [HP: 98] → MOVE_UP (Found tree!)
Agent 1 [HP: 97] → EXPLORE
Agent 2 [HP: 99] → MOVE_RIGHT

... (simulation continues) ...

╔════════════════════════════════════════════════════════════╗
║                    FINAL RESULTS                           ║
╠════════════════════════════════════════════════════════════╣
║ 🏆 WINNER: Agent 2 (strategist_model.pth)                 ║
║    Final Health: 87 | Survived: 8 days                    ║
║                                                            ║
║ Leaderboard:                                               ║
║  1. Agent 2 - ALIVE (HP: 87)                              ║
║  2. Agent 0 - Day 6 (DEHYDRATION)                         ║
║  3. Agent 1 - Day 4 (HYPOTHERMIA)                         ║
╚════════════════════════════════════════════════════════════╝
```

**Features of CLI Version:**
- Text-based simulation output with progress bars
- Real-time statistics in terminal with colored output
- No GUI overhead (perfect for remote servers)
- Ideal for batch testing multiple models
- Supports headless environments
- JSON output option for data analysis

**System Requirements:**
- Docker Engine 20.10+
- 2GB RAM minimum
- Any OS (Windows, macOS, Linux)

---

### 🛠️ Option 4: Source Code Installation (Developers)

**For development, customization, or if prebuilt packages don't work**

#### Prerequisites:

```bash
Python 3.8 or higher
pip (Python package manager)
```

#### Steps:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/apex-ai-survival-arena.git
cd apex-ai-survival-arena

# 2. Install dependencies
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0

# Or use requirements file:
pip install -r requirements.txt

# 3. Prepare models directory
mkdir models
# Add your .pth model files to the models/ folder

# 4. (Optional) Add custom logo
# Place logo.jpg in root directory for custom branding

# 5. Run the application
python enhanced_aesthetic_survival_betting.py
```

**requirements.txt:**
```
pygame==2.5.2
numpy==1.24.3
torch==2.1.0
```

---

### 📦 Quick Start Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Windows .exe** | No setup, instant launch | Windows only | Windows users wanting quick start |
| **macOS .dmg** | Native app, dock integration | Requires dependencies | Mac users preferring native apps |
| **CLI Docker** | Cross-platform, consistent | No GUI | Servers, automation, debugging |
| **Source Code** | Full customization | Manual setup | Developers, tinkerers |

---

## 🎮 Usage

### Basic Execution

Run with models from the `models/` directory:

```bash
python enhanced_aesthetic_survival_betting.py
```

### Custom Model Paths

Specify models directly:

```bash
python enhanced_aesthetic_survival_betting.py model1.pth model2.pth model3.pth
```

### In-Game Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play/Pause simulation |
| `↑` | Increase simulation speed (up to 100x) |
| `↓` | Decrease simulation speed |
| `F` | Toggle focus mode (highlight bet agent) |
| `F11` | Toggle fullscreen |
| `ESC` | Exit to menu/quit |
| `Mouse Wheel` | Scroll leaderboard or agent selection |

### Gameplay Flow

1. **Home Screen**: Select an agent and bet amount
2. **Simulation**: Watch agents compete for survival
3. **Results Screen**: See outcome, collect winnings, play again

---

## 🏗️ Game Architecture

### Core Components

```
┌─────────────────────────────────────────┐
│         Game State Manager              │
│  (Credits, Bets, History, Persistence)  │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼──────┐       ┌────────▼────────┐
│  Home Screen │       │  Results Screen │
│   (Betting)  │       │   (Outcomes)    │
└───────┬──────┘       └─────────────────┘
        │
┌───────▼──────────────────────────────────┐
│      Main Visualizer                     │
│  ┌────────────────────────────────────┐  │
│  │  Environment (MultiAgentSurvivalEnv)│ │
│  │  - Grid World (12x12)              │  │
│  │  - Resource Generation             │  │
│  │  - Season/Time Systems             │  │
│  │  - Fire Mechanics                  │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │  Agent System (RLAgent + DQN)     │  │
│  │  - Neural Network (PyTorch)       │  │
│  │  - State Processing               │  │
│  │  - Action Selection               │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │  Visual System (Pygame)           │  │
│  │  - Particle Effects               │  │
│  │  - UI Rendering                   │  │
│  │  - Animation System               │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### Class Hierarchy

**Environment Classes:**
- `MultiAgentSurvivalEnv`: Core simulation engine
  - Manages grid world, resources, time, seasons
  - Handles agent state updates and action execution
  - Tracks fires, exploration, and global state

**Agent Classes:**
- `RLAgent`: Interface between models and environment
  - Loads PyTorch DQN models
  - Performs inference for action selection
  - Handles adaptive checkpoint loading
- `DQN`: Neural network architecture
  - 5-layer fully connected network
  - Dropout regularization (30%, 20%, 10%)
  - ReLU activations

**UI Classes:**
- `HomeScreen`: Agent selection and betting interface
- `ResultsScreen`: End-game statistics and outcomes
- `AestheticMultiAgentVisualizer`: Main game renderer
- `ParticleSystem`: Visual effects manager
- `Slider`: Interactive UI control component

**State Management:**
- `GameStateManager`: Persistent credit and bet tracking

---

## 🧠 Reinforcement Learning Implementation

### State Space (25 Dimensions)

```python
[
    pos_x,                    # Normalized position (0-1)
    pos_y,                    # Normalized position (0-1)
    health,                   # Health percentage (0-1)
    hunger,                   # Hunger level (0-1)
    thirst,                   # Thirst level (0-1)
    body_temperature,         # Normalized temp (30-45°C)
    consecutive_rest_count,   # Rest streak counter
    time_progress,            # Simulation progress (0-1)
    in_mandatory_rest,        # Boolean flag
    tree_known,               # Has found any tree
    water_known,              # Has found any water
    exploration_progress,     # Map completion (0-1)
    exploration_complete,     # Boolean flag
    nearest_tree_distance,    # Normalized distance
    nearest_water_distance,   # Normalized distance
    season_index,             # Current season (0-3)
    current_day,              # Day number
    day_progress,             # Time in day (0-1)
    is_morning,               # Boolean flag
    environment_temperature,  # Ambient temp
    temp_too_hot,             # Hyperthermia warning
    temp_too_cold,            # Hypothermia warning
    nearest_fire_distance,    # Distance to fire
    tree_discovery_ratio,     # Found/total trees
    water_discovery_ratio     # Found/total waters
]
```

### Action Space (11 Actions)

1. **IDLE**: Do nothing (minimal resource consumption)
2. **REST**: Recover health (requires mandatory rest state)
3. **MOVE_UP**: Navigate north
4. **MOVE_DOWN**: Navigate south
5. **MOVE_LEFT**: Navigate west
6. **MOVE_RIGHT**: Navigate east
7. **EAT**: Consume food at tree location
8. **DRINK**: Consume water at water source
9. **EXPLORE**: Actively search current cell
10. **SWIM**: Cool down in water
11. **START_FIRE**: Create warmth source

### DQN Architecture

```python
Input Layer (25) 
    → Linear(512) → ReLU → Dropout(0.3)
    → Linear(256) → ReLU → Dropout(0.2)
    → Linear(128) → ReLU → Dropout(0.1)
    → Linear(64)  → ReLU
    → Output Layer (11)
```

### Reward Structure

| Event | Reward |
|-------|--------|
| **Discovery Rewards** |
| Find new tree | +40 |
| Find new water source | +45 |
| Explore new cell | +20 |
| **Survival Actions** |
| Successful eat | +45 |
| Successful drink | +50 |
| Emergency swim (temp > 40°C) | +80 |
| Emergency fire (temp < 34°C) | +75 |
| Complete mandatory rest | +15 |
| **Penalties** |
| Failed action | -20 to -30 |
| Illegal rest | -30 |
| Position timeout (stuck) | -50 per tick |
| Not resting when mandatory | -40 |
| **Death Penalty** | -300 |

---

## 🌡️ Environment Dynamics

### Season System

| Season | Base Temp | Duration | Effects |
|--------|-----------|----------|---------|
| Summer | 35°C | 2 days | Increased thirst (+50%) |
| Autumn | 20°C | 2 days | Balanced |
| Winter | 5°C | 2 days | Increased hunger (+50%), hypothermia risk |
| Spring | 22°C | 2 days | Balanced |

**Night Modifier**: -8°C to base temperature

### Temperature Regulation

- **Optimal Range**: 35-39°C
- **Hypothermia**: < 34°C (2.0 HP/tick damage)
- **Hyperthermia**: > 40°C (2.5 HP/tick damage)
- **Natural Regulation**: 5% return to 37°C per tick
- **Environmental Influence**: 2% per tick
- **Fire Warming**: +0.4°C per tick near fire
- **Swim Cooling**: -2.0 to -3.0°C per swim action

### Resource Mechanics

**Hunger:**
- Base increase: 0.10/tick
- Winter increase: 0.15/tick
- Eating reduction: -50 hunger

**Thirst:**
- Base increase: 0.12/tick
- Summer increase: 0.18/tick
- Drinking reduction: -50 thirst

**Fire:**
- Duration: 180 ticks (3 minutes at 1x speed)
- Cannot place on water/trees
- Provides warmth in radius

### Death Causes

Agents die when health reaches 0, with causes determined by:

1. **SEVERE_DEHYDRATION**: Thirst ≥ 85
2. **STARVATION**: Hunger ≥ 90
3. **HYPOTHERMIA**: Temperature ≤ 34°C
4. **HYPERTHERMIA**: Temperature ≥ 40°C
5. **DUAL_NEGLECT**: Both hunger and thirst ≥ 75
6. **DEHYDRATION**: Thirst ≥ 70
7. **HUNGER**: Hunger ≥ 80
8. **MULTIPLE_FACTORS**: Complex combination

---

## 💸 Betting System

### Credit Economy

- **Starting Credits**: 1000
- **Bet Options**: 50, 100, 250, 500, 1000
- **Win Multiplier**: 2x (double your bet)
- **Loss**: Forfeit bet amount
- **Persistence**: Saved to `game_credits.json`

### Winning Conditions

The agent with the **highest survival priority** wins:

```python
Priority = (alive, health, survival_time)
```

1. Living agents always rank higher than dead
2. Among living: highest health wins
3. Among dead: longest survival time wins

### Strategy Considerations

- **Low Risk**: Bet on agents with proven survival records
- **High Risk**: Bet on explorers who may find resources faster
- **Balanced**: Medium bets on strategists with consistent performance

---

## 🎨 Visual System

### Particle Effects

**Discovery Particles:**
- Spawn on tree/water discovery
- Color-coded (green/blue)
- Radial burst pattern
- 8 particles per discovery

**Death Particles:**
- Red explosion effect
- 12 particles with high velocity
- Fades over 1 second

**Fire Particles:**
- Continuous upward drift
- Color variation (red/orange/yellow)
- Heat shimmer effect

**Water Particles:**
- Circular ripple patterns
- Gentle drift animation

### UI Components

**Modern Design Elements:**
- Rounded rectangles (10-15px radius)
- Gradient backgrounds
- Glassmorphic surfaces
- Glow effects on interactive elements
- Smooth color transitions

**Color Palette:**
- Background: `#000000` (Pure Black)
- Surface: `#191E28` (Dark Blue-Gray)
- Accent: `#40E0D0` (Turquoise)
- Secondary: `#FF6B6B` (Coral Red)
- Success: `#22C55E` (Green)
- Warning: `#FBBF24` (Yellow)
- Danger: `#EF4444` (Red)

### Animation System

- **Agent Pulse**: Sine wave scaling (90-100%)
- **Fire Flicker**: Random + sine wave intensity
- **Water Ripple**: Expanding circles with fade
- **UI Hover**: Smooth color interpolation
- **Glow Effects**: Pulsing alpha channels

---

## ⚙️ Configuration

### Customizable Parameters

**Home Screen Sliders:**
- **Days Duration**: 4-12 days (default: 8)
- **Resource Count**: 4-12 per type (default: 6)

**Environment Constants:**
```python
GRID_SIZE = 12                # World size
MAX_TIME = duration * 360     # Simulation length
FIRE_DURATION = 180           # Fire lifetime (ticks)
SECONDS_PER_DAY = 360         # Day length
```

**Agent Constants:**
```python
INITIAL_HEALTH = 100.0
INITIAL_TEMPERATURE = 37.0
MANDATORY_REST_THRESHOLD = 240
REST_DURATION = 2  # ticks
```

### Performance Settings

- **Base Speed**: 2x (adjustable 1-100x)
- **Target FPS**: 60
- **Particle Limits**: Dynamic culling
- **Render Optimization**: Clipping regions for scrollable areas

---

## 🎓 Model Training

While this project focuses on simulation, here's how to train compatible models:

### Training Environment Setup

```python
from enhanced_aesthetic_survival_betting import MultiAgentSurvivalEnv

env = MultiAgentSurvivalEnv(grid_size=12, max_time=2880)
state_size = 25
action_size = 11
```

### DQN Training Loop (Pseudocode)

```python
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            action = agent.act(state)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Train network
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = agent.train_step(batch)
        
        state = next_state
        total_reward += reward
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Save checkpoint
    if episode % save_interval == 0:
        torch.save(agent.model.state_dict(), f'model_ep{episode}.pth')
```

### Recommended Hyperparameters

```python
learning_rate = 0.0001
gamma = 0.99  # Discount factor
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
batch_size = 64
replay_buffer_size = 100000
target_update_frequency = 10  # episodes
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Additional Agent Architectures**: CNNs, Transformers, Actor-Critic methods
2. **Enhanced Environment**: Weather events, predators, cooperative mechanics
3. **Multiplayer Betting**: Online leaderboards, tournaments
4. **Model Zoo**: Pre-trained agents with different strategies
5. **Analytics Dashboard**: Detailed performance metrics and visualizations

### Development Setup

```bash
git checkout -b feature/your-feature-name
# Make changes
git commit -m "Add your feature"
git push origin feature/your-feature-name
# Create pull request
```

---

## 📊 Technical Specifications

### Performance Metrics

- **Simulation Speed**: 1-100x real-time
- **Max Agents**: Tested up to 16 simultaneous
- **State Updates**: 60 Hz base rate
- **AI Inference**: < 5ms per agent per step (CPU)
- **Render Time**: < 16ms per frame (1080p)

### System Requirements

**Minimum:**
- CPU: Intel i3 or equivalent
- RAM: 4GB
- GPU: Integrated graphics
- Storage: 100MB

**Recommended:**
- CPU: Intel i5 or equivalent
- RAM: 8GB
- GPU: NVIDIA GTX 1050 or equivalent (for faster inference)
- Storage: 500MB

### Platform Support

- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Linux (Ubuntu 20.04+, Fedora, Arch)

---

## 🔧 Troubleshooting

### Common Issues

#### Windows .exe Issues

**Issue: "Windows Defender blocked this app"**
```
Solution: 
1. Click "More info" in the SmartScreen dialog
2. Click "Run anyway"
3. Or add exception in Windows Security settings
```

**Issue: "Application failed to start"**
```
Solution:
1. Right-click the .exe → "Run as administrator"
2. Ensure Windows 10/11 (64-bit) is installed
3. Install Visual C++ Redistributables if prompted
```

---

#### macOS .dmg Issues

**Issue: "'APEX AI Survival Arena' is damaged and can't be opened"**
```bash
# Solution: Remove quarantine attribute
xattr -rd com.apple.quarantine "/Applications/APEX AI Survival Arena.app"
```

**Issue: "Python dependencies not found"**
```bash
# Solution: Reinstall exact versions
pip uninstall pygame numpy torch
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0
```

**Issue: "App crashes on launch (M1/M2 Macs)"**
```bash
# Solution: Ensure ARM64 compatible packages
arch -arm64 pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0
```

**Issue: "Cannot verify developer"**
```
Solution:
1. Go to System Preferences → Security & Privacy
2. Click "Open Anyway" next to the blocked app message
3. Alternatively, use the xattr command above
```

---

#### CLI Docker Issues

**Issue: "docker-compose: command not found"**
```bash
# Solution: Install Docker Compose
# Mac/Linux:
pip install docker-compose
# Or use Docker Desktop which includes Compose
```

**Issue: "Permission denied while trying to connect to Docker daemon"**
```bash
# Linux Solution: Add user to docker group
sudo usermod -aG docker $USER
# Then log out and back in

# Or run with sudo (temporary):
sudo docker-compose run --rm --build survival-cli
```

**Issue: "Build fails with network errors"**
```bash
# Solution: Check internet connection and retry
docker-compose build --no-cache survival-cli
docker-compose run --rm survival-cli
```

---

#### Source Code Issues

**Issue: "No .pth files found"**
```bash
# Solution: Create models directory and add model files
mkdir models
cp your_model.pth models/
```

**Issue: "Could not load icon image"**
```bash
# Solution: Ensure logo.jpg exists or ignore (non-critical)
# The game will run without the logo
```

**Issue: "Models not loading correctly"**
```bash
# Solution: Ensure models have compatible state_dict format
# The adaptive loader should handle most formats automatically
# Check model file isn't corrupted:
python -c "import torch; torch.load('models/yourmodel.pth')"
```

**Issue: "ImportError: No module named 'pygame'"**
```bash
# Solution: Install dependencies with exact versions
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0
```

---

#### Performance Issues

**Issue: Low FPS / Lag**
```bash
# Solution 1: Reduce number of agents (select fewer models)
# Solution 2: Lower simulation speed (press DOWN arrow)
# Solution 3: Close other applications
# Solution 4: Try CLI version for better performance
```

**Issue: "Window too small on high-DPI displays"**
```bash
# Solution: Toggle fullscreen with F11
# Or manually resize window (supports 1200x800 minimum)
```

**Issue: "GPU not being utilized"**
```bash
# Check CUDA availability:
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

#### Platform-Specific Issues

**Linux:**
```bash
# Missing dependencies:
sudo apt-get update
sudo apt-get install python3-pygame python3-numpy

# Audio issues:
sudo apt-get install libasound2-dev
```

**Windows:**
```bash
# Path issues with models:
# Use absolute paths or ensure models/ folder is in same directory as .exe
```

**macOS:**
```bash
# Homebrew Python conflicts:
# Use virtual environment:
python3 -m venv venv
source venv/bin/activate
pip install pygame==2.5.2 numpy==1.24.3 torch==2.1.0
```

---

### Getting Help

If you encounter issues not listed here:

1. Check [GitHub Issues](https://github.com/yourusername/apex-ai-survival-arena/issues)
2. Search existing issues or create a new one
3. Include:
   - OS version and architecture
   - Python version (if applicable)
   - Full error message
   - Steps to reproduce

**Logs Location:**
- Windows: Same folder as .exe
- macOS: `~/Library/Logs/APEX-AI-Survival-Arena/`
- Docker: Use `docker logs <container_id>`

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 👥 Credits

**Developed by:** [Your Name/Team]

**Technologies:**
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Pygame](https://www.pygame.org/) - Game development library
- [NumPy](https://numpy.org/) - Numerical computing

**Inspiration:**
- DeepMind's AlphaGo
- OpenAI's Multi-Agent Systems
- Survival game mechanics from Don't Starve, The Forest

---

## 📞 Contact

- **GitHub**: [Your GitHub Profile]
- **Email**: [Your Email]
- **Discord**: [Your Discord Server]

---

<div align="center">

**⭐ Star this repo if you found it interesting! ⭐**

Made with ❤️ for [Hackathon Name]

</div>
