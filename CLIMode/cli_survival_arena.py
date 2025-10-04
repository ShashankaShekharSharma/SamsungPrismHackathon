#!/usr/bin/env python3
"""

TO RUN: docker-compose run --rm --build survival-cli

CLI AI Survival Arena
Terminal-based multi-agent survival simulation with betting mechanics
Fixed version with proper state alignment and model loading
"""

import numpy as np
import torch
import torch.nn as nn
import random
import time
import os
import glob
import json
from typing import List, Dict, Tuple, Optional

# ANSI color codes for terminal
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_separator():
    print(f"{Colors.CYAN}{'-'*70}{Colors.RESET}")

# Environment
class MultiAgentSurvivalEnv:
    def __init__(self, grid_size=12, max_time=2880, randomize_resources=True):
        self.grid_size = grid_size
        self.max_time = max_time
        self.randomize_resources = randomize_resources
        self.seconds_per_minute = 60
        self.minutes_per_day = 6
        self.seconds_per_day = self.minutes_per_day * self.seconds_per_minute
        self.total_days = max_time // self.seconds_per_day
        self.season_names = ["Summer", "Autumn", "Winter", "Spring"]
        self.HOME = (6, 6)
        self.TREES = []
        self.WATERS = []
        self.generate_resources()
        self.actions = ["IDLE", "REST", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
                        "EAT", "DRINK", "EXPLORE", "SWIM", "START_FIRE"]
        self.active_fires = {}
        self.fire_duration = 180
        self.agents = []
        self.num_agents = 0
        self.reset()

    def add_agent(self, agent_name, model_path=None):
        agent_id = len(self.agents)
        agent_data = {
            'id': agent_id, 'name': agent_name, 'model_path': model_path,
            'pos': list(self.HOME), 'prev_pos': list(self.HOME),
            'target_pos': list(self.HOME), 'move_progress': 0.0,
            'health': 100.0, 'hunger': 0.0, 'thirst': 0.0,
            'body_temperature': 37.0, 'position_timeout': 0,
            'last_position': tuple(self.HOME),
            'known_tree_locations': set(), 'known_water_locations': set(),
            'explored_positions': set([tuple(self.HOME)]),
            'all_positions_explored': False, 'alive': True,
            'death_time': None, 'death_cause': None,
            'pulse': 0.0, 'last_action': 'IDLE', 'action_time': 0,
            'consecutive_rest_count': 0, 'in_mandatory_rest': False,
            'time_since_last_rest': 0, 'can_rest': False,
            'stats': {'steps': 0, 'trees_found': 0, 'waters_found': 0,
                     'successful_eats': 0, 'successful_drinks': 0,
                     'fires_started': 0, 'exploration_progress': 0}
        }
        self.agents.append(agent_data)
        self.num_agents = len(self.agents)
        return agent_id

    def generate_resources(self):
        if not self.randomize_resources:
            self.TREES = [(2, 8), (7, 1), (1, 3), (9, 6)]
            self.WATERS = [(8, 2), (0, 9), (6, 0), (3, 7)]
            return
        valid_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) != self.HOME]
        random.shuffle(valid_positions)
        self.TREES = valid_positions[:6]
        self.WATERS = valid_positions[6:12]

    def action_space(self):
        return len(self.actions)

    def get_current_season(self):
        current_day = min(self.time // self.seconds_per_day, self.total_days - 1)
        days_per_season = self.total_days // 4
        season_idx = min(3, current_day // days_per_season)
        return self.season_names[season_idx], season_idx

    def get_time_of_day(self):
        seconds_in_current_day = self.time % self.seconds_per_day
        minutes_in_day = seconds_in_current_day // self.seconds_per_minute
        return "Morning" if minutes_in_day < 3 else "Night"

    def get_base_temperature(self):
        season, _ = self.get_current_season()
        time_of_day = self.get_time_of_day()
        season_temps = {"Summer": 35, "Autumn": 20, "Winter": 5, "Spring": 22}
        base_temp = season_temps[season]
        if time_of_day == "Night":
            base_temp -= 8
        return base_temp

    def update_fires(self):
        expired = []
        for pos, rem in self.active_fires.items():
            if rem <= 1:
                expired.append(pos)
            else:
                self.active_fires[pos] = rem - 1
        for pos in expired:
            del self.active_fires[pos]

    def reset(self, regenerate_resources=False):
        if regenerate_resources and self.randomize_resources:
            self.generate_resources()
        self.time = 0
        self.active_fires = {}
        
        for agent in self.agents:
            agent.update({
                'pos': list(self.HOME), 'prev_pos': list(self.HOME),
                'target_pos': list(self.HOME), 'move_progress': 0.0,
                'health': 100.0, 'hunger': 0.0, 'thirst': 0.0,
                'body_temperature': 37.0, 'position_timeout': 0,
                'last_position': tuple(self.HOME),
                'known_tree_locations': set(), 'known_water_locations': set(),
                'explored_positions': set([tuple(self.HOME)]),
                'all_positions_explored': False, 'alive': True,
                'death_time': None, 'death_cause': None,
                'pulse': 0.0, 'last_action': 'IDLE', 'action_time': 0,
                'consecutive_rest_count': 0, 'in_mandatory_rest': False,
                'time_since_last_rest': 0, 'can_rest': False,
            })
            for key in agent['stats']:
                agent['stats'][key] = 0

        self.global_explored = set([tuple(self.HOME)])
        self.all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}

    def get_state_for_agent(self, agent_id):
        """FIXED: Now returns 25 features to match GUI version"""
        if agent_id >= len(self.agents):
            return None
        agent = self.agents[agent_id]
        season, season_idx = self.get_current_season()
        time_of_day = self.get_time_of_day()
        env_temp = self.get_base_temperature()
        tree_known = 1.0 if len(agent['known_tree_locations']) > 0 else 0.0
        water_known = 1.0 if len(agent['known_water_locations']) > 0 else 0.0
        exploration_progress = len(agent['explored_positions']) / (self.grid_size * self.grid_size)
        exploration_complete = 1.0 if agent['all_positions_explored'] else 0.0
        
        nearest_tree_dist = 1.0
        if agent['known_tree_locations']:
            distances = [abs(agent['pos'][0] - tx) + abs(agent['pos'][1] - ty) for tx, ty in agent['known_tree_locations']]
            nearest_tree_dist = min(distances) / (self.grid_size * 2)
        
        nearest_water_dist = 1.0
        if agent['known_water_locations']:
            distances = [abs(agent['pos'][0] - wx) + abs(agent['pos'][1] - wy) for wx, wy in agent['known_water_locations']]
            nearest_water_dist = min(distances) / (self.grid_size * 2)
        
        current_day = self.time // self.seconds_per_day
        day_progress = (self.time % self.seconds_per_day) / self.seconds_per_day
        temp_normalized = (agent['body_temperature'] - 30) / 15
        temp_too_hot = 1.0 if agent['body_temperature'] >= 40.0 else 0.0
        temp_too_cold = 1.0 if agent['body_temperature'] <= 34.0 else 0.0
        
        nearest_fire_dist = 1.0
        if self.active_fires:
            min_fire_dist = min(abs(agent['pos'][0] - fx) + abs(agent['pos'][1] - fy) for fx, fy in self.active_fires.keys())
            nearest_fire_dist = min_fire_dist / (self.grid_size * 2)
        
        nearby_agents = 0
        for other_agent in self.agents:
            if other_agent['id'] != agent_id and other_agent['alive']:
                dist = abs(agent['pos'][0] - other_agent['pos'][0]) + abs(agent['pos'][1] - other_agent['pos'][1])
                if dist <= 2:
                    nearby_agents += 1
        nearby_agent_density = 0.0 if self.num_agents <= 1 else nearby_agents / (self.num_agents - 1)

        return np.array([
            agent['pos'][0] / self.grid_size, agent['pos'][1] / self.grid_size,
            agent['health'] / 100, agent['hunger'] / 100, agent['thirst'] / 100,
            temp_normalized, agent['consecutive_rest_count'] / 2.0,
            self.time / self.max_time, float(agent['in_mandatory_rest']),
            tree_known, water_known, exploration_progress, exploration_complete,
            nearest_tree_dist, nearest_water_dist, season_idx / 3.0,
            current_day / 7.0, day_progress,
            1.0 if time_of_day == "Morning" else 0.0, env_temp / 50.0,
            temp_too_hot, temp_too_cold, nearest_fire_dist,
            len(agent['known_tree_locations']) / max(1, len(self.TREES)),
            len(agent['known_water_locations']) / max(1, len(self.WATERS))
        ], dtype=np.float32)

    def update_body_temperature(self, agent):
        env_temp = self.get_base_temperature()
        current_pos = tuple(agent['pos'])
        temp_diff = env_temp - agent['body_temperature']
        natural_regulation = (37.0 - agent['body_temperature']) * 0.05
        env_influence = temp_diff * 0.02
        location_effect = 0.4 if current_pos in self.active_fires else 0.0
        temp_change = natural_regulation + env_influence + location_effect
        agent['body_temperature'] += temp_change
        agent['body_temperature'] = max(30.0, min(45.0, agent['body_temperature']))

    def step_agent(self, agent_id, action_idx):
        if agent_id >= len(self.agents) or not self.agents[agent_id]['alive']:
            return None, 0, True, "Agent dead or invalid"

        agent = self.agents[agent_id]
        action = self.actions[action_idx]
        reward = 0.0
        info = f"Agent-{agent_id}: "
        current_pos = tuple(agent['pos'])
        
        agent['pulse'] += 0.1
        agent['last_action'] = action
        agent['action_time'] = 10
        agent['stats']['steps'] += 1
        self.update_body_temperature(agent)
        agent['explored_positions'].add(current_pos)
        self.global_explored.add(current_pos)

        if current_pos == agent['last_position']:
            agent['position_timeout'] += 1
        else:
            agent['position_timeout'] = 0
            agent['last_position'] = current_pos

        if not agent['all_positions_explored'] and agent['position_timeout'] >= 5:
            reward -= 50
            info += "[POSITION_TIMEOUT] "

        # Discovery bonuses
        discovery_bonus = 0
        for tree_pos in self.TREES:
            if current_pos == tree_pos and tree_pos not in agent['known_tree_locations']:
                agent['known_tree_locations'].add(tree_pos)
                agent['stats']['trees_found'] += 1
                discovery_bonus += 40
                info += f"[TREE_FOUND at {current_pos}] "
                break
        for water_pos in self.WATERS:
            if current_pos == water_pos and water_pos not in agent['known_water_locations']:
                agent['known_water_locations'].add(water_pos)
                agent['stats']['waters_found'] += 1
                discovery_bonus += 45
                info += f"[WATER_FOUND at {current_pos}] "
                break
        reward += discovery_bonus

        if len(agent['explored_positions']) >= len(self.all_positions):
            agent['all_positions_explored'] = True

        agent['stats']['exploration_progress'] = len(agent['explored_positions']) / len(self.all_positions)

        # Hunger/thirst increase
        if action != "REST" or not (agent['can_rest'] and agent['in_mandatory_rest']):
            season, _ = self.get_current_season()
            hunger_rate = 0.10
            thirst_rate = 0.12
            if season == "Summer":
                thirst_rate = 0.18
            elif season == "Winter":
                hunger_rate = 0.15
            agent['hunger'] = min(100, agent['hunger'] + hunger_rate)
            agent['thirst'] = min(100, agent['thirst'] + thirst_rate)

        agent['time_since_last_rest'] += 1
        
        if agent['time_since_last_rest'] >= 240 or agent['health'] < 20:
            if not agent['in_mandatory_rest']:
                agent['in_mandatory_rest'] = True
                agent['can_rest'] = True
                agent['consecutive_rest_count'] = 0
                info += "[MANDATORY_REST_TRIGGERED] "

        # Execute action
        if action == "REST":
            if agent['can_rest'] and agent['in_mandatory_rest']:
                agent['consecutive_rest_count'] += 1
                agent['hunger'] = min(100, agent['hunger'] + 0.03)
                agent['thirst'] = min(100, agent['thirst'] + 0.04)
                agent['health'] = min(100, agent['health'] + 4)
                
                if agent['consecutive_rest_count'] >= 2:
                    agent['in_mandatory_rest'] = False
                    agent['can_rest'] = False
                    agent['time_since_last_rest'] = 0
                    agent['consecutive_rest_count'] = 0
                    reward += 15
                    info += "REST_COMPLETE"
                else:
                    reward += 8
                    info += f"RESTING ({agent['consecutive_rest_count']}/2)"
            else:
                reward -= 30
                info += "ILLEGAL_REST"
        elif action.startswith("MOVE_"):
            agent['prev_pos'] = list(agent['pos'])
            if action == "MOVE_UP" and agent['pos'][1] > 0:
                agent['pos'][1] -= 1
                info += f"MOVE_UP to {tuple(agent['pos'])}"
            elif action == "MOVE_DOWN" and agent['pos'][1] < self.grid_size - 1:
                agent['pos'][1] += 1
                info += f"MOVE_DOWN to {tuple(agent['pos'])}"
            elif action == "MOVE_LEFT" and agent['pos'][0] > 0:
                agent['pos'][0] -= 1
                info += f"MOVE_LEFT to {tuple(agent['pos'])}"
            elif action == "MOVE_RIGHT" and agent['pos'][0] < self.grid_size - 1:
                agent['pos'][0] += 1
                info += f"MOVE_RIGHT to {tuple(agent['pos'])}"
        elif action == "EAT":
            if current_pos in agent['known_tree_locations'] and agent['hunger'] > 15:
                agent['hunger'] = max(0, agent['hunger'] - 50)
                agent['health'] = min(100, agent['health'] + 6)
                agent['stats']['successful_eats'] += 1
                info += "EAT_SUCCESS"
                reward += 45
            else:
                info += "EAT_FAILED"
                reward -= 20
        elif action == "DRINK":
            if current_pos in agent['known_water_locations'] and agent['thirst'] > 15:
                agent['thirst'] = max(0, agent['thirst'] - 50)
                agent['health'] = min(100, agent['health'] + 5)
                agent['stats']['successful_drinks'] += 1
                info += "DRINK_SUCCESS"
                reward += 50
            else:
                info += "DRINK_FAILED"
                reward -= 20
        elif action == "SWIM":
            if current_pos in agent['known_water_locations']:
                if agent['body_temperature'] >= 40.0:
                    agent['body_temperature'] = max(35.0, agent['body_temperature'] - 3.0)
                    reward += 80
                    info += "EMERGENCY_SWIM"
                elif agent['body_temperature'] >= 38.0:
                    agent['body_temperature'] = max(35.0, agent['body_temperature'] - 2.0)
                    reward += 35
                    info += "COOLING_SWIM"
                else:
                    agent['body_temperature'] = max(30.0, agent['body_temperature'] - 2.0)
                    reward -= 15
                    info += "UNNECESSARY_SWIM"
            else:
                info += "SWIM_FAILED"
                reward -= 20
        elif action == "START_FIRE":
            if (current_pos not in self.WATERS and current_pos not in self.TREES and current_pos not in self.active_fires):
                self.active_fires[current_pos] = self.fire_duration
                agent['stats']['fires_started'] += 1
                if agent['body_temperature'] <= 34.0:
                    agent['body_temperature'] = min(42.0, agent['body_temperature'] + 2.5)
                    reward += 75
                    info += "EMERGENCY_FIRE"
                else:
                    reward += 30
                    info += "FIRE_STARTED"
            else:
                info += "FIRE_FAILED"
                reward -= 25
        elif action == "EXPLORE":
            if current_pos not in agent['explored_positions']:
                reward += 20
                info += "EXPLORING_NEW"
            else:
                reward += 5
                info += "EXPLORING"
        else:
            info += "IDLE"

        if agent['in_mandatory_rest'] and action != "REST":
            reward -= 40
            info += " [MUST_REST] "

        # Health penalties
        health_penalty = 0
        if agent['body_temperature'] >= 40.0:
            health_penalty += 2.5
            info += " HYPERTHERMIA"
        elif agent['body_temperature'] <= 34.0:
            health_penalty += 2.0
            info += " HYPOTHERMIA"
        if agent['hunger'] >= 80:
            health_penalty += 1.5
            info += " CRITICAL_HUNGER"
        if agent['thirst'] >= 75:
            health_penalty += 2.0
            info += " CRITICAL_THIRST"

        agent['health'] = max(0, agent['health'] - health_penalty)
        
        if agent['health'] <= 0:
            agent['alive'] = False
            agent['death_time'] = self.time
            agent['death_cause'] = self.determine_death_cause(agent)
            reward -= 300
            info += " DEATH"

        done = agent['health'] <= 0
        return self.get_state_for_agent(agent_id), reward, done, info

    def determine_death_cause(self, agent):
        if agent['thirst'] >= 85:
            return "SEVERE_DEHYDRATION"
        elif agent['hunger'] >= 90:
            return "STARVATION"
        elif agent['body_temperature'] <= 34:
            return "HYPOTHERMIA"
        elif agent['body_temperature'] >= 40:
            return "HYPERTHERMIA"
        elif agent['thirst'] >= 75 and agent['hunger'] >= 75:
            return "DUAL_NEGLECT"
        elif agent['thirst'] >= 70:
            return "DEHYDRATION"
        elif agent['hunger'] >= 80:
            return "HUNGER"
        else:
            return "MULTIPLE_FACTORS"

    def step_all_agents(self, actions):
        self.time += 1
        self.update_fires()
        
        for agent in self.agents:
            if agent['action_time'] > 0:
                agent['action_time'] -= 1
        
        results = []
        for i, action in enumerate(actions):
            if i < len(self.agents) and self.agents[i]['alive']:
                result = self.step_agent(i, action)
                results.append(result)
            else:
                results.append((None, 0, True, "Agent dead/invalid"))
        
        all_dead = all(not agent['alive'] for agent in self.agents)
        time_up = self.time >= self.max_time
        done = all_dead or time_up
        
        return results, done

    def get_living_agents(self):
        return sum(1 for agent in self.agents if agent['alive'])

    def get_leaderboard(self):
        def sort_key(agent):
            if agent['alive']:
                return (1, agent['health'], self.time)
            else:
                return (0, 0, agent['death_time'] or 0)
        return sorted(self.agents, key=sort_key, reverse=True)


# AI Agent
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class RLAgent:
    def __init__(self, state_size, action_size, model_path=None, name="Agent"):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, path):
        """FIXED: Now uses adaptive loading like GUI version"""
        try:
            if not os.path.exists(path):
                print(f"{Colors.YELLOW}Warning: Model file {path} not found. Using random weights.{Colors.RESET}")
                return False
            raw = torch.load(path, map_location=self.device)
            if isinstance(raw, dict) and ("state_dict" in raw or "model_state_dict" in raw):
                state_dict = raw.get("state_dict", raw.get("model_state_dict"))
            elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
                state_dict = raw
            else:
                state_dict = raw
            
            adapted_sd, adapted_flag = self.adapt_checkpoint_to_model(state_dict, self.model)
            self.model.load_state_dict(adapted_sd)
            status = "adapted" if adapted_flag else "loaded"
            print(f"{Colors.GREEN}Model {status} successfully: {self.name}{Colors.RESET}")
            return True
        except Exception as e:
            print(f"{Colors.YELLOW}Error loading model {path}: {e}. Using random weights.{Colors.RESET}")
            return False
    
    def adapt_checkpoint_to_model(self, ckpt_sd, model):
        """Adaptive model loading from GUI version"""
        model_sd = model.state_dict()
        new_sd = {}
        adapted = False
        
        for k, m_tensor in model_sd.items():
            if k in ckpt_sd:
                v = ckpt_sd[k]
                if isinstance(v, torch.Tensor):
                    v_tensor = v
                else:
                    try:
                        v_tensor = torch.tensor(v)
                    except Exception:
                        v_tensor = None
                if v_tensor is None:
                    new_sd[k] = m_tensor
                    continue
                if v_tensor.shape == m_tensor.shape:
                    new_sd[k] = v_tensor
                else:
                    adapted = True
                    if v_tensor.dim() == 2 and m_tensor.dim() == 2:
                        if v_tensor.shape[1] != m_tensor.shape[1]:
                            if v_tensor.shape[1] > m_tensor.shape[1]:
                                v_tensor = v_tensor[:, :m_tensor.shape[1]]
                            else:
                                pad = torch.zeros((v_tensor.shape[0], m_tensor.shape[1] - v_tensor.shape[1]))
                                v_tensor = torch.cat([v_tensor, pad], dim=1)
                        if v_tensor.shape[0] != m_tensor.shape[0]:
                            if v_tensor.shape[0] > m_tensor.shape[0]:
                                v_tensor = v_tensor[:m_tensor.shape[0], :]
                            else:
                                pad = torch.zeros((m_tensor.shape[0] - v_tensor.shape[0], v_tensor.shape[1]))
                                v_tensor = torch.cat([v_tensor, pad], dim=0)
                        new_sd[k] = v_tensor
                    elif v_tensor.dim() == 1 and m_tensor.dim() == 1:
                        if v_tensor.shape[0] > m_tensor.shape[0]:
                            new_sd[k] = v_tensor[:m_tensor.shape[0]]
                        else:
                            pad = torch.zeros(m_tensor.shape[0] - v_tensor.shape[0])
                            new_sd[k] = torch.cat([v_tensor, pad])
                    else:
                        new_sd[k] = m_tensor
            else:
                new_sd[k] = m_tensor
        return new_sd, adapted

    def act_greedy(self, state):
        with torch.no_grad():
            if state is None:
                return 0
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()


# Game State Manager
class GameStateManager:
    def __init__(self):
        self.credits = 1000
        self.selected_agent = None
        self.bet_amount = 100
        self.load_credits()
    
    def load_credits(self):
        try:
            if os.path.exists('game_credits.json'):
                with open('game_credits.json', 'r') as f:
                    data = json.load(f)
                    self.credits = data.get('credits', 1000)
        except:
            self.credits = 1000
    
    def save_credits(self):
        try:
            with open('game_credits.json', 'w') as f:
                json.dump({'credits': self.credits}, f)
        except:
            pass
    
    def place_bet(self, agent_id, amount):
        if amount <= self.credits:
            self.selected_agent = agent_id
            self.bet_amount = amount
            self.credits -= amount
            return True
        return False
    
    def win_bet(self):
        self.credits += self.bet_amount * 2
        self.save_credits()
    
    def lose_bet(self):
        self.save_credits()


# CLI Interface
def display_status(env, agents, game_state, update_interval=30):
    """Display current game status with enhanced info"""
    if env.time % update_interval != 0:
        return
    
    clear_screen()
    print_header("AI SURVIVAL ARENA - LIVE STATUS")
    
    # Time info
    current_day = (env.time // env.seconds_per_day) + 1
    total_days = env.max_time // env.seconds_per_day
    season, _ = env.get_current_season()
    time_of_day = env.get_time_of_day()
    env_temp = env.get_base_temperature()
    
    print(f"{Colors.BOLD}Day {current_day}/{total_days} | {season} | {time_of_day} | Temp: {env_temp}°C | Time: {env.time}s{Colors.RESET}")
    print(f"{Colors.BOLD}Your Bet: {Colors.YELLOW}{game_state.bet_amount}{Colors.RESET} credits on {Colors.CYAN}Agent #{game_state.selected_agent}{Colors.RESET}\n")
    
    print_separator()
    
    # Leaderboard
    leaderboard = env.get_leaderboard()
    living = env.get_living_agents()
    print(f"{Colors.BOLD}{Colors.GREEN}Living Agents: {living}/{len(env.agents)}{Colors.RESET}\n")
    
    for rank, agent in enumerate(leaderboard[:8], 1):  # Show top 8
        if agent['alive']:
            status_color = Colors.GREEN
            # Create health bar
            health_bar = create_bar(agent['health'], 100, 10, '█', '░')
            hunger_bar = create_bar(100 - agent['hunger'], 100, 10, '█', '░')
            thirst_bar = create_bar(100 - agent['thirst'], 100, 10, '█', '░')
            temp_status = f"{agent['body_temperature']:.1f}°C"
            
            status = f"HP:{health_bar} H:{hunger_bar} T:{thirst_bar} Temp:{temp_status}"
        else:
            status_color = Colors.RED
            days = (agent['death_time'] or 0) // env.seconds_per_day
            status = f"DEAD Day {days} | {agent['death_cause']}"
        
        bet_marker = f" {Colors.YELLOW}★ BET{Colors.RESET}" if agent['id'] == game_state.selected_agent else ""
        
        print(f"{Colors.BOLD}#{rank}{Colors.RESET} {status_color}{agent['name'][:25]}{Colors.RESET}{bet_marker}")
        print(f"   {status}")
        print(f"   Trees: {agent['stats']['trees_found']}/{len(env.TREES)} | "
              f"Waters: {agent['stats']['waters_found']}/{len(env.WATERS)} | "
              f"Map: {agent['stats']['exploration_progress']:.0%} | "
              f"Action: {agent['last_action']}\n")
    
    if len(leaderboard) > 8:
        print(f"{Colors.CYAN}... and {len(leaderboard) - 8} more agents{Colors.RESET}\n")
    
    # Active fires count
    if env.active_fires:
        print(f"{Colors.YELLOW}Active Fires: {len(env.active_fires)}{Colors.RESET}\n")
    
    print_separator()


def create_bar(value, max_value, length, filled_char='█', empty_char='░'):
    """Create a visual progress bar"""
    filled = int((value / max_value) * length)
    bar = filled_char * filled + empty_char * (length - filled)
    
    # Color based on value
    if value >= 70:
        color = Colors.GREEN
    elif value >= 40:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    return f"{color}{bar}{Colors.RESET}"


def run_simulation(env, agents, game_state, speed=10, show_interval=30):
    """Run the simulation with improved display"""
    print_header("SIMULATION START")
    print(f"{Colors.CYAN}Running at {speed}x speed... Press Ctrl+C to abort{Colors.RESET}\n")
    
    done = False
    last_update = 0
    step_times = []
    
    try:
        while not done:
            step_start = time.time()
            
            # Get actions from all agents
            actions = []
            for i, agent_data in enumerate(env.agents):
                if agent_data['alive'] and i < len(agents):
                    state = env.get_state_for_agent(i)
                    if state is not None:
                        # Verify state size matches
                        if len(state) != agents[i].state_size:
                            print(f"{Colors.RED}WARNING: State size mismatch for agent {i}! "
                                  f"Expected {agents[i].state_size}, got {len(state)}{Colors.RESET}")
                            # Pad or truncate to match
                            padded_state = np.zeros(agents[i].state_size, dtype=np.float32)
                            copy_len = min(len(state), agents[i].state_size)
                            padded_state[:copy_len] = state[:copy_len]
                            state = padded_state
                        action = agents[i].act_greedy(state)
                    else:
                        action = 0
                else:
                    action = 0
                actions.append(action)
            
            # Step simulation
            results, done = env.step_all_agents(actions)
            
            step_times.append(time.time() - step_start)
            
            # Display status periodically
            if env.time - last_update >= show_interval:
                display_status(env, agents, game_state, show_interval)
                avg_step_time = np.mean(step_times[-100:]) if step_times else 0
                print(f"{Colors.CYAN}Performance: {1/avg_step_time:.1f} steps/sec{Colors.RESET}")
                last_update = env.time
            
            # Speed control
            if speed < 100:
                time.sleep(0.01 / speed)
            
            # Check if all agents dead
            if env.get_living_agents() == 0:
                done = True
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Simulation interrupted by user{Colors.RESET}")
        done = True
    
    # Final display
    display_status(env, agents, game_state, 1)
    return env.get_leaderboard()


def show_results(leaderboard, game_state, env):
    """Display final results with detailed statistics"""
    clear_screen()
    print_header("GAME OVER - FINAL RESULTS")
    
    winner = leaderboard[0]
    player_won = winner['id'] == game_state.selected_agent
    
    # Win/Loss announcement
    if player_won:
        game_state.win_bet()
        print(f"{Colors.GREEN}{Colors.BOLD}{'★' * 30}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'VICTORY!'.center(30)}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'★' * 30}{Colors.RESET}")
        print(f"{Colors.GREEN}You won {game_state.bet_amount * 2} credits!{Colors.RESET}")
        print(f"{Colors.GREEN}Profit: +{game_state.bet_amount}{Colors.RESET}\n")
    else:
        game_state.lose_bet()
        print(f"{Colors.RED}{Colors.BOLD}{'✗' * 30}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}{'DEFEAT'.center(30)}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}{'✗' * 30}{Colors.RESET}")
        print(f"{Colors.RED}You lost {game_state.bet_amount} credits{Colors.RESET}")
        print(f"{Colors.RED}Your agent: {env.agents[game_state.selected_agent]['name']}{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}Total Credits: {Colors.YELLOW}{game_state.credits}{Colors.RESET}\n")
    
    print_separator()
    
    # Winner details
    print(f"{Colors.BOLD}{Colors.GREEN}WINNER: {winner['name']}{Colors.RESET}\n")
    if winner['alive']:
        print(f"{Colors.GREEN}Status: SURVIVED{Colors.RESET}")
        print(f"Final Health: {winner['health']:.1f}/100")
        print(f"Hunger Level: {winner['hunger']:.1f}/100")
        print(f"Thirst Level: {winner['thirst']:.1f}/100")
        print(f"Body Temp: {winner['body_temperature']:.1f}°C")
    else:
        survival_days = (winner['death_time'] or 0) // env.seconds_per_day
        survival_time = winner['death_time'] or 0
        print(f"{Colors.RED}Status: DIED{Colors.RESET}")
        print(f"Death Cause: {winner['death_cause']}")
        print(f"Survived: {survival_days} days ({survival_time} seconds)")
    
    print(f"\n{Colors.BOLD}Resource Discovery:{Colors.RESET}")
    print(f"Trees Found: {winner['stats']['trees_found']}/{len(env.TREES)}")
    print(f"Waters Found: {winner['stats']['waters_found']}/{len(env.WATERS)}")
    print(f"Map Explored: {winner['stats']['exploration_progress']:.1%}")
    
    print(f"\n{Colors.BOLD}Actions Taken:{Colors.RESET}")
    print(f"Total Steps: {winner['stats']['steps']}")
    print(f"Successful Eats: {winner['stats']['successful_eats']}")
    print(f"Successful Drinks: {winner['stats']['successful_drinks']}")
    print(f"Fires Started: {winner['stats']['fires_started']}")
    
    print_separator()
    print(f"\n{Colors.BOLD}FULL LEADERBOARD:{Colors.RESET}\n")
    
    for rank, agent in enumerate(leaderboard, 1):
        if agent['alive']:
            status = f"{Colors.GREEN}Alive{Colors.RESET} (HP: {agent['health']:.0f})"
        else:
            survival_days = (agent['death_time'] or 0) // env.seconds_per_day
            status = f"{Colors.RED}Dead{Colors.RESET} (Day {survival_days}: {agent['death_cause']})"
        
        bet_marker = f" {Colors.YELLOW}★{Colors.RESET}" if agent['id'] == game_state.selected_agent else ""
        
        print(f"{Colors.BOLD}#{rank}{Colors.RESET} {agent['name'][:30]}{bet_marker}")
        print(f"   Status: {status}")
        print(f"   Resources: T:{agent['stats']['trees_found']}/{len(env.TREES)} "
              f"W:{agent['stats']['waters_found']}/{len(env.WATERS)} "
              f"Exp:{agent['stats']['exploration_progress']:.0%}\n")
    
    # Simulation statistics
    print_separator()
    print(f"\n{Colors.BOLD}SIMULATION STATISTICS:{Colors.RESET}\n")
    print(f"Total Duration: {env.time} seconds ({env.time // 360} days)")
    print(f"Final Survivors: {env.get_living_agents()}/{len(env.agents)}")
    print(f"Total Resources: {len(env.TREES)} trees, {len(env.WATERS)} water sources")
    print(f"Total Map Tiles: {len(env.all_positions)}")
    print(f"Tiles Explored: {len(env.global_explored)} ({len(env.global_explored)/len(env.all_positions):.1%})")


def show_agent_selection_menu(model_files):
    """Enhanced agent selection menu"""
    clear_screen()
    print_header("SELECT YOUR CHAMPION")
    
    print(f"{Colors.BOLD}Available Agents:{Colors.RESET}\n")
    
    for i, model_file in enumerate(model_files):
        name = os.path.splitext(os.path.basename(model_file))[0]
        # Try to extract generation if in filename
        if 'gen' in name.lower():
            gen_info = f" {Colors.CYAN}[{name.split('gen')[-1][:10]}]{Colors.RESET}"
        else:
            gen_info = ""
        
        print(f"{Colors.YELLOW}{i:2d}{Colors.RESET}. {name[:40]}{gen_info}")
        
        # Show file size
        try:
            size = os.path.getsize(model_file)
            size_kb = size / 1024
            print(f"    {Colors.CYAN}Size: {size_kb:.1f} KB{Colors.RESET}")
        except:
            pass
        print()
    
    print_separator()
    return len(model_files)


def main():
    """Main game loop with enhanced features"""
    clear_screen()
    print_header("APEX AI SURVIVAL ARENA")
    print(f"{Colors.CYAN}CLI Edition v2.0 - Enhanced & Fixed{Colors.RESET}")
    print(f"{Colors.GREEN}✓ Proper state alignment (25 features){Colors.RESET}")
    print(f"{Colors.GREEN}✓ Adaptive model loading{Colors.RESET}")
    print(f"{Colors.GREEN}✓ Complete environment parity with GUI{Colors.RESET}\n")
    
    # Load models
    model_files = glob.glob("models/*.pth")
    if not model_files:
        print(f"{Colors.RED}No .pth files found in models/ directory{Colors.RESET}")
        print(f"{Colors.YELLOW}Please place your trained model files in a 'models' folder{Colors.RESET}")
        return
    
    print(f"{Colors.GREEN}Found {len(model_files)} AI agents{Colors.RESET}\n")
    
    # Initialize game state
    game_state = GameStateManager()
    print(f"{Colors.BOLD}Your Credits: {Colors.YELLOW}{game_state.credits}{Colors.RESET}\n")
    
    # Show agents
    num_agents = show_agent_selection_menu(model_files)
    
    # Select agent to bet on
    while True:
        try:
            agent_id = int(input(f"\n{Colors.YELLOW}Select agent to bet on (0-{num_agents-1}): {Colors.RESET}"))
            if 0 <= agent_id < num_agents:
                break
            print(f"{Colors.RED}Invalid selection. Please choose 0-{num_agents-1}{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Game cancelled{Colors.RESET}")
            return
    
    # Select bet amount
    clear_screen()
    print_header("PLACE YOUR BET")
    print(f"{Colors.BOLD}Selected Agent: {Colors.CYAN}{os.path.basename(model_files[agent_id])}{Colors.RESET}\n")
    print(f"{Colors.BOLD}Available Credits: {Colors.YELLOW}{game_state.credits}{Colors.RESET}\n")
    
    bet_amounts = [50, 100, 250, 500, 1000]
    print(f"{Colors.BOLD}Bet Amounts:{Colors.RESET}\n")
    for i, amount in enumerate(bet_amounts):
        if amount <= game_state.credits:
            status = f"{Colors.GREEN}✓ Available{Colors.RESET}"
        else:
            status = f"{Colors.RED}✗ Insufficient credits{Colors.RESET}"
        
        potential_win = amount * 2
        print(f"{i}. {Colors.YELLOW}{amount:4d}{Colors.RESET} credits "
              f"(Win: {Colors.GREEN}{potential_win}{Colors.RESET}) {status}")
    
    print_separator()
    
    while True:
        try:
            bet_idx = int(input(f"\n{Colors.YELLOW}Select bet amount (0-{len(bet_amounts)-1}): {Colors.RESET}"))
            if 0 <= bet_idx < len(bet_amounts):
                bet_amount = bet_amounts[bet_idx]
                if game_state.place_bet(agent_id, bet_amount):
                    break
                print(f"{Colors.RED}Not enough credits for this bet{Colors.RESET}")
            else:
                print(f"{Colors.RED}Invalid selection{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Game cancelled{Colors.RESET}")
            return
    
    # Game settings
    clear_screen()
    print_header("GAME CONFIGURATION")
    
    duration = 8  # days
    resources = 6
    speed = 10
    
    print(f"{Colors.BOLD}Simulation Settings:{Colors.RESET}\n")
    print(f"Duration: {duration} days ({duration * 360} seconds)")
    print(f"Resources: {resources} trees, {resources} water sources")
    print(f"Grid Size: 12x12")
    print(f"Total Agents: {len(model_files)}")
    print(f"Speed: {speed}x\n")
    
    print(f"{Colors.BOLD}Your Bet:{Colors.RESET}\n")
    print(f"Agent: {Colors.CYAN}{os.path.basename(model_files[agent_id])}{Colors.RESET}")
    print(f"Amount: {Colors.YELLOW}{bet_amount}{Colors.RESET} credits")
    print(f"Potential Win: {Colors.GREEN}{bet_amount * 2}{Colors.RESET} credits\n")
    
    print_separator()
    input(f"\n{Colors.YELLOW}Press Enter to start simulation...{Colors.RESET}")
    
    # Initialize environment
    max_time = duration * 360
    env = MultiAgentSurvivalEnv(grid_size=12, max_time=max_time, randomize_resources=True)
    
    # Override resources
    valid_positions = [(x, y) for x in range(12) for y in range(12) if (x, y) != (6, 6)]
    random.shuffle(valid_positions)
    env.TREES = valid_positions[:resources]
    env.WATERS = valid_positions[resources:resources*2]
    
    print(f"\n{Colors.CYAN}Loading AI agents...{Colors.RESET}\n")
    
    # Load all agents
    agents = []
    for i, model_file in enumerate(model_files):
        name = os.path.splitext(os.path.basename(model_file))[0]
        if len(name) > 20:
            name = name[:20] + "..."
        
        agent_id_env = env.add_agent(name, model_file)
        dummy_state = env.get_state_for_agent(agent_id_env)
        state_size = len(dummy_state) if dummy_state is not None else 25
        
        print(f"{Colors.CYAN}Loading agent {i+1}/{len(model_files)}: {name}{Colors.RESET}")
        agent = RLAgent(state_size, env.action_space(), model_file, name)
        agents.append(agent)
    
    print(f"\n{Colors.GREEN}All agents loaded successfully!{Colors.RESET}\n")
    time.sleep(1)
    
    # Run simulation
    leaderboard = run_simulation(env, agents, game_state, speed=speed, show_interval=30)
    
    # Show results
    show_results(leaderboard, game_state, env)
    
    # Play again?
    print_separator()
    try:
        play_again = input(f"\n{Colors.YELLOW}Play again? (y/n): {Colors.RESET}").lower()
        if play_again == 'y':
            main()
        else:
            print(f"\n{Colors.GREEN}Thanks for playing!{Colors.RESET}")
            print(f"{Colors.BOLD}Final Credits: {Colors.YELLOW}{game_state.credits}{Colors.RESET}")
            print(f"{Colors.CYAN}Credits saved to game_credits.json{Colors.RESET}\n")
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}Thanks for playing!{Colors.RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Game interrupted by user{Colors.RESET}")
        print(f"{Colors.GREEN}Thanks for playing!{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Fatal Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        print(f"\n{Colors.YELLOW}Please report this error with the traceback above{Colors.RESET}\n")