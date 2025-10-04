import pygame
import numpy as np
import torch
import torch.nn as nn
import random
import time
import traceback
import os
import glob
import math

pygame.init()

WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 1000
GRID_SIZE = 12
CELL_SIZE = 45
GRID_OFFSET_X = 80
GRID_OFFSET_Y = 120


BACKGROUND = (15, 18, 25)        
SURFACE = (25, 30, 40)           
ACCENT = (64, 224, 208)          
SECONDARY = (255, 107, 107)      
TEXT_PRIMARY = (245, 245, 245)  
TEXT_SECONDARY = (156, 163, 175)
BORDER = (55, 65, 81)           
SUCCESS = (34, 197, 94)          
WARNING = (251, 191, 36)       
DANGER = (239, 68, 68)       
WHITE = (255, 255, 255)         
BLACK = (0, 0, 0)                

UNEXPLORED = (30, 35, 45)
EXPLORED = (45, 52, 65)
HOME_COLOR = (147, 51, 234)    
TREE_COLOR = (34, 197, 94)      
WATER_COLOR = (59, 130, 246)    
FIRE_COLOR = (251, 146, 60)      
AGENT_COLORS = [
    (255, 71, 87),    
    (72, 207, 173),  
    (116, 125, 251), 
    (255, 184, 0),   
    (255, 105, 180),  
    (154, 230, 180),  
    (255, 159, 67),   
    (162, 155, 254), 
]

SEASON_COLORS = {
    "Summer": (255, 223, 186),
    "Autumn": (255, 183, 77),
    "Winter": (186, 230, 253),
    "Spring": (187, 247, 208)
}

FONT_PATH = None  
FONT_SIZES = {
    'small': 14,
    'medium': 16,
    'large': 20,
    'xlarge': 28,
    'title': 36
}

def load_fonts():
    """Load modern fonts with fallbacks"""
    fonts = {}
    font_names = ['SF Pro Display', 'Segoe UI', 'Arial']
    
    for size_name, size in FONT_SIZES.items():
        font_loaded = False
        for font_name in font_names:
            try:
                fonts[size_name] = pygame.font.SysFont(font_name, size)
                font_loaded = True
                break
            except:
                continue
        
        if not font_loaded:
            fonts[size_name] = pygame.font.Font(None, size)
    
    return fonts

FONTS = load_fonts()

class ParticleSystem:
    """Simple particle system for visual effects"""
    def __init__(self):
        self.particles = []
    
    def add_discovery_particles(self, x, y, color, count=8):
        """Add particles for resource discovery"""
        for _ in range(count):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-2, 2),
                'life': 1.0,
                'color': color,
                'size': random.uniform(2, 4)
            })
    
    def add_death_particles(self, x, y, count=12):
        """Add particles for agent death"""
        for _ in range(count):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-3, 3),
                'life': 1.0,
                'color': DANGER,
                'size': random.uniform(3, 6)
            })
    
    def update(self):
        """Update all particles"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 0.02
            particle['size'] *= 0.98
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        """Draw all particles"""
        for particle in self.particles:
            alpha = int(255 * particle['life'])
            color = (*particle['color'], alpha)
            size = max(1, int(particle['size']))
            
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size, size), size)
            screen.blit(surf, (particle['x'] - size, particle['y'] - size))

def draw_rounded_rect(surface, color, rect, radius=10, border_width=0, border_color=None):
    """Draw a rounded rectangle"""
    if border_width > 0 and border_color:
        pygame.draw.rect(surface, border_color, rect.inflate(border_width * 2, border_width * 2), border_radius=radius)
    
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def lerp_color(color1, color2, t):
    """Linear interpolation between two colors"""
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))

def draw_gradient_rect(surface, color1, color2, rect, horizontal=True):
    """Draw a gradient rectangle"""
    if horizontal:
        for i in range(rect.width):
            t = i / rect.width
            color = lerp_color(color1, color2, t)
            pygame.draw.line(surface, color, 
                           (rect.left + i, rect.top), 
                           (rect.left + i, rect.bottom))
    else:
        for i in range(rect.height):
            t = i / rect.height
            color = lerp_color(color1, color2, t)
            pygame.draw.line(surface, color, 
                           (rect.left, rect.top + i), 
                           (rect.right, rect.top + i))

class MultiAgentSurvivalEnv:
    def __init__(self, grid_size=12, max_time=2880, randomize_resources=True):
        self.grid_size = grid_size
        self.max_time = max_time
        self.randomize_resources = randomize_resources


        self.seconds_per_minute = 60
        self.minutes_per_day = 6
        self.seconds_per_day = self.minutes_per_day * self.seconds_per_minute
        self.total_days = 8

        self.season_names = ["Summer", "Autumn", "Winter", "Spring"]
        self.season_durations = [2, 2, 2, 2]

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
        """Add a new agent to the environment"""
        agent_id = len(self.agents)
        agent_data = {
            'id': agent_id,
            'name': agent_name,
            'model_path': model_path,
            'pos': list(self.HOME),
            'prev_pos': list(self.HOME),  
            'target_pos': list(self.HOME),
            'move_progress': 0.0,
            'health': 100.0,
            'hunger': 0.0,
            'thirst': 0.0,
            'body_temperature': 37.0,
            'position_timeout': 0,
            'last_position': tuple(self.HOME),
            'known_tree_locations': set(),
            'known_water_locations': set(),
            'explored_positions': set([tuple(self.HOME)]),
            'all_positions_explored': False,
            'alive': True,
            'death_time': None,
            'death_cause': None,
            'color': AGENT_COLORS[agent_id % len(AGENT_COLORS)],
            'pulse': 0.0,  
            'last_action': 'IDLE',
            'action_time': 0,
            'consecutive_rest_count': 0,
            'in_mandatory_rest': False,
            'time_since_last_rest': 0,
            'can_rest': False,
            'stats': {
                'steps': 0,
                'trees_found': 0,
                'waters_found': 0,
                'successful_eats': 0,
                'successful_drinks': 0,
                'fires_started': 0,
                'exploration_progress': 0
            }
        }
        self.agents.append(agent_data)
        self.num_agents = len(self.agents)
        return agent_id

    def generate_resources(self):
        if not self.randomize_resources:
            self.TREES = [(2, 8), (7, 1), (1, 3), (9, 6)]
            self.WATERS = [(8, 2), (0, 9), (6, 0), (3, 7)]
            return

        self.TREES = []
        self.WATERS = []
        valid_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) != self.HOME]
        random.shuffle(valid_positions)
        self.TREES = valid_positions[:6] 
        self.WATERS = valid_positions[6:12]

    def action_space(self):
        return len(self.actions)

    def get_current_season(self):
        current_day = min(self.time // self.seconds_per_day, self.total_days - 1)
        if current_day < 2:
            return "Summer", 0
        elif current_day < 4:
            return "Autumn", 1
        elif current_day < 6:
            return "Winter", 2
        else:
            return "Spring", 3

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
            agent['pos'] = list(self.HOME)
            agent['prev_pos'] = list(self.HOME)
            agent['target_pos'] = list(self.HOME)
            agent['move_progress'] = 0.0
            agent['health'] = 100.0
            agent['hunger'] = 0.0
            agent['thirst'] = 0.0
            agent['body_temperature'] = 37.0
            agent['position_timeout'] = 0
            agent['last_position'] = tuple(self.HOME)
            agent['known_tree_locations'] = set()
            agent['known_water_locations'] = set()
            agent['explored_positions'] = set([tuple(self.HOME)])
            agent['all_positions_explored'] = False
            agent['alive'] = True
            agent['death_time'] = None
            agent['death_cause'] = None
            agent['pulse'] = 0.0
            agent['last_action'] = 'IDLE'
            agent['action_time'] = 0
            agent['consecutive_rest_count'] = 0
            agent['in_mandatory_rest'] = False
            agent['time_since_last_rest'] = 0
            agent['can_rest'] = False
            for key in agent['stats']:
                agent['stats'][key] = 0

        self.global_explored = set([tuple(self.HOME)])
        self.all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}

    def get_state_for_agent(self, agent_id):
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
        nearest_water_dist = 1.0
        if agent['known_tree_locations']:
            distances = [abs(agent['pos'][0] - tx) + abs(agent['pos'][1] - ty) for tx, ty in agent['known_tree_locations']]
            nearest_tree_dist = min(distances) / (self.grid_size * 2)
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
            agent['pos'][0] / self.grid_size,        # 0: Agent X position
            agent['pos'][1] / self.grid_size,        # 1: Agent Y position  
            agent['health'] / 100,                   # 2: Health
            agent['hunger'] / 100,                   # 3: Hunger
            agent['thirst'] / 100,                   # 4: Thirst
            temp_normalized,                         # 5: Body temperature
            agent['consecutive_rest_count'] / 2.0,   # 6: Rest count
            self.time / self.max_time,               # 7: Time progress
            float(agent['in_mandatory_rest']),       # 8: In mandatory rest
            tree_known,                              # 9: Tree location known
            water_known,                             # 10: Water location known
            exploration_progress,                    # 11: Exploration progress
            exploration_complete,                    # 12: Exploration complete
            nearest_tree_dist,                       # 13: Distance to nearest tree
            nearest_water_dist,                      # 14: Distance to nearest water
            season_idx / 3.0,                        # 15: Season (0-1)
            current_day / 7.0,                       # 16: Day progress
            day_progress,                            # 17: Day time progress
            1.0 if time_of_day == "Morning" else 0.0, # 18: Is morning
            env_temp / 50.0,                         # 19: Environmental temp
            temp_too_hot,                            # 20: Temperature critical high
            temp_too_cold,                           # 21: Temperature critical low
            nearest_fire_dist,                       # 22: Distance to nearest active fire
            len(agent['known_tree_locations']) / max(1, len(self.TREES)),  # 23: Trees discovered ratio
            len(agent['known_water_locations']) / max(1, len(self.WATERS)) # 24: Waters discovered ratio
        ], dtype=np.float32)

    def update_body_temperature(self, agent):
        env_temp = self.get_base_temperature()
        current_pos = tuple(agent['pos'])
        temp_diff = env_temp - agent['body_temperature']
        natural_regulation = (37.0 - agent['body_temperature']) * 0.05
        env_influence = temp_diff * 0.02
        location_effect = 0.0
        
        if current_pos in self.active_fires:
            location_effect = 0.4
        
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
        
        if agent['time_since_last_rest'] >= 240 or agent['health'] < 20:  # 4 minutes = 240 seconds
            if not agent['in_mandatory_rest']:
                agent['in_mandatory_rest'] = True
                agent['can_rest'] = True
                agent['consecutive_rest_count'] = 0
                info += "[MANDATORY_REST_TRIGGERED] "

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

    def get_valid_actions_for_agent(self, agent_id):
        """
        Returns a list of valid action indices for the current agent state.
        For example, prevents REST if not allowed, EAT if not on tree, DRINK if not on water, etc.
        """
        if agent_id >= len(self.agents) or not self.agents[agent_id]['alive']:
            return []
            
        agent = self.agents[agent_id]
        valid = []
        for idx, action in enumerate(self.actions):
            if action == "REST":
                if agent['can_rest'] and agent['in_mandatory_rest']:
                    valid.append(idx)
            elif action == "EAT":
                if tuple(agent['pos']) in self.TREES:
                    valid.append(idx)
            elif action == "DRINK":
                if tuple(agent['pos']) in self.WATERS:
                    valid.append(idx)
            else:
                valid.append(idx)
        return valid

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

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
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
        try:
            if not os.path.exists(path):
                print(f"Warning: Model file {path} not found. Using random weights.")
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
            print(f"Model {status} successfully: {self.name} from {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model {path}: {e}")
            print(f"Agent {self.name} will use random weights.")
            return False

    def adapt_checkpoint_to_model(self, ckpt_sd, model):
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

class AestheticMultiAgentVisualizer:
    def __init__(self, model_files=None):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("✨ Multi-Agent Survival Arena - Windowed Mode")
        
        try:
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = 'centered'
        except:
            pass
        
        try:
            icon = pygame.Surface((32, 32))
            icon.fill(ACCENT)
            pygame.display.set_icon(icon)
        except:
            pass
            
        self.clock = pygame.time.Clock()
        
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.min_width = 1200
        self.min_height = 800
        self.is_fullscreen = False
        self.scale_factor = 1.0

        self.env = MultiAgentSurvivalEnv(randomize_resources=True)
        self.agents = []
        self.particles = ParticleSystem()
        
        if model_files:
            self.load_agents_from_files(model_files)
        else:
            pth_files = glob.glob("*.pth")
            if pth_files:
                self.load_agents_from_files(pth_files[:min(8, len(pth_files))])
            else:
                print("No .pth files found. Creating random agents.")
                self.create_random_agents(4)

        self.paused = False
        self.speed = 1
        self.last_step_time = 0
        self.step_count = 0
        self.simulation_count = 0
        self.time_since_start = 0
        
        self.season_transition = 0.0
        self.ui_animations = {}
        
        self.reset_simulation()

    def load_agents_from_files(self, model_files):
        print(f"🤖 Loading {len(model_files)} AI agents...")
        
        for i, model_file in enumerate(model_files):
            name = os.path.splitext(os.path.basename(model_file))[0]
            if len(name) > 12:
                name = name[:12] + "..."
            
            agent_id = self.env.add_agent(name, model_file)
            dummy_state = self.env.get_state_for_agent(agent_id)
            state_size = len(dummy_state) if dummy_state is not None else 25
            
            agent = RLAgent(state_size, self.env.action_space(), model_file, name)
            self.agents.append(agent)
            print(f"  🎯 Agent {i}: {name}")

    def create_random_agents(self, num_agents):
        print(f"🎲 Creating {num_agents} random agents...")
        for i in range(num_agents):
            name = f"Random-{i+1}"
            agent_id = self.env.add_agent(name)
            dummy_state = self.env.get_state_for_agent(agent_id)
            state_size = len(dummy_state) if dummy_state is not None else 25
            
            agent = RLAgent(state_size, self.env.action_space(), None, name)
            self.agents.append(agent)

    def reset_simulation(self, new_environment=False):
        self.env.reset(regenerate_resources=new_environment)
        self.step_count = 0
        self.time_since_start = 0
        self.particles = ParticleSystem()
        
        if new_environment:
            self.simulation_count += 1
            print(f"\n🚀 NEW SIMULATION #{self.simulation_count}")

    def step_forward(self):
        if self.env.time >= self.env.max_time:
            return

        living_agents = [i for i, agent in enumerate(self.env.agents) if agent['alive']]
        if not living_agents:
            return

        actions = []
        for i, agent_data in enumerate(self.env.agents):
            if agent_data['alive'] and i < len(self.agents):
                state = self.env.get_state_for_agent(i)
                if state is not None:
                    if len(state) != self.agents[i].state_size:
                        padded_state = np.zeros(self.agents[i].state_size, dtype=np.float32)
                        copy_len = min(len(state), self.agents[i].state_size)
                        padded_state[:copy_len] = state[:copy_len]
                        state = padded_state
                    action = self.agents[i].act_greedy(state)
                else:
                    action = 0
            else:
                action = 0
            actions.append(action)

        prev_discoveries = {}
        prev_alive = {}
        for i, agent_data in enumerate(self.env.agents):
            prev_discoveries[i] = (len(agent_data['known_tree_locations']), len(agent_data['known_water_locations']))
            prev_alive[i] = agent_data['alive']

        results, done = self.env.step_all_agents(actions)
        self.step_count += 1

        for i, agent_data in enumerate(self.env.agents):
            new_trees = len(agent_data['known_tree_locations']) - prev_discoveries[i][0]
            new_waters = len(agent_data['known_water_locations']) - prev_discoveries[i][1]
            
            if new_trees > 0 or new_waters > 0:
                x = GRID_OFFSET_X + agent_data['pos'][0] * CELL_SIZE + CELL_SIZE//2
                y = GRID_OFFSET_Y + agent_data['pos'][1] * CELL_SIZE + CELL_SIZE//2
                color = TREE_COLOR if new_trees > 0 else WATER_COLOR
                self.particles.add_discovery_particles(x, y, color)
            
            if prev_alive[i] and not agent_data['alive']:
                x = GRID_OFFSET_X + agent_data['pos'][0] * CELL_SIZE + CELL_SIZE//2
                y = GRID_OFFSET_Y + agent_data['pos'][1] * CELL_SIZE + CELL_SIZE//2
                self.particles.add_death_particles(x, y)

        return done

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.window_width = max(event.w, self.min_width)
                self.window_height = max(event.h, self.min_height)
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
                print(f"Window resized to: {self.window_width}x{self.window_height}")
                
                self.update_scaling_factors()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_simulation(new_environment=False)
                elif event.key == pygame.K_n:
                    self.reset_simulation(new_environment=True)
                elif event.key == pygame.K_UP:
                    self.speed = min(10, self.speed + 1)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 1)
                elif event.key == pygame.K_1:
                    self.speed = 1
                elif event.key == pygame.K_2:
                    self.speed = 2
                elif event.key == pygame.K_3:
                    self.speed = 5
                elif event.key == pygame.K_4:
                    self.speed = 10
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_F10:
                    self.reset_window_size()
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True

    def update_scaling_factors(self):
        """Update UI scaling factors based on window size"""
        width_ratio = self.window_width / WINDOW_WIDTH
        height_ratio = self.window_height / WINDOW_HEIGHT
        
        self.scale_factor = min(width_ratio, height_ratio)
        print(f"Scale factor updated to: {self.scale_factor:.2f}")

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        if hasattr(self, 'is_fullscreen') and self.is_fullscreen:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            self.window_width = WINDOW_WIDTH
            self.window_height = WINDOW_HEIGHT
            self.is_fullscreen = False
            pygame.display.set_caption("✨ Multi-Agent Survival Arena - Windowed Mode")
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.window_width = info.current_w
            self.window_height = info.current_h
            self.is_fullscreen = True
            pygame.display.set_caption("✨ Multi-Agent Survival Arena - Fullscreen Mode")
        print(f"Toggled to {'fullscreen' if self.is_fullscreen else 'windowed'} mode")

    def reset_window_size(self):
        """Reset window to default size"""
        if not hasattr(self, 'is_fullscreen') or not self.is_fullscreen:
            self.window_width = WINDOW_WIDTH
            self.window_height = WINDOW_HEIGHT
            self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
            print(f"Window reset to default size: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")

    def draw_beautiful_grid(self):
        """Draw the game grid with modern aesthetics"""
        season, _ = self.env.get_current_season()
        season_color = SEASON_COLORS[season]
        
        grid_rect = pygame.Rect(GRID_OFFSET_X - 5, GRID_OFFSET_Y - 5, 
                               GRID_SIZE * CELL_SIZE + 10, GRID_SIZE * CELL_SIZE + 10)
        draw_rounded_rect(self.screen, SURFACE, grid_rect, 15, 2, BORDER)

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                cell_x = GRID_OFFSET_X + x * CELL_SIZE
                cell_y = GRID_OFFSET_Y + y * CELL_SIZE
                cell_rect = pygame.Rect(cell_x + 1, cell_y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                pos = (x, y)

                if pos in self.env.global_explored:
                    cell_color = lerp_color(EXPLORED, season_color, 0.15)
                else:
                    cell_color = UNEXPLORED
                
                draw_rounded_rect(self.screen, cell_color, cell_rect, 8)
                
                center_x = cell_x + CELL_SIZE // 2
                center_y = cell_y + CELL_SIZE // 2
                
                # Trees
                if pos in self.env.TREES:
                    any_knows = any(pos in agent['known_tree_locations'] for agent in self.env.agents)
                    if any_knows or pos in self.env.global_explored:
                        tree_size = 12
                        pygame.draw.circle(self.screen, TREE_COLOR, 
                                         (center_x, center_y - 3), tree_size)
                        trunk_rect = pygame.Rect(center_x - 2, center_y + 5, 4, 8)
                        pygame.draw.rect(self.screen, (101, 67, 33), trunk_rect)
                        
                        glow_surf = pygame.Surface((tree_size * 3, tree_size * 3), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surf, (*TREE_COLOR, 30), 
                                         (tree_size * 3 // 2, tree_size * 3 // 2), tree_size)
                        self.screen.blit(glow_surf, (center_x - tree_size * 3 // 2, 
                                                   center_y - 3 - tree_size * 3 // 2))

                # Water
                if pos in self.env.WATERS:
                    any_knows = any(pos in agent['known_water_locations'] for agent in self.env.agents)
                    if any_knows or pos in self.env.global_explored:
                        water_size = 10
                        ripple_time = (self.time_since_start * 0.1) % (math.pi * 2)
                        ripple_offset = int(math.sin(ripple_time) * 2)
                        
                        pygame.draw.circle(self.screen, WATER_COLOR, 
                                         (center_x, center_y), water_size + ripple_offset)
                        pygame.draw.circle(self.screen, lerp_color(WATER_COLOR, WHITE, 0.3), 
                                         (center_x - 2, center_y - 2), water_size//2)

                if pos in self.env.active_fires:
                    fire_time = self.env.active_fires[pos]
                    intensity = min(1.0, fire_time / self.env.fire_duration)
                    
                    flicker = math.sin(self.time_since_start * 0.3) * 0.2 + 0.8
                    fire_size = int(15 * intensity * flicker)
                    
                    fire_color = lerp_color(FIRE_COLOR, (255, 255, 100), flicker)
                    pygame.draw.circle(self.screen, fire_color, (center_x, center_y), fire_size)
                    pygame.draw.circle(self.screen, (255, 255, 150), (center_x, center_y), fire_size//2)
                    
                    timer_text = FONTS['small'].render(f"{fire_time//10}", True, TEXT_PRIMARY)
                    timer_rect = timer_text.get_rect(center=(center_x, center_y + 20))
                    draw_rounded_rect(self.screen, (0, 0, 0, 128), timer_rect.inflate(4, 2), 3)
                    self.screen.blit(timer_text, timer_rect)

                if pos == self.env.HOME:
                    home_size = 16
                    draw_rounded_rect(self.screen, HOME_COLOR, 
                                    pygame.Rect(center_x - home_size//2, center_y - home_size//2, 
                                               home_size, home_size), 4)

                    pygame.draw.polygon(self.screen, (255, 215, 0), [
                        (center_x, center_y - 8),
                        (center_x - 6, center_y - 2),
                        (center_x + 6, center_y - 2)
                    ])

    def draw_beautiful_agents(self):
        """Draw agents with smooth animations and modern styling"""
        for i, agent_data in enumerate(self.env.agents):
            if not agent_data['alive']:
                continue
            
            target_x = GRID_OFFSET_X + agent_data['pos'][0] * CELL_SIZE + CELL_SIZE//2
            target_y = GRID_OFFSET_Y + agent_data['pos'][1] * CELL_SIZE + CELL_SIZE//2
            
            agents_in_cell = [a for a in self.env.agents if tuple(a['pos']) == tuple(agent_data['pos']) and a['alive']]
            if len(agents_in_cell) > 1:
                index_in_cell = [a['id'] for a in agents_in_cell].index(agent_data['id'])
                angle = (index_in_cell * 2 * math.pi) / len(agents_in_cell)
                offset = 8
                target_x += int(math.cos(angle) * offset)
                target_y += int(math.sin(angle) * offset)
            
            pulse = math.sin(agent_data['pulse']) * 0.1 + 0.9
            health_ratio = agent_data['health'] / 100.0
            agent_size = int(16 * health_ratio * pulse)
            
            agent_color = (*agent_data['color'], int(255 * health_ratio))
            
            agent_surf = pygame.Surface((agent_size * 3, agent_size * 3), pygame.SRCALPHA)
            
            glow_color = (*agent_data['color'], 60)
            pygame.draw.circle(agent_surf, glow_color, 
                             (agent_size * 3 // 2, agent_size * 3 // 2), agent_size + 6)
            
            pygame.draw.circle(agent_surf, agent_data['color'], 
                             (agent_size * 3 // 2, agent_size * 3 // 2), agent_size)
            
            highlight_color = lerp_color(agent_data['color'], WHITE, 0.4)
            pygame.draw.circle(agent_surf, highlight_color, 
                             (agent_size * 3 // 2 - 2, agent_size * 3 // 2 - 2), agent_size//2)
            
            self.screen.blit(agent_surf, (target_x - agent_size * 3 // 2, target_y - agent_size * 3 // 2))
            
            id_text = FONTS['small'].render(str(i), True, TEXT_PRIMARY)
            id_rect = pygame.Rect(target_x - 8, target_y - agent_size - 18, 16, 14)
            draw_rounded_rect(self.screen, agent_data['color'], id_rect, 7)
            draw_rounded_rect(self.screen, (0, 0, 0, 100), id_rect, 7)
            
            id_text_rect = id_text.get_rect(center=id_rect.center)
            self.screen.blit(id_text, id_text_rect)
            
            temp_ratio = (agent_data['body_temperature'] - 30) / 15
            temp_color = lerp_color((100, 150, 255), (255, 100, 100), temp_ratio)
            temp_rect = pygame.Rect(target_x + agent_size + 5, target_y - 6, 3, 12)
            draw_rounded_rect(self.screen, temp_color, temp_rect, 2)
            
            if agent_data['action_time'] > 0:
                action_text = FONTS['small'].render(agent_data['last_action'][:4], True, TEXT_SECONDARY)
                action_rect = action_text.get_rect(center=(target_x, target_y + agent_size + 15))
                action_bg = action_rect.inflate(6, 2)
                draw_rounded_rect(self.screen, (0, 0, 0, 120), action_bg, 3)
                self.screen.blit(action_text, action_rect)

    def draw_elegant_stats_bar(self, x, y, width, height, value, max_value, color, label, icon=None):
        """Draw an elegant progress bar with modern styling"""

        bg_rect = pygame.Rect(x, y, width, height)
        draw_rounded_rect(self.screen, (0, 0, 0, 50), bg_rect, height//2)
        
        fill_ratio = value / max_value
        fill_width = int(width * fill_ratio)
        if fill_width > 0:
            fill_rect = pygame.Rect(x, y, fill_width, height)
            
            grad_surf = pygame.Surface((fill_width, height))
            for i in range(fill_width):
                t = i / max(fill_width, 1)
                grad_color = lerp_color(color, lerp_color(color, WHITE, 0.3), t * 0.3)
                pygame.draw.line(grad_surf, grad_color, (i, 0), (i, height))
            
            grad_surf = pygame.transform.smoothscale(grad_surf, (fill_width, height))
            
            mask_surf = pygame.Surface((fill_width, height), pygame.SRCALPHA)
            draw_rounded_rect(mask_surf, (255, 255, 255), pygame.Rect(0, 0, fill_width, height), height//2)
            grad_surf.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
            
            self.screen.blit(grad_surf, (x, y))
        
        label_text = f"{label}: {value:.0f}"
        text_surf = FONTS['small'].render(label_text, True, TEXT_SECONDARY)
        self.screen.blit(text_surf, (x + width + 10, y + (height - text_surf.get_height()) // 2))

    def draw_modern_info_panel(self):
        """Draw the information panel with modern, clean design"""
        panel_x = GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + 40
        panel_y = GRID_OFFSET_Y
        panel_width = WINDOW_WIDTH - panel_x - 30
        panel_height = WINDOW_HEIGHT - panel_y - 80
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        draw_gradient_rect(self.screen, SURFACE, lerp_color(SURFACE, BACKGROUND, 0.3), panel_rect, False)
        draw_rounded_rect(self.screen, BACKGROUND, panel_rect, 20, 2, BORDER)
        
        y_offset = panel_y + 25
        
        title_text = FONTS['title'].render("Multi-Agent Arena", True, TEXT_PRIMARY)
        title_shadow = FONTS['title'].render("Multi-Agent Arena", True, (0, 0, 0, 60))
        self.screen.blit(title_shadow, (panel_x + 22, y_offset + 2))
        self.screen.blit(title_text, (panel_x + 20, y_offset))
        y_offset += 50
        
        current_day = (self.env.time // self.env.seconds_per_day) + 1
        season, _ = self.env.get_current_season()
        time_of_day = self.env.get_time_of_day()
        
        season_color = SEASON_COLORS[season]
        season_rect = pygame.Rect(panel_x + 20, y_offset, panel_width - 40, 45)
        draw_gradient_rect(self.screen, season_color, lerp_color(season_color, WHITE, 0.3), season_rect)
        draw_rounded_rect(self.screen, BACKGROUND, season_rect, 10, 2, season_color)
        
        season_text = f"Day {current_day}/8 • {season} • {time_of_day}"
        season_surf = FONTS['large'].render(season_text, True, TEXT_PRIMARY)
        season_text_rect = season_surf.get_rect(center=season_rect.center)
        self.screen.blit(season_surf, season_text_rect)
        y_offset += 65
        living_count = self.env.get_living_agents()
        status_text = f"Survivors: {living_count}/{len(self.env.agents)} • Time: {self.env.time//60}m {self.env.time%60}s"
        status_surf = FONTS['medium'].render(status_text, True, TEXT_SECONDARY)
        self.screen.blit(status_surf, (panel_x + 20, y_offset))
        y_offset += 35
    
        leaderboard_title = FONTS['large'].render("🏆 LEADERBOARD", True, TEXT_PRIMARY)
        self.screen.blit(leaderboard_title, (panel_x + 20, y_offset))
        y_offset += 35
        
        leaderboard = self.env.get_leaderboard()
        for rank, agent in enumerate(leaderboard[:6]):  
            card_height = 85
            card_rect = pygame.Rect(panel_x + 20, y_offset, panel_width - 40, card_height)
            
            card_bg_color = SURFACE if agent['alive'] else lerp_color(SURFACE, BACKGROUND, 0.5)
            draw_rounded_rect(self.screen, card_bg_color, card_rect, 12, 2, 
                            agent['color'] if agent['alive'] else BORDER)
            
            rank_size = 24
            rank_rect = pygame.Rect(card_rect.left + 15, card_rect.top + 15, rank_size, rank_size)
            rank_color = agent['color'] if agent['alive'] else BORDER
            draw_rounded_rect(self.screen, rank_color, rank_rect, rank_size//2)
            
            rank_text = FONTS['medium'].render(str(rank + 1), True, TEXT_PRIMARY)
            rank_text_rect = rank_text.get_rect(center=rank_rect.center)
            self.screen.blit(rank_text, rank_text_rect)
            
            name_x = card_rect.left + 50
            name_text = FONTS['medium'].render(agent['name'], True, TEXT_PRIMARY)
            self.screen.blit(name_text, (name_x, card_rect.top + 12))
            
            if agent['alive']:
                status_line = f"Health: {agent['health']:.0f} • Hunger: {agent['hunger']:.0f} • Thirst: {agent['thirst']:.0f}"
                status_color = TEXT_SECONDARY
            else:
                survival_time = agent['death_time'] or 0
                days = survival_time // self.env.seconds_per_day
                status_line = f"Died: {agent['death_cause']} • Survived: {days}d {survival_time}s"
                status_color = lerp_color(TEXT_SECONDARY, DANGER, 0.3)
            
            status_surf = FONTS['small'].render(status_line, True, status_color)
            self.screen.blit(status_surf, (name_x, card_rect.top + 32))
            
            if agent['alive']:
                bar_width = 60
                bar_height = 4
                bar_y = card_rect.top + 55
                
                health_color = lerp_color(DANGER, SUCCESS, agent['health'] / 100)
                self.draw_elegant_stats_bar(name_x, bar_y, bar_width, bar_height, 
                                          agent['health'], 100, health_color, "")
                
                hunger_color = lerp_color(SUCCESS, WARNING, agent['hunger'] / 100)
                self.draw_elegant_stats_bar(name_x + bar_width + 5, bar_y, bar_width, bar_height, 
                                          100 - agent['hunger'], 100, hunger_color, "")
                
                thirst_color = lerp_color(SUCCESS, DANGER, agent['thirst'] / 100)
                self.draw_elegant_stats_bar(name_x + (bar_width + 5) * 2, bar_y, bar_width, bar_height, 
                                          100 - agent['thirst'], 100, thirst_color, "")
            
            resources_text = f"🌲 {agent['stats']['trees_found']}/{len(self.env.TREES)} • 💧 {agent['stats']['waters_found']}/{len(self.env.WATERS)} • 🗺️ {agent['stats']['exploration_progress']:.1%}"
            resources_surf = FONTS['small'].render(resources_text, True, TEXT_SECONDARY)
            self.screen.blit(resources_surf, (name_x, card_rect.bottom - 18))
            
            y_offset += card_height + 10

        if self.env.time >= self.env.max_time or living_count == 0:
            y_offset += 20
            end_rect = pygame.Rect(panel_x + 20, y_offset, panel_width - 40, 50)
            
            if living_count > 0:
                end_text = "🎉 MISSION COMPLETE!"
                end_color = SUCCESS
                message = f"{living_count} agent(s) survived 8 days!"
            else:
                end_text = "💀 TOTAL EXTINCTION"
                end_color = DANGER
                message = "All agents perished in the wilderness"
            
            draw_rounded_rect(self.screen, end_color, end_rect, 15, 3, lerp_color(end_color, WHITE, 0.3))
            
            end_surf = FONTS['large'].render(end_text, True, TEXT_PRIMARY)
            end_text_rect = end_surf.get_rect(center=(end_rect.centerx, end_rect.centery - 8))
            self.screen.blit(end_surf, end_text_rect)
            
            message_surf = FONTS['small'].render(message, True, TEXT_SECONDARY)
            message_rect = message_surf.get_rect(center=(end_rect.centerx, end_rect.centery + 10))
            self.screen.blit(message_surf, message_rect)

    def draw_modern_controls(self):
        """Draw modern control panel at bottom"""
        controls_height = 60
        controls_y = WINDOW_HEIGHT - controls_height
        controls_rect = pygame.Rect(0, controls_y, WINDOW_WIDTH, controls_height)

        draw_gradient_rect(self.screen, SURFACE, BACKGROUND, controls_rect)
        pygame.draw.line(self.screen, BORDER, (0, controls_y), (WINDOW_WIDTH, controls_y), 2)
        
        y_center = controls_y + controls_height // 2

        status_text = "PAUSED" if self.paused else f"PLAYING {self.speed}x"
        status_color = WARNING if self.paused else SUCCESS
        
        status_surf = FONTS['medium'].render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status_surf, (20, y_center - status_surf.get_height()//2))

        controls_text = "SPACE: Play/Pause • R: Reset • N: New Environment • ↑↓: Speed • F11: Fullscreen • F10: Reset Size • ESC: Exit"
        controls_surf = FONTS['small'].render(controls_text, True, TEXT_SECONDARY)
        controls_rect = controls_surf.get_rect(center=(WINDOW_WIDTH//2, y_center))
        self.screen.blit(controls_surf, controls_rect)

        stats_text = f"Step: {self.step_count} • Agents: {len(self.agents)}"
        stats_surf = FONTS['small'].render(stats_text, True, TEXT_SECONDARY)
        stats_rect = stats_surf.get_rect(right=WINDOW_WIDTH - 20, centery=y_center)
        self.screen.blit(stats_surf, stats_rect)

    def run(self):
        """Main game loop with beautiful rendering"""
        running = True
        print("Starting Aesthetic Multi-Agent Survival Arena...")
        
        while running:
            current_time = time.time()
            self.time_since_start += 1/60
            
            running = self.handle_events()
            if not running:
                break

            if not self.paused and (current_time - self.last_step_time) >= (1.0 / self.speed):
                living_count = self.env.get_living_agents()
                if living_count > 0 and self.env.time < self.env.max_time:
                    self.step_forward()
                self.last_step_time = current_time

            self.particles.update()

            self.screen.fill(BACKGROUND)

            self.draw_beautiful_grid()
            self.draw_beautiful_agents()

            
            self.particles.draw(self.screen)

            
            self.draw_modern_info_panel()
            self.draw_modern_controls()

            pygame.display.flip()
            self.clock.tick(60) 

        pygame.quit()

def main():
    """Main function to start the aesthetic multi-agent simulation"""
    import sys
    
    if len(sys.argv) > 1:
        model_files = sys.argv[1:]
        print(f"Loading models from command line: {model_files}")
    else:
        model_files = glob.glob("*.pth")
        if not model_files:
            print("No .pth files found in current directory.")
            print("Usage: python aesthetic_multi_agent_survival.py [model1.pth] [model2.pth] ...")
            print("Or place .pth files in current directory for auto-detection.")
            return
        print(f"Auto-detected models: {model_files}")

    try:
        visualizer = AestheticMultiAgentVisualizer(model_files)
        visualizer.run()
    except Exception as e:
        print("Error running aesthetic multi-agent visualizer:", e)
        traceback.print_exc()
    finally:
        pygame.quit()

if __name__ == "__main__":

    main()
