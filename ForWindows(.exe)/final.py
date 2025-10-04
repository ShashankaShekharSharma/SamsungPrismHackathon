# enhanced_aesthetic_survival_betting.py
# Beautiful Multi-agent survival simulation with enhanced UI

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
import json

# Initialize Pygame
pygame.init()

try:
    icon = pygame.image.load('logo.jpg')  # Change to your logo filename
    pygame.display.set_icon(icon)
except:
    print("Could not load icon image")



# ---------- Modern Design Constants ----------
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
GRID_SIZE = 12
CELL_SIZE = 45
GRID_OFFSET_X = 80
GRID_OFFSET_Y = 120

# Modern Color Palette
BACKGROUND = (0, 0, 0)
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

# Grid and environment colors
UNEXPLORED = (30, 35, 45)
EXPLORED = (45, 52, 65)
HOME_COLOR = (147, 51, 234)
TREE_COLOR = (34, 197, 94)
WATER_COLOR = (59, 130, 246)
FIRE_COLOR = (251, 146, 60)

# Agent colors
AGENT_COLORS = [
    (255, 71, 87), (72, 207, 173), (116, 125, 251), (255, 184, 0),
    (255, 105, 180), (154, 230, 180), (255, 159, 67), (162, 155, 254),
]

SEASON_COLORS = {
    "Summer": (255, 223, 186),
    "Autumn": (255, 183, 77),
    "Winter": (186, 230, 253),
    "Spring": (187, 247, 208)
}

# Typography
FONT_SIZES = {
    'tiny': 12,
    'small': 14,
    'medium': 16,
    'large': 20,
    'xlarge': 28,
    'title': 36,
    'huge': 48
}

def load_fonts():
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

# Agent personality traits
AGENT_TRAITS = [
    {"name": "Explorer",  "desc": "Curious wanderer who maps the unknown",   "icon": ":)"},
    {"name": "Survivor",  "desc": "Resilient fighter with strong instincts", "icon": ":D"},
    {"name": "Strategist","desc": "Calculates every move with precision",    "icon": "B)"},
    {"name": "Gatherer",  "desc": "Expert at finding resources quickly",     "icon": ":P"},
    {"name": "Nomad",     "desc": "Always on the move, never stays still",   "icon": ";)"},
    {"name": "Guardian",  "desc": "Protects the home base at all costs",     "icon": ":]"},
    {"name": "Hunter",    "desc": "Aggressive seeker of food and water",     "icon": ">:("},
    {"name": "Monk",      "desc": "Balanced approach to survival",           "icon": ":|"},
]




class ParticleSystem:
    def __init__(self):
        self.particles = []
    
    def add_discovery_particles(self, x, y, color, count=8):
        for _ in range(count):
            self.particles.append({
                'x': x, 'y': y,
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-2, 2),
                'life': 1.0, 'color': color,
                'size': random.uniform(2, 4)
            })
    
    def add_death_particles(self, x, y, count=12):
        for _ in range(count):
            self.particles.append({
                'x': x, 'y': y,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-3, 3),
                'life': 1.0, 'color': DANGER,
                'size': random.uniform(3, 6)
            })
    
    def add_fire_particles(self, x, y, count=3):
        for _ in range(count):
            self.particles.append({
                'x': x + random.uniform(-5, 5),
                'y': y,
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(-2, -0.5),
                'life': 1.0,
                'color': random.choice([FIRE_COLOR, (255, 200, 0), (255, 100, 0)]),
                'size': random.uniform(2, 5),
                'type': 'fire'
            })
    
    def add_water_particles(self, x, y, count=2):
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(0.5, 1.5)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': 1.0,
                'color': WATER_COLOR,
                'size': random.uniform(1, 3),
                'type': 'water'
            })
    
    def update(self):
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            if particle.get('type') == 'fire':
                particle['vy'] -= 0.1
                particle['life'] -= 0.03
            elif particle.get('type') == 'water':
                particle['vy'] += 0.05
                particle['life'] -= 0.02
            else:
                particle['life'] -= 0.02
            
            particle['size'] *= 0.98
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        for particle in self.particles:
            alpha = int(255 * particle['life'])
            color = (*particle['color'][:3], alpha)
            size = max(1, int(particle['size']))
            
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size, size), size)
            screen.blit(surf, (particle['x'] - size, particle['y'] - size))

def draw_rounded_rect(surface, color, rect, radius=10, border_width=0, border_color=None):
    if border_width > 0 and border_color:
        pygame.draw.rect(surface, border_color, rect.inflate(border_width * 2, border_width * 2), border_radius=radius)
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def lerp_color(color1, color2, t):
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))

def draw_gradient_rect(surface, color1, color2, rect, horizontal=True):
    if horizontal:
        for i in range(rect.width):
            t = i / rect.width
            color = lerp_color(color1, color2, t)
            pygame.draw.line(surface, color, (rect.left + i, rect.top), (rect.left + i, rect.bottom))
    else:
        for i in range(rect.height):
            t = i / rect.height
            color = lerp_color(color1, color2, t)
            pygame.draw.line(surface, color, (rect.left, rect.top + i), (rect.right, rect.top + i))

class Slider:
    def __init__(self, x, y, width, min_val, max_val, initial_val, label, step=1):
        self.rect = pygame.Rect(x, y, width, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.step = step
        self.dragging = False
        self.handle_radius = 12
        
    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        handle_rect = pygame.Rect(handle_x - self.handle_radius, self.rect.y - self.handle_radius, 
                                  self.handle_radius * 2, self.handle_radius * 2)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if handle_rect.collidepoint(mouse_pos) or self.rect.collidepoint(mouse_pos):
                self.dragging = True
                self.update_value(mouse_pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_value(mouse_pos[0])
    
    def update_value(self, mouse_x):
        rel_x = max(0, min(self.rect.width, mouse_x - self.rect.x))
        raw_value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
        self.value = round(raw_value / self.step) * self.step
        self.value = max(self.min_val, min(self.max_val, self.value))
    
    def draw(self, screen):
        draw_rounded_rect(screen, BORDER, self.rect, 5)
        
        filled_width = int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        filled_rect = pygame.Rect(self.rect.x, self.rect.y, filled_width, self.rect.height)
        draw_rounded_rect(screen, ACCENT, filled_rect, 5)
        
        handle_x = self.rect.x + filled_width
        handle_y = self.rect.y + self.rect.height // 2
        
        glow_surf = pygame.Surface((self.handle_radius * 4, self.handle_radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*ACCENT, 60), (self.handle_radius * 2, self.handle_radius * 2), self.handle_radius * 2)
        screen.blit(glow_surf, (handle_x - self.handle_radius * 2, handle_y - self.handle_radius * 2))
        
        pygame.draw.circle(screen, ACCENT, (handle_x, handle_y), self.handle_radius)
        pygame.draw.circle(screen, WHITE, (handle_x, handle_y), self.handle_radius - 4)
        
        label_text = FONTS['medium'].render(f"{self.label}: {int(self.value)}", True, TEXT_PRIMARY)
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))

# ---------- Environment Classes ----------
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
            'color': AGENT_COLORS[agent_id % len(AGENT_COLORS)],
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
        
        if agent['time_since_last_rest'] >= 240 or agent['health'] < 20:
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

# ---------- AI Agent Classes ----------
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

# ---------- Game State Manager ----------
class GameStateManager:
    def __init__(self):
        self.credits = 1000
        self.selected_agent = None
        self.bet_amount = 100
        self.game_history = []
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

# ---------- Home Screen ----------
class HomeScreen:
    def __init__(self, screen, model_files):
        self.screen = screen
        self.model_files = model_files
        self.game_state = GameStateManager()
        self.selected_index = None
        self.bet_amounts = [50, 100, 250, 500, 1000]
        self.selected_bet_index = 1
        self.animation_time = 0
        self.hover_index = None
        self.scroll_offset = 0
        self.max_scroll = 0
        self.scrolling = False
        
        # Settings sliders (removed num_agents_slider)
        self.duration_slider = Slider(WINDOW_WIDTH // 2 - 200, 200, 400, 4, 12, 8, "Days Duration", 1)
        self.resources_slider = Slider(WINDOW_WIDTH // 2 - 200, 270, 400, 4, 12, 6, "Resource Count", 1)
        
        self.agent_info = []
        for i, model_file in enumerate(model_files):
            name = os.path.splitext(os.path.basename(model_file))[0]
            if len(name) > 15:
                name = name[:15] + "..."
            trait = AGENT_TRAITS[i % len(AGENT_TRAITS)]
            self.agent_info.append({
                'name': name,
                'model_path': model_file,
                'trait': trait,
                'color': AGENT_COLORS[i % len(AGENT_COLORS)],
                'id': i
            })
        
        self.agent_rects = []
        self.bet_rects = []
        self.start_button_rect = None
        self.quit_button_rect = None
    
    def handle_event(self, event):
        # Handle slider events
        self.duration_slider.handle_event(event)
        self.resources_slider.handle_event(event)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Check agent selection (scrollable area)
            for i, rect in enumerate(self.agent_rects):
                adjusted_rect = rect.move(0, -self.scroll_offset)
                if adjusted_rect.collidepoint(mouse_pos) and i < len(self.model_files):
                    if adjusted_rect.top >= 360 and adjusted_rect.bottom <= 640:
                        self.selected_index = i
                        return None
            
            # Check bet selection
            for i, rect in enumerate(self.bet_rects):
                if rect.collidepoint(mouse_pos):
                    self.selected_bet_index = i
                    return None
            
            # Check start button
            if hasattr(self, 'start_button_rect') and self.start_button_rect.collidepoint(mouse_pos):
                if self.selected_index is not None:
                    bet_amount = self.bet_amounts[self.selected_bet_index]
                    if self.game_state.place_bet(self.selected_index, bet_amount):
                        return ('start_game', {
                            'num_agents': len(self.model_files),
                            'duration': int(self.duration_slider.value),
                            'resources': int(self.resources_slider.value)
                        })
            
            # Check quit button
            if hasattr(self, 'quit_button_rect') and self.quit_button_rect.collidepoint(mouse_pos):
                return 'quit'
        
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            self.hover_index = None
            for i, rect in enumerate(self.agent_rects):
                adjusted_rect = rect.move(0, -self.scroll_offset)
                if adjusted_rect.collidepoint(mouse_pos) and i < len(self.model_files):
                    if adjusted_rect.top >= 360 and adjusted_rect.bottom <= 640:
                        self.hover_index = i
                        break
        
        elif event.type == pygame.MOUSEWHEEL:
            self.scroll_offset -= event.y * 30
            self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset))
        
        return None
    
    def draw(self):
        self.screen.fill(BACKGROUND)
        self.animation_time += 0.05
        
        # Title
        title = FONTS['huge'].render("APEX AI SURVIVAL ARENA", True, TEXT_PRIMARY)
        subtitle = FONTS['large'].render("Configure Your Battle", True, ACCENT)
        
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 60))
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 120))
        
        self.screen.blit(title, title_rect)
        self.screen.blit(subtitle, subtitle_rect)
        
        # Credits display
        credits_text = FONTS['xlarge'].render(f"Credits: {self.game_state.credits}", True, SUCCESS)
        credits_rect = credits_text.get_rect(topright=(WINDOW_WIDTH - 40, 40))
        
        credits_bg = credits_rect.inflate(30, 20)
        draw_rounded_rect(self.screen, SURFACE, credits_bg, 12, 2, SUCCESS)
        self.screen.blit(credits_text, credits_rect)
        
        # Draw sliders
        self.duration_slider.draw(self.screen)
        self.resources_slider.draw(self.screen)
        
        # Agent count display (static info)
        agent_count_text = FONTS['medium'].render(f"All {len(self.model_files)} agents will compete", True, TEXT_SECONDARY)
        agent_count_rect = agent_count_text.get_rect(center=(WINDOW_WIDTH // 2, 325))
        self.screen.blit(agent_count_text, agent_count_rect)
        
        # Agent selection title
        agent_select_title = FONTS['large'].render("Select Your Champion", True, TEXT_PRIMARY)
        agent_select_rect = agent_select_title.get_rect(center=(WINDOW_WIDTH // 2, 355))
        self.screen.blit(agent_select_title, agent_select_rect)
        
        # Scrollable agent cards area
        card_width = 220
        card_height = 140
        cards_per_row = 6
        spacing = 15
        start_x = (WINDOW_WIDTH - (cards_per_row * card_width + (cards_per_row - 1) * spacing)) // 2
        start_y = 380
        
        # Create clipping region for scrollable area
        scroll_area = pygame.Rect(0, start_y, WINDOW_WIDTH, 260)
        self.screen.set_clip(scroll_area)
        
        self.agent_rects = []
        num_agents = len(self.model_files)
        
        for i in range(num_agents):
            agent = self.agent_info[i]
            row = i // cards_per_row
            col = i % cards_per_row
            x = start_x + col * (card_width + spacing)
            y = start_y + row * (card_height + spacing) - self.scroll_offset
            
            card_rect = pygame.Rect(x, y, card_width, card_height)
            self.agent_rects.append(pygame.Rect(x, y + self.scroll_offset, card_width, card_height))
            
            # Skip if not visible
            if y + card_height < start_y or y > start_y + 260:
                continue
            
            is_selected = self.selected_index == i
            is_hover = self.hover_index == i
            
            border_color = agent['color'] if is_selected else BORDER
            border_width = 4 if is_selected else 2
            
            bg_color = SURFACE if not is_selected else lerp_color(SURFACE, agent['color'], 0.2)
            if is_hover and not is_selected:
                bg_color = lerp_color(SURFACE, agent['color'], 0.1)
            
            draw_rounded_rect(self.screen, bg_color, card_rect, 12, border_width, border_color)
            
            # Avatar
            avatar_size = 40
            avatar_center = (card_rect.centerx, card_rect.top + 35)
            
            pulse = math.sin(self.animation_time + i * 0.5) * 0.1 + 0.9
            avatar_radius = int(avatar_size * pulse / 2)
            
            pygame.draw.circle(self.screen, agent['color'], avatar_center, avatar_radius)
            pygame.draw.circle(self.screen, lerp_color(agent['color'], WHITE, 0.3), 
                             (avatar_center[0] - 5, avatar_center[1] - 5), avatar_radius // 2)
            
            # Name
            name_text = FONTS['medium'].render(agent['name'], True, TEXT_PRIMARY)
            name_rect = name_text.get_rect(center=(card_rect.centerx, card_rect.top + 70))
            self.screen.blit(name_text, name_rect)
            
            # Trait
            trait_text = FONTS['small'].render(f"{agent['trait']['icon']} {agent['trait']['name']}", True, agent['color'])
            trait_rect = trait_text.get_rect(center=(card_rect.centerx, card_rect.top + 95))
            self.screen.blit(trait_text, trait_rect)
            
            # ID badge
            id_text = FONTS['tiny'].render(f"#{i}", True, TEXT_SECONDARY)
            id_rect = id_text.get_rect(topright=(card_rect.right - 10, card_rect.top + 10))
            self.screen.blit(id_text, id_rect)
        
        # Calculate max scroll
        total_rows = (num_agents + cards_per_row - 1) // cards_per_row
        total_height = total_rows * (card_height + spacing)
        self.max_scroll = max(0, total_height - 260)
        
        # Remove clipping
        self.screen.set_clip(None)
        
        # Scroll indicators
        if self.scroll_offset > 0:
            up_arrow = FONTS['xlarge'].render("▲", True, ACCENT)
            up_rect = up_arrow.get_rect(center=(WINDOW_WIDTH // 2, start_y - 10))
            self.screen.blit(up_arrow, up_rect)
        
        if self.scroll_offset < self.max_scroll:
            down_arrow = FONTS['xlarge'].render("▼", True, ACCENT)
            down_rect = down_arrow.get_rect(center=(WINDOW_WIDTH // 2, start_y + 270))
            self.screen.blit(down_arrow, down_rect)
        
        # Bet selection
        bet_label = FONTS['large'].render("Select Bet Amount:", True, TEXT_PRIMARY)
        bet_label_rect = bet_label.get_rect(center=(WINDOW_WIDTH // 2, 670))
        self.screen.blit(bet_label, bet_label_rect)
        
        self.bet_rects = []
        bet_button_width = 110
        bet_button_height = 45
        total_bet_width = len(self.bet_amounts) * bet_button_width + (len(self.bet_amounts) - 1) * 12
        bet_start_x = (WINDOW_WIDTH - total_bet_width) // 2
        bet_y = 705
        
        for i, amount in enumerate(self.bet_amounts):
            x = bet_start_x + i * (bet_button_width + 12)
            bet_rect = pygame.Rect(x, bet_y, bet_button_width, bet_button_height)
            self.bet_rects.append(bet_rect)
            
            is_selected = self.selected_bet_index == i
            can_afford = amount <= self.game_state.credits
            
            if not can_afford:
                bg_color = lerp_color(SURFACE, BACKGROUND, 0.5)
                text_color = TEXT_SECONDARY
                border_color = BORDER
            elif is_selected:
                bg_color = lerp_color(SURFACE, WARNING, 0.3)
                text_color = TEXT_PRIMARY
                border_color = WARNING
            else:
                bg_color = SURFACE
                text_color = TEXT_PRIMARY
                border_color = BORDER
            
            draw_rounded_rect(self.screen, bg_color, bet_rect, 10, 2, border_color)
            
            amount_text = FONTS['medium'].render(f"{amount}", True, text_color)
            amount_rect = amount_text.get_rect(center=bet_rect.center)
            self.screen.blit(amount_text, amount_rect)
        
        # Action buttons
        button_width = 200
        button_height = 55
        button_y = 775
        
        start_x = WINDOW_WIDTH // 2 - button_width - 15
        self.start_button_rect = pygame.Rect(start_x, button_y, button_width, button_height)
        
        can_start = (self.selected_index is not None and 
                    self.bet_amounts[self.selected_bet_index] <= self.game_state.credits)
        
        if can_start:
            start_bg = SUCCESS
            start_text_color = TEXT_PRIMARY
        else:
            start_bg = lerp_color(SURFACE, BACKGROUND, 0.5)
            start_text_color = TEXT_SECONDARY
        
        draw_rounded_rect(self.screen, start_bg, self.start_button_rect, 12)
        start_text = FONTS['large'].render("START BATTLE", True, start_text_color)
        start_text_rect = start_text.get_rect(center=self.start_button_rect.center)
        self.screen.blit(start_text, start_text_rect)
        
        quit_x = WINDOW_WIDTH // 2 + 15
        self.quit_button_rect = pygame.Rect(quit_x, button_y, button_width, button_height)
        
        draw_rounded_rect(self.screen, DANGER, self.quit_button_rect, 12)
        quit_text = FONTS['large'].render("QUIT", True, TEXT_PRIMARY)
        quit_text_rect = quit_text.get_rect(center=self.quit_button_rect.center)
        self.screen.blit(quit_text, quit_text_rect)
        
        # Status message
        if self.selected_index is None:
            instruction = "Select an agent and bet to start"
        elif not can_start:
            instruction = "Not enough credits for this bet!"
        else:
            instruction = f"Ready! Betting {self.bet_amounts[self.selected_bet_index]} on {self.agent_info[self.selected_index]['name']}"
        
        instruction_text = FONTS['medium'].render(instruction, True, TEXT_SECONDARY)
        instruction_rect = instruction_text.get_rect(center=(WINDOW_WIDTH // 2, 850))
        self.screen.blit(instruction_text, instruction_rect)

# ---------- Results Screen ----------
class ResultsScreen:
    def __init__(self, screen, game_state, winner_id, agents_data):
        self.screen = screen
        self.game_state = game_state
        self.winner_id = winner_id
        self.agents_data = agents_data
        self.animation_time = 0
        
        self.player_won = (winner_id == game_state.selected_agent)
        
        if self.player_won:
            game_state.win_bet()
        else:
            game_state.lose_bet()
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            if hasattr(self, 'retry_button_rect') and self.retry_button_rect.collidepoint(mouse_pos):
                return 'home'
            
            if hasattr(self, 'quit_button_rect') and self.quit_button_rect.collidepoint(mouse_pos):
                return 'quit'
        
        return None
    
    def draw(self):
        self.screen.fill(BACKGROUND)
        self.animation_time += 0.05
        
        if self.player_won:
            title = FONTS['huge'].render("VICTORY!", True, SUCCESS)
            subtitle_text = f"You won {self.game_state.bet_amount * 2} credits!"
            subtitle_color = SUCCESS
        else:
            title = FONTS['huge'].render("DEFEAT", True, DANGER)
            subtitle_text = f"You lost {self.game_state.bet_amount} credits"
            subtitle_color = DANGER
        
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(title, title_rect)
        
        subtitle = FONTS['xlarge'].render(subtitle_text, True, subtitle_color)
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 160))
        self.screen.blit(subtitle, subtitle_rect)
        
        credits_text = FONTS['xlarge'].render(f"Total Credits: {self.game_state.credits}", True, TEXT_PRIMARY)
        credits_rect = credits_text.get_rect(center=(WINDOW_WIDTH // 2, 220))
        
        credits_bg = credits_rect.inflate(40, 25)
        draw_rounded_rect(self.screen, SURFACE, credits_bg, 12, 2, ACCENT)
        self.screen.blit(credits_text, credits_rect)
        
        winner_data = self.agents_data[self.winner_id]
        winner_label = FONTS['large'].render("SURVIVOR:", True, TEXT_PRIMARY)
        winner_label_rect = winner_label.get_rect(center=(WINDOW_WIDTH // 2, 290))
        self.screen.blit(winner_label, winner_label_rect)
        
        card_width = 400
        card_height = 180
        card_rect = pygame.Rect((WINDOW_WIDTH - card_width) // 2, 330, card_width, card_height)
        
        pulse = math.sin(self.animation_time) * 0.02 + 1.0
        card_rect = card_rect.inflate(int(10 * pulse), int(10 * pulse))
        
        draw_rounded_rect(self.screen, SURFACE, card_rect, 15, 3, winner_data['color'])
        
        avatar_size = 70
        avatar_center = (card_rect.left + 80, card_rect.centery)
        pygame.draw.circle(self.screen, winner_data['color'], avatar_center, avatar_size)
        pygame.draw.circle(self.screen, lerp_color(winner_data['color'], WHITE, 0.3),
                         (avatar_center[0] - 15, avatar_center[1] - 15), avatar_size // 2)
        
        text_x = card_rect.left + 170
        name_text = FONTS['xlarge'].render(winner_data['name'], True, TEXT_PRIMARY)
        self.screen.blit(name_text, (text_x, card_rect.top + 30))
        
        health_text = FONTS['medium'].render(f"Final Health: {winner_data['health']:.0f}", True, SUCCESS)
        self.screen.blit(health_text, (text_x, card_rect.top + 70))
        
        survival_days = (winner_data.get('death_time') or 2880) // 360
        survival_text = FONTS['medium'].render(f"Survived: {survival_days} days", True, TEXT_SECONDARY)
        self.screen.blit(survival_text, (text_x, card_rect.top + 100))
        
        stats_text = FONTS['small'].render(
            f"Trees: {winner_data['stats']['trees_found']} | Waters: {winner_data['stats']['waters_found']} | "
            f"Exploration: {winner_data['stats']['exploration_progress']:.0%}",
            True, TEXT_SECONDARY
        )
        self.screen.blit(stats_text, (text_x, card_rect.top + 130))
        
        leaderboard_y = 550
        leaderboard_title = FONTS['large'].render("FINAL STANDINGS", True, TEXT_PRIMARY)
        leaderboard_title_rect = leaderboard_title.get_rect(center=(WINDOW_WIDTH // 2, leaderboard_y))
        self.screen.blit(leaderboard_title, leaderboard_title_rect)
        
        sorted_agents = sorted(self.agents_data, 
                             key=lambda a: (a['alive'], a.get('death_time', 2880), a['health']), 
                             reverse=True)
        
        y_offset = leaderboard_y + 50
        for i, agent in enumerate(sorted_agents[:5]):
            rank_text = FONTS['medium'].render(f"#{i+1}", True, TEXT_SECONDARY)
            name_text = FONTS['medium'].render(agent['name'], True, agent['color'])
            
            if agent['alive']:
                status_text = FONTS['small'].render(f"Survived! HP: {agent['health']:.0f}", True, SUCCESS)
            else:
                days_survived = (agent.get('death_time') or 0) // 360
                status_text = FONTS['small'].render(f"Day {days_survived}: {agent.get('death_cause', 'Unknown')}", True, DANGER)
            
            self.screen.blit(rank_text, (WINDOW_WIDTH // 2 - 300, y_offset))
            self.screen.blit(name_text, (WINDOW_WIDTH // 2 - 240, y_offset))
            self.screen.blit(status_text, (WINDOW_WIDTH // 2 - 50, y_offset))
            
            y_offset += 35
        
        button_width = 250
        button_height = 60
        button_y = 800
        
        retry_x = WINDOW_WIDTH // 2 - button_width - 20
        self.retry_button_rect = pygame.Rect(retry_x, button_y, button_width, button_height)
        draw_rounded_rect(self.screen, ACCENT, self.retry_button_rect, 12)
        retry_text = FONTS['large'].render("PLAY AGAIN", True, TEXT_PRIMARY)
        retry_text_rect = retry_text.get_rect(center=self.retry_button_rect.center)
        self.screen.blit(retry_text, retry_text_rect)
        
        quit_x = WINDOW_WIDTH // 2 + 20
        self.quit_button_rect = pygame.Rect(quit_x, button_y, button_width, button_height)
        draw_rounded_rect(self.screen, DANGER, self.quit_button_rect, 12)
        quit_text = FONTS['large'].render("QUIT", True, TEXT_PRIMARY)
        quit_text_rect = quit_text.get_rect(center=self.quit_button_rect.center)
        self.screen.blit(quit_text, quit_text_rect)

# ---------- Main Visualizer ----------
class AestheticMultiAgentVisualizer:


    





    def __init__(self, model_files, game_state, num_agents, duration, resources):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("AI Survival Arena - Enhanced Edition")
        self.clock = pygame.time.Clock()
        self.game_state = game_state
        
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.min_width = 1200
        self.min_height = 800
        self.is_fullscreen = False
        self.scale_factor = 1.0
        
        # Leaderboard scrolling
        self.leaderboard_scroll = 0
        self.leaderboard_max_scroll = 0

        max_time = duration * 360
        self.env = MultiAgentSurvivalEnv(grid_size=12, max_time=max_time, randomize_resources=True)
        
        # Override resource generation with custom count
        valid_positions = [(x, y) for x in range(12) for y in range(12) if (x, y) != (6, 6)]
        random.shuffle(valid_positions)
        self.env.TREES = valid_positions[:resources]
        self.env.WATERS = valid_positions[resources:resources*2]
        
        self.agents = []
        self.particles = ParticleSystem()
        
        self.load_agents_from_files(model_files[:num_agents])

        self.paused = False
        self.speed = 2
        self.last_step_time = 0
        self.step_count = 0
        self.time_since_start = 0

        self.show_only_bet_agent = False

        
        
        self.reset_simulation()

    def load_agents_from_files(self, model_files):
        for i, model_file in enumerate(model_files):
            name = os.path.splitext(os.path.basename(model_file))[0]
            if len(name) > 12:
                name = name[:12] + "..."
            
            agent_id = self.env.add_agent(name, model_file)
            dummy_state = self.env.get_state_for_agent(agent_id)
            state_size = len(dummy_state) if dummy_state is not None else 25
            
            agent = RLAgent(state_size, self.env.action_space(), model_file, name)
            self.agents.append(agent)

    def reset_simulation(self):
        self.env.reset(regenerate_resources=False)
        self.step_count = 0
        self.time_since_start = 0
        self.particles = ParticleSystem()
        self.leaderboard_scroll = 0

    def step_forward(self):
        if self.env.time >= self.env.max_time:
            return True

        living_agents = [i for i, agent in enumerate(self.env.agents) if agent['alive']]
        if not living_agents:
            return True

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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed = min(100, self.speed + 10)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 10)
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_f:  # Press F to toggle focus mode
                    self.show_only_bet_agent = not self.show_only_bet_agent

            elif event.type == pygame.MOUSEWHEEL:
                # Check if mouse is over leaderboard area
                mouse_pos = pygame.mouse.get_pos()
                panel_x = GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + 40
                if mouse_pos[0] >= panel_x:
                    self.leaderboard_scroll -= event.y * 20
                    self.leaderboard_scroll = max(0, min(self.leaderboard_max_scroll, self.leaderboard_scroll))
        return True

    def toggle_fullscreen(self):
        if hasattr(self, 'is_fullscreen') and self.is_fullscreen:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            self.window_width = WINDOW_WIDTH
            self.window_height = WINDOW_HEIGHT
            self.is_fullscreen = False
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.window_width = info.current_w
            self.window_height = info.current_h
            self.is_fullscreen = True

    def draw_beautiful_grid(self):
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
                
                if pos in self.env.TREES:
                    any_knows = any(pos in agent['known_tree_locations'] for agent in self.env.agents)
                    if any_knows or pos in self.env.global_explored:
                        sway = math.sin(self.time_since_start * 0.5 + x * 0.3) * 2
                        tree_size = 14
                        
                        for layer in range(3):
                            layer_size = tree_size - layer * 3
                            layer_y = center_y - 5 + layer * 2
                            pygame.draw.circle(self.screen, lerp_color(TREE_COLOR, (20, 120, 40), layer * 0.3), 
                                             (int(center_x + sway), int(layer_y)), layer_size)
                        
                        trunk_rect = pygame.Rect(center_x - 3 + sway, center_y + 5, 6, 10)
                        pygame.draw.rect(self.screen, (101, 67, 33), trunk_rect, border_radius=2)
                        
                        glow_surf = pygame.Surface((tree_size * 4, tree_size * 4), pygame.SRCALPHA)
                        pulse = (math.sin(self.time_since_start * 0.3 + x) + 1) * 0.5
                        pygame.draw.circle(glow_surf, (*TREE_COLOR, int(20 + pulse * 15)), 
                                         (tree_size * 2, tree_size * 2), tree_size + 5)
                        self.screen.blit(glow_surf, (center_x - tree_size * 2, center_y - 5 - tree_size * 2))

                if pos in self.env.WATERS:
                    any_knows = any(pos in agent['known_water_locations'] for agent in self.env.agents)
                    if any_knows or pos in self.env.global_explored:
                        water_time = self.time_since_start * 0.15 + y * 0.5
                        
                        for i in range(3):
                            ripple_phase = (water_time + i * 0.8) % (math.pi * 2)
                            ripple_radius = 8 + int(math.sin(ripple_phase) * 4)
                            ripple_alpha = int((1 - ripple_phase / (math.pi * 2)) * 100)
                            
                            ripple_surf = pygame.Surface((ripple_radius * 3, ripple_radius * 3), pygame.SRCALPHA)
                            pygame.draw.circle(ripple_surf, (*WATER_COLOR, ripple_alpha), 
                                             (ripple_radius * 3 // 2, ripple_radius * 3 // 2), ripple_radius, 2)
                            self.screen.blit(ripple_surf, (center_x - ripple_radius * 3 // 2, 
                                                          center_y - ripple_radius * 3 // 2))
                        
                        water_size = 12
                        pygame.draw.circle(self.screen, WATER_COLOR, (center_x, center_y), water_size)
                        pygame.draw.circle(self.screen, lerp_color(WATER_COLOR, WHITE, 0.4), 
                                         (center_x - 3, center_y - 3), water_size//2)
                        
                        if random.random() < 0.05:
                            self.particles.add_water_particles(center_x, center_y, 1)

                if pos in self.env.active_fires:
                    fire_time = self.env.active_fires[pos]
                    intensity = min(1.0, fire_time / self.env.fire_duration)
                    
                    base_flicker = math.sin(self.time_since_start * 5 + x * y) * 0.3 + 0.7
                    random_flicker = random.uniform(0.8, 1.2)
                    flicker = base_flicker * random_flicker
                    
                    fire_size = int(18 * intensity * flicker)
                    
                    fire_colors = [
                        (255, 100, 0),
                        FIRE_COLOR,
                        (255, 200, 0)
                    ]
                    
                    for i, color in enumerate(fire_colors):
                        layer_size = fire_size - i * 4
                        if layer_size > 0:
                            layer_y_offset = i * 2
                            pygame.draw.circle(self.screen, color, 
                                             (center_x, center_y - layer_y_offset), layer_size)
                    
                    core_size = max(3, fire_size // 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), 
                                     (center_x, center_y - 5), core_size)
                    
                    glow_surf = pygame.Surface((fire_size * 4, fire_size * 4), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*FIRE_COLOR, 40), 
                                     (fire_size * 2, fire_size * 2), fire_size * 2)
                    self.screen.blit(glow_surf, (center_x - fire_size * 2, center_y - fire_size * 2))
                    
                    if random.random() < 0.3:
                        self.particles.add_fire_particles(center_x, center_y, 2)
                    
                    timer_text = FONTS['small'].render(f"{fire_time//10}", True, TEXT_PRIMARY)
                    timer_rect = timer_text.get_rect(center=(center_x, center_y + 22))
                    timer_bg = timer_rect.inflate(8, 4)
                    draw_rounded_rect(self.screen, (0, 0, 0, 180), timer_bg, 4)
                    self.screen.blit(timer_text, timer_rect)

                if pos == self.env.HOME:
                    home_size = 18
                    pulse = (math.sin(self.time_since_start * 0.5) + 1) * 0.5
                    
                    glow_surf = pygame.Surface((home_size * 3, home_size * 3), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*HOME_COLOR, int(30 + pulse * 20)), 
                                     (home_size * 3 // 2, home_size * 3 // 2), home_size + 5)
                    self.screen.blit(glow_surf, (center_x - home_size * 3 // 2, center_y - home_size * 3 // 2))
                    
                    draw_rounded_rect(self.screen, HOME_COLOR, 
                                    pygame.Rect(center_x - home_size//2, center_y - home_size//2, 
                                               home_size, home_size), 5)
                    
                    pygame.draw.polygon(self.screen, (255, 215, 0), [
                        (center_x, center_y - 10),
                        (center_x - 8, center_y - 2),
                        (center_x + 8, center_y - 2)
                    ])
                    
                    door_rect = pygame.Rect(center_x - 3, center_y + 2, 6, 6)
                    pygame.draw.rect(self.screen, (139, 69, 19), door_rect, border_radius=2)

    def draw_beautiful_agents(self):
        for i, agent_data in enumerate(self.env.agents):
            if not agent_data['alive']:
                continue
            if self.show_only_bet_agent and i != self.game_state.selected_agent:
                continue
            
            target_x = GRID_OFFSET_X + agent_data['pos'][0] * CELL_SIZE + CELL_SIZE//2
            target_y = GRID_OFFSET_Y + agent_data['pos'][1] * CELL_SIZE + CELL_SIZE//2
            
            agents_in_cell = [a for a in self.env.agents if tuple(a['pos']) == tuple(agent_data['pos']) and a['alive']]
            if len(agents_in_cell) > 1:
                index_in_cell = [a['id'] for a in agents_in_cell].index(agent_data['id'])
                angle = (index_in_cell * 2 * math.pi) / len(agents_in_cell)
                offset = 10
                target_x += int(math.cos(angle) * offset)
                target_y += int(math.sin(angle) * offset)
            
            pulse = math.sin(agent_data['pulse']) * 0.1 + 0.9
            health_ratio = agent_data['health'] / 100.0
            agent_size = int(16 * health_ratio * pulse)
            
            if i == self.game_state.selected_agent:
                glow_size = agent_size + 10
                glow_surf = pygame.Surface((glow_size * 4, glow_size * 4), pygame.SRCALPHA)
                glow_pulse = (math.sin(self.time_since_start * 2) + 1) * 0.5
                glow_color = (*ACCENT, int(80 + glow_pulse * 50))
                pygame.draw.circle(glow_surf, glow_color, (glow_size * 2, glow_size * 2), glow_size + 5)
                self.screen.blit(glow_surf, (target_x - glow_size * 2, target_y - glow_size * 2))
            
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

    def draw_modern_info_panel(self):
        panel_x = GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + 40
        panel_y = GRID_OFFSET_Y
        panel_width = WINDOW_WIDTH - panel_x - 30
        panel_height = WINDOW_HEIGHT - panel_y - 80
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        draw_gradient_rect(self.screen, SURFACE, lerp_color(SURFACE, BACKGROUND, 0.3), panel_rect, False)
        draw_rounded_rect(self.screen, BACKGROUND, panel_rect, 20, 2, BORDER)
        
        y_offset = panel_y + 25
        
        title_text = FONTS['title'].render("Betting Arena", True, TEXT_PRIMARY)
        title_shadow = FONTS['title'].render("Betting Arena", True, (0, 0, 0, 60))
        self.screen.blit(title_shadow, (panel_x + 22, y_offset + 2))
        self.screen.blit(title_text, (panel_x + 20, y_offset))
        y_offset += 50
        
        bet_text = FONTS['large'].render(f"Your Bet: {self.game_state.bet_amount}", True, WARNING)
        self.screen.blit(bet_text, (panel_x + 20, y_offset))
        y_offset += 35
        
        selected_agent = self.env.agents[self.game_state.selected_agent]
        your_agent_text = FONTS['medium'].render(f"Betting on: {selected_agent['name']}", True, ACCENT)
        self.screen.blit(your_agent_text, (panel_x + 20, y_offset))
        y_offset += 40
        
        current_day = (self.env.time // self.env.seconds_per_day) + 1
        season, _ = self.env.get_current_season()
        time_of_day = self.env.get_time_of_day()
        season_color = SEASON_COLORS[season]
        
        season_rect = pygame.Rect(panel_x + 20, y_offset, panel_width - 40, 45)
        draw_gradient_rect(self.screen, season_color, lerp_color(season_color, WHITE, 0.3), season_rect)
        draw_rounded_rect(self.screen, BACKGROUND, season_rect, 10, 2, season_color)
        
        total_days = self.env.max_time // self.env.seconds_per_day
        season_text = f"Day {current_day}/{total_days} • {season} • {time_of_day}"
        season_surf = FONTS['large'].render(season_text, True, TEXT_PRIMARY)
        season_text_rect = season_surf.get_rect(center=season_rect.center)
        self.screen.blit(season_surf, season_text_rect)
        y_offset += 65
        
        living_count = self.env.get_living_agents()
        status_text = f"Survivors: {living_count}/{len(self.env.agents)} • Time: {self.env.time//60}m {self.env.time%60}s"
        status_surf = FONTS['medium'].render(status_text, True, TEXT_SECONDARY)
        self.screen.blit(status_surf, (panel_x + 20, y_offset))
        y_offset += 35
        
        leaderboard_title = FONTS['large'].render("LEADERBOARD", True, TEXT_PRIMARY)
        self.screen.blit(leaderboard_title, (panel_x + 20, y_offset))
        y_offset += 35
        
        # Scrollable leaderboard area
        leaderboard_start_y = y_offset
        leaderboard_area_height = panel_height - (y_offset - panel_y) - 70
        leaderboard_rect = pygame.Rect(panel_x, leaderboard_start_y, panel_width, leaderboard_area_height)
        
        # Set clipping region
        self.screen.set_clip(leaderboard_rect)
        
        leaderboard = self.env.get_leaderboard()
        card_height = 85
        card_spacing = 10
        
        for rank, agent in enumerate(leaderboard):
            card_y = leaderboard_start_y + rank * (card_height + card_spacing) - self.leaderboard_scroll
            
            # Skip if not visible
            if card_y + card_height < leaderboard_start_y or card_y > leaderboard_start_y + leaderboard_area_height:
                continue
            
            card_rect = pygame.Rect(panel_x + 20, card_y, panel_width - 40, card_height)
            
            card_bg_color = SURFACE if agent['alive'] else lerp_color(SURFACE, BACKGROUND, 0.5)
            border_color = agent['color'] if agent['alive'] else BORDER
            
            if agent['id'] == self.game_state.selected_agent:
                border_width = 4
                border_color = ACCENT
            else:
                border_width = 2
            
            draw_rounded_rect(self.screen, card_bg_color, card_rect, 12, border_width, border_color)
            
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
                health_rect = pygame.Rect(name_x, bar_y, int(bar_width * agent['health'] / 100), bar_height)
                draw_rounded_rect(self.screen, health_color, health_rect, 2)
                
                hunger_color = lerp_color(SUCCESS, WARNING, agent['hunger'] / 100)
                hunger_rect = pygame.Rect(name_x + bar_width + 5, bar_y, int(bar_width * (100 - agent['hunger']) / 100), bar_height)
                draw_rounded_rect(self.screen, hunger_color, hunger_rect, 2)
                
                thirst_color = lerp_color(SUCCESS, DANGER, agent['thirst'] / 100)
                thirst_rect = pygame.Rect(name_x + (bar_width + 5) * 2, bar_y, int(bar_width * (100 - agent['thirst']) / 100), bar_height)
                draw_rounded_rect(self.screen, thirst_color, thirst_rect, 2)
            
            resources_text = f"Trees: {agent['stats']['trees_found']}/{len(self.env.TREES)} • Waters: {agent['stats']['waters_found']}/{len(self.env.WATERS)} • Map: {agent['stats']['exploration_progress']:.1%}"
            resources_surf = FONTS['small'].render(resources_text, True, TEXT_SECONDARY)
            self.screen.blit(resources_surf, (name_x, card_rect.bottom - 18))
        
        # Calculate max scroll
        total_leaderboard_height = len(leaderboard) * (card_height + card_spacing)
        self.leaderboard_max_scroll = max(0, total_leaderboard_height - leaderboard_area_height)
        
        # Remove clipping
        self.screen.set_clip(None)
        
        # Scroll indicators
        if self.leaderboard_scroll > 0:
            up_arrow = FONTS['medium'].render("▲", True, ACCENT)
            up_rect = up_arrow.get_rect(center=(panel_x + panel_width // 2, leaderboard_start_y + 5))
            self.screen.blit(up_arrow, up_rect)
        
        if self.leaderboard_scroll < self.leaderboard_max_scroll:
            down_arrow = FONTS['medium'].render("▼", True, ACCENT)
            down_rect = down_arrow.get_rect(center=(panel_x + panel_width // 2, leaderboard_start_y + leaderboard_area_height - 15))
            self.screen.blit(down_arrow, down_rect)
        
        # End game message
        if self.env.time >= self.env.max_time or living_count == 0:
            end_y = panel_y + panel_height - 60
            end_rect = pygame.Rect(panel_x + 20, end_y, panel_width - 40, 50)
            
            if living_count > 0:
                end_text = "MISSION COMPLETE!"
                end_color = SUCCESS
                message = f"{living_count} agent(s) survived!"
            else:
                end_text = "TOTAL EXTINCTION"
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
        
        controls_text = "SPACE: Play/Pause • Up or Down: Speed • F: Focus Mode • F11: Fullscreen • ESC: Exit"
        controls_surf = FONTS['small'].render(controls_text, True, TEXT_SECONDARY)
        controls_rect = controls_surf.get_rect(center=(WINDOW_WIDTH//2, y_center))
        self.screen.blit(controls_surf, controls_rect)
        
        stats_text = f"Step: {self.step_count} • Agents: {len(self.agents)}"
        stats_surf = FONTS['small'].render(stats_text, True, TEXT_SECONDARY)
        stats_rect = stats_surf.get_rect(right=WINDOW_WIDTH - 20, centery=y_center)
        self.screen.blit(stats_surf, stats_rect)

    def run(self):
        running = True
        game_done = False
        
        while running:
            current_time = time.time()
            self.time_since_start += 1/60
            
            running = self.handle_events()
            if not running:
                return None

            if not self.paused and not game_done and (current_time - self.last_step_time) >= (1.0 / self.speed):
                living_count = self.env.get_living_agents()
                if living_count > 0 and self.env.time < self.env.max_time:
                    game_done = self.step_forward()
                else:
                    game_done = True
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
            
            if game_done:
                time.sleep(2)
                
                leaderboard = self.env.get_leaderboard()
                winner_id = leaderboard[0]['id']
                
                return winner_id

        return None


def show_splash_screen(screen, duration=3):
        """Display a splash screen before the main menu."""
        clock = pygame.time.Clock()
        start_time = time.time()

        # Try loading logo
        try:
            logo = pygame.image.load('logo.jpg')
            logo = pygame.transform.smoothscale(logo, (400, 400))
        except:
            logo = None

        title_font = pygame.font.SysFont('Segoe UI', 72)
        subtitle_font = pygame.font.SysFont('Segoe UI', 32)

        title_text = title_font.render("APEX AI SURVIVAL ARENA", True, (255, 255, 255))
        subtitle_text = subtitle_font.render("Loading Simulation...", True, (200, 200, 200))

        while time.time() - start_time < duration:
            screen.fill((0, 0, 0))

            if logo:
                rect = logo.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
                screen.blit(logo, rect)

            title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 150))
            screen.blit(title_text, title_rect)

            subtitle_rect = subtitle_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 210))
            screen.blit(subtitle_text, subtitle_rect)

            pygame.display.flip()
            clock.tick(60)



# ---------- Main Game Loop ----------
def main():
    import sys
    
    if len(sys.argv) > 1:
        model_files = sys.argv[1:]
    else:
        model_files = glob.glob("models/*.pth")
        if not model_files:
            print("No .pth files found in current directory.")
            print("Usage: python script.py [model1.pth] [model2.pth] ...")
            return
    
    if len(model_files) < 2:
        print("Need at least 2 agents to play!")
        return
    
    print(f"Found {len(model_files)} AI agents")
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    show_splash_screen(screen, duration=3)
    
    current_state = 'home'
    game_state = GameStateManager()
    home_screen = HomeScreen(screen, model_files)
    
    try:
        while True:
            if current_state == 'home':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    
                    result = home_screen.handle_event(event)
                    if result == 'quit':
                        return
                    elif isinstance(result, tuple) and result[0] == 'start_game':
                        current_state = 'game'
                        game_settings = result[1]
                        break
                
                if current_state == 'home':
                    home_screen.draw()
                    pygame.display.flip()
                    clock.tick(60)
            
            elif current_state == 'game':
                visualizer = AestheticMultiAgentVisualizer(
                    home_screen.model_files, 
                    home_screen.game_state,
                    game_settings['num_agents'],
                    game_settings['duration'],
                    game_settings['resources']
                )
                winner_id = visualizer.run()
                
                if winner_id is None:
                    return
                
                agents_data = visualizer.env.agents
                
                current_state = 'results'
                results_screen = ResultsScreen(screen, home_screen.game_state, winner_id, agents_data)
            
            elif current_state == 'results':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    
                    result = results_screen.handle_event(event)
                    if result == 'home':
                        home_screen.selected_index = None
                        home_screen.selected_bet_index = 1
                        current_state = 'home'
                        break
                    elif result == 'quit':
                        return
                
                if current_state == 'results':
                    results_screen.draw()
                    pygame.display.flip()
                    clock.tick(60)
    
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()