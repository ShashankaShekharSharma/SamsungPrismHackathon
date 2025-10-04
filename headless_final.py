# headless_survival.py
# Headless Multi-agent survival simulation for Docker/automation

import numpy as np
import torch
import torch.nn as nn
import random
import time
import os
import glob
import json
import argparse
from datetime import datetime

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
            'health': 100.0, 'hunger': 0.0, 'thirst': 0.0,
            'body_temperature': 37.0, 'position_timeout': 0,
            'last_position': tuple(self.HOME),
            'known_tree_locations': set(), 'known_water_locations': set(),
            'explored_positions': set([tuple(self.HOME)]),
            'all_positions_explored': False, 'alive': True,
            'death_time': None, 'death_cause': None,
            'last_action': 'IDLE', 'consecutive_rest_count': 0, 
            'in_mandatory_rest': False, 'time_since_last_rest': 0, 'can_rest': False,
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
                'health': 100.0, 'hunger': 0.0, 'thirst': 0.0,
                'body_temperature': 37.0, 'position_timeout': 0,
                'last_position': tuple(self.HOME),
                'known_tree_locations': set(), 'known_water_locations': set(),
                'explored_positions': set([tuple(self.HOME)]),
                'all_positions_explored': False, 'alive': True,
                'death_time': None, 'death_cause': None,
                'last_action': 'IDLE', 'consecutive_rest_count': 0,
                'in_mandatory_rest': False, 'time_since_last_rest': 0, 'can_rest': False,
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
        current_pos = tuple(agent['pos'])
        
        agent['last_action'] = action
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

        for tree_pos in self.TREES:
            if current_pos == tree_pos and tree_pos not in agent['known_tree_locations']:
                agent['known_tree_locations'].add(tree_pos)
                agent['stats']['trees_found'] += 1
                reward += 40
                break
        for water_pos in self.WATERS:
            if current_pos == water_pos and water_pos not in agent['known_water_locations']:
                agent['known_water_locations'].add(water_pos)
                agent['stats']['waters_found'] += 1
                reward += 45
                break

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
                else:
                    reward += 8
            else:
                reward -= 30
        elif action.startswith("MOVE_"):
            agent['prev_pos'] = list(agent['pos'])
            if action == "MOVE_UP" and agent['pos'][1] > 0:
                agent['pos'][1] -= 1
            elif action == "MOVE_DOWN" and agent['pos'][1] < self.grid_size - 1:
                agent['pos'][1] += 1
            elif action == "MOVE_LEFT" and agent['pos'][0] > 0:
                agent['pos'][0] -= 1
            elif action == "MOVE_RIGHT" and agent['pos'][0] < self.grid_size - 1:
                agent['pos'][0] += 1
        elif action == "EAT":
            if current_pos in agent['known_tree_locations'] and agent['hunger'] > 15:
                agent['hunger'] = max(0, agent['hunger'] - 50)
                agent['health'] = min(100, agent['health'] + 6)
                agent['stats']['successful_eats'] += 1
                reward += 45
            else:
                reward -= 20
        elif action == "DRINK":
            if current_pos in agent['known_water_locations'] and agent['thirst'] > 15:
                agent['thirst'] = max(0, agent['thirst'] - 50)
                agent['health'] = min(100, agent['health'] + 5)
                agent['stats']['successful_drinks'] += 1
                reward += 50
            else:
                reward -= 20
        elif action == "SWIM":
            if current_pos in agent['known_water_locations']:
                if agent['body_temperature'] >= 40.0:
                    agent['body_temperature'] = max(35.0, agent['body_temperature'] - 3.0)
                    reward += 80
                elif agent['body_temperature'] >= 38.0:
                    agent['body_temperature'] = max(35.0, agent['body_temperature'] - 2.0)
                    reward += 35
                else:
                    agent['body_temperature'] = max(30.0, agent['body_temperature'] - 2.0)
                    reward -= 15
            else:
                reward -= 20
        elif action == "START_FIRE":
            if (current_pos not in self.WATERS and current_pos not in self.TREES and current_pos not in self.active_fires):
                self.active_fires[current_pos] = self.fire_duration
                agent['stats']['fires_started'] += 1
                if agent['body_temperature'] <= 34.0:
                    agent['body_temperature'] = min(42.0, agent['body_temperature'] + 2.5)
                    reward += 75
                else:
                    reward += 30
            else:
                reward -= 25

        if agent['in_mandatory_rest'] and action != "REST":
            reward -= 40

        health_penalty = 0
        if agent['body_temperature'] >= 40.0:
            health_penalty += 2.5
        elif agent['body_temperature'] <= 34.0:
            health_penalty += 2.0
        if agent['hunger'] >= 80:
            health_penalty += 1.5
        if agent['thirst'] >= 75:
            health_penalty += 2.0

        agent['health'] = max(0, agent['health'] - health_penalty)
        
        if agent['health'] <= 0:
            agent['alive'] = False
            agent['death_time'] = self.time
            agent['death_cause'] = self.determine_death_cause(agent)
            reward -= 300

        done = agent['health'] <= 0
        return self.get_state_for_agent(agent_id), reward, done, ""

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
            return True
        except Exception as e:
            print(f"Error loading model {path}: {e}")
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

# ---------- Headless Simulation Runner ----------
class HeadlessSimulation:
    def __init__(self, model_files, duration=8, num_runs=1, output_dir="results"):
        self.model_files = model_files
        self.duration = duration
        self.num_runs = num_runs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        max_time = duration * 360
        self.env = MultiAgentSurvivalEnv(grid_size=12, max_time=max_time, randomize_resources=True)
        self.agents = []
        self.load_agents()
    
    def load_agents(self):
        for i, model_file in enumerate(self.model_files):
            name = os.path.splitext(os.path.basename(model_file))[0]
            agent_id = self.env.add_agent(name, model_file)
            dummy_state = self.env.get_state_for_agent(agent_id)
            state_size = len(dummy_state) if dummy_state is not None else 25
            agent = RLAgent(state_size, self.env.action_space(), model_file, name)
            self.agents.append(agent)
    
    def run_simulation(self, run_id):
        print(f"\n=== Starting Simulation {run_id + 1}/{self.num_runs} ===")
        self.env.reset(regenerate_resources=True)
        
        step_count = 0
        start_time = time.time()
        
        while True:
            living_agents = [i for i, agent in enumerate(self.env.agents) if agent['alive']]
            if not living_agents or self.env.time >= self.env.max_time:
                break
            
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
            
            results, done = self.env.step_all_agents(actions)
            step_count += 1
            
            if step_count % 500 == 0:
                living = self.env.get_living_agents()
                print(f"  Step {step_count}: {living}/{len(self.agents)} alive, Time: {self.env.time}s")
            
            if done:
                break
        
        elapsed_time = time.time() - start_time
        
        results = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'duration_days': self.duration,
            'total_steps': step_count,
            'simulation_time_seconds': self.env.time,
            'elapsed_real_time': elapsed_time,
            'agents': []
        }
        
        for agent_data in self.env.agents:
            agent_result = {
                'id': agent_data['id'],
                'name': agent_data['name'],
                'model_path': agent_data['model_path'],
                'alive': agent_data['alive'],
                'final_health': agent_data['health'],
                'final_hunger': agent_data['hunger'],
                'final_thirst': agent_data['thirst'],
                'final_temperature': agent_data['body_temperature'],
                'death_time': agent_data['death_time'],
                'death_cause': agent_data['death_cause'],
                'survival_days': (agent_data['death_time'] or self.env.time) // self.env.seconds_per_day,
                'stats': agent_data['stats']
            }
            results['agents'].append(agent_result)
        
        sorted_agents = sorted(results['agents'], 
                              key=lambda a: (a['alive'], a.get('death_time', self.env.max_time), a['final_health']), 
                              reverse=True)
        results['winner'] = sorted_agents[0]
        results['leaderboard'] = sorted_agents
        
        print(f"\n=== Simulation {run_id + 1} Complete ===")
        print(f"Winner: {results['winner']['name']}")
        print(f"Survivors: {sum(1 for a in results['agents'] if a['alive'])}/{len(self.agents)}")
        print(f"Time: {elapsed_time:.2f}s for {step_count} steps")
        
        return results
    
    def run_all(self):
        all_results = []
        overall_start = time.time()
        
        for i in range(self.num_runs):
            result = self.run_simulation(i)
            all_results.append(result)
            
            filename = os.path.join(self.output_dir, f"simulation_{i+1}.json")
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {filename}")
        
        overall_elapsed = time.time() - overall_start
        
        summary = {
            'total_runs': self.num_runs,
            'total_elapsed_time': overall_elapsed,
            'duration_days': self.duration,
            'num_agents': len(self.agents),
            'agent_names': [agent.name for agent in self.agents],
            'win_counts': {},
            'average_survival_days': {},
            'results': all_results
        }
        
        for agent in self.agents:
            summary['win_counts'][agent.name] = 0
            summary['average_survival_days'][agent.name] = []
        
        for result in all_results:
            winner_name = result['winner']['name']
            summary['win_counts'][winner_name] += 1
            
            for agent_result in result['agents']:
                summary['average_survival_days'][agent_result['name']].append(agent_result['survival_days'])
        
        for name in summary['average_survival_days']:
            days = summary['average_survival_days'][name]
            summary['average_survival_days'][name] = sum(days) / len(days) if days else 0
        
        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        print(f"Total Simulations: {self.num_runs}")
        print(f"Total Time: {overall_elapsed:.2f}s")
        print(f"\nWin Counts:")
        for name, count in sorted(summary['win_counts'].items(), key=lambda x: x[1], reverse=True):
            win_rate = (count / self.num_runs) * 100
            print(f"  {name}: {count} wins ({win_rate:.1f}%)")
        print(f"\nAverage Survival Days:")
        for name, avg_days in sorted(summary['average_survival_days'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {avg_days:.1f} days")
        print(f"\nResults saved to: {self.output_dir}/")
        print("="*60)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Headless AI Survival Simulation')
    parser.add_argument('--models', type=str, nargs='+', help='Paths to model files')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing model files')
    parser.add_argument('--duration', type=int, default=8, help='Duration in days (default: 8)')
    parser.add_argument('--runs', type=int, default=1, help='Number of simulations to run (default: 1)')
    parser.add_argument('--output', type=str, default='results', help='Output directory (default: results)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Get model files
    if args.models:
        model_files = args.models
    else:
        model_files = glob.glob(os.path.join(args.model_dir, "*.pth"))
        if not model_files:
            print(f"Error: No .pth files found in {args.model_dir}/")
            print("Usage: python headless_survival.py --models model1.pth model2.pth ...")
            print("   or: python headless_survival.py --model-dir path/to/models/")
            return
    
    if len(model_files) < 2:
        print("Error: Need at least 2 agents to run simulation!")
        return
    
    print("="*60)
    print("HEADLESS AI SURVIVAL SIMULATION")
    print("="*60)
    print(f"Agents: {len(model_files)}")
    print(f"Duration: {args.duration} days")
    print(f"Simulations: {args.runs}")
    print(f"Output: {args.output}/")
    print("="*60)
    
    # Create and run simulation
    sim = HeadlessSimulation(
        model_files=model_files,
        duration=args.duration,
        num_runs=args.runs,
        output_dir=args.output
    )
    
    summary = sim.run_all()
    
    print("\nSimulation complete! Check the output directory for detailed results.")

if __name__ == "__main__":
    main()