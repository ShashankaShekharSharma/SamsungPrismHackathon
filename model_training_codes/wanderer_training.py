# survival_10x10_seasons_temperature_enhanced.py
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ====== Enhanced Survival Environment with Multiple Resources and Fire Duration ======
class SurvivalEnv:
    def __init__(self, grid_size=10, max_time=2880):  # 8 days * 6 minutes * 60 seconds
        self.grid_size = grid_size
        self.max_time = max_time
        
        # Time constants
        self.seconds_per_minute = 60
        self.minutes_per_day = 6  # 3 morning + 3 night
        self.seconds_per_day = self.minutes_per_day * self.seconds_per_minute  # 360 seconds
        self.total_days = 8
        
        # Season system - 2 days each
        self.season_names = ["Summer", "Autumn", "Winter", "Spring"]
        self.season_durations = [2, 2, 2, 2]  # days each
        
        # Fixed resource locations for 10x10 grid - MULTIPLE SOURCES
        self.HOME = (5, 5)  # Center of grid
        
        # Multiple food sources (trees)
        self.TREES = [(2, 8), (7, 1), (1, 3), (9, 6)]
        
        # Multiple water sources
        self.WATERS = [(8, 2), (0, 9), (6, 0), (3, 7)]

        print("="*80)
        print("10x10 GRID RESOURCE LAYOUT:")
        print(f"HOME (Starting Point): {self.HOME}")
        print(f"TREE LOCATIONS: {self.TREES}")
        print(f"WATER LOCATIONS: {self.WATERS}")
        print("="*80)

        # Actions - added temperature management actions
        self.actions = ["IDLE", "REST", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
                        "EAT", "DRINK", "EXPLORE", "SWIM", "START_FIRE"]

        # Fire system with duration
        self.active_fires = {}  # {(x, y): remaining_duration}
        self.fire_duration = 180  # 3 minutes = 180 seconds
        
        self.reset()

    def action_space(self):
        return len(self.actions)

    def get_current_season(self):
        """Get current season based on time"""
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
        """Get current time of day"""
        seconds_in_current_day = self.time % self.seconds_per_day
        minutes_in_day = seconds_in_current_day // self.seconds_per_minute
        
        if minutes_in_day < 3:
            return "Morning"
        else:
            return "Night"

    def get_base_temperature(self):
        """Get base environmental temperature based on season and time"""
        season, season_idx = self.get_current_season()
        time_of_day = self.get_time_of_day()
        
        # Base temperatures by season (Celsius)
        season_temps = {
            "Summer": 35,   # Hot
            "Autumn": 20,   # Mild
            "Winter": 5,    # Cold
            "Spring": 22    # Pleasant
        }
        
        base_temp = season_temps[season]
        
        # Time of day modifier
        if time_of_day == "Night":
            base_temp -= 8  # Cooler at night
            
        return base_temp

    def update_fires(self):
        """Update fire durations and remove expired fires"""
        expired_fires = []
        for fire_pos, remaining_time in self.active_fires.items():
            if remaining_time <= 1:
                expired_fires.append(fire_pos)
            else:
                self.active_fires[fire_pos] = remaining_time - 1
        
        for fire_pos in expired_fires:
            del self.active_fires[fire_pos]

    def reset(self):
        self.time = 0
        self.agent_pos = list(self.HOME)  # Start at home

        self.health = 100.0
        self.hunger = 0.0
        self.thirst = 0.0
        
        # Body temperature system (36-38°C is normal range)
        self.body_temperature = 37.0  # Normal body temp
        self.temp_min_critical = 34.0  # Hypothermia threshold
        self.temp_max_critical = 40.0  # Hyperthermia threshold
        self.temp_comfortable_min = 36.0
        self.temp_comfortable_max = 38.0
        
        # Rest management
        self.consecutive_rest_count = 0
        self.in_mandatory_rest = False
        self.time_since_last_rest = 0
        self.can_rest = False

        # Exploration memory - for multiple resources
        self.known_tree_locations = set()
        self.known_water_locations = set()
        self.explored_positions = set()
        self.all_positions_explored = False
        
        # Add starting position to explored
        self.explored_positions.add(tuple(self.HOME))
        
        # Generate all possible positions for systematic exploration
        self.all_positions = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.all_positions.add((x, y))
        
        # Grid state for visualization
        self.grid_explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.grid_explored[self.HOME[0], self.HOME[1]] = True
        
        # Reset fire system
        self.active_fires = {}

        return self.get_state()

    def get_state(self):
        # Enhanced state with exploration info, season, and temperature
        season, season_idx = self.get_current_season()
        time_of_day = self.get_time_of_day()
        env_temp = self.get_base_temperature()
        
        tree_known = 1.0 if len(self.known_tree_locations) > 0 else 0.0
        water_known = 1.0 if len(self.known_water_locations) > 0 else 0.0
        exploration_progress = len(self.explored_positions) / (self.grid_size * self.grid_size)
        exploration_complete = 1.0 if self.all_positions_explored else 0.0
        
        # Distance to nearest known resources (if any known)
        nearest_tree_dist = 1.0
        nearest_water_dist = 1.0
        
        if self.known_tree_locations:
            distances = [abs(self.agent_pos[0] - tx) + abs(self.agent_pos[1] - ty) 
                        for tx, ty in self.known_tree_locations]
            nearest_tree_dist = min(distances) / (self.grid_size * 2)
            
        if self.known_water_locations:
            distances = [abs(self.agent_pos[0] - wx) + abs(self.agent_pos[1] - wy) 
                        for wx, wy in self.known_water_locations]
            nearest_water_dist = min(distances) / (self.grid_size * 2)

        # Time and season info
        current_day = self.time // self.seconds_per_day
        day_progress = (self.time % self.seconds_per_day) / self.seconds_per_day
        
        # Temperature info
        temp_normalized = (self.body_temperature - 30) / 15  # Normalize 30-45°C range
        temp_too_hot = 1.0 if self.body_temperature >= self.temp_max_critical else 0.0
        temp_too_cold = 1.0 if self.body_temperature <= self.temp_min_critical else 0.0
        
        # Fire availability (nearest active fire distance)
        nearest_fire_dist = 1.0  # Default if no fires
        if self.active_fires:
            min_fire_dist = min(abs(self.agent_pos[0] - fx) + abs(self.agent_pos[1] - fy) 
                               for fx, fy in self.active_fires.keys())
            nearest_fire_dist = min_fire_dist / (self.grid_size * 2)

        return np.array([
            self.agent_pos[0] / self.grid_size,      # 0: Agent X position
            self.agent_pos[1] / self.grid_size,      # 1: Agent Y position  
            self.health / 100,                       # 2: Health
            self.hunger / 100,                       # 3: Hunger
            self.thirst / 100,                       # 4: Thirst
            temp_normalized,                         # 5: Body temperature
            self.consecutive_rest_count / 2.0,       # 6: Rest count
            self.time / self.max_time,               # 7: Time progress
            float(self.in_mandatory_rest),           # 8: In mandatory rest
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
            len(self.known_tree_locations) / len(self.TREES),  # 23: Trees discovered ratio
            len(self.known_water_locations) / len(self.WATERS) # 24: Waters discovered ratio
        ], dtype=np.float32)

    def update_body_temperature(self):
        """Update body temperature based on environment and time"""
        env_temp = self.get_base_temperature()
        current_pos = tuple(self.agent_pos)
        
        # Base temperature change toward environmental temperature
        temp_diff = env_temp - self.body_temperature
        
        # Natural temperature regulation (body tries to maintain 37°C)
        natural_regulation = (37.0 - self.body_temperature) * 0.05
        
        # Environmental influence (slower)
        env_influence = temp_diff * 0.02
        
        # Special location effects
        location_effect = 0.0
        if current_pos in self.WATERS:
            location_effect = -0.3  # Water cools you down
        elif current_pos in self.active_fires:
            location_effect = 0.4   # Fire warms you up
        
        # Apply temperature change
        temp_change = natural_regulation + env_influence + location_effect
        self.body_temperature += temp_change
        
        # Clamp to realistic range
        self.body_temperature = max(30.0, min(45.0, self.body_temperature))

    def step(self, action_idx):
        if self.time >= self.max_time:
            final_bonus = 200 if self.health > 50 else -100
            return self.get_state(), final_bonus, True, f"8 days survived! Final health: {self.health:.1f}"
            
        action = self.actions[action_idx]
        reward = 0.0
        info = ""
        current_pos = tuple(self.agent_pos)
        
        # Get current season and time info
        season, season_idx = self.get_current_season()
        time_of_day = self.get_time_of_day()
        current_day = (self.time // self.seconds_per_day) + 1

        # Time progression
        self.time += 1
        self.time_since_last_rest += 1
        
        # Update fire durations
        self.update_fires()
        
        # Update body temperature
        self.update_body_temperature()
        
        # Track current position as explored
        self.explored_positions.add(current_pos)
        self.grid_explored[current_pos[0], current_pos[1]] = True
        
        # AUTOMATIC RESOURCE DISCOVERY - Multiple sources
        discovery_bonus = 0
        for tree_pos in self.TREES:
            if current_pos == tree_pos and tree_pos not in self.known_tree_locations:
                self.known_tree_locations.add(tree_pos)
                discovery_bonus += 40
                info += f" [TREE_FOUND at {current_pos}]"
                break
                
        for water_pos in self.WATERS:
            if current_pos == water_pos and water_pos not in self.known_water_locations:
                self.known_water_locations.add(water_pos)
                discovery_bonus += 45
                info += f" [WATER_FOUND at {current_pos}]"
                break
        
        reward += discovery_bonus
        
        # Check if all positions are explored
        if len(self.explored_positions) >= len(self.all_positions):
            self.all_positions_explored = True
        
        # Check if mandatory rest is needed - RELAXED conditions
        if self.time_since_last_rest >= 240 or self.health < 20:  # 4 minutes = 240 seconds
            if not self.in_mandatory_rest:
                self.in_mandatory_rest = True
                self.can_rest = True
                self.consecutive_rest_count = 0
                info += " [MANDATORY_REST_TRIGGERED] "

        # TEMPERATURE CRITICAL CHECKS
        temp_critical_hot = self.body_temperature >= self.temp_max_critical
        temp_critical_cold = self.body_temperature <= self.temp_min_critical
        temp_uncomfortable_hot = self.body_temperature >= self.temp_comfortable_max
        temp_uncomfortable_cold = self.body_temperature <= self.temp_comfortable_min
        
        # CRITICAL SURVIVAL CHECKS - ADJUSTED thresholds for longer survival
        critical_thirst = self.thirst >= 70  # Reduced from 85 to prevent early death
        critical_hunger = self.hunger >= 75  # Reduced from 90
        
        # Execute actions
        if action == "REST":
            if self.can_rest and self.in_mandatory_rest:
                self.consecutive_rest_count += 1
                
                # Much slower need progression during rest
                self.hunger = min(100, self.hunger + 0.03)  # Very slow during rest
                self.thirst = min(100, self.thirst + 0.04)  # Very slow during rest
                
                # Better health recovery during rest
                self.health = min(100, self.health + 4)  # Better recovery
                
                if self.consecutive_rest_count >= 2:
                    self.in_mandatory_rest = False
                    self.can_rest = False
                    self.time_since_last_rest = 0
                    self.consecutive_rest_count = 0
                    reward += 15
                    info += f" REST_COMPLETE"
                else:
                    reward += 8
                    info += f" RESTING ({self.consecutive_rest_count}/2)"
                    
            else:
                reward -= 30
                info += " ILLEGAL_REST"
                self.hunger = min(100, self.hunger + 0.12)
                self.thirst = min(100, self.thirst + 0.15)
                
        else:
            if not self.in_mandatory_rest:
                self.consecutive_rest_count = 0
            
            # Seasonal need progression - FURTHER REDUCED rates
            hunger_rate = 0.10  # Further reduced
            thirst_rate = 0.12  # Further reduced
            
            if season == "Summer":
                thirst_rate = 0.18  # Reduced
            elif season == "Winter":
                hunger_rate = 0.15  # Reduced
            
            self.hunger = min(100, self.hunger + hunger_rate)
            self.thirst = min(100, self.thirst + thirst_rate)

            # Penalty for not resting during mandatory rest
            if self.in_mandatory_rest:
                reward -= 40
                info += " [MUST_REST] "

            if action == "IDLE":
                # Enhanced idle penalties with temperature consideration
                if temp_critical_hot or temp_critical_cold:
                    reward -= 50
                    info += "IDLE (CRITICAL_TEMP)"
                elif not self.all_positions_explored and not (critical_thirst or critical_hunger):
                    reward -= 25
                    info += "IDLE (MUST_EXPLORE)"
                elif critical_thirst or critical_hunger:
                    reward -= 40
                    info += "IDLE (CRITICAL_NEEDS)"
                elif self.hunger < 15 and self.thirst < 15 and not (temp_uncomfortable_hot or temp_uncomfortable_cold):
                    reward += 10
                    info += "IDLE (excellent_mgmt)"
                else:
                    reward -= 15
                    info += "IDLE (needs_attention)"

            elif action == "EXPLORE":
                explore_reward, explore_info = self.handle_exploration()
                reward += explore_reward
                info += explore_info

            elif action == "SWIM":
                swim_reward, swim_info = self.handle_swim()
                reward += swim_reward
                info += swim_info

            elif action == "START_FIRE":
                fire_reward, fire_info = self.handle_start_fire()
                reward += fire_reward
                info += fire_info

            elif action == "MOVE_UP" and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
                move_reward, move_info = self.evaluate_movement()
                reward += move_reward
                info += f" MOVE_UP to {tuple(self.agent_pos)}{move_info}"
            elif action == "MOVE_DOWN" and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1
                move_reward, move_info = self.evaluate_movement()
                reward += move_reward
                info += f" MOVE_DOWN to {tuple(self.agent_pos)}{move_info}"
            elif action == "MOVE_LEFT" and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
                move_reward, move_info = self.evaluate_movement()
                reward += move_reward
                info += f" MOVE_LEFT to {tuple(self.agent_pos)}{move_info}"
            elif action == "MOVE_RIGHT" and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
                move_reward, move_info = self.evaluate_movement()
                reward += move_reward
                info += f" MOVE_RIGHT to {tuple(self.agent_pos)}{move_info}"

            elif action == "EAT":
                if current_pos in self.known_tree_locations:
                    if self.hunger > 15:  # Lower threshold
                        hunger_before = self.hunger
                        self.hunger = max(0, self.hunger - 50)  # Less food per eat
                        self.health = min(100, self.health + 6)
                        info += f" EAT_SUCCESS: Hunger {hunger_before:.1f} -> {self.hunger:.1f}"
                        reward += 45
                    else:
                        info += " Not_hungry_enough"
                        reward -= 5
                elif len(self.known_tree_locations) == 0:
                    info += f" EAT_FAILED: No_trees_known"
                    reward -= 25
                else:
                    nearest_tree = min(self.known_tree_locations, 
                                     key=lambda t: abs(self.agent_pos[0] - t[0]) + abs(self.agent_pos[1] - t[1]))
                    info += f" EAT_FAILED: Not_at_tree (nearest: {nearest_tree})"
                    reward -= 20

            elif action == "DRINK":
                if current_pos in self.known_water_locations:
                    if self.thirst > 15:  # Lower threshold
                        thirst_before = self.thirst
                        self.thirst = max(0, self.thirst - 50)  # Less water per drink
                        self.health = min(100, self.health + 5)
                        info += f" DRINK_SUCCESS: Thirst {thirst_before:.1f} -> {self.thirst:.1f}"
                        reward += 50
                    else:
                        info += " Not_thirsty_enough"
                        reward -= 5
                elif len(self.known_water_locations) == 0:
                    info += f" DRINK_FAILED: No_water_known"
                    reward -= 25
                else:
                    nearest_water = min(self.known_water_locations, 
                                      key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                    info += f" DRINK_FAILED: Not_at_water (nearest: {nearest_water})"
                    reward -= 20

        # TEMPERATURE HEALTH EFFECTS - REDUCED penalties
        temp_health_penalty = 0
        if temp_critical_hot:
            temp_health_penalty += 2.5  # Reduced
            info += " HYPERTHERMIA"
        elif temp_critical_cold:
            temp_health_penalty += 2.0  # Reduced
            info += " HYPOTHERMIA"
        elif temp_uncomfortable_hot:
            temp_health_penalty += 0.8  # Reduced
            info += " Too_hot"
        elif temp_uncomfortable_cold:
            temp_health_penalty += 0.8  # Reduced
            info += " Too_cold"

        # Health penalties from unmet needs - SIGNIFICANTLY REDUCED
        health_penalty = temp_health_penalty
        if self.hunger >= 80:  # Higher threshold
            health_penalty += 1.5  # Reduced
            info += " CRITICAL_HUNGER"
        elif self.hunger >= 60:
            health_penalty += 0.5  # Reduced
            info += " High_hunger"
            
        if self.thirst >= 75:  # Higher threshold  
            health_penalty += 2.0  # Reduced from 2.5
            info += " CRITICAL_THIRST"
        elif self.thirst >= 50:
            health_penalty += 0.8  # Reduced
            info += " High_thirst"

        self.health = max(0, self.health - health_penalty)

        # ENHANCED PRIORITY SYSTEM with multiple resources
        if not self.in_mandatory_rest:
            
            # CRITICAL TEMPERATURE OVERRIDE - Highest priority
            if temp_critical_hot and len(self.known_water_locations) > 0:
                if action == "SWIM" and current_pos in self.known_water_locations:
                    reward += 100  # Massive bonus for addressing critical overheating
                    info += " [EMERGENCY_COOLING]"
                elif action.startswith("MOVE"):
                    nearest_water = min(self.known_water_locations, 
                                      key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                    water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                    old_dist = self.calculate_old_distance(action, nearest_water)
                    if water_dist < old_dist:
                        reward += 50
                        info += f" (EMERGENCY->WATER)"
                    else:
                        reward -= 35
                else:
                    reward -= 50
                    
            elif temp_critical_cold:
                # Need fire - either go to existing fire or start new one
                if self.active_fires:
                    nearest_fire = min(self.active_fires.keys(), 
                                     key=lambda f: abs(self.agent_pos[0] - f[0]) + abs(self.agent_pos[1] - f[1]))
                    if action.startswith("MOVE"):
                        fire_dist = abs(self.agent_pos[0] - nearest_fire[0]) + abs(self.agent_pos[1] - nearest_fire[1])
                        old_dist = self.calculate_old_distance(action, nearest_fire)
                        if fire_dist < old_dist:
                            reward += 50
                            info += f" (EMERGENCY->FIRE)"
                        elif fire_dist == 0:  # At fire
                            reward += 30
                            info += " (WARMING_AT_FIRE)"
                        else:
                            reward -= 35
                    else:
                        reward -= 45
                else:
                    # No fires exist - must start one
                    if action == "START_FIRE":
                        pass  # Reward handled in handle_start_fire
                    else:
                        reward -= 55
                        info += " (MUST_START_FIRE)"
                        
            # CRITICAL SURVIVAL OVERRIDE - Second priority
            elif critical_thirst and len(self.known_water_locations) > 0:
                if action == "DRINK" and current_pos in self.known_water_locations:
                    reward += 80
                elif action.startswith("MOVE"):
                    nearest_water = min(self.known_water_locations, 
                                      key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                    water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                    old_dist = self.calculate_old_distance(action, nearest_water)
                    if water_dist < old_dist:
                        reward += 40
                        info += f" (EMERGENCY->WATER)"
                    else:
                        reward -= 25
                else:
                    reward -= 40
                    
            elif critical_hunger and len(self.known_tree_locations) > 0:
                if action == "EAT" and current_pos in self.known_tree_locations:
                    reward += 70
                elif action.startswith("MOVE"):
                    nearest_tree = min(self.known_tree_locations, 
                                     key=lambda t: abs(self.agent_pos[0] - t[0]) + abs(self.agent_pos[1] - t[1]))
                    tree_dist = abs(self.agent_pos[0] - nearest_tree[0]) + abs(self.agent_pos[1] - nearest_tree[1])
                    old_dist = self.calculate_old_distance(action, nearest_tree)
                    if tree_dist < old_dist:
                        reward += 35
                        info += f" (EMERGENCY->FOOD)"
                    else:
                        reward -= 20
                else:
                    reward -= 35
                    
            # TEMPERATURE MANAGEMENT - Third priority
            elif temp_uncomfortable_hot and len(self.known_water_locations) > 0 and not (critical_hunger or critical_thirst):
                if action == "SWIM" and current_pos in self.known_water_locations:
                    reward += 40
                elif action.startswith("MOVE"):
                    nearest_water = min(self.known_water_locations, 
                                      key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                    water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                    old_dist = self.calculate_old_distance(action, nearest_water)
                    if water_dist < old_dist:
                        reward += 20
                        info += f" (->COOL)"
                        
            elif temp_uncomfortable_cold and not (critical_hunger or critical_thirst):
                if self.active_fires and current_pos in self.active_fires:
                    reward += 25  # Bonus for staying at fire when cold
                    info += " (WARMING)"
                elif not self.active_fires and action == "START_FIRE":
                    pass  # Handled in handle_start_fire
                elif self.active_fires and action.startswith("MOVE"):
                    nearest_fire = min(self.active_fires.keys(), 
                                     key=lambda f: abs(self.agent_pos[0] - f[0]) + abs(self.agent_pos[1] - f[1]))
                    fire_dist = abs(self.agent_pos[0] - nearest_fire[0]) + abs(self.agent_pos[1] - nearest_fire[1])
                    old_dist = self.calculate_old_distance(action, nearest_fire)
                    if fire_dist < old_dist:
                        reward += 15
                        info += f" (->WARM)"
                        
            # EXPLORATION PRIORITY - When not in critical survival mode
            elif not self.all_positions_explored and not (critical_thirst or critical_hunger or temp_critical_hot or temp_critical_cold):
                if action == "EXPLORE":
                    pass  # Reward handled in handle_exploration()
                elif action.startswith("MOVE"):
                    new_pos = tuple(self.agent_pos)
                    if new_pos not in self.explored_positions:
                        reward += 20
                        info += " (new_area)"
                    else:
                        reward += 3
                        info += " (searching)"
                else:
                    reward -= 12
                    info += " (must_explore)"
            
            # RESOURCE MANAGEMENT - Normal priority when exploration complete
            elif self.all_positions_explored:
                if self.thirst >= 45 and self.hunger < 60 and len(self.known_water_locations) > 0:
                    if action == "DRINK" and current_pos in self.known_water_locations:
                        reward += 40
                    elif action.startswith("MOVE"):
                        nearest_water = min(self.known_water_locations, 
                                          key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                        water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                        old_dist = self.calculate_old_distance(action, nearest_water)
                        if water_dist < old_dist:
                            reward += 20
                            info += f" (->WATER)"
                            
                elif self.hunger >= 35 and self.thirst < 45 and len(self.known_tree_locations) > 0:
                    if action == "EAT" and current_pos in self.known_tree_locations:
                        reward += 35
                    elif action.startswith("MOVE"):
                        nearest_tree = min(self.known_tree_locations, 
                                         key=lambda t: abs(self.agent_pos[0] - t[0]) + abs(self.agent_pos[1] - t[1]))
                        tree_dist = abs(self.agent_pos[0] - nearest_tree[0]) + abs(self.agent_pos[1] - nearest_tree[1])
                        old_dist = self.calculate_old_distance(action, nearest_tree)
                        if tree_dist < old_dist:
                            reward += 15
                            info += f" (->TREE)"

        # Base survival rewards
        reward += (self.health / 100.0) * 3
        
        # Temperature comfort bonus
        if self.temp_comfortable_min <= self.body_temperature <= self.temp_comfortable_max:
            reward += 2
        
        # Penalize unmet needs - REDUCED penalties
        if self.hunger > 40:
            reward -= (self.hunger / 100.0) * 1.2  # Reduced
        if self.thirst > 30:
            reward -= (self.thirst / 100.0) * 1.5  # Reduced
            
        # Temperature penalties
        if temp_uncomfortable_hot or temp_uncomfortable_cold:
            reward -= 2  # Reduced
        if temp_critical_hot or temp_critical_cold:
            reward -= 6  # Reduced

        done = self.health <= 0 or self.time >= self.max_time
        
        if done:
            if self.health <= 0:
                reward -= 300
                info += " DEATH"
            elif self.time >= self.max_time:
                if self.health >= 50:
                    need_bonus = max(0, (100 - self.hunger) + (100 - self.thirst)) / 2
                    exploration_bonus = 100 if self.all_positions_explored else -75
                    temp_bonus = 25 if self.temp_comfortable_min <= self.body_temperature <= self.temp_comfortable_max else -25
                    resource_bonus = (len(self.known_tree_locations) * 10) + (len(self.known_water_locations) * 12)
                    survival_bonus = 200 + need_bonus + exploration_bonus + temp_bonus + resource_bonus
                    reward += survival_bonus
                    info += f" 8DAY_SUCCESS Exp:{exploration_bonus} Mgmt:{need_bonus:.1f} Temp:{temp_bonus} Res:{resource_bonus}"
                else:
                    reward -= 100
                    info += f" MISSION_FAILED: Final_health {self.health:.1f} < 50"
        
        # Add season and day info to status
        info = f"D{current_day} {season} {time_of_day} | Temp:{self.body_temperature:.1f}C |" + info
            
        return self.get_state(), reward, done, info

    def handle_swim(self):
        """Handle swimming action for cooling down"""
        current_pos = tuple(self.agent_pos)
        
        if len(self.known_water_locations) == 0:
            return -25, " SWIM_FAILED: No_water_known"
            
        if current_pos not in self.known_water_locations:
            nearest_water = min(self.known_water_locations, 
                              key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
            return -20, f" SWIM_FAILED: Not_at_water (nearest: {nearest_water})"
            
        # Swimming effects
        if self.body_temperature >= self.temp_max_critical:
            # Critical cooling needed
            temp_before = self.body_temperature
            self.body_temperature = max(35.0, self.body_temperature - 3.0)
            reward = 80
            info = f" EMERGENCY_SWIM Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C"
        elif self.body_temperature >= self.temp_comfortable_max:
            # Regular cooling
            temp_before = self.body_temperature
            self.body_temperature = max(35.0, self.body_temperature - 2.0)
            reward = 35
            info = f" COOLING_SWIM Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C"
        else:
            # Not hot enough - might make you too cold
            temp_before = self.body_temperature
            self.body_temperature = max(30.0, self.body_temperature - 1.5)
            reward = -15
            info = f" UNNECESSARY_SWIM Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C (now_cold)"
            
        return reward, info

    def handle_start_fire(self):
        """Handle starting a fire for warming up - with duration system"""
        current_pos = tuple(self.agent_pos)
        
        # Can't start fire at water or tree locations
        if current_pos in self.WATERS:
            return -30, " FIRE_FAILED: Cant_start_in_water"
        if current_pos in self.TREES:
            return -25, " FIRE_FAILED: Cant_start_at_tree"
            
        # Can't start fire where one already exists
        if current_pos in self.active_fires:
            remaining_time = self.active_fires[current_pos]
            return -20, f" FIRE_FAILED: Fire_exists (time_left: {remaining_time}s)"
            
        # Start fire with duration
        self.active_fires[current_pos] = self.fire_duration
        
        if self.body_temperature <= self.temp_min_critical:
            # Critical warming needed
            temp_before = self.body_temperature
            self.body_temperature = min(42.0, self.body_temperature + 2.5)
            reward = 75
            info = f" EMERGENCY_FIRE Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C (3min_duration)"
        elif self.body_temperature <= self.temp_comfortable_min:
            # Regular warming
            temp_before = self.body_temperature
            self.body_temperature = min(40.0, self.body_temperature + 2.0)
            reward = 30
            info = f" WARMING_FIRE Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C (3min_duration)"
        else:
            # Not cold enough - might overheat
            temp_before = self.body_temperature
            self.body_temperature = min(45.0, self.body_temperature + 1.5)
            reward = -10
            info = f" UNNECESSARY_FIRE Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C (getting_hot)"
            
        return reward, info

    def evaluate_movement(self):
        """Evaluate movement rewards based on current priorities"""
        current_pos = tuple(self.agent_pos)
        reward = 0
        info = ""
        
        # During exploration phase
        if not self.all_positions_explored:
            if current_pos not in self.explored_positions:
                reward += 15  # Bonus for reaching unexplored area
                info += " (new_area)"
            else:
                reward += 1   # Small bonus for movement during exploration
                
        # Temperature-based movement evaluation
        if current_pos in self.WATERS and self.body_temperature >= self.temp_comfortable_max:
            reward += 10
            info += " (at_cooling_water)"
        elif current_pos in self.active_fires and self.body_temperature <= self.temp_comfortable_min:
            reward += 10
            info += " (at_warming_fire)"
                
        return reward, info

    def handle_exploration(self):
        """Handle the EXPLORE action with systematic exploration"""
        current_pos = tuple(self.agent_pos)
        reward = 8  # Base exploration reward
        info = " EXPLORING"
        
        # If all positions are explored, discourage further exploration
        if self.all_positions_explored:
            reward = -20
            info += " - ALL_MAPPED Focus_on_survival"
            return reward, info
        
        # Check for resource discovery - multiple sources
        for tree_pos in self.TREES:
            if current_pos == tree_pos and tree_pos not in self.known_tree_locations:
                self.known_tree_locations.add(tree_pos)
                reward += 40
                info += f" - TREE_FOUND at {current_pos}"
                break
                
        for water_pos in self.WATERS:
            if current_pos == water_pos and water_pos not in self.known_water_locations:
                self.known_water_locations.add(water_pos)
                reward += 45
                info += f" - WATER_FOUND at {current_pos}"
                break
        
        # Systematic exploration rewards
        if current_pos not in self.explored_positions:
            reward += 12
            info += " (new_area_mapped)"
        else:
            reward -= 18
            info += " (already_mapped_MOVE)"
            
        # Completion bonus
        if len(self.explored_positions) >= len(self.all_positions):
            if not self.all_positions_explored:
                reward += 75
                info += " - FULL_MAP_COMPLETE Focus_on_survival"
                self.all_positions_explored = True
            
        return reward, info

    def calculate_old_distance(self, action, target):
        """Calculate what the distance would have been before this move"""
        old_pos = list(self.agent_pos)
        
        if action == "MOVE_UP":
            old_pos[1] += 1
        elif action == "MOVE_DOWN":
            old_pos[1] -= 1
        elif action == "MOVE_LEFT":
            old_pos[0] += 1
        elif action == "MOVE_RIGHT":
            old_pos[0] -= 1
            
        return abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])

# ====== Enhanced DQN Network ======
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

# ====== RL Agent with GPU Support ======
class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.lr = 0.0008

        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = []
        self.target_update_frequency = 100

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def act_greedy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 50000:
            self.memory.pop(0)

    def replay(self, batch_size=128):
        if len(self.memory) < batch_size:
            return 0
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ====== Training Function with Performance Optimizations ======
def train_agent(episodes=6000):
    env = SurvivalEnv()
    agent = RLAgent(state_size=25, action_size=len(env.actions))

    batch_size = 64
    recent_rewards = deque(maxlen=100)

    # ✅ Make checkpoint folder
    os.makedirs("checkpoints", exist_ok=True)

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)

            state = next_state
            total_reward += reward

        recent_rewards.append(total_reward)

        print(f"Episode {e+1}/{episodes} | Total Reward: {total_reward:.2f}")

        # ✅ Save checkpoint every 200 episodes
        if (e + 1) % 200 == 0:
            checkpoint_name = f"checkpoints/survival_agent_ep{e+1}.pth"
            torch.save({
                'episode': e + 1,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_name)
            print(f"💾 Checkpoint saved: {checkpoint_name}")

    # ✅ Save final model
    final_name = "survival_agent_dynamic_winter.pth"
    torch.save(agent.model.state_dict(), final_name)
    print(f"✅ Final model saved as {final_name}")


# ====== Testing Function ======
def test_agent(agent, num_tests=3):
    print(f"\n{'='*120}")
    print(f"TESTING ENHANCED 10x10 SURVIVAL AGENT - {num_tests} DETAILED RUNS")
    print(f"Expected: Explore -> Find multiple resources -> Manage needs/temperature -> Survive 8 days")
    print(f"{'='*120}")
    
    for test_num in range(1, num_tests + 1):
        print(f"\n--- TEST {test_num} ---")
        env = SurvivalEnv()
        state = env.reset()
        total_reward = 0
        
        print(f"Grid: Starting at {env.agent_pos}")
        print(f"Hidden Resources: TREES={env.TREES}, WATERS={env.WATERS}")
        print(f"Mission: Survive 8 days through Summer->Autumn->Winter->Spring")
        
        step_count = 0
        last_day_printed = -1
        
        for step in range(env.max_time):
            action = agent.act_greedy(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Print every 60 seconds (1 minute) or on important events
            current_day = (env.time // env.seconds_per_day) + 1
            current_minute = (env.time % env.seconds_per_day) // env.seconds_per_minute + 1
            
            # Print day transitions and critical events
            if (current_day != last_day_printed or 
                step % 60 == 0 or  # Every minute
                "FOUND" in info or 
                "CRITICAL" in info or 
                "EMERGENCY" in info or
                "FIRE" in info or
                "SWIM" in info or
                "SUCCESS" in info or
                done):
                
                if current_day != last_day_printed:
                    season, _ = env.get_current_season()
                    time_of_day = env.get_time_of_day()
                    env_temp = env.get_base_temperature()
                    print(f"\nDAY {current_day} {season.upper()} BEGINS - Env Temp: {env_temp}C")
                    last_day_printed = current_day
                
                # Status display - cleaner format
                rest_status = f"MR:{env.consecutive_rest_count}/2" if env.in_mandatory_rest else "Active"
                
                # Resource discovery status
                trees_found = f"{len(env.known_tree_locations)}/{len(env.TREES)}"
                waters_found = f"{len(env.known_water_locations)}/{len(env.WATERS)}"
                exploration_status = "COMPLETE" if env.all_positions_explored else f"{len(env.explored_positions)}/100"
                
                # Temperature status
                if env.body_temperature >= env.temp_max_critical:
                    temp_status = "CRIT_HOT"
                elif env.body_temperature <= env.temp_min_critical:
                    temp_status = "CRIT_COLD"
                elif env.body_temperature >= env.temp_comfortable_max:
                    temp_status = "HOT"
                elif env.body_temperature <= env.temp_comfortable_min:
                    temp_status = "COLD"
                else:
                    temp_status = "GOOD"
                
                # Active fires status
                fire_count = len(env.active_fires)
                fire_info = f"Fires:{fire_count}"
                if fire_count > 0 and tuple(env.agent_pos) in env.active_fires:
                    fire_time_left = env.active_fires[tuple(env.agent_pos)]
                    fire_info += f"(at_fire:{fire_time_left}s)"
                
                print(f"[D{current_day}M{current_minute}:{step:4d}] Pos:{tuple(env.agent_pos)} | "
                      f"Act:{env.actions[action]:11s} | "
                      f"H:{env.health:5.1f} | "
                      f"Hun:{env.hunger:4.1f} | "
                      f"Thi:{env.thirst:4.1f} | "
                      f"Temp:{env.body_temperature:5.1f}C {temp_status:8s} | "
                      f"{rest_status:8s} | "
                      f"Exp:{exploration_status:8s} | "
                      f"T:{trees_found} W:{waters_found} | "
                      f"{fire_info:12s} | "
                      f"R:{reward:6.2f}")
                
                # Show info for important events
                if ("FOUND" in info or "CRITICAL" in info or "EMERGENCY" in info or 
                    "FIRE" in info or "SWIM" in info or "SUCCESS" in info or "FAILED" in info):
                    print(f"         INFO: {info}")
            
            state = next_state
            step_count += 1
            
            if done:
                break
        
        # Final results
        final_success = "SUCCESS" if env.health >= 50 and env.health > 0 else "FAILED"
        
        print(f"\nTEST {test_num} RESULTS: {final_success}")
        print(f"Total Reward: {total_reward:.2f} | Steps Taken: {step_count}")
        print(f"Final Stats:")
        print(f"  Health: {env.health:.1f}/100")
        print(f"  Hunger: {env.hunger:.1f}/100") 
        print(f"  Thirst: {env.thirst:.1f}/100")
        print(f"  Body Temperature: {env.body_temperature:.1f}C")
        print(f"  Exploration: {'COMPLETE (100/100)' if env.all_positions_explored else f'INCOMPLETE ({len(env.explored_positions)}/100)'}")
        print(f"  Trees Found: {len(env.known_tree_locations)}/{len(env.TREES)} - {sorted(list(env.known_tree_locations))}")
        print(f"  Waters Found: {len(env.known_water_locations)}/{len(env.WATERS)} - {sorted(list(env.known_water_locations))}")
        print(f"  Active Fires: {len(env.active_fires)} at {sorted(list(env.active_fires.keys())) if env.active_fires else 'None'}")
        print(f"  Survival Days: {(step_count // env.seconds_per_day) + 1}/8")
        
        # Season summary
        final_season, _ = env.get_current_season()
        print(f"  Final Season: {final_season}")
        
        if final_success == "SUCCESS":
            print(f"Agent successfully survived 8 days through all seasons with multiple resources!")
        else:
            print(f"Agent failed to complete the 8-day survival challenge.")

# ====== Main Execution ======
if __name__ == "__main__":
    print("ENHANCED SURVIVAL SIMULATION - 10x10 Grid with Multiple Resources & Fire Duration")
    print("=" * 80)
    
    trained_agent = train_agent(episodes=6000)
    
    torch.save(trained_agent.model.state_dict(), "survival_agent_10x10_enhanced.pth")
    print(f"\nModel saved as 'survival_agent_10x10_enhanced.pth'")
    
    test_agent(trained_agent, num_tests=3)