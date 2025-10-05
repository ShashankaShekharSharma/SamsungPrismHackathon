# survival_10x10_seasons_temperature_enhanced.py
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
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

        # Multiple food sources (trees) - CONVERT TO TUPLES
        self.TREES = [(2, 8), (7, 1), (1, 3), (9, 6)]
        self.TREES = [tuple(pos) for pos in self.TREES]  # Ensure all are tuples

        # Multiple water sources - CONVERT TO TUPLES
        self.WATERS = [(8, 2), (0, 9), (6, 0), (3, 7)]
        self.WATERS = [tuple(pos) for pos in self.WATERS]  # Ensure all are tuples

        # Also ensure HOME is tuple
        self.HOME = tuple(self.HOME)

        # Add after self.fire_duration = 180
        self.exploration_phase_duration = 720  # 12 minutes for exploration
        self.exploration_need_multiplier = 0.2  # Much slower need growth during exploration

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

        # Need threshold constants for consistency
        self.hunger_critical_exploration = 90     # During exploration phase
        self.hunger_critical_normal = 80          # After exploration phase
        self.hunger_high_normal = 70              # High hunger threshold
        self.hunger_penalty_threshold = 40        # Penalty threshold

        self.thirst_critical_exploration = 90     # During exploration phase  
        self.thirst_critical_normal = 75          # After exploration phase
        self.thirst_high_normal = 65              # High thirst threshold
        self.thirst_penalty_threshold = 30        # Penalty threshold

        # Health penalty thresholds
        self.health_critical_hunger_threshold = 90
        self.health_high_hunger_threshold = 70
        self.health_critical_thirst_threshold = 85
        self.health_high_thirst_threshold = 65
        
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

        # Add validation for resource positions
        for tree in self.TREES:
            if not isinstance(tree, tuple):
                print(f"WARNING: Tree position {tree} is not a tuple")
        for water in self.WATERS:
            if not isinstance(water, tuple):
                print(f"WARNING: Water position {water} is not a tuple")
        
        return self.get_state()

    def get_state(self):
        current_pos = tuple(self.agent_pos)  # ADD THIS LINE
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
            nearest_tree_dist = min(distances) / max(1, (self.grid_size * 2))
            
        if self.known_water_locations:
            distances = [abs(self.agent_pos[0] - wx) + abs(self.agent_pos[1] - wy) 
                        for wx, wy in self.known_water_locations]
            nearest_water_dist = min(distances) / max(1, (self.grid_size * 2))

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
            len(self.known_tree_locations) / max(1, len(self.TREES)),  # 23: Trees discovered ratio
            len(self.known_water_locations) / max(1, len(self.WATERS)), # 24: Waters discovered ratio
            # New fire-related state information
            # New fire-related state information
            1.0 if (current_pos in self.active_fires) else 0.0,  # 25: Currently at fire
            max([self.active_fires.get(pos, 0) for pos in self.active_fires], default=0) / 180.0,  # 26: Max fire duration remaining (FIXED)
            min(len(self.active_fires) / 3.0, 1.0)  # 27: Number of active fires (normalized and capped)
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
            location_effect = 0.8   # Increased fire warming effect
        
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

        # Calculate exploration phase once to avoid duplicate calculations
        in_exploration_phase = self.time < self.exploration_phase_duration and not self.all_positions_explored

        # Time progression
        self.time += 1
        self.time_since_last_rest += 1

        # DEFINE ALL CRITICAL VARIABLES EARLY TO PREVENT REFERENCE ERRORS
        temp_critical_hot = self.body_temperature >= self.temp_max_critical
        temp_critical_cold = self.body_temperature <= self.temp_min_critical
        temp_uncomfortable_hot = self.body_temperature >= self.temp_comfortable_max
        temp_uncomfortable_cold = self.body_temperature <= self.temp_comfortable_min

        # DEFINE NEED CRITICALITY VARIABLES TOO
        if in_exploration_phase:
            critical_thirst = self.thirst >= self.thirst_critical_exploration
            critical_hunger = self.hunger >= self.hunger_critical_exploration
        else:
            critical_thirst = self.thirst >= self.thirst_critical_normal
            critical_hunger = self.hunger >= self.hunger_critical_normal

        # Track current position if no movement happened
        if not action.startswith("MOVE"):
            self.explored_positions.add(current_pos)
            self.grid_explored[current_pos[0], current_pos[1]] = True
        # AUTOMATIC RESOURCE DISCOVERY - Multiple sources (MOVED TO DEDICATED METHOD)
        # AUTOMATIC RESOURCE DISCOVERY - Multiple sources
        discovery_bonus = self.check_resource_discovery()
        if discovery_bonus > 0:
            final_pos = tuple(self.agent_pos)  # Use current position after any movement
            if discovery_bonus == 40:
                info += f" [TREE_FOUND at {final_pos}]"
            elif discovery_bonus == 45:
                info += f" [WATER_FOUND at {final_pos}]"
            elif discovery_bonus == 85:  # Both found (unlikely but possible)
                info += f" [TREE+WATER_FOUND at {final_pos}]"
        reward += discovery_bonus
        
        # Check if all positions are explored (this line should already exist)
        if len(self.explored_positions) >= len(self.all_positions):
            self.all_positions_explored = True
        
        # Check if mandatory rest is needed - RELAXED conditions
        if self.time_since_last_rest >= 240 or self.health < 20:  # 4 minutes = 240 seconds
            if not self.in_mandatory_rest:
                self.in_mandatory_rest = True
                self.can_rest = True
                self.consecutive_rest_count = 0
                info += " [MANDATORY_REST_TRIGGERED] "

        # Execute actions
        if action == "REST":
            if self.can_rest and self.in_mandatory_rest:
                self.consecutive_rest_count += 1
                
                # Much slower need progression during rest
                self.hunger = min(100, self.hunger + 0.03)  # Very slow during rest
                self.thirst = min(100, self.thirst + 0.04)  # Very slow during rest
                
                # Better health recovery during rest
                self.health = min(100, self.health + 6)  # Better recovery
                
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
            
            # Track whether eating/drinking was successful this step
            successful_eat = False
            successful_drink = False

            # Adaptive need progression based on exploration phase
            if in_exploration_phase:
                hunger_rate = 0.02 * self.exploration_need_multiplier  # Much slower
                thirst_rate = 0.025 * self.exploration_need_multiplier  # Much slower
            else:
                hunger_rate = 0.04  # Slightly reduced from original
                thirst_rate = 0.06  # Slightly reduced from original

            # Apply seasonal modifiers only after exploration phase
            if not in_exploration_phase:
                if season == "Summer":
                    thirst_rate *= 1.5
                elif season == "Winter":
                    hunger_rate *= 1.4

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
                new_pos = tuple(self.agent_pos)
                position_was_new_after_move = new_pos not in self.explored_positions
                # Track the new position immediately
                self.explored_positions.add(new_pos)
                self.grid_explored[new_pos[0], new_pos[1]] = True
                move_reward, move_info = self.evaluate_movement(position_was_new_after_move)
                reward += move_reward
                info += f" MOVE_UP to {new_pos}{move_info}"
            elif action == "MOVE_DOWN" and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1
                new_pos = tuple(self.agent_pos)
                position_was_new_after_move = new_pos not in self.explored_positions
                # Track the new position immediately
                self.explored_positions.add(new_pos)
                self.grid_explored[new_pos[0], new_pos[1]] = True
                move_reward, move_info = self.evaluate_movement(position_was_new_after_move)
                reward += move_reward
                info += f" MOVE_DOWN to {new_pos}{move_info}"
            elif action == "MOVE_LEFT" and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
                new_pos = tuple(self.agent_pos)
                position_was_new_after_move = new_pos not in self.explored_positions
                # Track the new position immediately
                self.explored_positions.add(new_pos)
                self.grid_explored[new_pos[0], new_pos[1]] = True
                move_reward, move_info = self.evaluate_movement(position_was_new_after_move)
                reward += move_reward
                info += f" MOVE_LEFT to {new_pos}{move_info}"
            elif action == "MOVE_RIGHT" and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
                new_pos = tuple(self.agent_pos)
                position_was_new_after_move = new_pos not in self.explored_positions
                # Track the new position immediately
                self.explored_positions.add(new_pos)
                self.grid_explored[new_pos[0], new_pos[1]] = True
                move_reward, move_info = self.evaluate_movement(position_was_new_after_move)
                reward += move_reward
                info += f" MOVE_RIGHT to {new_pos}{move_info}"

            elif action == "EAT":
                if current_pos in self.known_tree_locations:
                    if self.hunger > 15:  # Lower threshold
                        hunger_before = self.hunger
                        self.hunger = max(0, self.hunger - 65)  # Less food per eat
                        self.health = min(100, self.health + 6)
                        info += f" EAT_SUCCESS: Hunger {hunger_before:.1f} -> {self.hunger:.1f}"
                        reward += 45
                        successful_eat = True  # Mark that eating was successful
                    else:
                        info += " Not_hungry_enough"
                        reward -= 5
                elif len(self.known_tree_locations) == 0:
                    info += f" EAT_FAILED: No_trees_known"
                    reward -= 25
                else:
                    if self.known_tree_locations:  # ADD THIS CHECK
                        nearest_tree = min(self.known_tree_locations, 
                                        key=lambda t: abs(self.agent_pos[0] - t[0]) + abs(self.agent_pos[1] - t[1]))
                        info += f" EAT_FAILED: Not_at_tree (nearest: {nearest_tree})"
                    else:
                        info += " EAT_FAILED: No_trees_found"
                    reward -= 20

            elif action == "DRINK":
                if current_pos in self.known_water_locations:
                    if self.thirst > 15:  # Lower threshold
                        thirst_before = self.thirst
                        self.thirst = max(0, self.thirst - 65)  # Less water per drink
                        self.health = min(100, self.health + 5)
                        info += f" DRINK_SUCCESS: Thirst {thirst_before:.1f} -> {self.thirst:.1f}"
                        reward += 50
                        successful_drink = True  # This is the ONLY new line
                    else:
                        info += " Not_thirsty_enough"
                        reward -= 5
                elif len(self.known_water_locations) == 0:
                    info += f" DRINK_FAILED: No_water_known"
                    reward -= 25
                else:
                    if self.known_water_locations:  # ADD THIS CHECK
                        nearest_water = min(self.known_water_locations, 
                                        key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                        info += f" DRINK_FAILED: Not_at_water (nearest: {nearest_water})"
                    else:
                        info += " DRINK_FAILED: No_water_found"
                    reward -= 20

            # Apply hunger/thirst progression only if not successfully eating/drinking this step
            if not successful_eat:
                self.hunger = min(100, self.hunger + hunger_rate)
            if not successful_drink:
                self.thirst = min(100, self.thirst + thirst_rate)

        # Update body temperature AFTER all actions are processed
        # Update body temperature AFTER all actions are processed
        self.update_body_temperature()

        # UPDATE TEMPERATURE STATUS AFTER BODY TEMPERATURE CHANGES
        temp_critical_hot = self.body_temperature >= self.temp_max_critical
        temp_critical_cold = self.body_temperature <= self.temp_min_critical
        temp_uncomfortable_hot = self.body_temperature >= self.temp_comfortable_max
        temp_uncomfortable_cold = self.body_temperature <= self.temp_comfortable_min

        # TEMPERATURE HEALTH EFFECTS - REDUCED penalties
        temp_health_penalty = 0
        if temp_critical_hot:
            temp_health_penalty += 1.0  # Reduced
            info += " HYPERTHERMIA"
        elif temp_critical_cold:
            temp_health_penalty += 0.8  # Reduced
            info += " HYPOTHERMIA"
        elif temp_uncomfortable_hot:
            temp_health_penalty += 0.3  # Reduced
            info += " Too_hot"
        elif temp_uncomfortable_cold:
            temp_health_penalty += 0.3  # Reduced
            info += " Too_cold"

        # Health penalties from unmet needs - SIGNIFICANTLY REDUCED
        health_penalty = temp_health_penalty
        if self.hunger >= self.health_critical_hunger_threshold:
            health_penalty += 0.6  # Reduced
            info += " CRITICAL_HUNGER"
        elif self.hunger >= self.health_high_hunger_threshold:
            health_penalty += 0.2  # Reduced
            info += " High_hunger"
            
        if self.thirst >= self.health_critical_thirst_threshold:
            health_penalty += 0.8  # Reduced from 2.5
            info += " CRITICAL_THIRST"
        elif self.thirst >= self.health_high_thirst_threshold:
            health_penalty += 0.3  # Reduced
            info += " High_thirst"

        self.health = max(0, self.health - health_penalty)

        # ENHANCED EMERGENCY PRIORITIZATION SYSTEM with Fine-tuned Weights
        if not self.in_mandatory_rest:

            # Calculate emergency severity levels for dynamic scaling
            temp_critical_severity = 0
            if temp_critical_hot:
                temp_critical_severity = max(0, (self.body_temperature - self.temp_max_critical) * 15)
            elif temp_critical_cold:
                temp_critical_severity = max(0, (self.temp_min_critical - self.body_temperature) * 15)
            
            # Calculate need severity with weighted multipliers
            thirst_severity = max(0, (self.thirst - 70) * 2) if self.thirst > 70 else 0
            hunger_severity = max(0, (self.hunger - 65) * 1.5) if self.hunger > 65 else 0
            
            # TIER 1: LIFE-THREATENING TEMPERATURE (Highest Priority - 150-250 reward range)
            if temp_critical_hot and len(self.known_water_locations) > 0:
                if action == "SWIM" and current_pos in self.known_water_locations:
                    reward += 150 + temp_critical_severity
                    info += " [TIER1_EMERGENCY_COOLING]"
                elif action.startswith("MOVE") and self.known_water_locations:
                    nearest_water = min(self.known_water_locations, 
                                    key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                    water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                    old_dist = self.calculate_old_distance(action, nearest_water)
                    if water_dist < old_dist:
                        reward += 100 + temp_critical_severity
                        info += f" (TIER1_EMERGENCY->WATER)"
                    else:
                        reward -= 80 + temp_critical_severity
                        info += f" (TIER1_WRONG_DIRECTION)"
                else:
                    reward -= 100 + temp_critical_severity
                    info += f" (TIER1_MUST_COOL)"
                    
            elif temp_critical_cold:
                # Enhanced fire priority system
                if self.active_fires:
                    nearest_fire = min(self.active_fires.keys(), 
                                    key=lambda f: abs(self.agent_pos[0] - f[0]) + abs(self.agent_pos[1] - f[1]))
                    fire_dist = abs(self.agent_pos[0] - nearest_fire[0]) + abs(self.agent_pos[1] - nearest_fire[1])
                    
                    if fire_dist == 0:  # At fire location
                        fire_time_left = self.active_fires[current_pos]
                        if action == "IDLE" or action == "REST":  # Add REST here too
                            # Scale reward by remaining fire time and temperature severity
                            base_reward = 150 + temp_critical_severity
                            time_bonus = min(50, fire_time_left / 4)
                            reward += base_reward + time_bonus
                            info += f" [TIER1_CRITICAL_FIRE_STAYING_{fire_time_left}s]"
                        elif action.startswith("MOVE") or action == "EXPLORE":
                            if fire_time_left > 30:  # Don't leave good fires
                                penalty = 120 + temp_critical_severity + (fire_time_left / 3)
                                reward -= penalty
                                info += f" [TIER1_DONT_LEAVE_FIRE_{fire_time_left}s]"
                            else:
                                reward += 30  # OK to leave dying fire
                                info += " [TIER1_FIRE_DYING_OK]"
                        else:
                            reward -= 80 + temp_critical_severity
                            info += f" [TIER1_STAY_AT_FIRE]"
                    elif action.startswith("MOVE"):
                        old_dist = self.calculate_old_distance(action, nearest_fire)
                        if fire_dist < old_dist:
                            reward += 100 + temp_critical_severity
                            info += f" (TIER1_EMERGENCY->FIRE_DIST{fire_dist})"
                        else:
                            reward -= 80 + temp_critical_severity
                            info += f" (TIER1_WRONG_DIR_TO_FIRE)"
                    else:
                        reward -= 90 + temp_critical_severity
                        info += f" (TIER1_MUST_REACH_FIRE)"
                else:
                    # No fires exist - must start one immediately
                    if action == "START_FIRE":
                        # Extra reward for emergency fire starting
                        reward += 50 + temp_critical_severity
                        info += " [TIER1_EMERGENCY_FIRE_START]"
                    else:
                        reward -= 100 + temp_critical_severity
                        info += " [TIER1_MUST_START_FIRE]"

            # TIER 2: CRITICAL SURVIVAL NEEDS (100-150 reward range)
            elif critical_thirst and len(self.known_water_locations) > 0:
                if action == "DRINK" and current_pos in self.known_water_locations:
                    reward += 100 + thirst_severity
                    info += " [TIER2_CRITICAL_DRINK]"
                elif action.startswith("MOVE") and self.known_water_locations:
                    nearest_water = min(self.known_water_locations, 
                                    key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                    water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                    old_dist = self.calculate_old_distance(action, nearest_water)
                    if water_dist < old_dist:
                        reward += 60 + thirst_severity
                        info += f" (TIER2_EMERGENCY->WATER)"
                    else:
                        reward -= 40 + thirst_severity
                        info += f" (TIER2_WRONG_DIR_WATER)"
                else:
                    reward -= 60 + thirst_severity
                    info += " [TIER2_MUST_DRINK]"
                    
            elif critical_hunger and len(self.known_tree_locations) > 0:
                if action == "EAT" and current_pos in self.known_tree_locations:
                    reward += 90 + hunger_severity
                    info += " [TIER2_CRITICAL_EAT]"
                elif action.startswith("MOVE") and self.known_tree_locations:
                    nearest_tree = min(self.known_tree_locations, 
                                    key=lambda t: abs(self.agent_pos[0] - t[0]) + abs(self.agent_pos[1] - t[1]))
                    tree_dist = abs(self.agent_pos[0] - nearest_tree[0]) + abs(self.agent_pos[1] - nearest_tree[1])
                    old_dist = self.calculate_old_distance(action, nearest_tree)
                    if tree_dist < old_dist:
                        reward += 50 + hunger_severity
                        info += f" (TIER2_EMERGENCY->FOOD)"
                    else:
                        reward -= 35 + hunger_severity
                        info += f" (TIER2_WRONG_DIR_FOOD)"
                else:
                    reward -= 50 + hunger_severity
                    info += " [TIER2_MUST_EAT]"

            # TIER 3: TEMPERATURE COMFORT (50-80 reward range)
            elif temp_uncomfortable_hot and len(self.known_water_locations) > 0 and not (critical_hunger or critical_thirst):
                if action == "SWIM" and current_pos in self.known_water_locations:
                    reward += 60
                    info += " [TIER3_COMFORT_COOLING]"
                elif action.startswith("MOVE") and self.known_water_locations:
                    nearest_water = min(self.known_water_locations, 
                                    key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                    water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                    old_dist = self.calculate_old_distance(action, nearest_water)
                    if water_dist < old_dist:
                        reward += 30
                        info += f" (TIER3->COOL)"
                        
            elif temp_uncomfortable_cold and not (critical_hunger or critical_thirst):
                if self.active_fires and current_pos in self.active_fires:
                    fire_time_left = self.active_fires[current_pos]
                    base_reward = 70
                    time_bonus = min(30, fire_time_left / 6)  # Bonus for longer fires
                    reward += base_reward + time_bonus
                    info += f" [TIER3_WARMING_AT_FIRE_{fire_time_left}s]"
                    
                    # Enhanced warming effect for staying at fire
                    if self.body_temperature < self.temp_comfortable_min:
                        self.body_temperature = min(40.0, self.body_temperature + 0.4)
                elif not self.active_fires and action == "START_FIRE":
                    reward += 40
                    info += " [TIER3_COMFORT_FIRE]"
                elif self.active_fires and action.startswith("MOVE"):
                    nearest_fire = min(self.active_fires.keys(), 
                                    key=lambda f: abs(self.agent_pos[0] - f[0]) + abs(self.agent_pos[1] - f[1]))
                    fire_dist = abs(self.agent_pos[0] - nearest_fire[0]) + abs(self.agent_pos[1] - nearest_fire[1])
                    old_dist = self.calculate_old_distance(action, nearest_fire)
                    if fire_dist < old_dist:
                        reward += 25
                        info += f" (TIER3->WARM)"

            # TIER 4: EXPLORATION PRIORITY (20-40 reward range) 
            elif not self.all_positions_explored and not (critical_thirst or critical_hunger or temp_critical_hot or temp_critical_cold):
                if in_exploration_phase:
                    if action == "EXPLORE":
                        reward += 35
                        info += " [TIER4_EXPLORATION_PHASE]"
                    elif action.startswith("MOVE"):
                        new_pos = tuple(self.agent_pos)
                        if new_pos not in self.explored_positions:
                            reward += 25
                            info += " [TIER4_NEW_AREA]"
                        else:
                            reward += 8
                            info += " [TIER4_SEARCHING]"
                    else:
                        reward -= 15
                        info += " [TIER4_MUST_EXPLORE]"
                else:
                    # Post-exploration phase - lighter exploration rewards
                    if action == "EXPLORE":
                        reward += 15
                        info += " [TIER4_POST_EXPLORATION]"
                    elif action.startswith("MOVE"):
                        new_pos = tuple(self.agent_pos)
                        if new_pos not in self.explored_positions:
                            reward += 15
                            info += " [TIER4_LATE_DISCOVERY]"
                        else:
                            reward += 3
                            info += " [TIER4_MOVEMENT]"
                    else:
                        reward -= 8
                        info += " [TIER4_SHOULD_EXPLORE]"

            # TIER 5: RESOURCE MANAGEMENT (15-35 reward range)
            elif self.all_positions_explored:
                if self.thirst >= 45 and self.hunger < 60 and len(self.known_water_locations) > 0:
                    if action == "DRINK" and current_pos in self.known_water_locations:
                        reward += 35
                        info += " [TIER5_MAINTENANCE_DRINK]"
                    elif action.startswith("MOVE") and self.known_water_locations:
                        nearest_water = min(self.known_water_locations, 
                                        key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                        water_dist = abs(self.agent_pos[0] - nearest_water[0]) + abs(self.agent_pos[1] - nearest_water[1])
                        old_dist = self.calculate_old_distance(action, nearest_water)
                        if water_dist < old_dist:
                            reward += 20
                            info += f" (TIER5->WATER)"
                            
                elif self.hunger >= 35 and self.thirst < 45 and len(self.known_tree_locations) > 0:
                    if action == "EAT" and current_pos in self.known_tree_locations:
                        reward += 30
                        info += " [TIER5_MAINTENANCE_EAT]"
                    elif action.startswith("MOVE") and self.known_tree_locations:
                        nearest_tree = min(self.known_tree_locations, 
                                        key=lambda t: abs(self.agent_pos[0] - t[0]) + abs(self.agent_pos[1] - t[1]))
                        tree_dist = abs(self.agent_pos[0] - nearest_tree[0]) + abs(self.agent_pos[1] - nearest_tree[1])
                        old_dist = self.calculate_old_distance(action, nearest_tree)
                        if tree_dist < old_dist:
                            reward += 15
                            info += f" (TIER5->TREE)"

            # TIER 6: DEFAULT/IDLE MANAGEMENT (lowest priority)
            else:
                if action == "IDLE":
                    if (self.hunger < 20 and self.thirst < 20 and 
                        self.temp_comfortable_min <= self.body_temperature <= self.temp_comfortable_max):
                        reward += 15
                        info += " [TIER6_OPTIMAL_IDLE]"
                    else:
                        reward -= 10
                        info += " [TIER6_SUBOPTIMAL_IDLE]"

        # Base survival rewards
        reward += (self.health / 100.0) * 3
        
        # Temperature comfort bonus
        if self.temp_comfortable_min <= self.body_temperature <= self.temp_comfortable_max:
            reward += 2

        if in_exploration_phase:
            # Very light penalties during exploration
            if self.hunger > 80:
                reward -= (self.hunger / 100.0) * 0.5
            if self.thirst > 75:
                reward -= (self.thirst / 100.0) * 0.6
        else:
            # Normal penalties after exploration
            if self.hunger > self.hunger_penalty_threshold:
                reward -= (self.hunger / 100.0) * 1.0
            if self.thirst > self.thirst_penalty_threshold:
                reward -= (self.thirst / 100.0) * 1.2
            
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

        # Update fire durations at end of step
        self.update_fires()
        return self.get_state(), reward, done, info

    def handle_swim(self):
        """Handle swimming action for cooling down"""
        current_pos = tuple(self.agent_pos)
        
        if len(self.known_water_locations) == 0:
            return -25, " SWIM_FAILED: No_water_known"
            
        if current_pos not in self.known_water_locations:
            if self.known_water_locations:
                nearest_water = min(self.known_water_locations, 
                                key=lambda w: abs(self.agent_pos[0] - w[0]) + abs(self.agent_pos[1] - w[1]))
                return -20, f" SWIM_FAILED: Not_at_water (nearest: {nearest_water})"
            else:
                return -20, " SWIM_FAILED: Not_at_water"

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
            self.body_temperature = min(40.0, self.body_temperature + 4.0)  # Large immediate boost for critical
            reward = 100  # Higher reward for emergency fire
            info = f" EMERGENCY_FIRE Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C (3min_duration)"
        elif self.body_temperature <= self.temp_comfortable_min:
            # Regular warming
            temp_before = self.body_temperature
            self.body_temperature = min(39.0, self.body_temperature + 2.5)
            reward = 50
            info = f" WARMING_FIRE Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C (3min_duration)"
        else:
            # Not cold enough - might overheat
            temp_before = self.body_temperature
            self.body_temperature = min(45.0, self.body_temperature + 1.5)
            reward = -10
            info = f" UNNECESSARY_FIRE Temp {temp_before:.1f}C -> {self.body_temperature:.1f}C (getting_hot)"
            
        return reward, info

    def evaluate_movement(self, position_was_new=False):
        """Evaluate movement rewards based on current priorities"""
        current_pos = tuple(self.agent_pos)
        reward = 0
        info = ""
        
        # During exploration phase
        if not self.all_positions_explored:
            if position_was_new:
                reward += 15  # Bonus for reaching unexplored area
                info += " (new_area)"
            else:
                reward += 1   # Small bonus for movement during exploration
                
        # Temperature-based movement evaluation with fire proximity
        if current_pos in self.WATERS and self.body_temperature >= self.temp_comfortable_max:
            reward += 10
            info += " (at_cooling_water)"
        elif current_pos in self.active_fires:
            if self.body_temperature <= self.temp_comfortable_min:
                reward += 25  # Increased reward for being at fire when cold
                info += " (at_warming_fire)"
            fire_time_left = self.active_fires[current_pos]
            if fire_time_left > 60:
                reward += 15  # Extra bonus for being at fire with good duration
                info += f"_good_duration({fire_time_left}s)"
                
        return reward, info

    def handle_exploration(self):
        """Handle the EXPLORE action with systematic exploration"""
        current_pos = tuple(self.agent_pos)
        in_exploration_phase = self.time < self.exploration_phase_duration and not self.all_positions_explored
        reward = 15 if in_exploration_phase else 5  # Higher reward during exploration phase
        info = " EXPLORING"
        
        # If all positions are explored, discourage further exploration
        if self.all_positions_explored:
            reward = -20
            info += " - ALL_MAPPED Focus_on_survival"
            return reward, info
        
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
        
        # Undo the movement to get the previous position
        if action == "MOVE_UP":
            old_pos[1] += 1      # We moved up (y decreased), so add 1 to get old position
        elif action == "MOVE_DOWN":  
            old_pos[1] -= 1      # We moved down (y increased), so subtract 1
        elif action == "MOVE_LEFT":
            old_pos[0] += 1      # We moved left (x decreased), so add 1
        elif action == "MOVE_RIGHT":
            old_pos[0] -= 1      # We moved right (x increased), so subtract 1
        else:
            # For non-movement actions, return current distance
            return abs(self.agent_pos[0] - target[0]) + abs(self.agent_pos[1] - target[1])
        
        # Ensure old_pos stays within bounds
        old_pos[0] = max(0, min(self.grid_size - 1, old_pos[0]))
        old_pos[1] = max(0, min(self.grid_size - 1, old_pos[1]))
            
        return abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])
    
    def check_resource_discovery(self):
        """Centralized resource discovery with info updates - FIXED VERSION"""
        current_pos = tuple(self.agent_pos)
        discovery_bonus = 0
        
        # DEBUG: Print current position for troubleshooting
        # print(f"DEBUG: Checking discovery at position {current_pos}")
        
        # Check tree discovery - ENSURED TUPLE COMPARISON
        for tree_pos in self.TREES:
            # Ensure both are tuples for proper comparison
            if current_pos == tuple(tree_pos):
                if tree_pos not in self.known_tree_locations:
                    self.known_tree_locations.add(tree_pos)
                    discovery_bonus += 40
                    # print(f"DEBUG: Tree found at {tree_pos}")
                    break
                    
        # Check water discovery - ENSURED TUPLE COMPARISON  
        for water_pos in self.WATERS:
            # Ensure both are tuples for proper comparison
            if current_pos == tuple(water_pos):
                if water_pos not in self.known_water_locations:
                    self.known_water_locations.add(water_pos)
                    discovery_bonus += 45
                    # print(f"DEBUG: Water found at {water_pos}")
                    break
            
        return discovery_bonus

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
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995 # Slower decay for longer exploration
        self.lr = 0.0005

        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=50000)
        self.target_update_frequency = 50

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
        # deque automatically handles max length, no manual removal needed

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states_np = np.array([e[0] for e in batch])
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states_np = np.array([e[3] for e in batch])
        next_states = torch.FloatTensor(next_states_np).to(self.device)
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
        
        # Step the learning rate scheduler
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
            
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_checkpoint(self, checkpoint_path, episode, scores, losses, optimizer_state=True):
        """Save complete training checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if optimizer_state else None,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if hasattr(self, 'lr_scheduler') else None,
            'epsilon': self.epsilon,
            'scores': scores,
            'losses': losses,
            'memory_size': len(self.memory),
            'hyperparameters': {
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'lr': self.lr,
                'target_update_frequency': self.target_update_frequency
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at episode {episode}: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, load_optimizer=True, load_memory=False):
        """Load training checkpoint"""
        if not Path(checkpoint_path).exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load learning rate scheduler
        if hasattr(self, 'lr_scheduler') and checkpoint['lr_scheduler_state_dict'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Load training parameters
        self.epsilon = checkpoint['epsilon']
        
        print(f"Checkpoint loaded from episode {checkpoint['episode']}")
        print(f"Epsilon restored to: {self.epsilon:.4f}")
        print(f"Memory size was: {checkpoint['memory_size']}")
        
        return checkpoint

def create_checkpoint_dir(base_name="survival_checkpoints"):
    """Create checkpoint directory with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"{base_name}_{timestamp}")
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_episode_*.pth"))
    if not checkpoint_files:
        return None
        
    # Sort by episode number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoint_files[-1]

def save_training_config(checkpoint_dir, config):
    """Save training configuration"""
    config_path = checkpoint_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

# ====== Training Function with Performance Optimizations ======
def train_agent(episodes=3000, resume_from_checkpoint=None, save_every=500):
    """
    Enhanced training with checkpointing support
    
    Args:
        episodes: Total episodes to train
        resume_from_checkpoint: Path to checkpoint directory to resume from
        save_every: Save checkpoint every N episodes
    """
    
    # Setup checkpoint directory
    if resume_from_checkpoint:
        checkpoint_dir = Path(resume_from_checkpoint)
        print(f"Attempting to resume from: {checkpoint_dir}")
    else:
        checkpoint_dir = create_checkpoint_dir()
        print(f"Created new checkpoint directory: {checkpoint_dir}")
    
    env = SurvivalEnv()
    agent = RLAgent(state_size=28, action_size=env.action_space())
    
    # Training state variables
    start_episode = 0
    scores = []
    losses = []
    
    # Try to load checkpoint
    if resume_from_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint_data = agent.load_checkpoint(latest_checkpoint, load_optimizer=True)
            if checkpoint_data:
                start_episode = checkpoint_data['episode'] + 1
                scores = checkpoint_data['scores']
                losses = checkpoint_data['losses']
                print(f"Resuming from episode {start_episode}/{episodes}")
            else:
                print("Failed to load checkpoint, starting fresh")
        else:
            print("No checkpoints found in directory, starting fresh")
    
    # Save training configuration
    training_config = {
        'total_episodes': episodes,
        'start_episode': start_episode,
        'save_every': save_every,
        'environment': 'SurvivalEnv 10x10',
        'agent_config': {
            'state_size': 28,
            'action_size': env.action_space(),
            'epsilon_start': agent.epsilon,
            'epsilon_min': agent.epsilon_min,
            'epsilon_decay': agent.epsilon_decay,
            'learning_rate': agent.lr,
            'target_update_freq': agent.target_update_frequency
        }
    }
    save_training_config(checkpoint_dir, training_config)
    
    # Performance tracking
    start_time = time.time()
    
    print(f"Training from episode {start_episode} to {episodes}")
    print(f"Saving checkpoints every {save_every} episodes to: {checkpoint_dir}")
    print(f"Current epsilon: {agent.epsilon:.4f}")
    
    # Training loop
    replay_frequency = 4
    step_counter = 0
    
    try:
        for episode in range(start_episode, episodes):
            state = env.reset()
            total_reward = 0
            episode_loss = []
            
            for step in range(env.max_time):
                action = agent.act(state)
                try:
                    next_state, reward, done, info = env.step(action)
                except Exception as e:
                    print(f"ERROR at step {step}, position {env.agent_pos}: {e}")
                    # Reset environment on error
                    state = env.reset()
                    break
                agent.remember(state, action, reward, next_state, done)
                
                step_counter += 1
                if step_counter % replay_frequency == 0:
                    loss = agent.replay(batch_size=256)
                    if loss > 0:
                        episode_loss.append(loss)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            scores.append(total_reward)
            if episode_loss:
                losses.append(np.mean(episode_loss))
            
            # Update target model
            if episode % agent.target_update_frequency == 0:
                agent.update_target_model()
            
            # Save checkpoint
            if (episode + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode + 1}.pth"
                agent.save_checkpoint(checkpoint_path, episode, scores, losses)
                
                # Keep only last 3 checkpoints to save disk space
                checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_episode_*.pth"), 
                                        key=lambda x: int(x.stem.split('_')[-1]))
                if len(checkpoint_files) > 3:
                    for old_checkpoint in checkpoint_files[:-3]:
                        old_checkpoint.unlink()
                        print(f"Removed old checkpoint: {old_checkpoint.name}")
            
            # Progress reporting
            if (episode + 1) % 200 == 0:
                avg_score = np.mean(scores[-300:]) if len(scores) >= 300 else np.mean(scores)
                success_rate = len([s for s in scores[-300:] if s > 500]) / min(300, len(scores)) * 100
                elapsed_time = time.time() - start_time
                episodes_per_sec = (episode + 1 - start_episode) / elapsed_time
                eta_minutes = ((episodes - episode - 1) / episodes_per_sec) / 60 if episodes_per_sec > 0 else 0
                
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Avg Score: {avg_score:.2f} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Speed: {episodes_per_sec:.1f} ep/s | "
                      f"ETA: {eta_minutes:.1f}m")
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    memory_cached = torch.cuda.memory_reserved() / 1e9
                    print(f"         GPU Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
    
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at episode {episode + 1}")
        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode + 1}_interrupted.pth"
        agent.save_checkpoint(checkpoint_path, episode, scores, losses)
        print(f"Emergency checkpoint saved: {checkpoint_path}")
        return agent, checkpoint_dir
    
    # Final checkpoint
    final_checkpoint = checkpoint_dir / f"checkpoint_final_episode_{episodes}.pth"
    agent.save_checkpoint(final_checkpoint, episodes - 1, scores, losses)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average training speed: {(episodes - start_episode)/total_time:.1f} episodes/second")
    print(f"All checkpoints saved in: {checkpoint_dir}")
    
    return agent, checkpoint_dir

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
    print("ENHANCED SURVIVAL SIMULATION - 10x10 Grid with Checkpointing")
    print("=" * 80)
    
    # Configuration
    TOTAL_EPISODES = 3000
    RESUME_FROM = None  # Set to checkpoint directory path to resume
    # RESUME_FROM = "survival_checkpoints_20231201_143022"  # Example
    
    try:
        trained_agent, checkpoint_dir = train_agent(
            episodes=TOTAL_EPISODES, 
            resume_from_checkpoint=RESUME_FROM,
            save_every=500
        )
        
        # Save final model
        final_model_path = checkpoint_dir / "survival_agent_10x10_enhanced_final.pth"
        torch.save(trained_agent.model.state_dict(), final_model_path)
        print(f"Final model saved: {final_model_path}")
        
        test_agent(trained_agent, num_tests=3)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Check the latest checkpoint in the directory to resume training")