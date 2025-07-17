#include <emscripten.h>
#include <emscripten/bind.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <iostream>
#include <string>
#include <chrono>
#include <unordered_map>

using namespace emscripten;

// Constants
const int GRID_SIZE = 20;
const int CELL_SIZE = 20;
const int INITIAL_SPEED = 150;
const double INITIAL_EXPLORATION = 1.0;
const double MIN_EXPLORATION = 0.01;
const double EXPLORATION_DECAY = 0.995;
const double LEARNING_RATE = 0.1;
const double DISCOUNT_FACTOR = 0.9;

// Game state
struct GameState {
    int score = 0;
    int lifetime_score = 0;
    bool game_over = false;
    std::vector<std::pair<int, int>> snake;
    std::pair<int, int> food;
    std::pair<int, int> direction = {1, 0};
    int speed = INITIAL_SPEED;
};

// Q-learning agent
struct QLearningAgent {
    std::unordered_map<std::string, std::vector<double>> q_table;
    double exploration_rate = INITIAL_EXPLORATION;
    int episodes = 0;
    
    std::string get_state_key(const GameState& state) {
        std::string key;
        
        // Snake head position
        key += std::to_string(state.snake.front().first) + ",";
        key += std::to_string(state.snake.front().second) + ",";
        
        // Food position relative to head
        key += std::to_string(state.food.first - state.snake.front().first) + ",";
        key += std::to_string(state.food.second - state.snake.front().second) + ",";
        
        // Danger directions
        key += std::to_string(is_danger(state, {1, 0})) + ",";  // Right
        key += std::to_string(is_danger(state, {-1, 0})) + ","; // Left
        key += std::to_string(is_danger(state, {0, 1})) + ",";  // Down
        key += std::to_string(is_danger(state, {0, -1}));       // Up
        
        return key;
    }
    
    bool is_danger(const GameState& state, const std::pair<int, int>& dir) {
        auto new_head = state.snake.front();
        new_head.first += dir.first;
        new_head.second += dir.second;
        
        // Check wall collision
        if (new_head.first < 0 || new_head.first >= GRID_SIZE ||
            new_head.second < 0 || new_head.second >= GRID_SIZE) {
            return true;
        }
        
        // Check self collision
        for (size_t i = 0; i < state.snake.size(); i++) {
            if (new_head == state.snake[i]) {
                return true;
            }
        }
        
        return false;
    }
    
    std::pair<int, int> get_action(GameState& state) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // Exploration: random action
        if (dis(gen) < exploration_rate) {
            std::vector<std::pair<int, int>> possible_actions = {
                {1, 0},  // Right
                {-1, 0},  // Left
                {0, 1},   // Down
                {0, -1}   // Up
            };
            
            // Remove current opposite direction to prevent instant death
            auto opposite_dir = std::make_pair(-state.direction.first, -state.direction.second);
            possible_actions.erase(
                std::remove(possible_actions.begin(), possible_actions.end(), opposite_dir),
                possible_actions.end()
            );
            
            std::uniform_int_distribution<> action_dis(0, possible_actions.size() - 1);
            return possible_actions[action_dis(gen)];
        }
        
        // Exploitation: best action from Q-table
        std::string state_key = get_state_key(state);
        if (q_table.find(state_key) == q_table.end()) {
            q_table[state_key] = {0, 0, 0, 0};
        }
        
        auto& actions = q_table[state_key];
        int best_action = std::distance(actions.begin(), std::max_element(actions.begin(), actions.end()));
        
        std::vector<std::pair<int, int>> possible_actions = {
            {1, 0},  // Right
            {-1, 0}, // Left
            {0, 1},  // Down
            {0, -1}  // Up
        };
        
        return possible_actions[best_action];
    }
    
    void learn(const std::string& state_key, int action_idx, double reward, const std::string& new_state_key) {
        if (q_table.find(state_key) == q_table.end()) {
            q_table[state_key] = {0, 0, 0, 0};
        }
        
        if (q_table.find(new_state_key) == q_table.end()) {
            q_table[new_state_key] = {0, 0, 0, 0};
        }
        
        double max_future_q = *std::max_element(q_table[new_state_key].begin(), q_table[new_state_key].end());
        double current_q = q_table[state_key][action_idx];
        
        double new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - current_q);
        q_table[state_key][action_idx] = new_q;
    }
};

// Global variables
GameState game;
QLearningAgent q_learning;
double avg_q = 0.0;
std::chrono::time_point<std::chrono::system_clock> last_update;

// Helper functions
double getMinExploration() {
    return MIN_EXPLORATION;
}

void updateCharts(int episodes, int score, double avg_q, double exploration_rate, int lifetime_score) {
    EM_ASM({
        updateCharts($0, $1, $2, $3, $4);
    }, episodes, score, avg_q, exploration_rate, lifetime_score);
}

void place_food() {
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, GRID_SIZE - 1);
    
    while (true) {
        game.food = {dis(gen), dis(gen)};
        bool valid = true;
        
        for (const auto& segment : game.snake) {
            if (segment == game.food) {
                valid = false;
                break;
            }
        }
        
        if (valid) break;
    }
}

void reset_game() {
    game.snake.clear();
    game.snake.push_back({GRID_SIZE / 2, GRID_SIZE / 2});
    game.direction = {1, 0};
    game.score = 0;
    game.game_over = false;
    game.speed = INITIAL_SPEED;
    place_food();
}

void update() {
    if (game.game_over) return;
    
    // Move snake
    auto new_head = game.snake.front();
    new_head.first += game.direction.first;
    new_head.second += game.direction.second;
    
    // Check wall collision
    if (new_head.first < 0 || new_head.first >= GRID_SIZE ||
        new_head.second < 0 || new_head.second >= GRID_SIZE) {
        game.game_over = true;
        return;
    }
    
    // Check self collision
    for (const auto& segment : game.snake) {
        if (segment == new_head) {
            game.game_over = true;
            return;
        }
    }
    
    game.snake.insert(game.snake.begin(), new_head);
    
    // Check food collision
    if (new_head == game.food) {
        game.score++;
        game.lifetime_score++;
        place_food();
        
        // Increase speed slightly
        game.speed = std::max(50, game.speed - 5);
    } else {
        game.snake.pop_back();
    }
    
    // Update Q-learning
    std::string old_state_key = q_learning.get_state_key(game);
    int action_idx = 0;
    if (game.direction.first == 1) action_idx = 0;
    else if (game.direction.first == -1) action_idx = 1;
    else if (game.direction.second == 1) action_idx = 2;
    else action_idx = 3;
    
    double reward = 0.0;
    if (new_head == game.food) {
        reward = 10.0;
    } else if (game.game_over) {
        reward = -10.0;
    } else {
        reward = -0.1;  // Small negative reward for each step to encourage efficiency
    }
    
    std::string new_state_key = q_learning.get_state_key(game);
    q_learning.learn(old_state_key, action_idx, reward, new_state_key);
    
    // Calculate average Q-value for stats
    if (!q_learning.q_table.empty()) {
        double total_q = 0.0;
        int count = 0;
        for (const auto& entry : q_learning.q_table) {
            for (double q : entry.second) {
                total_q += q;
                count++;
            }
        }
        avg_q = total_q / count;
    }
}

void game_loop() {
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
    
    if (elapsed >= game.speed) {
        last_update = now;
        
        if (!game.game_over) {
            update();
        } else {
            q_learning.episodes++;
            q_learning.exploration_rate = std::max(getMinExploration(), 
                                                  q_learning.exploration_rate * EXPLORATION_DECAY);
            reset_game();
        }
        
        updateCharts(q_learning.episodes, game.score, avg_q, q_learning.exploration_rate, game.lifetime_score);
    }
    
    emscripten_set_main_loop(game_loop, 0, 1);
}

EMSCRIPTEN_BINDINGS(aisnake) {
    emscripten::function("resetGame", &reset_game);
    emscripten::function("getGameState", &get_game_state);
    emscripten::function("setDirection", &set_direction);
    
    emscripten::value_object<GameState>("GameState")
        .field("score", &GameState::score)
        .field("gameOver", &GameState::game_over)
        .field("snake", &GameState::snake)
        .field("food", &GameState::food);
    
    emscripten::register_vector<std::pair<int, int>>("vector<pair<int,int>>");
    emscripten::value_object<std::pair<int, int>>("pair<int,int>")
        .field("first", &std::pair<int, int>::first)
        .field("second", &std::pair<int, int>::second);
}

// JS-interop functions
GameState get_game_state() {
    return game;
}

void set_direction(int x, int y) {
    // Prevent reversing direction
    if (game.direction.first != -x && game.direction.second != -y) {
        game.direction = {x, y};
    }
}

int main() {
    reset_game();
    last_update = std::chrono::system_clock::now();
    emscripten_set_main_loop(game_loop, 0, 1);
    return 0;
}
