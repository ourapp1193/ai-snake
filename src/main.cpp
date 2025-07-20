#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <queue>
#include <cmath>
#include <climits>
#include <SDL/SDL.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>

EM_JS(void, export_functions, (), {
    Module['getExplorationRate'] = Module.cwrap('getExplorationRate', 'number', []);
});
#endif

using namespace std;

// Enhanced game constants for longer play
const int WIDTH = 30;  // Increased from 20
const int HEIGHT = 30; // Increased from 20
const int CELL_SIZE = 20;
const int AI_UPDATE_INTERVAL = 5;
const int LOG_INTERVAL = 100;
const int MAX_TRAINING_EPISODES = 5000000;
const int STEP_LIMIT = 5000;  // Increased from 200
const int MAX_SNAKE_LENGTH = 10000; // Increased capacity

// Game state with enhanced parameters
struct GameState {
    int head_x = HEIGHT / 2;
    int head_y = WIDTH / 2;
    int score = 0;
    int length = 2;
    int food_x = 0, food_y = 0;
    bool crashed = false;
    int speed = 10;
    vector<vector<int>> body;
    vector<vector<int>> trail;
    int lifetime_score = 0;
    int steps_since_last_food = 0;
    int foods_eaten_this_life = 0;
};

// Q-learning with adjusted parameters
struct QLearning {
    vector<vector<float>> table;
    float learning_rate = 0.1f;
    float discount_factor = 0.98f;  // Increased from 0.95
    float exploration_rate = 1.0f;
    int episodes = 0;
    const float exploration_decay = 0.99999f;  // Slower decay
};

struct Performance {
    vector<int> scores;
    vector<float> avg_q_values;
    vector<int> lengths;
    vector<float> avg_rewards;
};

struct SDLResources {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
};

GameState game;
QLearning q_learning;
Performance performance;
SDLResources sdl;

// [Previous function declarations remain the same...]

float getMinExploration() {
    return q_learning.episodes < 1000000 ? 0.0001f : 0.00001f;  // Lower minimum exploration
}

#ifdef __EMSCRIPTEN__
// [Keep existing EM_JS functions...]
#endif

vector<vector<int>> generateAllPositions() {
    vector<vector<int>> positions(WIDTH * HEIGHT, vector<int>(2));
    int idx = 0;
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j, ++idx) {
            positions[idx] = {i, j};
        }
    }
    return positions;
}

// [Keep existing getFreePositions, isValidPosition, isBodyPosition functions...]

void initQTable() {
    q_learning.table.resize(WIDTH * HEIGHT * 128);
    for (auto& row : q_learning.table) {
        row.assign(4, 0.0f);
    }
}

// [Keep existing getStateIndex function...]

bool isTrapped(int x, int y) {
    // More permissive trapping detection
    if (game.length < WIDTH/2) return false; // Don't check for traps when snake is small
    
    vector<vector<bool>> visited(HEIGHT, vector<bool>(WIDTH, false));
    queue<pair<int, int>> q;
    q.push({x, y});
    visited[x][y] = true;
    
    int reachable = 0;
    vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    while (!q.empty()) {
        auto current = q.front();
        q.pop();
        reachable++;
        
        for (const auto& dir : directions) {
            int nx = current.first + dir.first;
            int ny = current.second + dir.second;
            
            if (isValidPosition(nx, ny)) {
                if (!isBodyPosition(nx, ny, false) && !visited[nx][ny]) {
                    visited[nx][ny] = true;
                    q.push({nx, ny});
                }
            }
        }
    }
    
    // Only consider trapped if reachable area is very small
    return reachable < game.length/4;  // Reduced threshold
}

vector<int> findSafeDirections(int x, int y, int current_dir) {
    vector<int> safe_directions;
    vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    for (int i = 0; i < 4; i++) {
        int new_x = x + directions[i].first;
        int new_y = y + directions[i].second;
        
        if (isValidPosition(new_x, new_y) && !isBodyPosition(new_x, new_y, false)) {
            safe_directions.push_back(i);
        }
    }
    
    if (safe_directions.empty()) {
        for (int i = 0; i < 4; i++) {
            int new_x = x + directions[i].first;
            int new_y = y + directions[i].second;
            
            if (isValidPosition(new_x, new_y)) {
                safe_directions.push_back(i);
            }
        }
    }
    
    if (safe_directions.empty()) {
        safe_directions.push_back(current_dir);
    }
    
    return safe_directions;
}

int chooseAction(int x, int y, int current_dir) {
    if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < q_learning.exploration_rate) {
        vector<int> safe_directions = findSafeDirections(x, y, current_dir);
        if (!safe_directions.empty()) {
            vector<int> non_trapping_directions;
            for (int dir : safe_directions) {
                int nx = x + (dir == 0 ? -1 : dir == 1 ? 1 : 0);
                int ny = y + (dir == 2 ? -1 : dir == 3 ? 1 : 0);
                if (!isTrapped(nx, ny)) {
                    non_trapping_directions.push_back(dir);
                }
            }
            
            if (!non_trapping_directions.empty()) {
                return non_trapping_directions[rand() % non_trapping_directions.size()];
            }
            return safe_directions[rand() % safe_directions.size()];
        }
        return rand() % 4;
    }

    int state = getStateIndex(x, y, current_dir);
    if (state >= 0 && state < q_learning.table.size()) {
        vector<int> safe_directions = findSafeDirections(x, y, current_dir);
        if (safe_directions.empty()) return current_dir;
        
        vector<int> non_trapping_directions;
        for (int dir : safe_directions) {
            int nx = x + (dir == 0 ? -1 : dir == 1 ? 1 : 0);
            int ny = y + (dir == 2 ? -1 : dir == 3 ? 1 : 0);
            if (!isTrapped(nx, ny)) {
                non_trapping_directions.push_back(dir);
            }
        }
        
        if (non_trapping_directions.empty()) {
            non_trapping_directions = safe_directions;
        }
        
        int best_action = non_trapping_directions[0];
        float best_value = q_learning.table[state][best_action];
        
        for (size_t i = 1; i < non_trapping_directions.size(); i++) {
            int action = non_trapping_directions[i];
            if (q_learning.table[state][action] > best_value) {
                best_value = q_learning.table[state][action];
                best_action = action;
            }
        }
        
        return best_action;
    }
    
    vector<int> safe_directions = findSafeDirections(x, y, current_dir);
    if (!safe_directions.empty()) {
        return safe_directions[rand() % safe_directions.size()];
    }
    return current_dir;
}

void updateQTable(int old_state, int action, int new_state, float reward) {
    if (old_state >= 0 && old_state < q_learning.table.size() && 
        new_state >= 0 && new_state < q_learning.table.size()) {
        float best_future = *max_element(q_learning.table[new_state].begin(), 
                                       q_learning.table[new_state].end());
        q_learning.table[old_state][action] = 
            (1 - q_learning.learning_rate) * q_learning.table[old_state][action] +
            q_learning.learning_rate * (reward + q_learning.discount_factor * best_future);
    }
}

float calculateDistanceToBody(int x, int y) {
    if (game.body.size() <= 1) return 1.0f;
    
    float min_distance = numeric_limits<float>::max();
    for (size_t i = 1; i < game.body.size(); i++) {
        const auto& seg = game.body[i];
        if (seg.size() == 2) {
            float dist = sqrt(pow(x - seg[0], 2) + pow(y - seg[1], 2));
            if (dist < min_distance) {
                min_distance = dist;
            }
        }
    }
    return min_distance;
}

float calculateReward(int prev_x, int prev_y, int x, int y, bool got_food, bool crashed) {
    if (crashed) return -50.0f;  // Reduced penalty
    if (got_food) return 100.0f;  // Increased reward
    
    float prev_dist = abs(prev_x - game.food_x) + abs(prev_y - game.food_y);
    float new_dist = abs(x - game.food_x) + abs(y - game.food_y);
    
    // More lenient body penalty
    float body_penalty = 0.0f;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            if (isBodyPosition(x + dx, y + dy, false)) {
                body_penalty -= 5.0f;  // Reduced from 10.0f
            }
        }
    }
    
    float body_distance = calculateDistanceToBody(x, y);
    float distance_reward = body_distance * 0.8f;  // Increased from 0.5f
    
    // Reduced circle penalty
    float circle_penalty = 0.0f;
    if (game.trail.size() > 20) {  // Increased from 10
        for (size_t i = 0; i < game.trail.size() - 1; i++) {
            if (game.trail[i][0] == x && game.trail[i][1] == y) {
                circle_penalty -= 2.0f * (game.trail.size() - i);  // Reduced from 5.0f
                break;
            }
        }
    }
    
    float exploration_reward = 0.0f;
    bool new_position = true;
    for (const auto& pos : game.trail) {
        if (pos[0] == x && pos[1] == y) {
            new_position = false;
            break;
        }
    }
    if (new_position) exploration_reward += 3.0f;  // Increased from 2.0f
    
    float trap_penalty = isTrapped(x, y) ? -20.0f : 0.0f;  // Reduced from -50.0f
    
    return (prev_dist - new_dist) * 8.0f + body_penalty + circle_penalty + 
           exploration_reward + trap_penalty + distance_reward;
}

void resetGame() {
    game.head_x = HEIGHT / 2;
    game.head_y = WIDTH / 2;
    game.length = 2;
    game.score = 0;
    game.steps_since_last_food = 0;
    game.foods_eaten_this_life = 0;
    game.body = {{game.head_x, game.head_y}};
    game.trail = {{game.head_x, game.head_y}};
    game.crashed = false;
    
    auto all_positions = generateAllPositions();
    auto free_positions = getFreePositions(game.trail, all_positions);
    if (!free_positions.empty()) {
        int k = rand() % free_positions.size();
        game.food_x = free_positions[k][0];
        game.food_y = free_positions[k][1];
    }
}

void spawnFood() {
    auto all_positions = generateAllPositions();
    auto free_positions = getFreePositions(game.body, all_positions);
    if (!free_positions.empty()) {
        int k = rand() % free_positions.size();
        game.food_x = free_positions[k][0];
        game.food_y = free_positions[k][1];
    }
}

bool moveSnake(int& direction) {
    static int frame = 0;
    frame++;

    int prev_x = game.head_x;
    int prev_y = game.head_y;
    int prev_dir = direction;

    if (frame % AI_UPDATE_INTERVAL == 0 || q_learning.episodes < MAX_TRAINING_EPISODES) {
        int action = chooseAction(game.head_x, game.head_y, direction);
        
        int new_x = game.head_x, new_y = game.head_y;
        switch (action) {
            case 0: new_x--; break;
            case 1: new_x++; break;
            case 2: new_y--; break;
            case 3: new_y++; break;
        }

        bool valid = isValidPosition(new_x, new_y) && !isBodyPosition(new_x, new_y, false);
        bool got_food = (new_x == game.food_x && new_y == game.food_y);
        bool crashed = !valid;

        float reward = calculateReward(prev_x, prev_y, new_x, new_y, got_food, crashed);
        int old_state = getStateIndex(prev_x, prev_y, prev_dir);
        int new_state = getStateIndex(new_x, new_y, action);
        updateQTable(old_state, action, new_state, reward);

        if (valid) {
            direction = action;
        } else {
            vector<int> safe_actions = findSafeDirections(prev_x, prev_y, prev_dir);
            if (!safe_actions.empty()) {
                direction = safe_actions[rand() % safe_actions.size()];
            } else {
                return true;
            }
        }
    }

    switch (direction) {
        case 0: game.head_x--; break;
        case 1: game.head_x++; break;
        case 2: game.head_y--; break;
        case 3: game.head_y++; break;
    }

    game.steps_since_last_food++;

    if (!isValidPosition(game.head_x, game.head_y) || isBodyPosition(game.head_x, game.head_y, false)) {
        return true;
    }

    game.trail.insert(game.trail.begin(), {game.head_x, game.head_y});
    if (game.trail.size() > MAX_SNAKE_LENGTH) {
        game.trail.resize(MAX_SNAKE_LENGTH);
    }

    game.body.insert(game.body.begin(), {game.head_x, game.head_y});
    if (game.body.size() > game.length) {
        game.body.resize(game.length);
    }

    if (game.head_x == game.food_x && game.head_y == game.food_y) {
        game.score++;
        game.lifetime_score++;
        game.foods_eaten_this_life++;
        game.length++;
        game.steps_since_last_food = 0;
        spawnFood();
    }

    if (game.steps_since_last_food > STEP_LIMIT || isTrapped(game.head_x, game.head_y)) {
        return true;
    }

    return false;
}

// [Keep existing initSDL, drawGame, logPerformance, mainLoop, cleanup functions...]

int main() {
    srand((unsigned)time(nullptr));

    #ifdef __EMSCRIPTEN__
    export_functions();
    initChartJS();
    #endif

    auto all_positions = generateAllPositions();
    game.body = {{HEIGHT/2, WIDTH/2}};
    game.trail = {{HEIGHT/2, WIDTH/2}};
    auto free_positions = getFreePositions(game.trail, all_positions);
    if (!free_positions.empty()) {
        game.food_x = free_positions[0][0];
        game.food_y = free_positions[0][1];
    }

    initQTable();
    initSDL();

    #ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(mainLoop, 0, 1);
    #else
    int direction = rand() % 4;
    int reset_timer = 0;
    
    bool running = true;
    while (running) {
        if (reset_timer > 0) {
            reset_timer--;
            if (reset_timer == 0) {
                resetGame();
            }
            SDL_Delay(game.speed);
            continue;
        }

        bool crashed = moveSnake(direction);
        drawGame();
        SDL_Delay(game.speed);

        if (q_learning.episodes < MAX_TRAINING_EPISODES) {
            q_learning.episodes++;
            q_learning.exploration_rate = max(getMinExploration(), 
                                            q_learning.exploration_rate * q_learning.exploration_decay);
            logPerformance();
        }

        if (crashed) {
            cout << "Foods eaten this life: " << game.foods_eaten_this_life << endl;
            reset_timer = 5;
        }

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }
    }
    #endif

    cleanup();
    return 0;
}
