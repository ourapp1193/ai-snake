#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <queue>
#include <cmath>
#include <climits>  // Added for FLT_MAX
#include <SDL/SDL.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>

EM_JS(void, export_functions, (), {
    Module['getExplorationRate'] = Module.cwrap('getExplorationRate', 'number', []);
});
#endif

using namespace std;

// Game constants
const int WIDTH = 20;
const int HEIGHT = 20;
const int CELL_SIZE = 20;
const int AI_UPDATE_INTERVAL = 5;
const int LOG_INTERVAL = 100;
const int MAX_TRAINING_EPISODES = 5000000;

// Game state
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
};

// Q-learning parameters
struct QLearning {
    vector<vector<float>> table;
    float learning_rate = 0.1f;
    float discount_factor = 0.95f;
    float exploration_rate = 1.0f;
    int episodes = 0;
    const float exploration_decay = 0.9999f;
};

// Performance tracking
struct Performance {
    vector<int> scores;
    vector<float> avg_q_values;
    vector<int> lengths;
    vector<float> avg_rewards;
};

// SDL resources
struct SDLResources {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
};

// Global game objects
GameState game;
QLearning q_learning;
Performance performance;
SDLResources sdl;

// Forward declarations
float calculateReward(int prev_x, int prev_y, int x, int y, bool got_food, bool crashed);
vector<int> findSafeDirections(int x, int y, int current_dir);
bool isTrapped(int x, int y);

// Function to get dynamic minimum exploration
float getMinExploration() {
    return q_learning.episodes < 1000000 ? 0.001f : 0.0001f;
}

#ifdef __EMSCRIPTEN__
EM_JS(void, initChartJS, (), {
    function initializeCharts() {
        if (typeof Chart === 'undefined' || !Module.canvas) {
            setTimeout(initializeCharts, 100);
            return;
        }

        var container = document.getElementById('chart-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'chart-container';
            container.style.width = '800px';
            container.style.margin = '20px auto';
            container.style.padding = '20px';
            container.style.backgroundColor = '#333';
            document.body.insertBefore(container, Module.canvas.nextSibling);
        }

        ['scoreChart', 'qValueChart', 'lifetimeChart'].forEach(id => {
            if (!document.getElementById(id)) {
                var canvas = document.createElement('canvas');
                canvas.id = id;
                canvas.style.marginBottom = '20px';
                container.appendChild(canvas);
            }
        });

        window.scoreChart = new Chart(document.getElementById('scoreChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Episode Score',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                fill: false
            }]},
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });

        window.qValueChart = new Chart(document.getElementById('qValueChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Average Q Value',
                data: [],
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1,
                fill: false
            }]},
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });

        window.lifetimeChart = new Chart(document.getElementById('lifetimeChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Lifetime Score',
                data: [],
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
                fill: false
            }]},
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
    }

    if (document.readyState === 'complete') {
        initializeCharts();
    } else {
        document.addEventListener('DOMContentLoaded', initializeCharts);
    }
});

EM_JS(void, updateCharts, (int episode, int score, float avg_q, float exploration, int lifetime_score), {
    try {
        var statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.innerHTML = 
                `Episode: ${episode} | Score: ${score} | Lifetime: ${lifetime_score} | Avg Q: ${avg_q.toFixed(2)} | Exploration: ${exploration.toFixed(4)}`;
        }
        
        if (window.scoreChart && window.qValueChart && window.lifetimeChart) {
            function updateChart(chart, value) {
                chart.data.labels.push(episode);
                chart.data.datasets[0].data.push(value);
                if (chart.data.labels.length > 500) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                }
                chart.update();
            }
            
            updateChart(window.scoreChart, score);
            updateChart(window.qValueChart, avg_q);
            updateChart(window.lifetimeChart, lifetime_score);
        }
    } catch(e) {
        console.error('Chart update error:', e);
    }
});
#endif

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    float getExplorationRate() {
        return q_learning.exploration_rate;
    }
}

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

vector<vector<int>> getFreePositions(const vector<vector<int>>& occupied, const vector<vector<int>>& all) {
    vector<vector<int>> free = all;
    for (const auto& pos : occupied) {
        free.erase(remove(free.begin(), free.end(), pos), free.end());
    }
    return free;
}

bool isValidPosition(int x, int y) {
    return x >= 0 && x < HEIGHT && y >= 0 && y < WIDTH;
}

bool isBodyPosition(int x, int y, bool include_head = true) {
    for (size_t i = include_head ? 0 : 1; i < game.body.size(); ++i) {
        const auto& seg = game.body[i];
        if (seg.size() == 2 && seg[0] == x && seg[1] == y) {
            return true;
        }
    }
    return false;
}

void initQTable() {
    q_learning.table.resize(WIDTH * HEIGHT * 128);
    for (auto& row : q_learning.table) {
        row.assign(4, 0.0f);
    }
}

int getStateIndex(int x, int y, int dir) {
    if (!isValidPosition(x, y)) return 0;
    
    int food_dir = 0;
    if (game.food_x > x) food_dir = 1;
    else if (game.food_x < x) food_dir = 2;
    if (game.food_y > y) food_dir |= 4;
    else if (game.food_y < y) food_dir |= 8;
    
    int danger = 0;
    if (!isValidPosition(x-1, y) || isBodyPosition(x-1, y, false)) danger |= 1;
    if (!isValidPosition(x+1, y) || isBodyPosition(x+1, y, false)) danger |= 2;
    if (!isValidPosition(x, y-1) || isBodyPosition(x, y-1, false)) danger |= 4;
    if (!isValidPosition(x, y+1) || isBodyPosition(x, y+1, false)) danger |= 8;
    
    return (y * WIDTH + x) * 128 + dir * 32 + food_dir * 4 + danger;
}

bool isTrapped(int x, int y) {
    // Check if the snake is in a position with no escape
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
    
    // If reachable area is less than the snake length, it's trapped
    return reachable < game.length;
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
    
    // If no safe directions, try to find the least dangerous one
    if (safe_directions.empty()) {
        for (int i = 0; i < 4; i++) {
            int new_x = x + directions[i].first;
            int new_y = y + directions[i].second;
            
            if (isValidPosition(new_x, new_y)) {
                safe_directions.push_back(i);
            }
        }
    }
    
    // If still no directions, just return current direction
    if (safe_directions.empty()) {
        safe_directions.push_back(current_dir);
    }
    
    return safe_directions;
}

int chooseAction(int x, int y, int current_dir) {
    if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < q_learning.exploration_rate) {
        vector<int> safe_directions = findSafeDirections(x, y, current_dir);
        if (!safe_directions.empty()) {
            // Prefer directions that don't lead to traps
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
        // Only consider safe directions when exploiting
        vector<int> safe_directions = findSafeDirections(x, y, current_dir);
        if (safe_directions.empty()) return current_dir;
        
        // Filter out directions that lead to traps
        vector<int> non_trapping_directions;
        for (int dir : safe_directions) {
            int nx = x + (dir == 0 ? -1 : dir == 1 ? 1 : 0);
            int ny = y + (dir == 2 ? -1 : dir == 3 ? 1 : 0);
            if (!isTrapped(nx, ny)) {
                non_trapping_directions.push_back(dir);
            }
        }
        
        // If all directions lead to traps, use the original safe directions
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
    if (game.body.size() <= 1) return 1.0f; // No body to avoid
    
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
    if (crashed) return -100.0f;
    if (got_food) return 50.0f;
    
    float prev_dist = abs(prev_x - game.food_x) + abs(prev_y - game.food_y);
    float new_dist = abs(x - game.food_x) + abs(y - game.food_y);
    
    // Stronger penalty for being near body segments
    float body_penalty = 0.0f;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            if (isBodyPosition(x + dx, y + dy, false)) {
                body_penalty -= 20.0f; // Increased from 10 to 20
            }
        }
    }
    
    // Add reward for keeping distance from body
    float body_distance = calculateDistanceToBody(x, y);
    float distance_reward = body_distance * 1.0f; // Increased from 0.5f
    
    // Add penalty for moving in circles
    float circle_penalty = 0.0f;
    if (game.trail.size() > 10) {
        for (size_t i = 0; i < game.trail.size() - 1; i++) {
            if (game.trail[i][0] == x && game.trail[i][1] == y) {
                circle_penalty -= 10.0f * (game.trail.size() - i); // Increased from 5.0f
                break;
            }
        }
    }
    
    // Add reward for exploring new areas
    float exploration_reward = 0.0f;
    bool new_position = true;
    for (const auto& pos : game.trail) {
        if (pos[0] == x && pos[1] == y) {
            new_position = false;
            break;
        }
    }
    if (new_position) exploration_reward += 3.0f; // Increased from 2.0f
    
    // Add penalty for entering a trapped state
    float trap_penalty = isTrapped(x, y) ? -100.0f : 0.0f; // Increased from -50.0f
    
    return (prev_dist - new_dist) * 5.0f + body_penalty + circle_penalty + 
           exploration_reward + trap_penalty + distance_reward;
}

void resetGame() {
    game.head_x = HEIGHT / 2;
    game.head_y = WIDTH / 2;
    game.length = 2;
    game.score = 0;
    game.steps_since_last_food = 0;
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
    auto free_positions = getFreePositions(game.trail, all_positions);
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
    if (game.trail.size() > game.length + 2) {
        game.trail.resize(game.length + 2);
    }

    game.body.insert(game.body.begin(), {game.head_x, game.head_y});
    if (game.body.size() > game.length) {
        game.body.resize(game.length);
    }

    if (game.head_x == game.food_x && game.head_y == game.food_y) {
        game.score++;
        game.lifetime_score++;
        game.length++;
        game.steps_since_last_food = 0;
        spawnFood();
    }

    if (game.steps_since_last_food > 2000000 || isTrapped(game.head_x, game.head_y)) {
        return true;
    }

    return false;
}

void initSDL() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        exit(1);
    }

    sdl.window = SDL_CreateWindow("AI Snake",
                                SDL_WINDOWPOS_CENTERED,
                                SDL_WINDOWPOS_CENTERED,
                                WIDTH * CELL_SIZE,
                                HEIGHT * CELL_SIZE,
                                SDL_WINDOW_SHOWN);
    if (!sdl.window) {
        cerr << "SDL_CreateWindow Error: " << SDL_GetError() << endl;
        SDL_Quit();
        exit(1);
    }

    sdl.renderer = SDL_CreateRenderer(sdl.window, -1, 
                                    SDL_RENDERER_ACCELERATED | 
                                    SDL_RENDERER_PRESENTVSYNC);
    if (!sdl.renderer) {
        cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(sdl.window);
        SDL_Quit();
        exit(1);
    }

    if (SDL_SetRenderDrawBlendMode(sdl.renderer, SDL_BLENDMODE_BLEND) != 0) {
        cerr << "SDL_SetRenderDrawBlendMode Error: " << SDL_GetError() << endl;
    }
}

void drawGame() {
    SDL_SetRenderDrawColor(sdl.renderer, 0, 0, 0, 255);
    SDL_RenderClear(sdl.renderer);

    SDL_SetRenderDrawColor(sdl.renderer, 50, 50, 50, 255);
    SDL_Rect border = {0, 0, WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE};
    SDL_RenderDrawRect(sdl.renderer, &border);

    SDL_SetRenderDrawColor(sdl.renderer, 255, 0, 0, 255);
    SDL_Rect food = {game.food_y * CELL_SIZE, game.food_x * CELL_SIZE, CELL_SIZE, CELL_SIZE};
    SDL_RenderFillRect(sdl.renderer, &food);

    for (size_t i = 0; i < game.body.size(); i++) {
        const auto& seg = game.body[i];
        if (seg.size() == 2) {
            if (i == 0) {
                SDL_SetRenderDrawColor(sdl.renderer, 0, 255, 0, 255);
            } else {
                int intensity = 100 + (155 * i / game.body.size());
                SDL_SetRenderDrawColor(sdl.renderer, 0, intensity, 0, 255);
            }
            SDL_Rect body = {seg[1] * CELL_SIZE, seg[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE};
            SDL_RenderFillRect(sdl.renderer, &body);
        }
    }

    SDL_RenderPresent(sdl.renderer);
}

void logPerformance() {
    if (q_learning.episodes % LOG_INTERVAL == 0) {
        float total_q = 0;
        int count = 0;
        for (const auto& row : q_learning.table) {
            for (float val : row) {
                total_q += val;
                count++;
            }
        }
        float avg_q = count > 0 ? total_q / count : 0;

        performance.scores.push_back(game.score);
        performance.avg_q_values.push_back(avg_q);
        performance.lengths.push_back(game.length);

        #ifdef __EMSCRIPTEN__
        updateCharts(q_learning.episodes, game.score, avg_q, q_learning.exploration_rate, game.lifetime_score);
        #else
        cout << "Episode: " << q_learning.episodes 
             << " | Score: " << game.score 
             << " | Lifetime: " << game.lifetime_score
             << " | Avg Q: " << avg_q
             << " | Exploration: " << q_learning.exploration_rate << endl;
        #endif
    }
}

void mainLoop() {
    static int direction = rand() % 4;
    static int reset_timer = 0;

    if (reset_timer > 0) {
        reset_timer--;
        if (reset_timer == 0) {
            resetGame();
        }
        return;
    }

    bool crashed = moveSnake(direction);
    drawGame();

    if (q_learning.episodes < MAX_TRAINING_EPISODES) {
        q_learning.episodes++;
        q_learning.exploration_rate = max(getMinExploration(), 
                                         q_learning.exploration_rate * q_learning.exploration_decay);
        logPerformance();
    }

    if (crashed) {
        reset_timer = 5;
    }
}

void cleanup() {
    SDL_DestroyRenderer(sdl.renderer);
    SDL_DestroyWindow(sdl.window);
    SDL_Quit();
}

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
