#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <queue>
#include <cmath>
#include <SDL/SDL.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>

EM_JS(void, export_functions, (), {
    Module['getExplorationRate'] = Module.cwrap('getExplorationRate', 'number', []);
    Module['getAverageScore'] = Module.cwrap('getAverageScore', 'number', []);
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
const int SCORE_HISTORY_SIZE = 1000;

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
    vector<int> score_history;
    float average_score = 0.0f;
};

// Q-learning parameters
struct QLearning {
    vector<vector<float>> table;
    float learning_rate = 0.2f;  // Increased learning rate
    float discount_factor = 0.98f;  // Increased discount factor
    float exploration_rate = 1.0f;
    int episodes = 0;
    const float exploration_decay = 0.99995f;  // Slower decay
    const float min_exploration = 0.0001f;
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
void updateAverageScore();

// Function to get dynamic minimum exploration
float getMinExploration() {
    return q_learning.min_exploration;
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

        ['scoreChart', 'qValueChart', 'lifetimeChart', 'avgScoreChart'].forEach(id => {
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

        window.avgScoreChart = new Chart(document.getElementById('avgScoreChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Average Score (1000 eps)',
                data: [],
                borderColor: 'rgba(54, 162, 235, 1)',
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

EM_JS(void, updateCharts, (int episode, int score, float avg_q, float exploration, int lifetime_score, float avg_score), {
    try {
        var statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.innerHTML = 
                `Episode: ${episode} | Score: ${score} | Lifetime: ${lifetime_score} | Avg Q: ${avg_q.toFixed(2)} | Exploration: ${exploration.toFixed(4)} | Avg Score: ${avg_score.toFixed(2)}`;
        }
        
        if (window.scoreChart && window.qValueChart && window.lifetimeChart && window.avgScoreChart) {
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
            updateChart(window.avgScoreChart, avg_score);
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
    
    EMSCRIPTEN_KEEPALIVE
    float getAverageScore() {
        return game.average_score;
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

bool hasNoEscape(int x, int y) {
    int blocked_directions = 0;
    if (!isValidPosition(x-1, y) || isBodyPosition(x-1, y, false)) blocked_directions++;
    if (!isValidPosition(x+1, y) || isBodyPosition(x+1, y, false)) blocked_directions++;
    if (!isValidPosition(x, y-1) || isBodyPosition(x, y-1, false)) blocked_directions++;
    if (!isValidPosition(x, y+1) || isBodyPosition(x, y+1, false)) blocked_directions++;
    
    return blocked_directions >= 3;
}

bool isInCorner(int x, int y) {
    return (x == 0 || x == HEIGHT-1) && (y == 0 || y == WIDTH-1);
}

bool isNearWall(int x, int y) {
    return x == 0 || x == HEIGHT-1 || y == 0 || y == WIDTH-1;
}

void initQTable() {
    q_learning.table.resize(WIDTH * HEIGHT * 256);  // Larger state space
    for (auto& row : q_learning.table) {
        row.assign(4, 0.0f);
    }
}

int getStateIndex(int x, int y, int dir) {
    if (!isValidPosition(x, y)) return 0;
    
    // Food direction (4 bits)
    int food_dir = 0;
    if (game.food_x > x) food_dir = 1;
    else if (game.food_x < x) food_dir = 2;
    if (game.food_y > y) food_dir |= 4;
    else if (game.food_y < y) food_dir |= 8;
    
    // Immediate danger (4 bits)
    int danger = 0;
    if (!isValidPosition(x-1, y) || isBodyPosition(x-1, y, false)) danger |= 1;
    if (!isValidPosition(x+1, y) || isBodyPosition(x+1, y, false)) danger |= 2;
    if (!isValidPosition(x, y-1) || isBodyPosition(x, y-1, false)) danger |= 4;
    if (!isValidPosition(x, y+1) || isBodyPosition(x, y+1, false)) danger |= 8;
    
    // Additional state features (3 bits)
    int features = 0;
    if (isNearWall(x, y)) features |= 1;
    if (isInCorner(x, y)) features |= 2;
    if (game.length > 10) features |= 4;
    
    return (y * WIDTH + x) * 256 + dir * 64 + food_dir * 8 + danger * 2 + features;
}

int chooseAction(int x, int y, int current_dir) {
    if (static_cast<float>(rand()) / RAND_MAX < q_learning.exploration_rate) {
        return rand() % 4;
    }

    int state = getStateIndex(x, y, current_dir);
    if (state >= 0 && state < q_learning.table.size()) {
        // Softmax selection for better exploration
        vector<float> q_values = q_learning.table[state];
        vector<float> exp_values(q_values.size());
        float sum = 0.0f;
        
        for (size_t i = 0; i < q_values.size(); i++) {
            exp_values[i] = exp(q_values[i] / (q_learning.exploration_rate + 0.1f));
            sum += exp_values[i];
        }
        
        float r = static_cast<float>(rand()) / RAND_MAX;
        float cumulative = 0.0f;
        for (size_t i = 0; i < exp_values.size(); i++) {
            cumulative += exp_values[i] / sum;
            if (r <= cumulative) {
                return i;
            }
        }
    }
    return rand() % 4;
}

void updateQTable(int old_state, int action, int new_state, float reward) {
    if (old_state >= 0 && old_state < q_learning.table.size() && 
        new_state >= 0 && new_state < q_learning.table.size()) {
        float best_future = *max_element(q_learning.table[new_state].begin(), 
                                       q_learning.table[new_state].end());
        
        // Adaptive learning rate based on episode count
        float adaptive_learning_rate = q_learning.learning_rate * 
                                     (1.0f - min(1.0f, q_learning.episodes / 1000000.0f));
        
        q_learning.table[old_state][action] = 
            (1 - adaptive_learning_rate) * q_learning.table[old_state][action] +
            adaptive_learning_rate * (reward + q_learning.discount_factor * best_future);
    }
}

float calculateReward(int prev_x, int prev_y, int x, int y, bool got_food, bool crashed) {
    if (crashed) return -200.0f;  // Larger penalty for crashing
    
    if (got_food) {
        // Reward for getting food scales with snake length
        return 50.0f + game.length * 2.0f;
    }
    
    float prev_dist = abs(prev_x - game.food_x) + abs(prev_y - game.food_y);
    float new_dist = abs(x - game.food_x) + abs(y - game.food_y);
    
    // Distance-based reward with non-linear scaling
    float distance_reward = 0.0f;
    if (new_dist < prev_dist) {
        distance_reward = 5.0f * (1.0f / (new_dist + 1.0f));
    } else if (new_dist > prev_dist) {
        distance_reward = -3.0f * (new_dist - prev_dist);
    }
    
    // Body proximity penalty (more severe closer to head)
    float body_penalty = 0.0f;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            if (isBodyPosition(x + dx, y + dy, false)) {
                float dist = abs(dx) + abs(dy);
                body_penalty -= 15.0f / dist;
            }
        }
    }
    
    // Wall proximity penalty
    float wall_penalty = 0.0f;
    if (isNearWall(x, y)) {
        wall_penalty -= 5.0f;
        if (isInCorner(x, y)) {
            wall_penalty -= 10.0f;  // Extra penalty for corners
        }
    }
    
    // Movement pattern penalty
    float pattern_penalty = 0.0f;
    if (game.trail.size() > 10) {
        for (size_t i = 0; i < min(game.trail.size(), static_cast<size_t>(20)); i++) {
            if (game.trail[i][0] == x && game.trail[i][1] == y) {
                pattern_penalty -= 10.0f * (1.0f - i/20.0f);
                break;
            }
        }
    }
    
    // No escape penalty
    float escape_penalty = 0.0f;
    if (hasNoEscape(x, y)) {
        escape_penalty -= 50.0f;  // Very large penalty for no escape
    }
    
    // Time penalty (encourage efficiency)
    float time_penalty = -0.1f;
    
    return distance_reward + body_penalty + wall_penalty + 
           pattern_penalty + escape_penalty + time_penalty;
}

void updateAverageScore() {
    game.score_history.push_back(game.score);
    if (game.score_history.size() > SCORE_HISTORY_SIZE) {
        game.score_history.erase(game.score_history.begin());
    }
    
    float sum = 0.0f;
    for (int score : game.score_history) {
        sum += score;
    }
    game.average_score = sum / game.score_history.size();
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
        // Prefer food placement away from walls
        vector<vector<int>> center_positions;
        for (const auto& pos : free_positions) {
            if (pos[0] > 2 && pos[0] < HEIGHT-3 && pos[1] > 2 && pos[1] < WIDTH-3) {
                center_positions.push_back(pos);
            }
        }
        
        if (!center_positions.empty()) {
            int k = rand() % center_positions.size();
            game.food_x = center_positions[k][0];
            game.food_y = center_positions[k][1];
        } else {
            int k = rand() % free_positions.size();
            game.food_x = free_positions[k][0];
            game.food_y = free_positions[k][1];
        }
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
            vector<int> safe_actions;
            for (int i = 0; i < 4; i++) {
                int test_x = game.head_x, test_y = game.head_y;
                switch (i) {
                    case 0: test_x--; break;
                    case 1: test_x++; break;
                    case 2: test_y--; break;
                    case 3: test_y++; break;
                }
                if (isValidPosition(test_x, test_y) && !isBodyPosition(test_x, test_y, false)) {
                    safe_actions.push_back(i);
                }
            }
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

    if (game.steps_since_last_food > 200 + game.length * 5) {  // Longer timeout for longer snakes
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

        updateAverageScore();
        performance.scores.push_back(game.score);
        performance.avg_q_values.push_back(avg_q);
        performance.lengths.push_back(game.length);

        #ifdef __EMSCRIPTEN__
        updateCharts(q_learning.episodes, game.score, avg_q, q_learning.exploration_rate, 
                    game.lifetime_score, game.average_score);
        #else
        cout << "Episode: " << q_learning.episodes 
             << " | Score: " << game.score 
             << " | Lifetime: " << game.lifetime_score
             << " | Avg Q: " << avg_q
             << " | Exploration: " << q_learning.exploration_rate
             << " | Avg Score: " << game.average_score << endl;
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
