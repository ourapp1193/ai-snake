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
});
#endif

using namespace std;

const int WIDTH = 20;
const int HEIGHT = 20;
const int CELL_SIZE = 20;
const int AI_UPDATE_INTERVAL = 5;
const int LOG_INTERVAL = 100;
const int MAX_TRAINING_EPISODES = 5000000;
const float MIN_EXPLORATION = 0.0001f;

struct GameState {
    int head_x = HEIGHT / 2;
    int head_y = WIDTH / 2;
    int score = 0;
    int length = 2;
    int food_x = -1, food_y = -1;
    bool crashed = false;
    int speed = 200;
    vector<vector<int>> body;
    vector<vector<int>> trail;
    int lifetime_score = 0;
    int steps_since_last_food = 0;
    bool has_food = false;
};

struct QLearning {
    vector<vector<float>> table;
    float learning_rate = 0.1f;
    float discount_factor = 0.95f;
    float exploration_rate = 1.0f;
    int episodes = 0;
    const float exploration_decay = 0.9995f;
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
        row.assign(4, 0.1f);
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
    
    int proximity = 0;
    const int directions[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    for (int i = 0; i < 4; i++) {
        for (int dist = 1; dist <= 3; dist++) {
            int nx = x + directions[i][0] * dist;
            int ny = y + directions[i][1] * dist;
            if (isValidPosition(nx, ny) && isBodyPosition(nx, ny, false)) {
                proximity |= (1 << (i + 4 * (dist - 1)));
                break;
            }
        }
    }
    
    return (y * WIDTH + x) * 128 + dir * 32 + food_dir * 4 + danger + proximity;
}

int chooseAction(int x, int y, int current_dir) {
    if (static_cast<float>(rand()) / RAND_MAX < q_learning.exploration_rate) {
        return rand() % 4;
    }

    int state = getStateIndex(x, y, current_dir);
    if (state >= 0 && state < q_learning.table.size()) {
        return distance(q_learning.table[state].begin(),
                      max_element(q_learning.table[state].begin(), q_learning.table[state].end()));
    }
    return rand() % 4;
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

float calculateReward(int prev_x, int prev_y, int x, int y, bool got_food, bool crashed) {
    if (crashed) return -100.0f;
    
    if (got_food) return 100.0f;
    
    float prev_dist = abs(prev_x-game.food_x) + abs(prev_y-game.food_y);
    float new_dist = abs(x-game.food_x) + abs(y-game.food_y);
    float dist_reward = (prev_dist - new_dist) * 1.0f;
    
    float body_penalty = 0.0f;
    const int directions[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    for (int i = 0; i < 4; i++) {
        for (int dist = 1; dist <= 3; dist++) {
            int nx = x + directions[i][0] * dist;
            int ny = y + directions[i][1] * dist;
            if (isValidPosition(nx, ny)) {
                if (isBodyPosition(nx, ny, false)) {
                    body_penalty -= 3.0f / dist;
                    break;
                }
            }
        }
    }
    
    float time_reward = 0.05f;
    
    return dist_reward + body_penalty + time_reward;
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
    game.has_food = false;
}

void spawnFood() {
    if (game.has_food) return; // Don't spawn if food already exists

    auto all_positions = generateAllPositions();
    auto free_positions = getFreePositions(game.trail, all_positions);
    if (!free_positions.empty() && !game.crashed) {
        int k = rand() % free_positions.size();
        game.food_x = free_positions[k][0];
        game.food_y = free_positions[k][1];
        game.has_food = true;
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
        bool got_food = game.has_food && (new_x == game.food_x && new_y == game.food_y);
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
                game.crashed = true;
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
        game.crashed = true;
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

    if (game.has_food && game.head_x == game.food_x && game.head_y == game.food_y) {
        game.score++;
        game.lifetime_score++;
        game.length++;
        game.steps_since_last_food = 0;
        game.has_food = false; // Mark food as eaten
        spawnFood(); // Spawn new food only when eaten
    }

    if (game.steps_since_last_food > 200) {
        game.crashed = true;
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

    if (game.has_food) {
        SDL_SetRenderDrawColor(sdl.renderer, 255, 0, 0, 255);
        SDL_Rect food = {game.food_y * CELL_SIZE, game.food_x * CELL_SIZE, CELL_SIZE, CELL_SIZE};
        SDL_RenderFillRect(sdl.renderer, &food);
    }

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
            spawnFood(); // Spawn initial food on reset
        }
        return;
    }

    bool crashed = moveSnake(direction);
    drawGame();

    if (q_learning.episodes < MAX_TRAINING_EPISODES) {
        q_learning.episodes++;
        q_learning.exploration_rate = max(MIN_EXPLORATION, 
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

    initQTable();
    initSDL();
    resetGame();
    spawnFood(); // Initial food spawn

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
                spawnFood();
            }
            SDL_Delay(game.speed);
            continue;
        }

        bool crashed = moveSnake(direction);
        drawGame();
        SDL_Delay(game.speed);

        if (q_learning.episodes < MAX_TRAINING_EPISODES) {
            q_learning.episodes++;
            q_learning.exploration_rate = max(MIN_EXPLORATION, 
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
