#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <SDL/SDL.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>

EM_JS(void, initChartJS, (), {
    if (typeof Chart === 'undefined') {
        console.log('Chart.js not loaded yet');
        setTimeout(() => initChartJS(), 100);
        return;
    }
    
    const container = document.createElement('div');
    container.id = 'chart-container';
    container.style.cssText = `
        position: absolute;
        top: 20px;
        left: 20px;
        background: rgba(0,0,0,0.7);
        padding: 10px;
        color: white;
        font-family: Arial;
    `;
    document.body.appendChild(container);
    
    const canvas1 = document.createElement('canvas');
    canvas1.id = 'scoreChart';
    canvas1.width = 400;
    canvas1.height = 200;
    container.appendChild(canvas1);
    
    const canvas2 = document.createElement('canvas');
    canvas2.id = 'distanceChart';
    canvas2.width = 400;
    canvas2.height = 200;
    container.appendChild(canvas2);
    
    window.scoreChart = new Chart(canvas1, {
        type: 'line',
        data: { labels: [], datasets: [{
            label: 'Score',
            data: [],
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false
        }]},
        options: { responsive: false }
    });
    
    window.distanceChart = new Chart(canvas2, {
        type: 'line',
        data: { labels: [], datasets: [
            {
                label: 'Avg Distance',
                data: [],
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
                fill: false
            },
            {
                label: 'Min Distance',
                data: [],
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                fill: false
            }
        ]},
        options: { responsive: false }
    });
});

EM_JS(void, updateCharts, (int episode, int score, float avg_dist, float min_dist), {
    if (!window.scoreChart || !window.distanceChart) return;
    
    const labels = window.scoreChart.data.labels;
    labels.push(episode);
    if (labels.length > 100) labels.shift();
    
    window.scoreChart.data.datasets[0].data.push(score);
    if (window.scoreChart.data.datasets[0].data.length > 100) {
        window.scoreChart.data.datasets[0].data.shift();
    }
    
    window.distanceChart.data.labels = labels;
    window.distanceChart.data.datasets[0].data.push(avg_dist);
    window.distanceChart.data.datasets[1].data.push(min_dist);
    
    if (window.distanceChart.data.datasets[0].data.length > 100) {
        window.distanceChart.data.datasets[0].data.shift();
        window.distanceChart.data.datasets[1].data.shift();
    }
    
    window.scoreChart.update();
    window.distanceChart.update();
    
    const status = document.getElementById('status') || document.createElement('div');
    status.id = 'status';
    status.textContent = `Episode: ${episode} | Score: ${score} | Avg Dist: ${avg_dist.toFixed(2)} | Min Dist: ${min_dist.toFixed(2)}`;
    if (!document.getElementById('status')) {
        document.body.appendChild(status);
    }
});
#endif

// Game constants
const int WIDTH = 20;
const int HEIGHT = 20;
const int CELL_SIZE = 20;
const int AI_UPDATE_INTERVAL = 5;
const int MAX_TRAINING_EPISODES = 5000000;
const float OPTIMAL_MIN_DISTANCE = 3.0f;
const float OPTIMAL_MAX_DISTANCE = 5.0f;

// Game state
struct GameState {
    int head_x = HEIGHT / 2;
    int head_y = WIDTH / 2;
    int score = 0;
    int length = 3;
    int food_x = 0, food_y = 0;
    bool crashed = false;
    int speed = 10;
    vector<vector<int>> body;
    int lifetime_score = 0;
    int steps_since_last_food = 0;
    float avg_head_body_distance = 0;
    float min_head_body_distance = 999;
    float max_head_body_distance = 0;
};

// Q-learning parameters
struct QLearning {
    vector<vector<float>> table;
    float learning_rate = 0.1f;
    float discount_factor = 0.95f;
    float exploration_rate = 1.0f;
    int episodes = 0;
    const float exploration_decay = 0.9999f;
    const float min_exploration = 0.01f;
};

// SDL resources
struct SDLResources {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Surface* screen = nullptr;
};

// Global game objects
GameState game;
QLearning q_learning;
SDLResources sdl;

// Helper functions
vector<vector<int>> generateAllPositions() {
    vector<vector<int>> positions;
    positions.reserve(WIDTH * HEIGHT);
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            positions.push_back({i, j});
        }
    }
    return positions;
}

vector<vector<int>> getFreePositions(const vector<vector<int>>& occupied) {
    auto all = generateAllPositions();
    vector<vector<int>> free;
    free.reserve(all.size());
    
    for (const auto& pos : all) {
        bool occupied_pos = false;
        for (const auto& occ : occupied) {
            if (pos[0] == occ[0] && pos[1] == occ[1]) {
                occupied_pos = true;
                break;
            }
        }
        if (!occupied_pos) {
            free.push_back(pos);
        }
    }
    return free;
}

bool isValidPosition(int x, int y) {
    return x >= 0 && x < HEIGHT && y >= 0 && y < WIDTH;
}

bool isBodyPosition(int x, int y, bool include_head = true) {
    for (size_t i = include_head ? 0 : 1; i < game.body.size(); ++i) {
        const auto& seg = game.body[i];
        if (seg[0] == x && seg[1] == y) {
            return true;
        }
    }
    return false;
}

void calculateDistanceMetrics() {
    if (game.body.size() <= 1) {
        game.avg_head_body_distance = 0;
        game.min_head_body_distance = 999;
        game.max_head_body_distance = 0;
        return;
    }
    
    float total_distance = 0;
    float current_min = 999;
    float current_max = 0;
    int count = 0;
    
    for (size_t i = 1; i < game.body.size(); ++i) {
        const auto& seg = game.body[i];
        float dx = game.head_x - seg[0];
        float dy = game.head_y - seg[1];
        float dist = sqrt(dx*dx + dy*dy);
        
        total_distance += dist;
        current_min = min(current_min, dist);
        current_max = max(current_max, dist);
        count++;
    }
    
    game.avg_head_body_distance = count > 0 ? total_distance / count : 0;
    game.min_head_body_distance = current_min;
    game.max_head_body_distance = current_max;
}

float calculateDistanceReward() {
    if (game.min_head_body_distance < 1.0f) {
        return -50.0f;
    }
    
    if (game.min_head_body_distance < OPTIMAL_MIN_DISTANCE) {
        return -10.0f * (OPTIMAL_MIN_DISTANCE - game.min_head_body_distance);
    }
    
    if (game.min_head_body_distance >= OPTIMAL_MIN_DISTANCE && 
        game.min_head_body_distance <= OPTIMAL_MAX_DISTANCE) {
        return 5.0f;
    }
    
    if (game.min_head_body_distance > OPTIMAL_MAX_DISTANCE) {
        return 2.0f;
    }
    
    return 0.0f;
}

void initQTable() {
    int state_space_size = WIDTH * HEIGHT * 4 * 8 * 8;
    q_learning.table.resize(state_space_size);
    
    for (auto& row : q_learning.table) {
        row.assign(4, 0.0f);
    }
}

int getStateIndex(int x, int y, int dir) {
    if (!isValidPosition(x, y)) return 0;
    
    int food_dir = 0;
    if (game.food_x > x) food_dir |= 1;
    else if (game.food_x < x) food_dir |= 2;
    if (game.food_y > y) food_dir |= 4;
    
    int danger = 0;
    if (!isValidPosition(x-1, y) || isBodyPosition(x-1, y, false)) danger |= 1;
    if (!isValidPosition(x+1, y) || isBodyPosition(x+1, y, false)) danger |= 2;
    if (!isValidPosition(x, y-1) || isBodyPosition(x, y-1, false)) danger |= 4;
    
    return ((y * WIDTH + x) * 4 + dir) * 8 + food_dir;
}

int chooseAction(int x, int y, int current_dir) {
    if (static_cast<float>(rand()) / RAND_MAX < q_learning.exploration_rate) {
        return rand() % 4;
    }

    int state = getStateIndex(x, y, current_dir);
    if (state >= 0 && state < static_cast<int>(q_learning.table.size())) {
        return distance(q_learning.table[state].begin(),
                       max_element(q_learning.table[state].begin(), 
                                   q_learning.table[state].end()));
    }
    
    return rand() % 4;
}

void updateQTable(int old_state, int action, int new_state, float reward) {
    if (old_state >= 0 && old_state < static_cast<int>(q_learning.table.size()) && 
        new_state >= 0 && new_state < static_cast<int>(q_learning.table.size())) {
        float best_future = *max_element(q_learning.table[new_state].begin(), 
                                        q_learning.table[new_state].end());
        q_learning.table[old_state][action] = 
            (1 - q_learning.learning_rate) * q_learning.table[old_state][action] +
            q_learning.learning_rate * (reward + q_learning.discount_factor * best_future);
    }
}

float calculateReward(int prev_x, int prev_y, int x, int y, bool got_food, bool crashed) {
    if (crashed) return -100.0f;
    if (got_food) return 50.0f;
    
    float prev_food_dist = abs(prev_x - game.food_x) + abs(prev_y - game.food_y);
    float new_food_dist = abs(x - game.food_x) + abs(y - game.food_y);
    float food_reward = (prev_food_dist - new_food_dist) * 2.0f;
    
    float distance_reward = calculateDistanceReward();
    float movement_penalty = -0.1f;
    
    return food_reward + distance_reward + movement_penalty;
}

void resetGame() {
    game.head_x = HEIGHT / 2;
    game.head_y = WIDTH / 2;
    game.length = 3;
    game.score = 0;
    game.steps_since_last_food = 0;
    game.avg_head_body_distance = 0;
    game.min_head_body_distance = 999;
    game.max_head_body_distance = 0;
    game.body = {{game.head_x, game.head_y}};
    game.crashed = false;
    
    auto free_positions = getFreePositions(game.body);
    if (!free_positions.empty()) {
        int k = rand() % free_positions.size();
        game.food_x = free_positions[k][0];
        game.food_y = free_positions[k][1];
    }
}

void spawnFood() {
    auto free_positions = getFreePositions(game.body);
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

    game.body.insert(game.body.begin(), {game.head_x, game.head_y});
    if (game.body.size() > game.length) {
        game.body.pop_back();
    }

    if (game.head_x == game.food_x && game.head_y == game.food_y) {
        game.score++;
        game.lifetime_score++;
        game.length++;
        game.steps_since_last_food = 0;
        spawnFood();
    }

    calculateDistanceMetrics();

    if (game.steps_since_last_food > 200) {
        return true;
    }

    return false;
}

void initSDL() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        exit(1);
    }

    sdl.window = SDL_CreateWindow("AI Snake with Distance Rewards",
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

    sdl.screen = SDL_GetWindowSurface(sdl.window);
    if (!sdl.screen) {
        cerr << "SDL_GetWindowSurface Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(sdl.window);
        SDL_Quit();
        exit(1);
    }
}

void drawGame() {
    SDL_FillRect(sdl.screen, NULL, SDL_MapRGB(sdl.screen->format, 0, 0, 0));

    SDL_Rect grid_rect;
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            grid_rect = {j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE};
            SDL_FillRect(sdl.screen, &grid_rect, 
                        SDL_MapRGB(sdl.screen->format, 30, 30, 30));
        }
    }

    SDL_Rect food_rect = {game.food_y * CELL_SIZE, game.food_x * CELL_SIZE, 
                         CELL_SIZE, CELL_SIZE};
    SDL_FillRect(sdl.screen, &food_rect, 
                SDL_MapRGB(sdl.screen->format, 255, 0, 0));

    for (size_t i = 0; i < game.body.size(); ++i) {
        const auto& seg = game.body[i];
        SDL_Rect body_rect = {seg[1] * CELL_SIZE, seg[0] * CELL_SIZE, 
                             CELL_SIZE, CELL_SIZE};
        
        if (i == 0) {
            SDL_FillRect(sdl.screen, &body_rect, 
                        SDL_MapRGB(sdl.screen->format, 0, 255, 0));
        } else {
            float dx = seg[0] - game.head_x;
            float dy = seg[1] - game.head_y;
            float dist = sqrt(dx*dx + dy*dy);
            
            int r = min(255, static_cast<int>(255 * (1 - dist/10.0f)));
            int g = 255;
            int b = 0;
            
            SDL_FillRect(sdl.screen, &body_rect, 
                        SDL_MapRGB(sdl.screen->format, r, g, b));
        }
    }

    SDL_UpdateWindowSurface(sdl.window);
}

void logPerformance() {
    if (q_learning.episodes % 100 == 0) {
        cout << "Episode: " << q_learning.episodes 
             << " | Score: " << game.score 
             << " | Length: " << game.length
             << " | Dist: " << game.min_head_body_distance << "-" 
             << game.max_head_body_distance << " (avg " 
             << game.avg_head_body_distance << ")"
             << " | Explore: " << q_learning.exploration_rate << endl;
             
        #ifdef __EMSCRIPTEN__
        updateCharts(q_learning.episodes, game.score, 
                    game.avg_head_body_distance, game.min_head_body_distance);
        #endif
    }
}

#ifdef __EMSCRIPTEN__
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    float getExplorationRate() {
        return q_learning.exploration_rate;
    }
    
    EMSCRIPTEN_KEEPALIVE
    float getAverageDistance() {
        return game.avg_head_body_distance;
    }
    
    EMSCRIPTEN_KEEPALIVE
    int getScore() {
        return game.score;
    }
}
#endif

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
        q_learning.exploration_rate = max(q_learning.min_exploration, 
                                        q_learning.exploration_rate * q_learning.exploration_decay);
        logPerformance();
    }

    if (crashed) {
        reset_timer = 10;
    }
}

void cleanup() {
    SDL_DestroyWindow(sdl.window);
    SDL_Quit();
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    #ifdef __EMSCRIPTEN__
    initChartJS();
    #endif

    game.body = {{HEIGHT/2, WIDTH/2}};
    auto free_positions = getFreePositions(game.body);
    if (!free_positions.empty()) {
        game.food_x = free_positions[0][0];
        game.food_y = free_positions[0][1];
    }

    initQTable();
    initSDL();

    #ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(mainLoop, 0, 1);
    #else
    bool running = true;
    while (running) {
        mainLoop();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }
        
        SDL_Delay(game.speed);
    }
    #endif

    cleanup();
    return 0;
}
