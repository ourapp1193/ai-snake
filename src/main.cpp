float calculateReward(int prev_x, int prev_y, int x, int y, bool got_food, bool crashed) {
    if (crashed) return -50.0f;
    if (got_food) return 100.0f;
    
    float prev_dist = abs(prev_x-game.food_x) + abs(prev_y-game.food_y);
    float new_dist = abs(x-game.food_x) + abs(y-game.food_y);
    
    // Add penalty for being near body segments
    float body_penalty = 0.0f;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            if (isBodyPosition(x + dx, y + dy, false)) {
                body_penalty -= 5.0f; // Penalty for each adjacent body segment
            }
        }
    }
    
    return (prev_dist - new_dist) * 2.0f + 0.1f + body_penalty;
}
