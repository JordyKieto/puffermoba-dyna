// Render real arena games with the actual PufferLib MOBA client, frame by frame, for video clips.
//
// moba_arena.c is included unchanged (same game, same shipped net, same controllers as the
// real-game evaluation), so a game driven from Python through ma_* can also be drawn. mr_frame()
// is upstream c_render() (third_party/pufferlib/ocean/moba/moba.h) with three changes:
//   - the camera is given (a map position in cells + cell size) instead of following the human
//     player, so a clip can frame a fight, a tower or an ancient;
//   - it draws into an offscreen render texture and copies the RGB pixels out (no window
//     input, no frame-rate throttle), and the shader clock is the clip's own frame time;
//   - an optional caption bar on top; the HUD (bars, Q/W/E timers) is upstream's, for hud_pid.
// Everything else (map shader, map texture, sprites, health/mana bars, attack lines, AoE
// squares, hit bloom, stun/slow/speed icons) is the client's own drawing code.
//
//   mr_init(w, h, resources_dir)    hidden window + render texture; resources_dir holds resources/moba/
//   mr_entities(out)                NUM_ENTITIES x mr_fields() (16) floats: alive, type, team, x, y, health,
//                                   max_health, target_pid, tier, level, mana, stun_timer, move_modifier,
//                                   q_timer, w_timer, e_timer
//   mr_frame(cx, cy, ts, frac, t, hud_pid, caption, caption_right, out)   one frame -> out (h x w x 3 RGB);
//                                   ts = pixels per cell (float: the camera zooms smoothly; the client uses 32)
//   mr_set_bloom(code)              upstream's bloom shader, keywords translated for core-profile GL (macOS)

#include <unistd.h>
#include "moba_arena.c"

static GameRenderer* mr = NULL;
static RenderTexture2D mr_rt;
static int mr_w, mr_h;
static int mr_ids[128 * 128];

MA_API int mr_init(int w, int h, const char* resources_dir) {
    if (chdir(resources_dir) != 0) return -1;            // the client loads resources/moba/... relative
    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    mr = init_game_renderer(32, w / 32 + 1, h / 32 + 1);
    SetTargetFPS(0);
    mr_w = w; mr_h = h;
    mr_rt = LoadRenderTexture(w, h);
    return IsRenderTextureValid(mr_rt) ? 0 : -2;
}

// Upstream's bloom_shader_330.fs is GLSL 1.x (varying/texture2D/gl_FragColor), which a macOS core-profile context
// rejects (raylib then draws hit entities without bloom). The caller passes the same shader with the keywords
// translated; the maths is unchanged.
MA_API int mr_set_bloom(const char* fs_code) {
    Shader s = LoadShaderFromMemory(0, fs_code);
    if (!IsShaderValid(s)) return -1;
    mr->bloom_shader = s;
    return 0;
}

#define MR_FIELDS 16
MA_API int mr_num_entities(void) { return NUM_ENTITIES; }
MA_API int mr_fields(void) { return MR_FIELDS; }

MA_API void mr_entities(float* out) {
    MOBA* env = &ma_game.env;
    for (int i = 0; i < NUM_ENTITIES; i++) {
        Entity* e = &env->entities[i];
        float* o = out + MR_FIELDS * i;
        o[0] = e->pid != -1; o[1] = e->entity_type; o[2] = e->team; o[3] = e->x; o[4] = e->y;
        o[5] = e->health; o[6] = e->max_health; o[7] = e->target_pid; o[8] = e->tier; o[9] = e->level;
        o[10] = e->mana; o[11] = e->stun_timer; o[12] = e->move_modifier; o[13] = e->q_timer; o[14] = e->w_timer; o[15] = e->e_timer;
    }
}

MA_API void mr_frame(float cx, float cy, float ts, float tick_frac, float t_sec, int hud_pid, const char* caption,
                     const char* caption_right, uint8_t* out) {
    MOBA* env = &ma_game.env;
    GameRenderer* renderer = mr;
    Map* map = env->map;
    float W = (float)mr_w / ts, H = (float)mr_h / ts;           // view size in cells
    renderer->camera.target.x = (cx - W / 2) * ts;
    renderer->camera.target.y = (cy - H / 2) * ts;
    renderer->camera.zoom = 1.0f;
    int r_min = (int)(cy - H / 2) - 1, r_max = (int)(cy + H / 2) + 1;
    int c_min = (int)(cx - W / 2) - 1, c_max = (int)(cx + W / 2) + 1;

    BeginTextureMode(mr_rt);
    ClearBackground(COLORS[0]);

    // Main environment shader (upstream)
    BeginShaderMode(renderer->shader);
    renderer->shader_y = (cy - H / 2) / 128;
    renderer->shader_x = (cx - W / 2) / 128;
    renderer->shader_seconds = t_sec;
    renderer->shader_resolution[0] = W;
    renderer->shader_resolution[1] = H;
    SetShaderValue(renderer->shader, renderer->shader_camera_x, &renderer->shader_x, SHADER_UNIFORM_FLOAT);
    SetShaderValue(renderer->shader, renderer->shader_camera_y, &renderer->shader_y, SHADER_UNIFORM_FLOAT);
    SetShaderValue(renderer->shader, renderer->shader_time, &renderer->shader_seconds, SHADER_UNIFORM_FLOAT);
    SetShaderValue(renderer->shader, renderer->shader_resolution_loc, renderer->shader_resolution, SHADER_UNIFORM_VEC3);
    SetShaderValueTexture(renderer->shader, renderer->shader_texture1, renderer->game_map);
    DrawTexture(renderer->shader_canvas, 0, 0, WHITE);
    EndShaderMode();

    BeginMode2D(renderer->camera);
    int n = 0;
    for (int y = r_min; y < r_max + 1; y++) {
        for (int x = c_min; x < c_max + 1; x++) {
            if (y < 0 || y >= 128 || x < 0 || x >= 128) continue;
            int adr = map_offset(map, y, x);
            unsigned char tile = map->grid[adr];
            if (tile == EMPTY || tile == WALL) continue;
            int pid = map->pids[adr];
            if (pid == -1) DrawRectangle(x * ts, y * ts, ceilf(ts), ceilf(ts), RED);
            mr_ids[n++] = pid;
        }
    }
    // Targeting overlays (upstream)
    for (int i = 0; i < n; i++) {
        int pid = mr_ids[i];
        if (pid == -1) continue;
        Entity* entity = &env->entities[pid];
        int target_pid = entity->target_pid;
        if (target_pid == -1) continue;
        Entity* target = &env->entities[target_pid];
        float entity_x = entity->last_x + tick_frac * (entity->x - entity->last_x);
        float entity_y = entity->last_y + tick_frac * (entity->y - entity->last_y);
        float target_x = target->last_x + tick_frac * (target->x - target->last_x);
        float target_y = target->last_y + tick_frac * (target->y - target->last_y);
        Color base, accent;
        if (entity->team == 0) { base = (Color){0, 128, 128, 255}; accent = (Color){0, 255, 255, 255}; }
        else if (entity->team == 1) { base = (Color){128, 0, 0, 255}; accent = (Color){255, 0, 0, 255}; }
        else { base = (Color){128, 128, 128, 255}; accent = (Color){255, 255, 255, 255}; }
        int target_px = target_x * ts + ts / 2, target_py = target_y * ts + ts / 2;
        int entity_px = entity_x * ts + ts / 2, entity_py = entity_y * ts + ts / 2;
        if (entity->attack_aoe == 0) {
            DrawLineEx((Vector2){entity_px, entity_py}, (Vector2){target_px, target_py}, ts / 16, accent);
        } else {
            int radius = entity->attack_aoe * ts;
            DrawRectangle(target_px - radius, target_py - radius, 2 * radius, 2 * radius, base);
            DrawRectangleLinesEx((Rectangle){target_px - radius, target_py - radius, 2 * radius, 2 * radius}, ts / 8, accent);
        }
    }
    // Entity renders (upstream)
    for (int i = 0; i < n; i++) {
        Color tint = (Color){255, 255, 255, 255};
        int pid = mr_ids[i];
        if (pid == -1) continue;
        Entity* entity = &env->entities[pid];
        int y = entity->y, x = entity->x;
        float entity_x = entity->last_x + tick_frac * (entity->x - entity->last_x);
        float entity_y = entity->last_y + tick_frac * (entity->y - entity->last_y);
        int tx = entity_x * ts, ty = entity_y * ts;
        draw_bars(entity, tx, ty - 8, ts, 4, false);
        int tile = map->grid[map_offset(map, y, x)];
        Rectangle source_rect = renderer->asset_map[tile];
        Rectangle dest_rect = (Rectangle){tx, ty, ts, ts};
        if (entity->is_hit) BeginShaderMode(renderer->bloom_shader);
        Vector2 origin = (Vector2){0, 0};
        DrawTexturePro(renderer->puffer, source_rect, dest_rect, origin, 0, tint);
        if (entity->is_hit) EndShaderMode();
        if (entity->stun_timer > 0) DrawTexturePro(renderer->puffer, renderer->stun_uv, dest_rect, origin, 0, tint);
        if (entity->move_timer > 0) {
            if (entity->move_modifier < 0) DrawTexturePro(renderer->puffer, renderer->slow_uv, dest_rect, origin, 0, tint);
            if (entity->move_modifier > 0) DrawTexturePro(renderer->puffer, renderer->speed_uv, dest_rect, origin, 0, tint);
        }
    }
    EndMode2D();

    // HUD (upstream) for the clip's focus hero
    if (hud_pid >= 0 && hud_pid < NUM_PLAYERS) {
        Entity* player = &env->entities[hud_pid];
        float (*actions)[6] = (float(*)[6])env->actions;
        float hud_y = mr_h - 2 * 32;
        draw_bars(player, 2 * 32, hud_y, 10 * 32, 24, true);
        Color off_color = (Color){255, 255, 255, 255};
        Color on_color = (player->team == 0) ? (Color){0, 255, 255, 255} : (Color){255, 0, 0, 255};
        DrawText(TextFormat("Q: %i", player->q_timer), 13 * 32, hud_y - 20, 40, actions[hud_pid][3] ? on_color : off_color);
        DrawText(TextFormat("W: %i", player->w_timer), 17 * 32, hud_y - 20, 40, actions[hud_pid][4] ? on_color : off_color);
        DrawText(TextFormat("E: %i", player->e_timer), 21 * 32, hud_y - 20, 40, actions[hud_pid][5] ? on_color : off_color);
    }
    if (caption && caption[0]) {
        DrawRectangle(0, 0, mr_w, 34, (Color){0, 0, 0, 170});
        DrawText(caption, 12, 8, 20, (Color){255, 255, 255, 255});
        if (caption_right && caption_right[0])
            DrawText(caption_right, mr_w - 12 - MeasureText(caption_right, 18), 9, 18, (Color){200, 200, 200, 255});
    }
    EndTextureMode();

    Image im = LoadImageFromTexture(mr_rt.texture);
    ImageFlipVertical(&im);
    ImageFormat(&im, PIXELFORMAT_UNCOMPRESSED_R8G8B8);
    memcpy(out, im.data, (size_t)mr_w * mr_h * 3);
    UnloadImage(im);
}
