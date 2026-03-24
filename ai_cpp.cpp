/*
 * ai_cpp.cpp — C++ port of ai.py for maximum search performance.
 *
 * Optimized variant: uses ankerl::unordered_dense flat hash maps
 * instead of flat_map for ~2-3x faster hash table operations.
 *
 * Build:
 *     python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include the ankerl stl prerequisites directly to avoid "stl.h" path
// collision with pybind11's stl.h on the include path.
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#define ANKERL_UNORDERED_DENSE_STD_MODULE 1
#include "ankerl_unordered_dense.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

// ── Alias flat hash containers ──
template <typename K, typename V, typename H = ankerl::unordered_dense::hash<K>>
using flat_map = ankerl::unordered_dense::map<K, V, H>;

template <typename K, typename H = ankerl::unordered_dense::hash<K>>
using flat_set = ankerl::unordered_dense::set<K, H>;

namespace py = pybind11;

// ═══════════════════════════════════════════════════════════════════════
//  Constants  (mirror ai.py hyperparameters exactly)
// ═══════════════════════════════════════════════════════════════════════
static constexpr int    CANDIDATE_CAP      = 11;
static constexpr int    ROOT_CANDIDATE_CAP = 13;
static constexpr int    NEIGHBOR_DIST      = 2;
static constexpr double DELTA_WEIGHT       = 1.5;
static constexpr int    MAX_QDEPTH         = 16;
static constexpr int    WIN_LENGTH         = 6;
static constexpr double WIN_SCORE          = 100000000.0;
static constexpr double INF_SCORE          = std::numeric_limits<double>::infinity();

// Player ids — match game.Player enum values
static constexpr int8_t P_NONE = 0;
static constexpr int8_t P_A    = 1;
static constexpr int8_t P_B    = 2;

// TT flags
static constexpr int8_t TT_EXACT = 0;
static constexpr int8_t TT_LOWER = 1;
static constexpr int8_t TT_UPPER = 2;

// ═══════════════════════════════════════════════════════════════════════
//  Coordinate packing
// ═══════════════════════════════════════════════════════════════════════
// A cell (q, r) is packed into a single int64_t.
using Coord = int64_t;

static inline Coord pack(int q, int r) {
    return (static_cast<int64_t>(static_cast<uint32_t>(q)) << 32) |
            static_cast<uint32_t>(r);
}
static inline int pack_q(Coord c) { return static_cast<int32_t>(static_cast<uint32_t>(c >> 32)); }
static inline int pack_r(Coord c) { return static_cast<int32_t>(static_cast<uint32_t>(c)); }

// Lexicographic comparison matching Python tuple comparison on (q, r).
static inline bool coord_lt(Coord a, Coord b) {
    int aq = pack_q(a), ar = pack_r(a), bq = pack_q(b), br = pack_r(b);
    return (aq < bq) || (aq == bq && ar < br);
}
static inline Coord coord_min(Coord a, Coord b) { return coord_lt(a, b) ? a : b; }
static inline Coord coord_max(Coord a, Coord b) { return coord_lt(a, b) ? b : a; }

// Window key: packs (d_idx, q, r) into int64_t.
// d_idx occupies the top byte; q and r are biased into 28-bit fields.
static constexpr int WKEY_BIAS = 0x8000000; // 2^27
static inline int64_t pack_wkey(int d_idx, int q, int r) {
    return (static_cast<int64_t>(static_cast<uint8_t>(d_idx)) << 56) |
           (static_cast<int64_t>(static_cast<uint32_t>(q + WKEY_BIAS) & 0x0FFFFFFFu) << 28) |
            static_cast<int64_t>(static_cast<uint32_t>(r + WKEY_BIAS) & 0x0FFFFFFFu);
}

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════
using Turn = std::pair<Coord, Coord>;

struct TurnHash {
    size_t operator()(const Turn& t) const {
        auto h = std::hash<int64_t>{};
        return h(t.first) ^ (h(t.second) * 0x9e3779b97f4a7c15ULL);
    }
};

struct WinOff  { int d_idx, oq, or_; };
struct EvalOff { int d_idx, k, oq, or_; };
struct NbOff   { int dq, dr; };

struct SavedState {
    int8_t cur_player;
    int8_t moves_left;
    int8_t winner;
    bool   game_over;
};

struct UndoStep {
    Coord      cell;
    SavedState state;
    int8_t     player;
};

struct TTEntry {
    int    depth;
    double score;
    int8_t flag;
    Turn   move;
    bool   has_move;
};

struct TimeUp {};  // thrown on time-out

// ═══════════════════════════════════════════════════════════════════════
//  Direction arrays
// ═══════════════════════════════════════════════════════════════════════
static constexpr int DIR_Q[3] = {1, 0, 1};
static constexpr int DIR_R[3] = {0, 1, -1};
static constexpr int COLONY_DQ[6] = { 1, -1,  0,  0,  1, -1};
static constexpr int COLONY_DR[6] = { 0,  0,  1, -1, -1,  1};

// ═══════════════════════════════════════════════════════════════════════
//  Precomputed offset tables (initialised once)
// ═══════════════════════════════════════════════════════════════════════
static std::vector<WinOff> g_win_offsets;
static std::vector<NbOff>  g_nb_offsets;

static inline int hex_distance(int dq, int dr) {
    return std::max({std::abs(dq), std::abs(dr), std::abs(dq + dr)});
}

static bool g_tables_ready = false;
static void ensure_tables() {
    if (g_tables_ready) return;
    for (int d = 0; d < 3; d++)
        for (int k = 0; k < WIN_LENGTH; k++)
            g_win_offsets.push_back({d, k * DIR_Q[d], k * DIR_R[d]});
    for (int dq = -NEIGHBOR_DIST; dq <= NEIGHBOR_DIST; dq++)
        for (int dr = -NEIGHBOR_DIST; dr <= NEIGHBOR_DIST; dr++)
            if ((dq || dr) && hex_distance(dq, dr) <= NEIGHBOR_DIST)
                g_nb_offsets.push_back({dq, dr});
    g_tables_ready = true;
}

// ═══════════════════════════════════════════════════════════════════════
//  Zobrist tables (lazy, global, never cleared)
// ═══════════════════════════════════════════════════════════════════════
static flat_map<Coord, uint64_t> g_zobrist_a, g_zobrist_b;
static std::mt19937_64 g_zobrist_rng(12345);

static inline uint64_t get_zobrist(Coord c, int8_t player) {
    auto& tbl = (player == P_A) ? g_zobrist_a : g_zobrist_b;
    auto it = tbl.find(c);
    if (it != tbl.end()) return it->second;
    uint64_t v = g_zobrist_rng();
    tbl[c] = v;
    return v;
}

// ═══════════════════════════════════════════════════════════════════════
//  MinimaxBot
// ═══════════════════════════════════════════════════════════════════════
namespace opt {
class MinimaxBot {
public:
    // ── Python-visible attributes ──
    bool   pair_moves = true;
    double time_limit;
    int    last_depth  = 0;
    int    _nodes      = 0;
    double last_score  = 0;
    double last_ebf    = 0;

    // ── Constructors ──
    MinimaxBot() : time_limit(0.05), _rng(std::random_device{}()) { ensure_tables(); }

    MinimaxBot(double tl, py::object pattern_path = py::none())
        : time_limit(tl), _rng(std::random_device{}())
    {
        ensure_tables();
        _load_patterns(pattern_path);
    }

    // ── Main entry point ──
    py::list get_move(py::object game);

    // ── Pickle support (needed for multiprocessing) ──
    py::tuple getstate() const {
        py::bytes pv_bytes(reinterpret_cast<const char*>(_pv.data()),
                           _pv.size() * sizeof(double));
        return py::make_tuple(time_limit, pv_bytes,
                              static_cast<int>(_pv.size()),
                              _eval_length, _pattern_path_str);
    }

    void setstate(py::tuple t) {
        ensure_tables();
        time_limit       = t[0].cast<double>();
        auto pv_str      = t[1].cast<std::string>();
        int  pv_size     = t[2].cast<int>();
        _eval_length     = t[3].cast<int>();
        _pattern_path_str = t[4].cast<std::string>();

        _pv.resize(pv_size);
        std::memcpy(_pv.data(), pv_str.data(), pv_size * sizeof(double));

        _rng = std::mt19937(std::random_device{}());
        _build_eval_tables();
    }

private:
    // ── Pattern data ──
    std::vector<double>  _pv;          // pattern_int -> eval value
    int                  _eval_length = 6;
    std::vector<EvalOff> _eval_offsets;
    std::vector<int>     _pow3;
    std::string          _pattern_path_str;

    // ── Internal game state ──
    flat_map<Coord, int8_t> _board;
    int8_t _cur_player  = P_A;
    int8_t _moves_left  = 1;
    int8_t _winner      = P_NONE;
    bool   _game_over   = false;
    int    _move_count   = 0;

    // ── Search state ──
    using Clock = std::chrono::steady_clock;
    Clock::time_point _deadline;
    uint64_t _hash      = 0;
    int8_t   _player    = P_A;   // side we are maximising for
    int8_t   _cell_a    = 1;     // pattern encoding for A stones
    int8_t   _cell_b    = 2;     // pattern encoding for B stones
    double   _eval_score = 0;

    // ── 6-cell window counts  (win/threat detection) ──
    flat_map<int64_t, std::pair<int8_t,int8_t>> _wc;
    flat_set<int64_t> _hot_a, _hot_b;

    // ── N-cell eval window patterns ──
    flat_map<int64_t, int> _wp;  // wkey -> pattern_int

    // ── Candidates ──
    flat_map<Coord, int> _cand_rc;   // refcount
    flat_set<Coord>      _cand_set;
    std::vector<int>               _rc_stack;

    // ── Transposition table & history ──
    flat_map<uint64_t, TTEntry> _tt;
    flat_map<Coord, int>        _history;

    // ── RNG (for colony direction) ──
    std::mt19937 _rng;

    // ────────────────────────────────────────────────────────────────
    //  Pattern loading  (calls back into Python ai._load_pattern_values)
    // ────────────────────────────────────────────────────────────────
    void _load_patterns(py::object pattern_path) {
        py::module_ ai_mod = py::module_::import("ai");
        std::string path;
        if (pattern_path.is_none())
            path = ai_mod.attr("_DEFAULT_PATTERN_PATH").cast<std::string>();
        else
            path = pattern_path.cast<std::string>();
        _pattern_path_str = path;

        py::tuple result = ai_mod.attr("_load_pattern_values")(path).cast<py::tuple>();
        py::list  pv_list = result[0].cast<py::list>();
        _eval_length      = result[1].cast<int>();

        _pv.resize(pv_list.size());
        for (size_t i = 0; i < pv_list.size(); i++)
            _pv[i] = pv_list[i].cast<double>();

        _build_eval_tables();
    }

    void _build_eval_tables() {
        _eval_offsets.clear();
        for (int d = 0; d < 3; d++)
            for (int k = 0; k < _eval_length; k++)
                _eval_offsets.push_back({d, k, k * DIR_Q[d], k * DIR_R[d]});
        _pow3.resize(_eval_length);
        _pow3[0] = 1;
        for (int i = 1; i < _eval_length; i++)
            _pow3[i] = _pow3[i - 1] * 3;
    }

    // ────────────────────────────────────────────────────────────────
    //  Time control
    // ────────────────────────────────────────────────────────────────
    inline void _check_time() {
        _nodes++;
        if ((_nodes & 1023) == 0 && Clock::now() >= _deadline)
            throw TimeUp{};
    }

    // ────────────────────────────────────────────────────────────────
    //  TT key
    // ────────────────────────────────────────────────────────────────
    inline uint64_t _tt_key() const {
        return _hash ^ (static_cast<uint64_t>(_cur_player) * 0x9e3779b97f4a7c15ULL)
                      ^ (static_cast<uint64_t>(_moves_left) * 0x517cc1b727220a95ULL);
    }

    // ────────────────────────────────────────────────────────────────
    //  Incremental make / undo
    // ────────────────────────────────────────────────────────────────
    void _make(int q, int r) {
        int8_t player = _cur_player;
        Coord  cell   = pack(q, r);

        // Zobrist
        _hash ^= get_zobrist(cell, player);

        // Cell value for pattern encoding
        int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;

        // ── 6-cell windows ──
        bool won = false;
        if (player == P_A) {
            for (const auto& wo : g_win_offsets) {
                int64_t wkey = pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];  // default {0,0}
                counts.first++;
                if (counts.first >= 4) _hot_a.insert(wkey);
                if (counts.first == WIN_LENGTH && counts.second == 0) won = true;
            }
        } else {
            for (const auto& wo : g_win_offsets) {
                int64_t wkey = pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];
                counts.second++;
                if (counts.second >= 4) _hot_b.insert(wkey);
                if (counts.second == WIN_LENGTH && counts.first == 0) won = true;
            }
        }

        // ── N-cell eval windows ──
        const double* pv = _pv.data();
        for (const auto& eo : _eval_offsets) {
            int64_t wkey8 = pack_wkey(3 + eo.d_idx, q - eo.oq, r - eo.or_);
            auto it = _wp.find(wkey8);
            int old_pi = (it != _wp.end()) ? it->second : 0;
            int new_pi = old_pi + cell_val * _pow3[eo.k];
            _eval_score += pv[new_pi] - pv[old_pi];
            if (it != _wp.end())
                it->second = new_pi;
            else
                _wp[wkey8] = new_pi;
        }

        // ── Candidates  (BEFORE placing stone, matching Python order) ──
        _cand_set.erase(cell);
        auto rc_it = _cand_rc.find(cell);
        _rc_stack.push_back((rc_it != _cand_rc.end()) ? rc_it->second : 0);
        if (rc_it != _cand_rc.end()) _cand_rc.erase(rc_it);

        for (const auto& nb : g_nb_offsets) {
            Coord nc = pack(q + nb.dq, r + nb.dr);
            _cand_rc[nc]++;
            if (!_board.count(nc))
                _cand_set.insert(nc);
        }

        // Place stone
        _board[cell] = player;
        _move_count++;

        if (won) {
            _winner   = player;
            _game_over = true;
        } else {
            _moves_left--;
            if (_moves_left <= 0) {
                _cur_player = (player == P_A) ? P_B : P_A;
                _moves_left = 2;
            }
        }
    }

    void _undo(int q, int r, const SavedState& st, int8_t player) {
        Coord cell = pack(q, r);

        // Remove stone
        _board.erase(cell);
        _move_count--;
        _cur_player = st.cur_player;
        _moves_left = st.moves_left;
        _winner     = st.winner;
        _game_over  = st.game_over;

        // Zobrist
        _hash ^= get_zobrist(cell, player);

        int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;

        // ── 6-cell windows ──
        if (player == P_A) {
            for (const auto& wo : g_win_offsets) {
                int64_t wkey = pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];
                counts.first--;
                if (counts.first < 4) _hot_a.erase(wkey);
            }
        } else {
            for (const auto& wo : g_win_offsets) {
                int64_t wkey = pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];
                counts.second--;
                if (counts.second < 4) _hot_b.erase(wkey);
            }
        }

        // ── N-cell eval windows ──
        const double* pv = _pv.data();
        for (const auto& eo : _eval_offsets) {
            int64_t wkey8 = pack_wkey(3 + eo.d_idx, q - eo.oq, r - eo.or_);
            int old_pi = _wp[wkey8];
            int new_pi = old_pi - cell_val * _pow3[eo.k];
            _eval_score += pv[new_pi] - pv[old_pi];
            if (new_pi == 0)
                _wp.erase(wkey8);
            else
                _wp[wkey8] = new_pi;
        }

        // ── Candidates ──
        for (const auto& nb : g_nb_offsets) {
            Coord nc = pack(q + nb.dq, r + nb.dr);
            auto it = _cand_rc.find(nc);
            int c = it->second - 1;
            if (c == 0) {
                _cand_rc.erase(it);
                _cand_set.erase(nc);
            } else {
                it->second = c;
            }
        }
        int saved_rc = _rc_stack.back();
        _rc_stack.pop_back();
        if (saved_rc > 0) {
            _cand_rc[cell] = saved_rc;
            _cand_set.insert(cell);
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Turn make / undo  (a turn = pair of moves)
    // ────────────────────────────────────────────────────────────────
    int _make_turn(const Turn& turn, UndoStep steps[2]) {
        int q1 = pack_q(turn.first),  r1 = pack_r(turn.first);
        int q2 = pack_q(turn.second), r2 = pack_r(turn.second);

        steps[0] = {turn.first, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
        _make(q1, r1);
        if (_game_over) return 1;

        steps[1] = {turn.second, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
        _make(q2, r2);
        return 2;
    }

    void _undo_turn(const UndoStep steps[], int n) {
        for (int i = n - 1; i >= 0; i--)
            _undo(pack_q(steps[i].cell), pack_r(steps[i].cell),
                  steps[i].state, steps[i].player);
    }

    // ────────────────────────────────────────────────────────────────
    //  Move delta  (read-only eval change for placing at q,r)
    // ────────────────────────────────────────────────────────────────
    double _move_delta(int q, int r, bool is_a) const {
        int8_t cell_val = is_a ? _cell_a : _cell_b;
        const double* pv = _pv.data();
        double delta = 0.0;
        for (const auto& eo : _eval_offsets) {
            int64_t wkey8 = pack_wkey(3 + eo.d_idx, q - eo.oq, r - eo.or_);
            int old_pi = 0;
            auto it = _wp.find(wkey8);
            if (it != _wp.end()) old_pi = it->second;
            int new_pi = old_pi + cell_val * _pow3[eo.k];
            delta += pv[new_pi] - pv[old_pi];
        }
        return delta;
    }

    // ────────────────────────────────────────────────────────────────
    //  Win / threat detection  (6-cell windows)
    // ────────────────────────────────────────────────────────────────
    // Returns {found, turn}.  If !found the turn is meaningless.
    std::pair<bool, Turn> _find_instant_win(int8_t player) const {
        int p_idx = (player == P_A) ? 0 : 1;
        const auto& hot = (player == P_A) ? _hot_a : _hot_b;

        for (int64_t wkey : hot) {
            auto wit = _wc.find(wkey);
            if (wit == _wc.end()) continue;
            int my_count  = (p_idx == 0) ? wit->second.first : wit->second.second;
            int opp_count = (p_idx == 0) ? wit->second.second : wit->second.first;

            if (my_count >= WIN_LENGTH - 2 && opp_count == 0) {
                // Extract window start
                int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
                int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - WKEY_BIAS;
                int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - WKEY_BIAS;
                int dq = DIR_Q[d_idx], dr = DIR_R[d_idx];

                Coord cells[WIN_LENGTH];
                int n = 0;
                for (int j = 0; j < WIN_LENGTH; j++) {
                    Coord c = pack(sq + j * dq, sr + j * dr);
                    if (!_board.count(c))
                        cells[n++] = c;
                }
                if (n == 1) {
                    Coord other = cells[0];
                    for (Coord c : _cand_set)
                        if (c != cells[0]) { other = c; break; }
                    return {true, {coord_min(cells[0], other),
                                   coord_max(cells[0], other)}};
                }
                if (n == 2) {
                    return {true, {coord_min(cells[0], cells[1]),
                                   coord_max(cells[0], cells[1])}};
                }
            }
        }
        return {false, {}};
    }

    flat_set<Coord> _find_threat_cells(int8_t player) const {
        flat_set<Coord> threats;
        int p_idx = (player == P_A) ? 0 : 1;
        const auto& hot = (player == P_A) ? _hot_a : _hot_b;

        for (int64_t wkey : hot) {
            auto wit = _wc.find(wkey);
            if (wit == _wc.end()) continue;
            int opp_count = (p_idx == 0) ? wit->second.second : wit->second.first;
            if (opp_count != 0) continue;

            int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
            int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - WKEY_BIAS;
            int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - WKEY_BIAS;
            int dq = DIR_Q[d_idx], dr = DIR_R[d_idx];

            for (int j = 0; j < WIN_LENGTH; j++) {
                Coord c = pack(sq + j * dq, sr + j * dr);
                if (!_board.count(c))
                    threats.insert(c);
            }
        }
        return threats;
    }

    std::vector<Turn> _filter_turns_by_threats(const std::vector<Turn>& turns) const {
        int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
        int p_idx = (opponent == P_A) ? 0 : 1;
        const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;

        // Collect must-hit sets
        std::vector<flat_set<Coord>> must_hit;
        for (int64_t wkey : hot) {
            auto wit = _wc.find(wkey);
            if (wit == _wc.end()) continue;
            int my_count  = (p_idx == 0) ? wit->second.first  : wit->second.second;
            int opp_count = (p_idx == 0) ? wit->second.second : wit->second.first;
            if (my_count < WIN_LENGTH - 2 || opp_count != 0) continue;

            int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
            int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - WKEY_BIAS;
            int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - WKEY_BIAS;
            int dq = DIR_Q[d_idx], dr = DIR_R[d_idx];

            flat_set<Coord> empties;
            for (int j = 0; j < WIN_LENGTH; j++) {
                Coord c = pack(sq + j * dq, sr + j * dr);
                if (!_board.count(c))
                    empties.insert(c);
            }
            must_hit.push_back(std::move(empties));
        }
        if (must_hit.empty()) return turns;

        std::vector<Turn> out;
        out.reserve(turns.size());
        for (const auto& t : turns) {
            bool ok = true;
            for (const auto& w : must_hit) {
                if (!w.count(t.first) && !w.count(t.second)) {
                    ok = false; break;
                }
            }
            if (ok) out.push_back(t);
        }
        return out;
    }

    // ────────────────────────────────────────────────────────────────
    //  Turn generation
    // ────────────────────────────────────────────────────────────────
    std::vector<Turn> _generate_turns() {
        auto [found, wt] = _find_instant_win(_cur_player);
        if (found) return {wt};

        std::vector<Coord> cands(_cand_set.begin(), _cand_set.end());
        if (cands.size() < 2) {
            if (!cands.empty()) return {{cands[0], cands[0]}};
            return {};
        }

        bool is_a = (_cur_player == P_A);
        bool maximizing = (_cur_player == _player);

        // Sort by move_delta
        std::vector<std::pair<double, Coord>> scored;
        scored.reserve(cands.size());
        for (Coord c : cands)
            scored.push_back({_move_delta(pack_q(c), pack_r(c), is_a), c});
        std::sort(scored.begin(), scored.end(), [maximizing](const auto& a, const auto& b) {
            return maximizing ? (a.first > b.first) : (a.first < b.first);
        });

        cands.clear();
        int cap = std::min(static_cast<int>(scored.size()), ROOT_CANDIDATE_CAP);
        for (int i = 0; i < cap; i++)
            cands.push_back(scored[i].second);

        // Colony candidate
        if (!_board.empty()) {
            int64_t sq = 0, sr = 0;
            for (const auto& kv : _board) { sq += pack_q(kv.first); sr += pack_r(kv.first); }
            int cq = static_cast<int>(sq / static_cast<int64_t>(_board.size()));
            int cr = static_cast<int>(sr / static_cast<int64_t>(_board.size()));
            int max_r = 0;
            for (const auto& kv : _board) {
                int d = hex_distance(pack_q(kv.first) - cq, pack_r(kv.first) - cr);
                if (d > max_r) max_r = d;
            }
            int cd = max_r + 3;
            std::uniform_int_distribution<int> dist(0, 5);
            int di = dist(_rng);
            Coord colony = pack(cq + COLONY_DQ[di] * cd, cr + COLONY_DR[di] * cd);
            if (!_board.count(colony))
                cands.push_back(colony);
        }

        // All pairs
        int n = static_cast<int>(cands.size());
        std::vector<Turn> turns;
        turns.reserve(n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                turns.push_back({cands[i], cands[j]});

        return _filter_turns_by_threats(turns);
    }

    std::vector<Turn> _generate_threat_turns(
            const flat_set<Coord>& my_threats,
            const flat_set<Coord>& opp_threats) {
        auto [found, wt] = _find_instant_win(_cur_player);
        if (found) return {wt};

        bool is_a = (_cur_player == P_A);
        bool maximizing = (_cur_player == _player);
        double sign = maximizing ? 1.0 : -1.0;

        std::vector<Coord> opp_cells, my_cells;
        for (Coord c : opp_threats) if (_cand_set.count(c)) opp_cells.push_back(c);
        for (Coord c : my_threats)  if (_cand_set.count(c)) my_cells.push_back(c);

        std::vector<Coord>* primary = nullptr;
        if (!opp_cells.empty())     primary = &opp_cells;
        else if (!my_cells.empty()) primary = &my_cells;
        else return {};

        if (primary->size() >= 2) {
            int n = static_cast<int>(primary->size());
            std::vector<Turn> pairs;
            pairs.reserve(n * (n - 1) / 2);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    pairs.push_back({(*primary)[i], (*primary)[j]});
            std::sort(pairs.begin(), pairs.end(),
                [&](const Turn& a, const Turn& b) {
                    double da = _move_delta(pack_q(a.first), pack_r(a.first), is_a)
                              + _move_delta(pack_q(a.second), pack_r(a.second), is_a);
                    double db = _move_delta(pack_q(b.first), pack_r(b.first), is_a)
                              + _move_delta(pack_q(b.second), pack_r(b.second), is_a);
                    return maximizing ? (da > db) : (da < db);
                });
            return pairs;
        }

        // Single threat cell — pair with best companion
        Coord tc = (*primary)[0];
        Coord best_comp = tc;
        double best_d = -INF_SCORE;
        for (Coord c : _cand_set) {
            if (c != tc) {
                double d = _move_delta(pack_q(c), pack_r(c), is_a) * sign;
                if (d > best_d) { best_d = d; best_comp = c; }
            }
        }
        if (best_comp == tc) return {};
        return {{coord_min(tc, best_comp), coord_max(tc, best_comp)}};
    }

    // ────────────────────────────────────────────────────────────────
    //  Quiescence search
    // ────────────────────────────────────────────────────────────────
    double _quiescence(double alpha, double beta, int qdepth) {
        _check_time();

        if (_game_over) {
            if (_winner == _player)    return  WIN_SCORE;
            if (_winner != P_NONE)     return -WIN_SCORE;
            return 0.0;
        }

        auto [found, wt] = _find_instant_win(_cur_player);
        if (found) {
            UndoStep steps[2];
            int n = _make_turn(wt, steps);
            double sc = (_winner == _player) ? WIN_SCORE : -WIN_SCORE;
            _undo_turn(steps, n);
            return sc;
        }

        double stand_pat = _eval_score;
        int8_t current  = _cur_player;
        int8_t opponent = (current == P_A) ? P_B : P_A;
        auto my_threats  = _find_threat_cells(current);
        auto opp_threats = _find_threat_cells(opponent);

        if ((my_threats.empty() && opp_threats.empty()) || qdepth <= 0)
            return stand_pat;

        bool maximizing = (current == _player);
        if (maximizing) {
            if (stand_pat >= beta) return stand_pat;
            alpha = std::max(alpha, stand_pat);
        } else {
            if (stand_pat <= alpha) return stand_pat;
            beta = std::min(beta, stand_pat);
        }

        auto threat_turns = _generate_threat_turns(my_threats, opp_threats);
        if (threat_turns.empty()) return stand_pat;

        double value = stand_pat;
        if (maximizing) {
            for (const auto& turn : threat_turns) {
                UndoStep steps[2];
                int nm = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? WIN_SCORE : -WIN_SCORE)
                    : _quiescence(alpha, beta, qdepth - 1);
                _undo_turn(steps, nm);
                if (cv > value) value = cv;
                alpha = std::max(alpha, value);
                if (alpha >= beta) break;
            }
        } else {
            for (const auto& turn : threat_turns) {
                UndoStep steps[2];
                int nm = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? WIN_SCORE : -WIN_SCORE)
                    : _quiescence(alpha, beta, qdepth - 1);
                _undo_turn(steps, nm);
                if (cv < value) value = cv;
                beta = std::min(beta, value);
                if (alpha >= beta) break;
            }
        }
        return value;
    }

    // ────────────────────────────────────────────────────────────────
    //  Root search
    // ────────────────────────────────────────────────────────────────
    std::pair<Turn, flat_map<Turn, double, TurnHash>>
    _search_root(std::vector<Turn>& turns, int depth) {
        bool maximizing = (_cur_player == _player);
        Turn best = turns[0];
        double alpha = -INF_SCORE, beta = INF_SCORE;

        flat_map<Turn, double, TurnHash> scores;
        scores.reserve(turns.size());

        for (const auto& turn : turns) {
            _check_time();
            UndoStep steps[2];
            int n = _make_turn(turn, steps);
            double sc;
            if (_game_over)
                sc = (_winner == _player) ? WIN_SCORE : -WIN_SCORE;
            else
                sc = _minimax(depth - 1, alpha, beta);
            _undo_turn(steps, n);
            scores[turn] = sc;

            if (maximizing && sc > alpha)  { alpha = sc; best = turn; }
            if (!maximizing && sc < beta)  { beta  = sc; best = turn; }
        }

        double best_sc = maximizing ? alpha : beta;
        _tt[_tt_key()] = {depth, best_sc, TT_EXACT, best, true};
        return {best, std::move(scores)};
    }

    // ────────────────────────────────────────────────────────────────
    //  Minimax with alpha-beta and TT
    // ────────────────────────────────────────────────────────────────
    double _minimax(int depth, double alpha, double beta) {
        _check_time();

        if (_game_over) {
            if (_winner == _player)    return  WIN_SCORE;
            if (_winner != P_NONE)     return -WIN_SCORE;
            return 0.0;
        }

        uint64_t ttk = _tt_key();
        Turn tt_move{};
        bool has_tt_move = false;

        auto tt_it = _tt.find(ttk);
        if (tt_it != _tt.end()) {
            const auto& e = tt_it->second;
            has_tt_move = e.has_move;
            tt_move     = e.move;
            if (e.depth >= depth) {
                if (e.flag == TT_EXACT) return e.score;
                if (e.flag == TT_LOWER) alpha = std::max(alpha, e.score);
                if (e.flag == TT_UPPER) beta  = std::min(beta,  e.score);
                if (alpha >= beta) return e.score;
            }
        }

        if (depth == 0) {
            double sc = _quiescence(alpha, beta, MAX_QDEPTH);
            _tt[ttk] = {0, sc, TT_EXACT, {}, false};
            return sc;
        }

        // Instant win for current player
        {
            auto [found, wt] = _find_instant_win(_cur_player);
            if (found) {
                UndoStep steps[2];
                int n = _make_turn(wt, steps);
                double sc = (_winner == _player) ? WIN_SCORE : -WIN_SCORE;
                _undo_turn(steps, n);
                _tt[ttk] = {depth, sc, TT_EXACT, wt, true};
                return sc;
            }
        }

        // Opponent instant win → check if blockable
        int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
        {
            auto [opp_found, opp_wt] = _find_instant_win(opponent);
            if (opp_found) {
                int p_idx = (opponent == P_A) ? 0 : 1;
                const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;
                std::vector<flat_set<Coord>> must_hit;
                for (int64_t wkey : hot) {
                    auto wit = _wc.find(wkey);
                    if (wit == _wc.end()) continue;
                    int mc = (p_idx == 0) ? wit->second.first  : wit->second.second;
                    int oc = (p_idx == 0) ? wit->second.second : wit->second.first;
                    if (mc < WIN_LENGTH - 2 || oc != 0) continue;

                    int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
                    int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - WKEY_BIAS;
                    int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - WKEY_BIAS;
                    int dq = DIR_Q[d_idx], dr = DIR_R[d_idx];
                    flat_set<Coord> empties;
                    for (int j = 0; j < WIN_LENGTH; j++) {
                        Coord c = pack(sq + j * dq, sr + j * dr);
                        if (!_board.count(c)) empties.insert(c);
                    }
                    must_hit.push_back(std::move(empties));
                }
                if (must_hit.size() > 1) {
                    flat_set<Coord> all_cells;
                    for (const auto& s : must_hit) all_cells.insert(s.begin(), s.end());
                    bool can_block = false;
                    for (Coord c1 : all_cells) {
                        for (Coord c2 : all_cells) {
                            bool ok = true;
                            for (const auto& w : must_hit)
                                if (!w.count(c1) && !w.count(c2)) { ok = false; break; }
                            if (ok) { can_block = true; break; }
                        }
                        if (can_block) break;
                    }
                    if (!can_block) {
                        double sc = (opponent != _player) ? -WIN_SCORE : WIN_SCORE;
                        _tt[ttk] = {depth, sc, TT_EXACT, {}, false};
                        return sc;
                    }
                }
            }
        }

        double orig_alpha = alpha, orig_beta = beta;
        bool maximizing = (_cur_player == _player);

        // Generate candidates and turns
        std::vector<Turn> turns;
        {
            std::vector<Coord> cands(_cand_set.begin(), _cand_set.end());
            if (cands.size() < 2) {
                if (cands.empty()) {
                    double sc = _eval_score;
                    _tt[ttk] = {depth, sc, TT_EXACT, {}, false};
                    return sc;
                }
                turns = {{cands[0], cands[0]}};
            } else {
                bool is_a = (_cur_player == P_A);
                double dsign = maximizing ? DELTA_WEIGHT : -DELTA_WEIGHT;

                std::vector<std::pair<double, Coord>> scored;
                scored.reserve(cands.size());
                for (Coord c : cands) {
                    double h = 0;
                    auto hi = _history.find(c);
                    if (hi != _history.end()) h = static_cast<double>(hi->second);
                    scored.push_back({h + _move_delta(pack_q(c), pack_r(c), is_a) * dsign, c});
                }
                std::sort(scored.begin(), scored.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                cands.clear();
                int cap = std::min(static_cast<int>(scored.size()), CANDIDATE_CAP);
                for (int i = 0; i < cap; i++) cands.push_back(scored[i].second);

                int n = static_cast<int>(cands.size());
                turns.reserve(n * (n - 1) / 2);
                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++)
                        turns.push_back({cands[i], cands[j]});
                turns = _filter_turns_by_threats(turns);
            }
        }

        if (turns.empty()) {
            double sc = _eval_score;
            _tt[ttk] = {depth, sc, TT_EXACT, {}, false};
            return sc;
        }

        // TT move ordering
        if (has_tt_move) {
            for (size_t i = 0; i < turns.size(); i++)
                if (turns[i] == tt_move) { std::swap(turns[0], turns[i]); break; }
        }

        Turn best_move{};
        double value;

        if (maximizing) {
            value = -INF_SCORE;
            for (const auto& turn : turns) {
                UndoStep steps[2];
                int n = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? WIN_SCORE : -WIN_SCORE)
                    : _minimax(depth - 1, alpha, beta);
                _undo_turn(steps, n);
                if (cv > value) { value = cv; best_move = turn; }
                alpha = std::max(alpha, value);
                if (alpha >= beta) {
                    _history[turn.first]  += depth * depth;
                    _history[turn.second] += depth * depth;
                    break;
                }
            }
        } else {
            value = INF_SCORE;
            for (const auto& turn : turns) {
                UndoStep steps[2];
                int n = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? WIN_SCORE : -WIN_SCORE)
                    : _minimax(depth - 1, alpha, beta);
                _undo_turn(steps, n);
                if (cv < value) { value = cv; best_move = turn; }
                beta = std::min(beta, value);
                if (alpha >= beta) {
                    _history[turn.first]  += depth * depth;
                    _history[turn.second] += depth * depth;
                    break;
                }
            }
        }

        int8_t flag;
        if      (value <= orig_alpha) flag = TT_UPPER;
        else if (value >= orig_beta)  flag = TT_LOWER;
        else                          flag = TT_EXACT;
        _tt[ttk] = {depth, value, flag, best_move, true};
        return value;
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  get_move — top-level entry point
// ═══════════════════════════════════════════════════════════════════════
py::list MinimaxBot::get_move(py::object game) {
    // ── Extract board from Python ──
    py::dict py_board = game.attr("board").cast<py::dict>();

    if (py_board.empty()) {
        py::list res;
        res.append(py::make_tuple(0, 0));
        return res;
    }

    py::module_ game_mod = py::module_::import("game");
    py::object  PyA = game_mod.attr("Player").attr("A");
    py::object  PyB = game_mod.attr("Player").attr("B");

    _board.clear();
    _board.reserve(py_board.size() + 64);
    for (auto item : py_board) {
        py::tuple key = item.first.cast<py::tuple>();
        int q = key[0].cast<int>(), r = key[1].cast<int>();
        int8_t p = item.second.is(PyA) ? P_A : P_B;
        _board[pack(q, r)] = p;
    }

    py::object py_cur = game.attr("current_player");
    _cur_player = py_cur.is(PyA) ? P_A : P_B;
    _moves_left = game.attr("moves_left_in_turn").cast<int8_t>();
    _move_count = game.attr("move_count").cast<int>();
    _winner     = P_NONE;
    _game_over  = false;

    // ── Deadline ──
    _deadline = Clock::now() + std::chrono::microseconds(
                    static_cast<int64_t>(time_limit * 2000000.0));

    // ── Player tracking / TT management ──
    if (_cur_player != _player) {
        _tt.clear();
        _history.clear();
    }
    _player    = _cur_player;
    _nodes     = 0;
    last_depth = 0;
    last_score = 0;
    last_ebf   = 0;
    if (_tt.size() > 1000000) _tt.clear();

    // ── Zobrist ──
    _hash = 0;
    for (const auto& kv : _board)
        _hash ^= get_zobrist(kv.first, kv.second);

    // ── Cell value mapping ──
    if (_player == P_A) { _cell_a = 1; _cell_b = 2; }
    else                { _cell_a = 2; _cell_b = 1; }

    // ── Init 6-cell windows ──
    _wc.clear();
    _hot_a.clear();
    _hot_b.clear();
    {
        flat_set<int64_t> seen;
        for (const auto& kv : _board) {
            int bq = pack_q(kv.first), br = pack_r(kv.first);
            for (const auto& wo : g_win_offsets) {
                int64_t wkey = pack_wkey(wo.d_idx, bq - wo.oq, br - wo.or_);
                if (!seen.insert(wkey).second) continue;
                int d = wo.d_idx;
                int sq = bq - wo.oq, sr = br - wo.or_;
                int ac = 0, bc = 0;
                for (int j = 0; j < WIN_LENGTH; j++) {
                    auto it = _board.find(pack(sq + j * DIR_Q[d], sr + j * DIR_R[d]));
                    if (it != _board.end()) { if (it->second == P_A) ac++; else bc++; }
                }
                if (ac || bc) _wc[wkey] = {static_cast<int8_t>(ac), static_cast<int8_t>(bc)};
            }
        }
        for (const auto& kv : _wc) {
            if (kv.second.first  >= 4) _hot_a.insert(kv.first);
            if (kv.second.second >= 4) _hot_b.insert(kv.first);
        }
    }

    // ── Init N-cell eval windows ──
    _wp.clear();
    _eval_score = 0.0;
    {
        const double* pv = _pv.data();
        flat_set<int64_t> seen;
        for (const auto& kv : _board) {
            int bq = pack_q(kv.first), br = pack_r(kv.first);
            for (const auto& eo : _eval_offsets) {
                int64_t wkey8 = pack_wkey(3 + eo.d_idx, bq - eo.oq, br - eo.or_);
                if (!seen.insert(wkey8).second) continue;
                int sq = bq - eo.oq, sr = br - eo.or_;
                int d = eo.d_idx;
                int pi = 0;
                bool has = false;
                for (int j = 0; j < _eval_length; j++) {
                    auto it = _board.find(pack(sq + j * DIR_Q[d], sr + j * DIR_R[d]));
                    if (it != _board.end()) {
                        pi += ((it->second == P_A) ? _cell_a : _cell_b) * _pow3[j];
                        has = true;
                    }
                }
                if (has) { _wp[wkey8] = pi; _eval_score += pv[pi]; }
            }
        }
    }

    // ── Init candidates ──
    _cand_rc.clear();
    _cand_set.clear();
    _rc_stack.clear();
    for (const auto& kv : _board) {
        int bq = pack_q(kv.first), br = pack_r(kv.first);
        for (const auto& nb : g_nb_offsets) {
            Coord nc = pack(bq + nb.dq, br + nb.dr);
            if (!_board.count(nc)) {
                _cand_rc[nc]++;
                _cand_set.insert(nc);
            }
        }
    }

    if (_cand_set.empty()) {
        py::list res;
        res.append(py::make_tuple(0, 0));
        return res;
    }

    bool maximizing = (_cur_player == _player);
    auto turns = _generate_turns();
    if (turns.empty()) {
        py::list res;
        res.append(py::make_tuple(0, 0));
        return res;
    }

    Turn best_move = turns[0];

    // ── Save state for TimeUp rollback ──
    auto saved_board    = _board;
    auto saved_st       = SavedState{_cur_player, _moves_left, _winner, _game_over};
    int  saved_mc       = _move_count;
    uint64_t saved_hash = _hash;
    double   saved_eval = _eval_score;
    auto saved_wc       = _wc;
    auto saved_wp       = _wp;
    auto saved_cs       = _cand_set;
    auto saved_cr       = _cand_rc;
    auto saved_ha       = _hot_a;
    auto saved_hb       = _hot_b;

    for (int depth = 1; depth < 200; depth++) {
        try {
            int nb4 = _nodes;
            auto root_result = _search_root(turns, depth);
            Turn result = root_result.first;
            auto& scores = root_result.second;
            best_move  = result;
            last_depth = depth;
            auto si = scores.find(result);
            last_score = (si != scores.end()) ? si->second : 0.0;
            int nthis = _nodes - nb4;
            if (nthis > 1)
                last_ebf = std::round(std::pow(static_cast<double>(nthis),
                                               1.0 / depth) * 10.0) / 10.0;
            // Re-order turns for next iteration
            std::sort(turns.begin(), turns.end(),
                [&scores, maximizing](const Turn& a, const Turn& b) {
                    double sa = 0, sb = 0;
                    auto ia = scores.find(a); if (ia != scores.end()) sa = ia->second;
                    auto ib = scores.find(b); if (ib != scores.end()) sb = ib->second;
                    return maximizing ? (sa > sb) : (sa < sb);
                });
            if (std::abs(last_score) >= WIN_SCORE) break;
        } catch (const TimeUp&) {
            _board      = std::move(saved_board);
            _move_count = saved_mc;
            _cur_player = saved_st.cur_player;
            _moves_left = saved_st.moves_left;
            _winner     = saved_st.winner;
            _game_over  = saved_st.game_over;
            _hash       = saved_hash;
            _eval_score = saved_eval;
            _wc         = std::move(saved_wc);
            _wp         = std::move(saved_wp);
            _cand_set   = std::move(saved_cs);
            _cand_rc    = std::move(saved_cr);
            _hot_a      = std::move(saved_ha);
            _hot_b      = std::move(saved_hb);
            break;
        }
    }

    // ── Build Python result ──
    py::list res;
    res.append(py::make_tuple(pack_q(best_move.first),  pack_r(best_move.first)));
    res.append(py::make_tuple(pack_q(best_move.second), pack_r(best_move.second)));
    return res;
}

} // namespace opt

// ═══════════════════════════════════════════════════════════════════════
//  pybind11 module
// ═══════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(ai_cpp, m) {
    m.doc() = "C++ port of ai.py MinimaxBot (optimized flat hash maps)";

    py::class_<opt::MinimaxBot>(m, "MinimaxBot")
        .def(py::init<double, py::object>(),
             py::arg("time_limit") = 0.05,
             py::arg("pattern_path") = py::none())
        .def("get_move", &opt::MinimaxBot::get_move, py::arg("game"))
        .def_readwrite("pair_moves",  &opt::MinimaxBot::pair_moves)
        .def_readwrite("time_limit",  &opt::MinimaxBot::time_limit)
        .def_readwrite("last_depth",  &opt::MinimaxBot::last_depth)
        .def_readwrite("_nodes",      &opt::MinimaxBot::_nodes)
        .def_readwrite("last_score",  &opt::MinimaxBot::last_score)
        .def_readwrite("last_ebf",    &opt::MinimaxBot::last_ebf)
        .def("__str__", [](const opt::MinimaxBot&) { return std::string("ai_cpp"); })
        .def(py::pickle(
            [](const opt::MinimaxBot& bot) { return bot.getstate(); },
            [](py::tuple t) {
                opt::MinimaxBot bot;
                bot.setstate(t);
                return bot;
            }
        ));
}
