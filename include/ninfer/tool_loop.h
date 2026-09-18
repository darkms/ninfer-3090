#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

// Protocol-neutral representation: callers adapt their wire-specific tool calls to this pair.
using ToolLoopCall = std::pair<std::string, std::string>;
using ToolLoopTurn = std::vector<ToolLoopCall>;

struct ToolLoopDetection {
    std::size_t period = 0;
    ToolLoopTurn expected_next;
};

inline std::string canonical_tool_loop_arguments(std::string_view arguments_json) {
    const nlohmann::json parsed = nlohmann::json::parse(arguments_json, nullptr, false);
    return parsed.is_discarded() ? std::string(arguments_json) : parsed.dump();
}

inline ToolLoopCall make_tool_loop_call(std::string_view name, std::string_view arguments_json) {
    return {std::string(name), canonical_tool_loop_arguments(arguments_json)};
}

inline bool same_tool_loop_turn(const ToolLoopTurn& actual, const ToolLoopTurn& expected) {
    return actual == expected;
}

inline std::optional<ToolLoopDetection>
repeated_tool_cycle(const std::vector<ToolLoopTurn>& history, const ToolLoopTurn& current) {
    if (current.empty()) { return std::nullopt; }

    // ponytail: bounded O(n^2) scan; the detector is deliberately small and only needs the
    // recent action tail, not the whole context window.
    constexpr std::size_t kMaxHistoryTurns = 20;
    const std::size_t history_begin = history.size() > kMaxHistoryTurns
                                          ? history.size() - kMaxHistoryTurns
                                          : 0;
    const std::size_t history_size = history.size() - history_begin;
    const std::size_t turn_count = history_size + 1U;
    for (std::size_t period = 1; period <= turn_count / 2U; ++period) {
        const std::size_t first = turn_count - period * 2U;
        bool repeated = true;
        for (std::size_t offset = 0; offset < period; ++offset) {
            const ToolLoopTurn& previous = history[history_begin + first + offset];
            const std::size_t repeated_index = first + period + offset;
            const ToolLoopTurn& repeated_turn = repeated_index == history_size
                                                    ? current
                                                    : history[history_begin + repeated_index];
            if (previous != repeated_turn) {
                repeated = false;
                break;
            }
        }
        if (repeated) {
            ToolLoopDetection detection;
            detection.period = period;
            const std::size_t expected_index = turn_count - period;
            detection.expected_next = expected_index == history_size
                                          ? current
                                          : history[history_begin + expected_index];
            return detection;
        }
    }
    return std::nullopt;
}

inline std::optional<std::size_t>
repeated_tool_cycle_period(const std::vector<ToolLoopTurn>& history,
                           const ToolLoopTurn& current) {
    const auto detection = repeated_tool_cycle(history, current);
    return detection ? std::optional<std::size_t>(detection->period) : std::nullopt;
}

} // namespace ninfer
