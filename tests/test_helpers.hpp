#pragma once

#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"

using namespace te;

inline NodeType make_type(const std::string& id, uint32_t n_in, uint32_t n_out,
                   uint32_t pass_count = 1,
                   const std::vector<SocketType>& input_types = {},
                   const std::string& glsl = "",
                   uint32_t n_params = 0) {
    NodeType t;
    t.id = id;
    t.display_name = id;
    t.pass_kind = PassKind::Compute;
    t.pass_count = pass_count;
    t.glsl_function = glsl;
    for (uint32_t i = 0; i < n_in; ++i) {
        te::Socket s;
        s.name = "in" + std::to_string(i);
        s.type = (i < input_types.size()) ? input_types[i] : SocketType::Vec4;
        t.inputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        te::Socket s;
        s.name = "out" + std::to_string(i);
        s.type = SocketType::Vec4;
        t.outputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_params; ++i) {
        NodeParam p;
        p.name = "param" + std::to_string(i);
        t.params.push_back(p);
    }
    return t;
}
