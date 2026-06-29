#pragma once
#include "engine/NodeLibrary.hpp"
#include "engine/Graph.hpp"
#include <string>

namespace te {
class NodeRegistryLoader {
public:
    // Scans `nodes_dir` for *.node.json, resolves includes from `glsl_dir`. Returns number of nodes loaded; appends to lib.
    static int load_from_directory(NodeLibrary& lib,
                                   const std::string& nodes_dir,
                                   const std::string& glsl_dir,
                                   std::string* error_out = nullptr);

    // Public so unit tests can pin the string-to-enum mapping without writing a temp .node.json to disk. Empty/unrecognized strings return Compute and emit a log_warn.
    static PassKind parse_pass_kind(const std::string& s);
};


} // namespace te