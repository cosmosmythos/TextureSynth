#pragma once
#include "engine/Graph.hpp"
#include <unordered_map>


namespace te {


class NodeLibrary {
public:
    NodeLibrary() = default;
    const NodeType* find(const std::string& type_id) const;
    const std::unordered_map<std::string, NodeType>& all() const { return types_; }
    void add_public(NodeType type) { add(std::move(type)); }
private:
    void add(NodeType type);
    std::unordered_map<std::string, NodeType> types_;
};


} // namespace te