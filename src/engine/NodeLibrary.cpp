#include "engine/NodeLibrary.hpp"

namespace te {

const NodeType* NodeLibrary::find(const std::string& type_id) const {
    auto it = types_.find(type_id);
    return it != types_.end() ? &it->second : nullptr;
}

void NodeLibrary::add(NodeType type) {
    types_[type.id] = std::move(type);
}

} // namespace te