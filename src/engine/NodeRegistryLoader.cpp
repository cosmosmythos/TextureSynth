#include "engine/NodeRegistryLoader.hpp"
#include "engine/Logging.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace te {
namespace fs = std::filesystem;
using json = nlohmann::json;


static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    // Strip UTF-8 BOM if present (EF BB BF) - shaderc rejects it mid-shader.
    if (s.size() >= 3 &&
        static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB &&
        static_cast<unsigned char>(s[2]) == 0xBF) {
        s.erase(0, 3);
    }
    return s;
}


static SocketType parse_socket_type(const std::string& s) {
    if (s == "float") return SocketType::Float;
    if (s == "sampler2D") return SocketType::Sampler2D;
    return SocketType::Vec4;
}


static ChannelFormat parse_channel_format(const std::string& s) {
    if (s == "mono") return ChannelFormat::Mono;
    if (s == "uv")   return ChannelFormat::UV;
    if (s == "rgb")  return ChannelFormat::RGB;
    if (s == "rgba") return ChannelFormat::RGBA;
    return ChannelFormat::RGBA;
}


// Stage 2: map .node.json "pass_kind" string -> PassKind enum. Defaults to Compute if key missing/unrecognized; logs invalid keys but does not abort.
PassKind NodeRegistryLoader::parse_pass_kind(const std::string& s) {
    if (s == "upload")   return PassKind::Upload;
    if (s == "readback") return PassKind::Readback;
    return PassKind::Compute;
}


// Normalize CRLF → LF and strip blank lines from GLSL content.
static std::string clean_glsl(std::string s) {
    // Normalize CRLF → LF.
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n') { ++i; out += '\n'; }
        else out += s[i];
    }
    // Strip blank lines.
    std::istringstream ss(out);
    std::ostringstream filtered;
    std::string line;
    while (std::getline(ss, line)) {
        // Trim trailing \r (already handled above, but defensive).
        while (!line.empty() && line.back() == '\r') line.pop_back();
        bool blank = true;
        for (char c : line) { if (c != ' ' && c != '\t') { blank = false; break; } }
        if (!blank) filtered << line << '\n';
    }
    return filtered.str();
}

// Resolve includes once per node, dedup symbols globally via #ifndef guards.
static std::string assemble_glsl(const json& manifest,
                                 const fs::path& shader_path,
                                 const fs::path& glsl_dir,
                                 std::unordered_set<std::string>& /*loaded_includes*/) {
    std::ostringstream out;
    if (manifest.contains("includes")) {
        for (auto& inc : manifest["includes"]) {
            auto p = glsl_dir / (inc.get<std::string>() + ".glsl");
            if (!fs::exists(p)) {
                log_warn("missing include: " + p.string());
                continue;
            }
            // Guards in the file prevent duplicate-symbol errors when multiple nodes pull the same include — final shader still gets the body once per #include site. For now, emit once by relying on the guard macros.
            out << clean_glsl(slurp(p)) << "\n";
        }
    }
    out << clean_glsl(slurp(shader_path)) << "\n";
    return out.str();
}


int NodeRegistryLoader::load_from_directory(NodeLibrary& lib,
                                            const std::string& nodes_dir,
                                            const std::string& glsl_dir,
                                            std::string* error_out) {
    int count = 0;
    if (!fs::exists(nodes_dir)) {
        if (error_out) *error_out = "nodes dir not found: " + nodes_dir;
        return 0;
    }
    for (auto& entry : fs::directory_iterator(nodes_dir)) {
        if (entry.path().extension() != ".json") continue;
        if (entry.path().stem().extension() != ".node") continue;

        try {
            json manifest = json::parse(slurp(entry.path()));
            NodeType n;
            n.id = manifest.at("id").get<std::string>();
            n.display_name = manifest.value("display_name", n.id);
            n.description  = manifest.value("description", "");

            for (auto& p : manifest.value("params", json::array())) {
                NodeParam np;
                np.name          = p.at("name").get<std::string>();
                np.display_name  = p.value("display_name", np.name);
                np.description   = p.value("description", "");
                np.default_value  = p.value("default", 0.0f);
                np.min_value      = p.value("min", 0.0f);
                np.max_value      = p.value("max", 1.0f);
                np.soft_min_value = p.value("soft_min", p.value("min", 0.0f));
                np.soft_max_value = p.value("soft_max", p.value("max", 1.0f));
                np.step           = p.value("step", 0.0f);
                np.is_integer     = p.value("integer", false);
                np.as_socket      = p.value("as_socket", false);
                n.params.push_back(std::move(np));
            }           
            for (auto& s : manifest.value("inputs", json::array())) {
                if (!s.contains("name")) { log_error("Input missing name in " + n.id); continue; }
                Socket socket;
                socket.name   = s.at("name").get<std::string>();
                socket.type   = parse_socket_type(s.value("type", "vec4"));
                socket.format = parse_channel_format(s.value("format", "rgba"));
                // "default" may be a scalar (broadcast to all 4) or a 4-array.
                if (s.contains("default")) {
                    const auto& d = s.at("default");
                    if (d.is_array()) {
                        for (int k = 0; k < 4; ++k)
                            socket.default_vec4[k] = d.at(k).get<float>();
                    } else {
                        float v = d.get<float>();
                        socket.default_vec4 = {v, v, v, v};
                    }
                    socket.default_value = socket.default_vec4[0];
                }
                n.inputs.push_back(std::move(socket));
            }
            for (auto& s : manifest.value("outputs", json::array())) {
                if (!s.contains("name")) { log_error("Output missing name in " + n.id); continue; }
                Socket socket;
                socket.name = s.at("name").get<std::string>();
                socket.type = parse_socket_type(s.value("type", "vec4"));
                socket.format = parse_channel_format(s.value("format", "rgba"));
                n.outputs.push_back(std::move(socket));
            }

            for (auto& f : manifest.value("variant_flags", json::array())) {
                n.variant_flags.push_back(f.get<std::string>());
            }

            // Stage 2: how this node participates in chain fusion (7-kind reference: shader_assets/nodes/README.md).
            n.pass_kind = parse_pass_kind(manifest.value("pass_kind", "pure_pixel"));

            // Multi-pass: how many compute dispatches this node needs.
            n.pass_count         = manifest.value("pass_count", 1);
            n.intermediate_count = manifest.value("intermediate_count", 0);

            auto shader_file = manifest.at("shader").get<std::string>();
            std::unordered_set<std::string> loaded;
            n.glsl_function = assemble_glsl(
                manifest, fs::path(nodes_dir) / shader_file, glsl_dir, loaded);

            lib.add_public(std::move(n));
            ++count;
        } catch (const std::exception& e) {
            log_error(std::string("node parse failed: ") + entry.path().string() + " - " + e.what());
        }
    }
    log_info("NodeRegistryLoader: loaded " + std::to_string(count) + " nodes");
    return count;
}


} // namespace te