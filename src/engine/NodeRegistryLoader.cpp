#include "engine/NodeRegistryLoader.hpp"
#include "engine/Logging.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <nlohmann/json.hpp>   // header-only

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
    if (s == "id")   return ChannelFormat::ID;
    if (s == "metadata")   return ChannelFormat::Metadata;
    return ChannelFormat::RGBA;
}


// Stage 2: map .node.json "pass_kind" string -> PassKind enum. Defaults to PurePixel if key missing/unrecognized; logs invalid keys but does not abort.
PassKind NodeRegistryLoader::parse_pass_kind(const std::string& s) {
    if (s == "pure_pixel")   return PassKind::PurePixel;
    if (s == "boundary")     return PassKind::Boundary;
    if (s == "reduction")    return PassKind::Reduction;
    if (s == "feedback")     return PassKind::Feedback;
    if (s == "upload")       return PassKind::Upload;
    if (s == "readback")     return PassKind::Readback;
    if (s == "debug_preview") return PassKind::DebugPreview;
    if (!s.empty()) {
        log_warn("Unknown pass_kind '" + s + "', defaulting to pure_pixel");
    }
    return PassKind::PurePixel;
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
            out << slurp(p) << "\n";
        }
    }
    out << slurp(shader_path) << "\n";
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
            json m = json::parse(slurp(entry.path()));
            NodeType n;
            n.id = m.at("id").get<std::string>();
            n.display_name = m.value("display_name", n.id);
            n.description  = m.value("description", "");

            for (auto& p : m.value("params", json::array())) {
                NodeParam np;
                np.name          = p.at("name").get<std::string>();
                np.display_name  = p.value("display_name", np.name);
                np.description   = p.value("description", "");
                np.default_value = p.value("default", 0.0f);
                np.min_value     = p.value("min", 0.0f);
                np.max_value     = p.value("max", 1.0f);
                np.step          = p.value("step", 0.0f);
                np.is_integer    = p.value("integer", false);
                np.as_socket     = p.value("as_socket", false);
                n.params.push_back(std::move(np));
            }           
            for (auto& s : m.value("inputs", json::array())) {
                if (!s.contains("name")) { log_error("Input missing name in " + n.id); continue; }
                Socket socket;
                socket.name   = s.at("name").get<std::string>();
                socket.type   = parse_socket_type(s.value("type", "vec4"));
                socket.format = parse_channel_format(s.value("format", "rgba"));
                n.inputs.push_back(std::move(socket));
            }
            for (auto& s : m.value("outputs", json::array())) {
                if (!s.contains("name")) { log_error("Output missing name in " + n.id); continue; }
                Socket socket;
                socket.name = s.at("name").get<std::string>();
                socket.type = parse_socket_type(s.value("type", "vec4"));
                socket.format = parse_channel_format(s.value("format", "rgba"));
                n.outputs.push_back(std::move(socket));
            }

            for (auto& f : m.value("variant_flags", json::array())) {
                n.variant_flags.push_back(f.get<std::string>());
            }

            // Opt-in flag for the format post-process (Graph.hpp:is_format_sensitive). Only noise generators set this. Default false.
            n.is_format_sensitive = m.value("format_sensitive", false);

            // Stage 2: how this node participates in chain fusion (7-kind reference: shader_assets/nodes/README.md).
            n.pass_kind = parse_pass_kind(m.value("pass_kind", "pure_pixel"));

            auto shader_file = m.at("shader").get<std::string>();
            std::unordered_set<std::string> loaded;
            n.glsl_function = assemble_glsl(
                m, fs::path(nodes_dir) / shader_file, glsl_dir, loaded);

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