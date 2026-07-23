#include "engine/Engine.hpp"
#include "engine/EngineBarriers.hpp"
#include "engine/Logging.hpp"
#include "engine/graphfusion/FusedGroupCompiler.hpp"

namespace te {

bool Engine::populate_groups_(const fusion::CompiledGroupBundle& compiled,
                              const GraphIR& ir) {
    group_execs_.clear();
    if (!compiled.ok() || compiled.groups.empty()) return true;

    group_execs_.reserve(compiled.groups.size());

    // Track intermediate images for multi-pass nodes: node_id → {image, storage_slot, sampled_slot, vk_image}.
    struct IntermediateInfo {
        std::unique_ptr<Image> image;
        uint32_t storage_slot = BindlessTable::INVALID_SLOT;
        uint32_t sampled_slot = BindlessTable::INVALID_SLOT;
        VkImageView view = VK_NULL_HANDLE;
        VkImage    vk_image = VK_NULL_HANDLE;
    };
    std::unordered_map<NodeId, IntermediateInfo> intermediates;

    for (size_t gi = 0; gi < compiled.groups.size(); ++gi) {
        const auto& compiled_group = compiled.groups[gi];
        GroupExec group_exec;
        group_exec.nodes = compiled_group.nodes;
        group_exec.param_base_slot = compiled_group.param_base_slot;
        group_exec.output_node = compiled_group.output_node;
        group_exec.pass_index = compiled_group.pass_index;
        group_exec.pass_count = compiled_group.pass_count;

        // Compile GLSL → SPIR-V → pipeline.
        if (!compiled_group.glsl.empty()) {
            CompileResult r = compiler_.compile_compute_sync(
                compiled_group.glsl, "group_" + std::to_string(gi) + "_pass" + std::to_string(compiled_group.pass_index));
            if (!r.success) {
                log_warn("group " + std::to_string(gi) + " compile failed: " + r.error_log);
                set_error_(EngineErrorCode::ShaderCompile,
                           "group " + std::to_string(gi) + ": " + r.error_log,
                            EnginePhase::GraphCompileFinish, compiled_group.output_node);
                return false;
            }

            // Build specialization info for ts_pass_index (layout(constant_id = 0)).
            ShaderVariantKey spec_key{};
            spec_key.specialization[0] = compiled_group.pass_index;
            spec_key.specialization_count = (compiled_group.pass_count > 1) ? 1u : 0u;
            VkSpecializationMapEntry entries[8]{};
            VkSpecializationInfo spec_info{};
            const VkSpecializationInfo* spec_ptr = build_spec_info(spec_key, entries, spec_info);

            auto pipe = std::make_unique<ComputePipeline>();
            pipe->set_name("pipe_group_" + std::to_string(gi) + "_pass" + std::to_string(compiled_group.pass_index));
            if (!pipe->create(ctx_, r.spirv, bindless_.pipeline_layout(), spec_ptr)) {
                log_warn("group " + std::to_string(gi) + " pipeline creation failed");
                set_error_(EngineErrorCode::PipelineCreation,
                           "group pipeline creation failed",
                           EnginePhase::GraphCompileFinish, compiled_group.output_node);
                return false;
            }
            ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                                (uint64_t)pipe->pipeline(), pipe->name());
            group_exec.pipeline = std::move(pipe);
        }

        // Allocate output image for this group.
        VkFormat fmt = storage_format_to_vk(
            resolve_node_storage(*ir.find(compiled_group.output_node), node_lib_));

        // Multi-pass pass 0: output goes to intermediate image.
        if (compiled_group.pass_count > 1 && compiled_group.pass_index == 0) {
            IntermediateInfo intermediate_info;
            intermediate_info.image = std::make_unique<Image>();
            if (!intermediate_info.image->create(ctx_, output_w_, output_h_, fmt)) {
                log_warn("group " + std::to_string(gi) + " intermediate image creation failed");
                return false;
            }
            intermediate_info.storage_slot = bindless_.alloc_storage_slot();
            intermediate_info.sampled_slot = bindless_.alloc_sampled_slot();
            intermediate_info.view = intermediate_info.image->view();
            intermediate_info.vk_image = intermediate_info.image->image();
            bindless_.write_storage(ctx_, intermediate_info.storage_slot, intermediate_info.image->view());
            bindless_.write_sampled(ctx_, intermediate_info.sampled_slot, intermediate_info.image->view(), VK_IMAGE_LAYOUT_GENERAL);

            group_exec.output_image = std::move(intermediate_info.image);
            group_exec.out_storage_slot = intermediate_info.storage_slot;

            intermediates[compiled_group.output_node] = std::move(intermediate_info);
            log_info("INTERMEDIATE CREATED: key=node_" + std::to_string(compiled_group.output_node)
                + " from group " + std::to_string(gi) + " nodes=[" + [&]() {
                    std::string s; for (auto n : compiled_group.nodes) { if (!s.empty()) s+=","; s+=std::to_string(n); } return s;
                }() + "]");
        }
        // Multi-pass pass 1: output goes to final image, intermediate bound as sampled input.
        else if (compiled_group.pass_count > 1 && compiled_group.pass_index == 1) {
            group_exec.output_image = std::make_unique<Image>();
            if (!group_exec.output_image->create(ctx_, output_w_, output_h_, fmt)) {
                log_warn("group " + std::to_string(gi) + " output image creation failed");
                return false;
            }
            group_exec.out_storage_slot = bindless_.alloc_storage_slot();
            bindless_.write_storage(ctx_, group_exec.out_storage_slot, group_exec.output_image->view());

            // Find the intermediate from pass 0.
            auto it = intermediates.find(compiled_group.output_node);
            log_info("INTERMEDIATE LOOKUP (pass1 alloc): group " + std::to_string(gi)
                + " lookup=node_" + std::to_string(compiled_group.output_node)
                + " found=" + (it != intermediates.end() ? "YES" : "NO")
                + " inter_count=" + std::to_string(intermediates.size()));
            if (it != intermediates.end()) {
                // The blur node's GLSL reads from ext_inputs[0] (sampler2D).
                // We need to override ext_inputs[0] to point to the intermediate.
                // This will be patched after ext_inputs allocation below.
            }
        }
        // Single-pass or pass_count=1: normal output image.
        else {
            group_exec.output_image = std::make_unique<Image>();
            if (!group_exec.output_image->create(ctx_, output_w_, output_h_, fmt)) {
                log_warn("group " + std::to_string(gi) + " output image creation failed");
                return false;
            }
            group_exec.out_storage_slot = bindless_.alloc_storage_slot();
            if (group_exec.out_storage_slot == BindlessTable::INVALID_SLOT) {
                log_warn("group " + std::to_string(gi) + " storage slot allocation failed");
                return false;
            }
            bindless_.write_storage(ctx_, group_exec.out_storage_slot, group_exec.output_image->view());
        }

        // Allocate bindless sampled slots for external inputs.
        group_exec.ext_inputs.resize(compiled_group.external_inputs.size());
        for (size_t ei = 0; ei < compiled_group.external_inputs.size(); ++ei) {
            const auto& ext = compiled_group.external_inputs[ei];
            auto& input = group_exec.ext_inputs[ei];

            uint32_t slot = bindless_.alloc_sampled_slot();
            if (slot == BindlessTable::INVALID_SLOT)
                slot = dummy_slot_;

            // Multi-pass pass 1: override ext_input[0] to use intermediate.
            // This MUST happen before the connected/unconnected branching because
            // the blur node's sampler2D input IS connected (cross-group to image node),
            // but pass 1 needs to read from the intermediate, not the original image.
            if (compiled_group.pass_count > 1 && compiled_group.pass_index == 1) {
                auto intermediate_it = intermediates.find(ext.dst_node);
                log_info("INTERMEDIATE OVERRIDE: group " + std::to_string(gi)
                    + " ei=" + std::to_string(ei) + " ext.dst_node=" + std::to_string(ext.dst_node)
                    + " found=" + (intermediate_it != intermediates.end() ? "YES" : "NO")
                    + " ext.src_node=" + std::to_string(ext.src_node)
                    + " inter_count=" + std::to_string(intermediates.size()));
                if (intermediate_it != intermediates.end()) {
                    bindless_.write_sampled(ctx_, intermediate_it->second.sampled_slot,
                                            intermediate_it->second.view, VK_IMAGE_LAYOUT_GENERAL);
                    input.sampled_slot  = intermediate_it->second.sampled_slot;
                    input.sampled_image = intermediate_it->second.vk_image;
                    input.node_id       = ext.dst_node;
                    continue;
                }
            }

            // Unconnected sampler2D: the image was uploaded by the user and lives in image_registry_.
            if (ext.src_node == 0) {
                auto entry = image_registry_.find(ext.dst_node);
                bool found = (entry != image_registry_.end() && entry->second);
                VkImageView view = found ? entry->second->view() : dummy_image_.view();
                VkImage    image = found ? entry->second->image() : dummy_image_.image();

                bindless_.write_sampled(ctx_, slot, view, VK_IMAGE_LAYOUT_GENERAL);
                input.sampled_slot  = slot;
                input.sampled_image = image;
                input.node_id       = ext.dst_node;
                continue;
            }

            // Cross-group connection: check cache first.
            ResourceUUID resource_id{ext.src_node, ext.src_socket};
            input.resource = resource_id;

            auto cached = res_sampled_slot_.find(resource_id);
            if (cached != res_sampled_slot_.end()) {
                input.sampled_slot = cached->second;
                continue;
            }

            // Not cached — find the image to bind.
            res_sampled_slot_[resource_id] = slot;
            input.sampled_slot = slot;

            // Priority: ResourceManager → image_registry_ → previous group output → dummy.
            VkImageView view = dummy_image_.view();
            VkImage    image = dummy_image_.image();

            if (auto* r = resources_.get(resource_id)) {
                view  = r->view;
                image = r->image;
            } else if (auto entry = image_registry_.find(ext.src_node);
                       entry != image_registry_.end() && entry->second) {
                view  = entry->second->view();
                image = entry->second->image();
            } else {
                for (auto& prev : group_execs_) {
                    if (prev.output_node == ext.src_node) {
                        view  = prev.output_image->view();
                        image = prev.output_image->image();
                        break;
                    }
                }
            }

            bindless_.write_sampled(ctx_, slot, view, VK_IMAGE_LAYOUT_GENERAL);
            input.sampled_image = image;
        }

        group_execs_.push_back(std::move(group_exec));
    }

    use_groups_ = true;
    return true;
}


bool Engine::record_group_dispatches_(VkCommandBuffer cmd, const PushConstants& pc) {
    if (group_execs_.empty()) return false;

    for (size_t gi = 0; gi < group_execs_.size(); ++gi) {
        GroupExec& group_exec = group_execs_[gi];
        if (!group_exec.pipeline) continue;

        // Barrier: ensure previous group's output storage writes are visible
        // as sampled reads for this group's external inputs.
        for (auto& ext : group_exec.ext_inputs) {
            if (ext.sampled_image != VK_NULL_HANDLE) {
                barrier_compute_write_to_sampled(cmd, ext.sampled_image);
            }
        }

        // Transition output image to general layout for storage write.
        if (group_exec.output_image && group_exec.output_image->image() != VK_NULL_HANDLE) {
            transition_output_to_general(cmd, group_exec.output_image->image(),
                                          VK_IMAGE_LAYOUT_UNDEFINED, 0);
        }

        VkDescriptorSet set = bindless_.set();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, group_exec.pipeline->pipeline());

        PassPushConstants pass_push_constants{};
        pass_push_constants.global = pc;
        pass_push_constants.global.resolution_x = output_w_;
        pass_push_constants.global.resolution_y = output_h_;
        pass_push_constants.param_base_slot = 0;  // emitter uses ctx.param_base[] directly
        pass_push_constants.input_count = static_cast<uint32_t>(group_exec.ext_inputs.size());
        pass_push_constants.pass_index = group_exec.pass_index;

        for (uint32_t k = 0; k < group_exec.ext_inputs.size() && k < MAX_PASS_INPUTS; ++k)
            pass_push_constants.in_sampled_slots[k] = group_exec.ext_inputs[k].sampled_slot;
        for (uint32_t k = static_cast<uint32_t>(group_exec.ext_inputs.size()); k < MAX_PASS_INPUTS; ++k)
            pass_push_constants.in_sampled_slots[k] = dummy_slot_;

        pass_push_constants.out_storage_slots[0] = group_exec.out_storage_slot;
        for (uint32_t t = 1; t < MAX_PASS_OUTPUTS; ++t)
            pass_push_constants.out_storage_slots[t] = BindlessTable::INVALID_SLOT;

        pass_push_constants.param_ring_idx = param_write_idx_;

        vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PassPushConstants), &pass_push_constants);

        vkCmdDispatch(cmd, (output_w_ + 7) / 8, (output_h_ + 7) / 8, 1);
        ++last_dispatch_count_;

        // Barrier: output storage write → sampled read for next group's inputs.
        if (group_exec.output_image && group_exec.output_image->image() != VK_NULL_HANDLE)
            barrier_compute_write_to_sampled_read(cmd, group_exec.output_image->image());
    }

    return true;
}

} // namespace te
