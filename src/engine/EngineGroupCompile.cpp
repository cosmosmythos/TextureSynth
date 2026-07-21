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

    // Track intermediate images for multi-pass nodes: node_id → {image, storage_slot, sampled_slot, vkimg}.
    struct IntermediateInfo {
        std::unique_ptr<Image> image;
        uint32_t storage_slot = BindlessTable::INVALID_SLOT;
        uint32_t sampled_slot = BindlessTable::INVALID_SLOT;
        VkImageView view = VK_NULL_HANDLE;
        VkImage    vkimg = VK_NULL_HANDLE;
    };
    std::unordered_map<NodeId, IntermediateInfo> intermediates;

    for (size_t gi = 0; gi < compiled.groups.size(); ++gi) {
        const auto& cg = compiled.groups[gi];
        GroupExec ge;
        ge.nodes = cg.nodes;
        ge.param_base_slot = cg.param_base_slot;
        ge.output_node = cg.output_node;
        ge.pass_index = cg.pass_index;
        ge.pass_count = cg.pass_count;

        // Compile GLSL → SPIR-V → pipeline, with disk cache.
        if (!cg.glsl.empty()) {
            std::vector<uint32_t> spirv;
            bool cache_hit = false;

            if (cache_) {
                auto cached = cache_->load(cg.fused_key);
                if (cached.has_value()) {
                    spirv = std::move(*cached);
                    cache_hit = true;
                }
            }

            if (!cache_hit) {
                CompileResult r = compiler_.compile_compute_sync(
                    cg.glsl, "group_" + std::to_string(gi) + "_pass" + std::to_string(cg.pass_index));
                if (!r.success) {
                    log_warn("group " + std::to_string(gi) + " compile failed: " + r.error_log);
                    set_error_(EngineErrorCode::ShaderCompile,
                               "group " + std::to_string(gi) + ": " + r.error_log,
                               EnginePhase::GraphCompileFinish, cg.output_node);
                    return false;
                }
                spirv = std::move(r.spirv);
                if (cache_)
                    cache_->store(cg.fused_key, spirv);
            }

            {
                std::string msg = "shader_cache: ";
                msg += cache_hit ? "HIT" : "MISS";
                msg += " group=" + std::to_string(gi);
                msg += " hash=" + std::to_string(cg.fused_key.hash());
                log_info(msg);
            }

            // Build specialization info for ts_pass_index (layout(constant_id = 0)).
            ShaderVariantKey spec_key{};
            spec_key.specialization[0] = cg.pass_index;
            spec_key.specialization_count = (cg.pass_count > 1) ? 1u : 0u;
            VkSpecializationMapEntry entries[8]{};
            VkSpecializationInfo spec_info{};
            const VkSpecializationInfo* spec_ptr = build_spec_info(spec_key, entries, spec_info);

            auto pipe = std::make_unique<ComputePipeline>();
            pipe->set_name("pipe_group_" + std::to_string(gi) + "_pass" + std::to_string(cg.pass_index));
            if (!pipe->create(ctx_, spirv, bindless_.pipeline_layout(), spec_ptr)) {
                log_warn("group " + std::to_string(gi) + " pipeline creation failed");
                set_error_(EngineErrorCode::PipelineCreation,
                           "group pipeline creation failed",
                           EnginePhase::GraphCompileFinish, cg.output_node);
                return false;
            }
            ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                                (uint64_t)pipe->pipeline(), pipe->name());
            ge.pipeline = std::move(pipe);
        }

        // Allocate output image for this group.
        VkFormat fmt = storage_format_to_vk(
            resolve_node_storage(*ir.find(cg.output_node), node_lib_));

        // Multi-pass pass 0: output goes to intermediate image.
        if (cg.pass_count > 1 && cg.pass_index == 0) {
            IntermediateInfo inter;
            inter.image = std::make_unique<Image>();
            if (!inter.image->create(ctx_, output_w_, output_h_, fmt)) {
                log_warn("group " + std::to_string(gi) + " intermediate image creation failed");
                return false;
            }
            inter.storage_slot = bindless_.alloc_storage_slot();
            inter.sampled_slot = bindless_.alloc_sampled_slot();
            inter.view = inter.image->view();
            inter.vkimg = inter.image->image();
            bindless_.write_storage(ctx_, inter.storage_slot, inter.image->view());
            bindless_.write_sampled(ctx_, inter.sampled_slot, inter.image->view(), VK_IMAGE_LAYOUT_GENERAL);

            ge.output_image = std::move(inter.image);
            ge.out_storage_slot = inter.storage_slot;

            intermediates[cg.output_node] = std::move(inter);
        }
        // Multi-pass pass 1: output goes to final image, intermediate bound as sampled input.
        else if (cg.pass_count > 1 && cg.pass_index == 1) {
            ge.output_image = std::make_unique<Image>();
            if (!ge.output_image->create(ctx_, output_w_, output_h_, fmt)) {
                log_warn("group " + std::to_string(gi) + " output image creation failed");
                return false;
            }
            ge.out_storage_slot = bindless_.alloc_storage_slot();
            bindless_.write_storage(ctx_, ge.out_storage_slot, ge.output_image->view());

            // Find the intermediate from pass 0.
            // If the group was merged with a downstream Vec4 consumer,
            // cg.output_node is the tail (e.g. blend) but the intermediate
            // was stored under the multi-pass node's original ID.
            // Scan the group to find the right key.
            auto it = [&]() -> decltype(intermediates.begin()) {
                auto c = intermediates.find(cg.output_node);
                if (c != intermediates.end()) return c;
                for (NodeId nid : cg.nodes) { c = intermediates.find(nid); if (c != intermediates.end()) return c; }
                return intermediates.end();
            }();
        }
        // Single-pass or pass_count=1: normal output image.
        else {
            ge.output_image = std::make_unique<Image>();
            if (!ge.output_image->create(ctx_, output_w_, output_h_, fmt)) {
                log_warn("group " + std::to_string(gi) + " output image creation failed");
                return false;
            }
            ge.out_storage_slot = bindless_.alloc_storage_slot();
            if (ge.out_storage_slot == BindlessTable::INVALID_SLOT) {
                log_warn("group " + std::to_string(gi) + " storage slot allocation failed");
                return false;
            }
            bindless_.write_storage(ctx_, ge.out_storage_slot, ge.output_image->view());
        }

        // Allocate bindless sampled slots for external inputs.
        ge.ext_inputs.resize(cg.external_inputs.size());
        for (size_t ei = 0; ei < cg.external_inputs.size(); ++ei) {
            const auto& ext = cg.external_inputs[ei];
            auto& input = ge.ext_inputs[ei];

            uint32_t slot = bindless_.alloc_sampled_slot();
            if (slot == BindlessTable::INVALID_SLOT)
                slot = dummy_slot_;

            // Multi-pass pass 1: override ext_input[0] to use intermediate.
            // This MUST happen before the connected/unconnected branching because
            // the blur node's sampler2D input IS connected (cross-group to image node),
            // but pass 1 needs to read from the intermediate, not the original image.
            if (cg.pass_count > 1 && cg.pass_index == 1 && ei == 0) {
                auto inter_it = [&]() -> decltype(intermediates.begin()) {
                    auto c = intermediates.find(cg.output_node);
                    if (c != intermediates.end()) return c;
                    for (NodeId nid : cg.nodes) { c = intermediates.find(nid); if (c != intermediates.end()) return c; }
                    return intermediates.end();
                }();
                if (inter_it != intermediates.end()) {
                    bindless_.write_sampled(ctx_, inter_it->second.sampled_slot,
                                            inter_it->second.view, VK_IMAGE_LAYOUT_GENERAL);
                    input.sampled_slot  = inter_it->second.sampled_slot;
                    input.sampled_image = inter_it->second.vkimg;
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
            ResourceUUID rid{ext.src_node, ext.src_socket};
            input.resource = rid;

            auto cached = res_sampled_slot_.find(rid);
            if (cached != res_sampled_slot_.end()) {
                input.sampled_slot = cached->second;
                continue;
            }

            // Not cached — find the image to bind.
            res_sampled_slot_[rid] = slot;
            input.sampled_slot = slot;

            // Priority: ResourceManager → image_registry_ → previous group output → dummy.
            VkImageView view = dummy_image_.view();
            VkImage    image = dummy_image_.image();

            if (auto* r = resources_.get(rid)) {
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

        group_execs_.push_back(std::move(ge));
    }

    use_groups_ = true;
    return true;
}


bool Engine::record_group_dispatches_(VkCommandBuffer cmd, const PushConstants& pc) {
    if (group_execs_.empty()) return false;

    for (size_t gi = 0; gi < group_execs_.size(); ++gi) {
        GroupExec& ge = group_execs_[gi];
        if (!ge.pipeline) continue;

        // Barrier: ensure previous group's output storage writes are visible
        // as sampled reads for this group's external inputs.
        for (auto& ext : ge.ext_inputs) {
            if (ext.sampled_image != VK_NULL_HANDLE) {
                barrier_compute_write_to_sampled(cmd, ext.sampled_image);
            }
        }

        // Transition output image to general layout for storage write.
        if (ge.output_image && ge.output_image->image() != VK_NULL_HANDLE) {
            transition_output_to_general(cmd, ge.output_image->image(),
                                          VK_IMAGE_LAYOUT_UNDEFINED, 0);
        }

        VkDescriptorSet set = bindless_.set();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ge.pipeline->pipeline());

        PassPushConstants ppc{};
        ppc.global = pc;
        ppc.global.resolution_x = output_w_;
        ppc.global.resolution_y = output_h_;
        ppc.param_base_slot = 0;  // emitter uses ctx.param_base[] directly
        ppc.input_count = static_cast<uint32_t>(ge.ext_inputs.size());
        ppc.pass_index = ge.pass_index;

        for (uint32_t k = 0; k < ge.ext_inputs.size() && k < MAX_PASS_INPUTS; ++k)
            ppc.in_sampled_slots[k] = ge.ext_inputs[k].sampled_slot;
        for (uint32_t k = static_cast<uint32_t>(ge.ext_inputs.size()); k < MAX_PASS_INPUTS; ++k)
            ppc.in_sampled_slots[k] = dummy_slot_;

        ppc.out_storage_slots[0] = ge.out_storage_slot;
        for (uint32_t t = 1; t < MAX_PASS_OUTPUTS; ++t)
            ppc.out_storage_slots[t] = BindlessTable::INVALID_SLOT;

        ppc.param_ring_idx = param_write_idx_;

        vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PassPushConstants), &ppc);

        vkCmdDispatch(cmd, (output_w_ + 7) / 8, (output_h_ + 7) / 8, 1);
        ++last_dispatch_count_;

        // Barrier: output storage write → sampled read for next group's inputs.
        if (ge.output_image && ge.output_image->image() != VK_NULL_HANDLE)
            barrier_compute_write_to_sampled_read(cmd, ge.output_image->image());
    }

    return true;
}

} // namespace te
