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

    for (size_t gi = 0; gi < compiled.groups.size(); ++gi) {
        const auto& cg = compiled.groups[gi];
        GroupExec ge;
        ge.nodes = cg.nodes;
        ge.param_base_slot = cg.param_base_slot;
        ge.output_node = cg.output_node;

        // Compile GLSL → SPIR-V → pipeline.
        if (!cg.glsl.empty()) {
            CompileResult r = compiler_.compile_compute_sync(
                cg.glsl, "group_" + std::to_string(gi));
            if (!r.success) {
                log_warn("group " + std::to_string(gi) + " compile failed: " + r.error_log);
                set_error_(EngineErrorCode::ShaderCompile,
                           "group " + std::to_string(gi) + ": " + r.error_log,
                           EnginePhase::GraphCompileFinish, cg.output_node);
                return false;
            }
            auto pipe = std::make_unique<ComputePipeline>();
            pipe->set_name("pipe_group_" + std::to_string(gi));
            if (!pipe->create(ctx_, r.spirv, bindless_.pipeline_layout(), nullptr)) {
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
        ge.output_image = std::make_unique<Image>();
        VkFormat fmt = storage_format_to_vk(
            resolve_node_storage(*ir.find(cg.output_node), node_lib_));
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

        // Allocate bindless sampled slots for external inputs.
        ge.ext_inputs.resize(cg.external_inputs.size());
        for (size_t ei = 0; ei < cg.external_inputs.size(); ++ei) {
            const auto& ext = cg.external_inputs[ei];
            ResourceUUID rid{ext.src_node, ext.src_socket};
            ge.ext_inputs[ei].resource = rid;

            auto sit = res_sampled_slot_.find(rid);
            if (sit != res_sampled_slot_.end()) {
                ge.ext_inputs[ei].sampled_slot = sit->second;
            } else {
                uint32_t slot = bindless_.alloc_sampled_slot();
                if (slot == BindlessTable::INVALID_SLOT) {
                    ge.ext_inputs[ei].sampled_slot = dummy_slot_;
                } else {
                    ge.ext_inputs[ei].sampled_slot = slot;
                    res_sampled_slot_[rid] = slot;

                    // Find the image to bind: try resource system first, then external uploads.
                    auto* r = resources_.get(rid);
                    if (r) {
                        bindless_.write_sampled(ctx_, slot, r->view, VK_IMAGE_LAYOUT_GENERAL);
                    } else {
                        auto eit = image_registry_.find(ext.src_node);
                        if (eit != image_registry_.end() && eit->second) {
                            bindless_.write_sampled(ctx_, slot,
                                                    eit->second->view(), VK_IMAGE_LAYOUT_GENERAL);
                        } else {
                            // Check if it's a previous group's output.
                            bool found = false;
                            for (auto& prev : group_execs_) {
                                if (prev.output_node == ext.src_node) {
                                    bindless_.write_sampled(ctx_, slot,
                                                            prev.output_image->view(),
                                                            VK_IMAGE_LAYOUT_GENERAL);
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                                bindless_.write_sampled(ctx_, slot,
                                                        dummy_image_.view(), VK_IMAGE_LAYOUT_GENERAL);
                        }
                    }
                }
            }
        }

        group_execs_.push_back(std::move(ge));
    }

    use_groups_ = true;
    log_info("populate_groups_: " + std::to_string(group_execs_.size()) + " groups compiled");
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
            auto* r = resources_.get(ext.resource);
            if (r && r->layout != VK_IMAGE_LAYOUT_GENERAL) {
                barrier_compute_write_to_sampled(cmd, r->image);
                r->layout = VK_IMAGE_LAYOUT_GENERAL;
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
        ppc.param_base_slot = ge.param_base_slot;
        ppc.input_count = static_cast<uint32_t>(ge.ext_inputs.size());
        ppc.pass_index = 0;

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
