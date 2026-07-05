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
        log_info("[mem]   group " + std::to_string(gi) + " output: node="
                 + std::to_string(cg.output_node)
                 + " " + std::to_string(output_w_) + "x" + std::to_string(output_h_)
                 + " fmt=" + std::to_string(fmt)
                 + " " + std::to_string((size_t)output_w_ * output_h_ * vk_format_bytes(fmt) / 1024) + " KB"
                 + " storage_slot=" + std::to_string(ge.out_storage_slot)
                 + " nodes_in_group=" + std::to_string(cg.nodes.size()));

        // Allocate bindless sampled slots for external inputs.
        ge.ext_inputs.resize(cg.external_inputs.size());
        for (size_t ei = 0; ei < cg.external_inputs.size(); ++ei) {
            const auto& ext = cg.external_inputs[ei];
            auto& input = ge.ext_inputs[ei];

            uint32_t slot = bindless_.alloc_sampled_slot();
            if (slot == BindlessTable::INVALID_SLOT)
                slot = dummy_slot_;
            log_info("[ext_slot_mismatch] group=" + std::to_string(gi)
                     + " ei=" + std::to_string(ei)
                     + " glsl_ext_slot=" + std::to_string(ext.slot)
                     + " bindless_alloc_slot=" + std::to_string(slot)
                     + " dst_node=" + std::to_string(ext.dst_node)
                     + " dst_socket=" + std::to_string(ext.dst_socket)
                     + " src_node=" + std::to_string(ext.src_node));

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
            log_info("[mem]   group " + std::to_string(gi) + " ext_input: node="
                     + std::to_string(ext.dst_node)
                     + " socket=" + std::to_string(ext.dst_socket)
                     + " sampled_slot=" + std::to_string(slot)
                     + " vkview=" + std::to_string((uint64_t)view)
                     + " vkimg=" + std::to_string((uint64_t)image)
                     + (found ? " [from image_registry]" : " [FALLBACK dummy]"));
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
        ppc.pass_index = 0;

        log_info("[dispatch] group output_node=" + std::to_string(ge.output_node)
                 + " param_base_slot=" + std::to_string(ge.param_base_slot)
                 + " param_ring_idx=" + std::to_string(param_write_idx_)
                 + " res_x=" + std::to_string(output_w_)
                 + " res_y=" + std::to_string(output_h_)
                 + " ext_count=" + std::to_string(ge.ext_inputs.size()));

        for (uint32_t k = 0; k < ge.ext_inputs.size() && k < MAX_PASS_INPUTS; ++k)
            ppc.in_sampled_slots[k] = ge.ext_inputs[k].sampled_slot;
        for (uint32_t k = static_cast<uint32_t>(ge.ext_inputs.size()); k < MAX_PASS_INPUTS; ++k)
            ppc.in_sampled_slots[k] = dummy_slot_;

        if (!ge.ext_inputs.empty()) {
            log_info("[dispatch] in_sampled_slots[0]=" + std::to_string(ppc.in_sampled_slots[0])
                     + " dummy_slot=" + std::to_string(dummy_slot_)
                     + " out_storage_slot=" + std::to_string(ppc.out_storage_slots[0]));
        }

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
