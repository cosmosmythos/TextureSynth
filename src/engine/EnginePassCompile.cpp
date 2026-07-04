#include "engine/Engine.hpp"
#include "engine/Logging.hpp"
#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace te {

void Engine::assign_bindless_slots_(PassExec& pe) {
    pe.output_count = static_cast<uint32_t>(pe.output_resources.size());
    for (uint32_t t = 0; t < pe.output_count && t < MAX_PASS_OUTPUTS; ++t) {
        const ResourceUUID& rid = pe.output_resources[t];
        auto it_o = res_storage_slot_.find(rid);
        if (it_o == res_storage_slot_.end()) {
            uint32_t slot = bindless_.alloc_storage_slot();
            if (slot == BindlessTable::INVALID_SLOT) {
                log_error("Bindless storage exhausted");
                slot = 0;
            }
            else {
                res_storage_slot_[rid] = slot;
                if (auto* r = resources_.get(rid))
                    bindless_.write_storage(ctx_, slot, r->view);
                else
                    bindless_.write_storage(ctx_, slot, dummy_image_.view());
            }
            pe.out_storage_slots[t] = slot;
        }
        else {
            pe.out_storage_slots[t] = it_o->second;
        }
    }

    pe.input_count = static_cast<uint32_t>(pe.input_resources.size());
    for (uint32_t i = 0; i < pe.input_count && i < MAX_PASS_INPUTS; ++i) {
        const ResourceUUID rid = pe.input_resources[i];
        uint32_t slot = BindlessTable::INVALID_SLOT;

        if (rid.node_id == 0) {
            auto eit = image_registry_.find(pe.node_id);
            if (eit != image_registry_.end() && eit->second) {
                auto sit = ext_sampled_slot_.find(pe.node_id);
                if (sit == ext_sampled_slot_.end()) {
                    slot = bindless_.alloc_sampled_slot();
                    if (slot == BindlessTable::INVALID_SLOT) slot = dummy_slot_;
                    else {
                        ext_sampled_slot_[pe.node_id] = slot;
                        bindless_.write_sampled(ctx_, slot, eit->second->view(),
                                                VK_IMAGE_LAYOUT_GENERAL);
                    }
                } else slot = sit->second;
            } else slot = dummy_slot_;
        } else {
            auto sit = res_sampled_slot_.find(rid);
            if (sit == res_sampled_slot_.end()) {
                slot = bindless_.alloc_sampled_slot();
                if (slot == BindlessTable::INVALID_SLOT) slot = dummy_slot_;
                else {
                    res_sampled_slot_[rid] = slot;
                    if (auto* r = resources_.get(rid))
                        bindless_.write_sampled(ctx_, slot, r->view, VK_IMAGE_LAYOUT_GENERAL);
                    else
                        bindless_.write_sampled(ctx_, slot, dummy_image_.view(), VK_IMAGE_LAYOUT_GENERAL);
                }
            } else slot = sit->second;
        }
        pe.in_sampled_slots[i] = slot;
    }
    for (uint32_t i = pe.input_count; i < MAX_PASS_INPUTS; ++i)
        pe.in_sampled_slots[i] = dummy_slot_;
}


bool Engine::create_pass_pipeline_(PassExec& pe,
                                   NodeId node_id,
                                   const std::string& type_id,
                                   const std::string& error_prefix,
                                   const ShaderVariantKey& variant_key,
                                   const std::vector<uint32_t>& spirv) {
    auto pipe = std::make_unique<ComputePipeline>();
    const std::string pipe_name = "pipe_node_" + std::to_string(node_id) + "_" + type_id;
    pipe->set_name(pipe_name);
    VkSpecializationMapEntry entries[8];
    VkSpecializationInfo     spec{};
    const VkSpecializationInfo* spec_ptr = build_spec_info(variant_key, entries, spec);
    if (!pipe->create(ctx_, spirv, bindless_.pipeline_layout(), spec_ptr)) {
        set_error_(EngineErrorCode::PipelineCreation, error_prefix,
                   EnginePhase::GraphCompileFinish, node_id);
        return false;
    }
    pe.pipeline = std::move(pipe);
    ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                        (uint64_t)pe.pipeline->pipeline(),
                        pe.pipeline->name());
    return true;
}


void Engine::poll_pending_compiles() {
    TE_GUARD_READY(;);

    poll_completed_uploads_();
    tick_retired();
    resources_.tick(ctx_);
    poll_timestamps_();

    if (!pending_active_) return;

    for (auto& pp : pending_passes_) {
        if (pp.kind != PassKind::Compute) continue;
        if (pp.bypassed || pp.chain_member) continue;
        if (!pp.fut.valid()) continue;
        if (pp.fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return;
    }

    if (pending_generation_ != compile_generation_) {
        for (auto& pp : pending_passes_) if (pp.fut.valid()) pp.fut.get();
        pending_passes_.clear();
        pending_active_     = false;
        pending_generation_ = 0;
        log_info("discarded stale PassPlan compile result");
        return;
    }

    std::vector<PassExec> new_passes;
    new_passes.reserve(pending_passes_.size());

    for (auto& pp : pending_passes_) {
        PassExec pe;
        pe.node_id          = pp.node_id;
        pe.output_resources = pp.output_resources;
        pe.input_resources  = std::move(pp.input_resources);
        pe.input_formats    = std::move(pp.input_formats);
        pe.bypassed         = pp.bypassed;
        pe.param_base_slot  = pp.param_base_slot;

        if (pp.kind == PassKind::Compute) {
            if (pp.bypassed || pp.chain_member) {
            } else {
                CompileResult r = pp.fut.get();
                if (!r.success) {
                    set_error_(EngineErrorCode::ShaderCompile,
                               "Node " + pp.name + ": " + r.error_log,
                               EnginePhase::GraphCompileFinish, pp.node_id);
                    pending_passes_.clear();
                    pending_active_     = false;
                    pending_generation_ = 0;
                    return;
                }
                cache_->store(pp.variant_key, r.spirv);
                if (!create_pass_pipeline_(pe, pp.node_id, pp.type_id,
                        "pipeline creation failed for " + pp.name,
                        pp.variant_key, r.spirv)) {
                    pending_passes_.clear();
                    pending_active_     = false;
                    pending_generation_ = 0;
                    return;
                }
            }
            assign_bindless_slots_(pe);
        }
        new_passes.push_back(std::move(pe));
    }

    retire_all_passes_();
    passes_                = std::move(new_passes);
    final_output_resource_ = pending_final_output_;
    installed_generation_  = pending_generation_;
    clear_error();

    populate_chains_(pending_pass_plan_);
    pending_passes_.clear();
    pending_active_     = false;
    pending_generation_ = 0;
    log_info("PassPlan installed (compiled), passes=" + std::to_string(passes_.size())
             + " generation=" + std::to_string(installed_generation_));
}


void Engine::poll_completed_uploads_() {
    auto completed = uploader_.poll(ctx_);
    for (auto& c : completed) {
        image_registry_[c.node_id] = std::move(c.image);
        images_needing_acquire_[c.node_id] = (ctx_.has_dedicated_transfer());
        pending_uploads_.erase(
            std::remove_if(pending_uploads_.begin(), pending_uploads_.end(),
                [&](const PendingUpload& p){ return p.ticket == c.ticket; }),
            pending_uploads_.end());

        auto sit = ext_sampled_slot_.find(c.node_id);
        uint32_t slot = (sit == ext_sampled_slot_.end())
                      ? bindless_.alloc_sampled_slot()
                      : sit->second;
        ext_sampled_slot_[c.node_id] = slot;
        if (auto* img = image_registry_[c.node_id].get())
            bindless_.write_sampled(ctx_, slot, img->view(), VK_IMAGE_LAYOUT_GENERAL);

        for (auto& pe : passes_) {
            if (pe.node_id != c.node_id) continue;
            for (uint32_t i = 0; i < pe.input_count && i < MAX_PASS_INPUTS; ++i)
                if (pe.input_resources[i].node_id == 0)
                    pe.in_sampled_slots[i] = slot;
        }
        mark_downstream_dirty_(c.node_id);
    }
}


void Engine::retire_all_passes_() {
    for (auto& pe : passes_) {
        retired_passes_.push_back({
            std::move(pe.pipeline),
            MAX_FRAMES_IN_FLIGHT + 2
        });
    }
    passes_.clear();
    for (auto& ce : chain_execs_) {
        if (ce.pipeline) {
            retired_passes_.push_back({
                std::move(ce.pipeline),
                MAX_FRAMES_IN_FLIGHT + 2
            });
        }
        // Retire sub-pass pipelines.
        for (auto& sp : ce.sub_pipelines) {
            if (sp) {
                retired_passes_.push_back({
                    std::move(sp),
                    MAX_FRAMES_IN_FLIGHT + 2
                });
            }
        }
        // Retire intermediate images and free their bindless slots.
        for (auto& intermedi : ce.intermediates) {
            if (intermedi.image) {
                RetiredImage ri;
                ri.img = std::move(intermedi.image);
                ri.frames_remaining = MAX_FRAMES_IN_FLIGHT + 2;
                retired_images_.push_back(std::move(ri));
            }
            if (intermedi.sampled_slot != BindlessTable::INVALID_SLOT)
                bindless_.free_sampled_slot(intermedi.sampled_slot);
            if (intermedi.storage_slot != BindlessTable::INVALID_SLOT)
                bindless_.free_storage_slot(intermedi.storage_slot);
        }
    }
    chain_execs_.clear();
    chain_id_of_pass_.clear();

    // Retire group exec pipelines and output images.
    for (auto& ge : group_execs_) {
        if (ge.pipeline) {
            retired_passes_.push_back({
                std::move(ge.pipeline),
                MAX_FRAMES_IN_FLIGHT + 2
            });
        }
        if (ge.output_image) {
            RetiredImage ri;
            ri.img = std::move(ge.output_image);
            ri.frames_remaining = MAX_FRAMES_IN_FLIGHT + 2;
            retired_images_.push_back(std::move(ri));
        }
        for (auto& ext : ge.ext_inputs) {
            // sampled slots for group inputs are freed by res_sampled_slot_ cleanup
        }
    }
    group_execs_.clear();
    use_groups_ = false;
}


void Engine::populate_chains_(const PassPlan& plan) {
    for (auto& ce : chain_execs_) {
        if (ce.pipeline) ce.pipeline->destroy(ctx_);
    }
    chain_execs_.clear();
    chain_execs_.reserve(plan.chains.size());
    chain_id_of_pass_ = plan.chain_index_of_pass;
    chain_id_of_pass_.resize(passes_.size(), UINT32_MAX);

    std::unordered_map<NodeId, uint32_t> pass_idx_by_node;
    pass_idx_by_node.reserve(plan.passes.size());
    for (uint32_t i = 0; i < (uint32_t)plan.passes.size(); ++i)
        pass_idx_by_node[plan.passes[i].node_id] = i;

    std::unordered_map<ChainId, std::vector<NodeId>> membership;
    membership.reserve(plan.chains.size());

    for (size_t ci = 0; ci < plan.chains.size(); ++ci) {
        const auto& ch = plan.chains[ci];
        ChainExec ce;
        ce.bypassed = ch.bypassed;

        std::vector<uint32_t> member_pis;
        for (NodeId n : ch.nodes) {
            auto it = pass_idx_by_node.find(n);
            if (it != pass_idx_by_node.end()) member_pis.push_back(it->second);
        }
        ce.member_pass_indices = member_pis;
        if (!member_pis.empty()) {
            ce.head_pass_index = member_pis.front();
            ce.tail_pass_index = member_pis.back();
            // Override head pass param_base_slot to chain base (min global slot).
            passes_[member_pis.front()].param_base_slot = ch.param_base_slot;
        }

        uint32_t ext_slot = 0;
        for (size_t mi = 0; mi < member_pis.size(); ++mi) {
            const auto& pe = passes_[member_pis[mi]];
            const auto* vn = current_ir_.find(ch.nodes[mi]);
            const auto* type = vn ? node_lib_.find(vn->type_id) : nullptr;
            if (!type) continue;
            const uint32_t inputs_n = (uint32_t)type->inputs.size();
            for (uint32_t s = 0; s < inputs_n && ext_slot < MAX_PASS_INPUTS; ++s) {
                NodeId src_node = (s < pe.input_resources.size()) ? pe.input_resources[s].node_id : 0;
                if (src_node == 0) {
                    if (s < type->inputs.size() && type->inputs[s].type == SocketType::Sampler2D) {
                        // Unconnected Sampler2D: not baked as constant, gets dummy slot in ext_slot
                    } else {
                        continue; // unconnected float/vec4 — baked as constant, no ext_slot
                    }
                }
                bool is_chain_member = false;
                for (NodeId cn : ch.nodes) {
                    if (cn == src_node) { is_chain_member = true; break; }
                }
                if (is_chain_member) continue;
                ce.chain_in_sampled_slots[ext_slot] = pe.in_sampled_slots[s];
                ce.slot_sources[ext_slot] = {(uint32_t)mi, s};
                ++ext_slot;
            }
            uint32_t as_socket_idx = 0;
            for (uint32_t p = 0; p < (uint32_t)type->params.size(); ++p) {
                if (!type->params[p].as_socket) continue;
                uint32_t socket_idx = inputs_n + as_socket_idx;
                if (socket_idx < pe.input_resources.size() &&
                    pe.input_resources[socket_idx].node_id != 0 &&
                    ext_slot < MAX_PASS_INPUTS) {
                    NodeId as_src = pe.input_resources[socket_idx].node_id;
                    bool as_in_chain = false;
                    for (NodeId cn : ch.nodes) {
                        if (cn == as_src) { as_in_chain = true; break; }
                    }
                    if (as_in_chain) { ++as_socket_idx; continue; }
                    ce.chain_in_sampled_slots[ext_slot] = pe.in_sampled_slots[socket_idx];
                    ce.slot_sources[ext_slot] = {(uint32_t)mi, socket_idx};
                    ++ext_slot;
                }
                ++as_socket_idx;
            }
        }
        ce.slot_source_count = ext_slot;

        if (!ch.glsl.empty() && !ch.bypassed) {
            std::optional<std::vector<uint32_t>> blob;
            if (cache_) blob = cache_->load(ch.variant_key);
            if (!blob) {
                CompileResult r = compiler_.compile_compute_sync(
                    ch.glsl, "chain_" + std::to_string(ci));
                if (r.success) {
                    if (cache_) cache_->store(ch.variant_key, r.spirv);
                    blob = std::move(r.spirv);
                } else {
                    log_warn("chain compile failed: " + r.error_log);
                }
            }
            if (blob) {
                auto pipe = std::make_unique<ComputePipeline>();
                const std::string pipe_name = "pipe_chain_" + std::to_string(ci);
                pipe->set_name(pipe_name);
                if (pipe->create(ctx_, *blob, bindless_.pipeline_layout(), nullptr)) {
                    ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                                        (uint64_t)pipe->pipeline(), pipe->name());
                    ce.pipeline = std::move(pipe);
                } else {
                    log_warn("chain pipeline creation failed: chain " + std::to_string(ci));
                }
            }
        }

        // Multi-pass: allocate intermediate images and compile per-sub-pass pipelines.
        if (ch.sub_pass_count > 1 && !ch.bypassed) {
            ce.sub_pass_count = ch.sub_pass_count;

            // Allocate intermediate images (pass_count - 1 temps between sub-passes).
            // Format must match the GLSL u_storage[] layout qualifier — single
            // source of truth is resolve_node_storage() (same as ResourceManager).
            VkFormat intermedi_fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
            if (auto* vn = current_ir_.find(ch.nodes[0])) {
                intermedi_fmt = storage_format_to_vk(
                    resolve_node_storage(*vn, node_lib_));
            }
            for (uint32_t i = 0; i < ch.intermediate_count; ++i) {
                ChainExec::Intermediate intermedi;
                intermedi.image = std::make_unique<Image>();
                if (!intermedi.image->create(ctx_, output_w_, output_h_, intermedi_fmt)) {
                    log_warn("[populate_chains] intermediate image creation failed for chain "
                             + std::to_string(ci) + " intermedi[" + std::to_string(i) + "]");
                }
                intermedi.sampled_slot = bindless_.alloc_sampled_slot();
                intermedi.storage_slot = bindless_.alloc_storage_slot();
                if (intermedi.sampled_slot != BindlessTable::INVALID_SLOT && intermedi.image->image() != VK_NULL_HANDLE)
                    bindless_.write_sampled(ctx_, intermedi.sampled_slot,
                                            intermedi.image->view(), VK_IMAGE_LAYOUT_GENERAL);
                if (intermedi.storage_slot != BindlessTable::INVALID_SLOT && intermedi.image->image() != VK_NULL_HANDLE)
                    bindless_.write_storage(ctx_, intermedi.storage_slot,
                                            intermedi.image->view());
                ce.intermediates.push_back(std::move(intermedi));
            }

            // Log intermediate slot assignments.
            for (uint32_t i = 0; i < (uint32_t)ce.intermediates.size(); ++i) {
                log_info("[blur-debug] chain " + std::to_string(ci)
                         + " intermediate[" + std::to_string(i) + "]"
                         + " sampled_slot=" + std::to_string(ce.intermediates[i].sampled_slot)
                         + " storage_slot=" + std::to_string(ce.intermediates[i].storage_slot));
            }

            // Compile per-sub-pass pipelines.
            ce.sub_pipelines.reserve(ch.sub_pass_count);
            for (uint32_t sp = 0; sp < ch.sub_pass_count; ++sp) {
                std::optional<std::vector<uint32_t>> blob;
                if (cache_) blob = cache_->load(ch.sub_pass_variant_keys[sp]);
                if (!blob) {
                    CompileResult r = compiler_.compile_compute_sync(
                        ch.sub_pass_glsl[sp],
                        "chain_" + std::to_string(ci) + "_sp" + std::to_string(sp));
                    if (r.success) {
                        if (cache_) cache_->store(ch.sub_pass_variant_keys[sp], r.spirv);
                        blob = std::move(r.spirv);
                    } else {
                        log_warn("sub-pass compile failed: chain " + std::to_string(ci)
                                 + " sp" + std::to_string(sp) + ": " + r.error_log);
                    }
                }
                if (blob) {
                    auto pipe = std::make_unique<ComputePipeline>();
                    const std::string pipe_name = "pipe_chain_" + std::to_string(ci) + "_sp" + std::to_string(sp);
                    pipe->set_name(pipe_name);
                    VkSpecializationMapEntry entries[8];
                    VkSpecializationInfo     spec{};
                    entries[0] = {0, 0, sizeof(uint32_t)};
                    spec.mapEntryCount = 1;
                    spec.pMapEntries = entries;
                    spec.dataSize = sizeof(uint32_t);
                    spec.pData = &sp;
                    log_info("[blur-debug] chain " + std::to_string(ci)
                             + " compiling sub-pass sp=" + std::to_string(sp)
                             + " specialization ts_pass_index=" + std::to_string(sp)
                             + " pipeline=" + pipe_name);
                    if (pipe->create(ctx_, *blob, bindless_.pipeline_layout(), &spec)) {
                        ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                                            (uint64_t)pipe->pipeline(), pipe->name());
                        ce.sub_pipelines.push_back(std::move(pipe));
                    } else {
                        log_warn("sub-pass pipeline creation failed: chain " + std::to_string(ci)
                                 + " sp" + std::to_string(sp));
                    }
                }
            }

            // Set up per-sub-pass output storage slots.
            // Intermediate sub-passes write to intermediates[sp].
            // Last sub-pass writes to the chain's final output.
            const PassExec& tail = passes_[ce.tail_pass_index];
            log_info("[blur-debug] chain " + std::to_string(ci)
                     + " tail.out_storage_slots[0]=" + std::to_string(tail.out_storage_slots[0])
                     + " tail.out_storage_slots[1]=" + std::to_string(tail.out_storage_slots[1])
                     + " tail.output_count=" + std::to_string(tail.output_count));
            for (uint32_t sp = 0; sp < ch.sub_pass_count; ++sp) {
                if (sp == ch.sub_pass_count - 1) {
                    // Last sub-pass: write to final output.
                    for (uint32_t t = 0; t < tail.output_count && t < MAX_PASS_OUTPUTS; ++t)
                        ce.sub_out_storage_slots[sp][t] = tail.out_storage_slots[t];
                } else {
                    // Intermediate: write to intermediate[sp].
                    if (sp < ce.intermediates.size())
                        ce.sub_out_storage_slots[sp][0] = ce.intermediates[sp].storage_slot;
                }
            }
        }

        chain_execs_.push_back(std::move(ce));
        membership[(ChainId)ci] = ch.nodes;
    }

    dirty_set_.set_chain_membership(std::move(membership));
}


void Engine::tick_retired() {
    for (auto& r : retired_images_) if (r.frames_remaining > 0) --r.frames_remaining;
    retired_images_.erase(
        std::remove_if(retired_images_.begin(), retired_images_.end(),
            [&](RetiredImage& r) {
                if (r.frames_remaining == 0) {
                    if (r.img) r.img->destroy(ctx_);
                    return true;
                }
                return false;
            }),
        retired_images_.end());

    for (auto& r : retired_passes_) if (r.frames_remaining > 0) --r.frames_remaining;
    retired_passes_.erase(
        std::remove_if(retired_passes_.begin(), retired_passes_.end(),
            [&](RetiredPass& r) {
                if (r.frames_remaining == 0) {
                    if (r.pipeline) r.pipeline->destroy(ctx_);
                    return true;
                }
                return false;
            }),
        retired_passes_.end());
}

} // namespace te
