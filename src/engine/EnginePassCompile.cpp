#include "engine/Engine.hpp"
#include "engine/Logging.hpp"
#include <algorithm>

namespace te {

void Engine::poll_pending_compiles() {
    TE_GUARD_READY(;);

    poll_completed_uploads_();
    tick_retired();
    resources_.tick(ctx_);
    poll_timestamps_();
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
        auto* img = image_registry_[c.node_id].get();
        if (img)
            bindless_.write_sampled(ctx_, slot, img->view(), VK_IMAGE_LAYOUT_GENERAL);

        log_info("[upload_complete] node=" + std::to_string(c.node_id)
                 + " bindless_slot=" + std::to_string(slot)
                 + " vkview=" + std::to_string((uint64_t)(img ? img->view() : VK_NULL_HANDLE))
                 + " vkimg=" + std::to_string((uint64_t)(img ? img->image() : VK_NULL_HANDLE)));

        // Patch any group ext_inputs that reference this node (stale snapshot fix).
        for (auto& ge : group_execs_) {
            for (auto& ei : ge.ext_inputs) {
                if (ei.node_id == c.node_id) {
                    ei.sampled_slot  = slot;
                    ei.sampled_image = img ? img->image() : VK_NULL_HANDLE;
                }
            }
        }

        mark_downstream_dirty_(c.node_id);
    }
}


void Engine::retire_all_passes_() {
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
    }
    group_execs_.clear();
    use_groups_ = false;
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
