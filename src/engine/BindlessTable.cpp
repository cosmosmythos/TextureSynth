#include "engine/BindlessTable.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
#include <array>

namespace te {

bool BindlessTable::init(VulkanContext& ctx,
                         VkSampler samp_repeat,
                         VkSampler samp_clamp,
                         VkSampler samp_mirror,
                         uint32_t push_constant_size) {
    // Pool (UPDATE_AFTER_BIND)
    std::array<VkDescriptorPoolSize, 4> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_SAMPLED};
    sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_STORAGE};
    sizes[2] = {VK_DESCRIPTOR_TYPE_SAMPLER,       3};
    sizes[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, PARAM_RING_SIZE};

    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pci.maxSets       = 1;
    pci.poolSizeCount = (uint32_t)sizes.size();
    pci.pPoolSizes    = sizes.data();
    if (vkCreateDescriptorPool(ctx.device(), &pci, nullptr, &pool_) != VK_SUCCESS) {
        log_error("BindlessTable: pool create failed"); return false;
    }
    ctx.set_debug_name(VK_OBJECT_TYPE_DESCRIPTOR_POOL, (uint64_t)pool_, "bindless_pool");

    // Set layout -- 6 bindings (arrays + samplers + SSBO)
    std::array<VkSampler, 1> imm_rep{samp_repeat};
    std::array<VkSampler, 1> imm_clp{samp_clamp};
    std::array<VkSampler, 1> imm_mir{samp_mirror};

    std::array<VkDescriptorSetLayoutBinding, 6> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  MAX_SAMPLED, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  MAX_STORAGE, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, imm_rep.data()};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, imm_clp.data()};
    b[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, imm_mir.data()};
    b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, PARAM_RING_SIZE, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    const VkDescriptorBindingFlags bindless_flags =
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    std::array<VkDescriptorBindingFlags, 6> flags{
        bindless_flags, bindless_flags, 0, 0, 0,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT   // SSBO rebound each ring swap; PARTIALLY_BOUND so unused tail slots are legal.
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfci.bindingCount  = (uint32_t)flags.size();
    bfci.pBindingFlags = flags.data();

    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    slci.bindingCount = (uint32_t)b.size();
    slci.pBindings    = b.data();
    slci.pNext        = &bfci;
    if (vkCreateDescriptorSetLayout(ctx.device(), &slci, nullptr, &set_layout_) != VK_SUCCESS) {
        log_error("BindlessTable: set layout create failed"); return false;
    }

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &set_layout_;
    if (vkAllocateDescriptorSets(ctx.device(), &ai, &set_) != VK_SUCCESS) {
        log_error("BindlessTable: set alloc failed"); return false;
    }
    ctx.set_debug_name(VK_OBJECT_TYPE_DESCRIPTOR_SET, (uint64_t)set_, "bindless_set_0");

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &set_layout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx.device(), &plci, nullptr, &pipe_layout_) != VK_SUCCESS) {
        log_error("BindlessTable: pipeline layout create failed"); return false;
    }

    // Reserve slot 0 as permanent sentinel -- never returned by alloc. Guarantees slot 0 always safe (reads zero pixel).
    sampled_next_ = 1;
    storage_next_ = 1;

    log_info("BindlessTable: ready (sampled cap=" + std::to_string(MAX_SAMPLED)
             + ", storage cap=" + std::to_string(MAX_STORAGE) + ")");
    return true;
}


void BindlessTable::shutdown(VulkanContext& ctx) {
    if (pipe_layout_) { vkDestroyPipelineLayout(ctx.device(), pipe_layout_, nullptr); pipe_layout_ = VK_NULL_HANDLE; }
    if (set_layout_)  { vkDestroyDescriptorSetLayout(ctx.device(), set_layout_, nullptr); set_layout_ = VK_NULL_HANDLE; }
    if (pool_)        { vkDestroyDescriptorPool(ctx.device(), pool_, nullptr); pool_ = VK_NULL_HANDLE; }
    set_ = VK_NULL_HANDLE;
    sampled_free_.clear();
    storage_free_.clear();
    sampled_next_ = 1;   // keep slot 0 reserved
    storage_next_ = 1;
}


uint32_t BindlessTable::alloc_sampled_slot() {
    if (!sampled_free_.empty()) { auto s = sampled_free_.back(); sampled_free_.pop_back(); return s; }
    if (sampled_next_ >= MAX_SAMPLED) { log_error("BindlessTable: sampled exhausted"); return INVALID_SLOT; }
    return sampled_next_++;
}


uint32_t BindlessTable::alloc_storage_slot() {
    if (!storage_free_.empty()) { auto s = storage_free_.back(); storage_free_.pop_back(); return s; }
    if (storage_next_ >= MAX_STORAGE) { log_error("BindlessTable: storage exhausted"); return INVALID_SLOT; }
    return storage_next_++;
}


void BindlessTable::free_sampled_slot(uint32_t s) {
    if (s != INVALID_SLOT && s < MAX_SAMPLED && s != 0) sampled_free_.push_back(s);
}


void BindlessTable::free_storage_slot(uint32_t s) {
    if (s != INVALID_SLOT && s < MAX_STORAGE) storage_free_.push_back(s);
}


void BindlessTable::write_sampled(VulkanContext& ctx, uint32_t slot, VkImageView view, VkImageLayout layout) {
    if (slot == INVALID_SLOT) return;
    VkDescriptorImageInfo ii{};
    ii.imageView   = view;
    ii.imageLayout = layout;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = set_;
    w.dstBinding      = 0;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w.pImageInfo      = &ii;
    vkUpdateDescriptorSets(ctx.device(), 1, &w, 0, nullptr);
}


void BindlessTable::write_storage(VulkanContext& ctx, uint32_t slot, VkImageView view) {
    if (slot == INVALID_SLOT) return;
    VkDescriptorImageInfo ii{};
    ii.imageView   = view;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = set_;
    w.dstBinding      = 1;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo      = &ii;
    vkUpdateDescriptorSets(ctx.device(), 1, &w, 0, nullptr);
}


void BindlessTable::write_param_ring(VulkanContext& ctx,
                                     const std::array<VkBuffer, PARAM_RING_SIZE>& bufs,
                                     VkDeviceSize range) {
    std::array<VkDescriptorBufferInfo, PARAM_RING_SIZE> infos{};
    std::array<VkWriteDescriptorSet, PARAM_RING_SIZE> writes{};
    for (uint32_t i = 0; i < PARAM_RING_SIZE; ++i) {
        infos[i].buffer = bufs[i];
        infos[i].offset = 0;
        infos[i].range  = range;

        writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet          = set_;
        writes[i].dstBinding      = 5;
        writes[i].dstArrayElement = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(ctx.device(), PARAM_RING_SIZE, writes.data(), 0, nullptr);
}


} // namespace te