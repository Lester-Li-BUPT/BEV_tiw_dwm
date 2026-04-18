_base_ = '../default_runtime.py'

# -------------------- basic --------------------
category_name = 'Pedestrian'   # ✅ 改成 Ped
data_dir = '/home/lishengjie/data/cxtrack/KITTI_tracking/training'

# ⚠️ 与 voxel/grid 保持一致
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]

# Ped 样本/点更少，batch 太大容易不稳；A40 一般 64/96 更稳
batch_size = 96

# make sure custom modules are registered
custom_imports = dict(
    imports=['models'],
    allow_failed_imports=False
)

# -------------------- model --------------------
model = dict(
    type='BEVTrack',
    backbone=dict(
        type='VoxelNet',
        point_cloud_range=point_cloud_range,
        voxel_size=[0.075, 0.075, 0.15],
        grid_size=[21, 128, 128],
    ),
    fuser=dict(type='BEVFuser'),
    head=dict(type='SimpleHead'),

    # ✅ TIW + DMW 全开
    use_dmw=True,
    dmw=dict(
        type='DMWModule',
        feat_dim=128,
        hidden=256,
        # Ped：遮挡/形变更频繁，motion 略加权，feat 略保守
        lambda_motion=1.5,
        lambda_feat=0.25,
        per_dof_weight=True,
        init_w_bias=-3.0,
    ),

    use_tiw=True,
    tiw=dict(
        type='TIWModule',
        feat_dim=128,
        # Ped：短期一致性更重要，L 不宜太长；5 是稳健默认
        L=5,
        alpha=1.0,
        beta=0.5,
        sigma=1.0,
        use_delta_norm=True,
    ),

    cfg=dict(point_cloud_range=point_cloud_range),
)

# -------------------- dataset --------------------
train_dataset = dict(
    type='TrainSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Train',
        category_name=category_name,
        # ✅ 不预加载
        preloading=False,
        preload_offset=10
    ),
    cfg=dict(
        # Ped：更容易出现 hard negative / 丢失，候选略多更稳
        num_candidates=10,

        # Ped：precision 优先 => 搜索别太大，避免引入干扰
        search_thr=5,

        # 给明确阈值（别 None），抑制伪正样本导致 precision 抖
        target_thr=10,

        point_cloud_range=point_cloud_range,
        time_flip=True,
        flip=True,
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Test',
        category_name=category_name,
        preloading=False,
        preload_offset=-1
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)

# -------------------- hooks --------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=1,
        save_best='precision',  # ✅ best 依据 precision
        rule='greater',
    )
)

# -------------------- (可选) stabilization --------------------
# 如果你后期 precision 抖，建议加上（不改 optimizer 类型）：
# optim_wrapper = dict(clip_grad=dict(max_norm=35.0, norm_type=2))
# param_scheduler = [dict(type='MultiStepLR', by_epoch=True, milestones=[210, 260, 320], gamma=0.1)]
# train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)
