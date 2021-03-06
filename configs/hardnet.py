cfg = dict(
    model_type='hardnet',
    num_aux_heads=0,
    n_epochs=100,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,
    n_classes=19,
    im_root='./datasets/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    test_im_anns='./datasets/cityscapes/test.txt',
    scales=[0.75, 2.],
    cropsize=[512, 512],
    ims_per_gpu=12,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res/cityscapes',
)