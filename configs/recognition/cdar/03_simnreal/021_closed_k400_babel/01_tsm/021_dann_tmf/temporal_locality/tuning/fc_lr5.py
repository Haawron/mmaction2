_base_ = [
    'k2b_tsm_dann_tmf_cross_dcx2.py'
]

optimizer = dict(paramwise_cfg=dict(fc_lr5=True))
