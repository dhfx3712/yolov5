1、构建gt
gt映射到特征图，并扩充正样本

2、计算损失
过滤出与gt相同位子的框，计算损失




loss model_pred : 3 pred_0 : torch.Size([1, 3, 80, 80, 85])  pred_1 : torch.Size([1, 3, 40, 40, 85])  pred_2 : torch.Size([1, 3, 20, 20, 85])  target : torch.Size([15, 6])
build_target : na - 3   nt - 15
build_target_xyxy : tensor([80, 80, 80, 80])
build_target_t :  t - torch.Size([3, 15, 7])
build_target_t-filter :  t - torch.Size([22, 7])  j - torch.Size([3, 15])
build_target_positive_sample : j-tensor([False,  True, False,  True, False,  True,  True,  True, False,  True,  True,  True,  True, False,  True, False,  True,  True,  True, False,  True,  True]) ,k-tensor([ True,  True, False,  True,  True, False,  True,  True, False, False,  True,  True,  True, False,  True,  True, False,  True, False,  True, False,  True]) ,l-tensor([ True, False,  True, False,  True, False, False, False,  True, False, False, False, False,  True, False,  True, False, False, False,  True, False, False]) ,m-tensor([False, False,  True, False, False,  True, False, False, False,  True, False, False, False,  True, False, False,  True, False,  True, False,  True, False])
build_target_repeat_t: torch.Size([65, 7]) 
build_target_xyxy : tensor([40, 40, 40, 40])
build_target_t :  t - torch.Size([3, 15, 7])
build_target_t-filter :  t - torch.Size([17, 7])  j - torch.Size([3, 15])
build_target_positive_sample : j-tensor([False,  True,  True,  True, False, False,  True, False,  True,  True,  True, False,  True, False,  True,  True,  True]) ,k-tensor([ True, False,  True, False,  True,  True, False,  True, False,  True, False,  True, False,  True, False,  True, False]) ,l-tensor([ True, False, False, False,  True,  True, False,  True, False, False, False,  True, False,  True, False, False, False]) ,m-tensor([False,  True, False,  True, False, False,  True, False,  True, False,  True, False,  True, False,  True, False,  True])
build_target_repeat_t: torch.Size([51, 7]) 
build_target_xyxy : tensor([20, 20, 20, 20])
build_target_t :  t - torch.Size([3, 15, 7])
build_target_t-filter :  t - torch.Size([13, 7])  j - torch.Size([3, 15])
build_target_positive_sample : j-tensor([ True, False, False, False,  True,  True, False, False, False,  True,  True, False, False]) ,k-tensor([False,  True,  True,  True, False, False,  True,  True,  True, False, False,  True,  True]) ,l-tensor([False,  True,  True,  True, False, False,  True,  True,  True, False, False,  True,  True]) ,m-tensor([ True, False, False, False,  True,  True, False, False, False,  True,  True, False, False])
build_target_repeat_t: torch.Size([39, 7]) 
build_target_return : 3,3,3,3
pbox_shape : torch.Size([65, 4])  tbox : torch.Size([65, 4])
pbox_shape : torch.Size([51, 4])  tbox : torch.Size([51, 4])
pbox_shape : torch.Size([39, 4])  tbox : torch.Size([39, 4])




NMS非极大值抑制（通过conf_thres过滤）
1、指标计算
2、显示中有重叠框



训练和val输出结构有区别
models/yolo.py -- Detect
    if not self.training -- self.na * nx * ny (grid组装在一起返回)
    
        
    



train 
model(im)

val 
model(im)
val.py-->DetectMultiBackend()
    models/common.py-->attempt_load()
        models/experimental.py-->Ensemble()
    





val_model : model.modules()
Ensemble(
  (0): DetectionModel(
    (model): Sequential(
      (0): Conv(
        (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        (act): SiLU()
      )
      (1): Conv(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (2): C3(
        (cv1): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (3): Conv(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (4): C3(
        (cv1): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (5): Conv(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (6): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
          (2): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (7): Conv(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (8): C3(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (9): SPPF(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
      )
      (10): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (11): Upsample(scale_factor=2.0, mode=nearest)
      (12): Concat()
      (13): C3(
        (cv1): Conv(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (14): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (15): Upsample(scale_factor=2.0, mode=nearest)
      (16): Concat()
      (17): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (18): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (19): Concat()
      (20): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (21): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (22): Concat()
      (23): C3(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (24): Detect(
        (m): ModuleList(
          (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
          (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
)






val_model : 
Ensemble(
  (0): DetectionModel(
    (model): Sequential(
      (0): Conv(
        (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        (act): SiLU()
      )
      (1): Conv(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (2): C3(
        (cv1): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (3): Conv(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (4): C3(
        (cv1): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (5): Conv(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (6): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
          (2): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (7): Conv(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (8): C3(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (9): SPPF(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
      )
      (10): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (11): Upsample(scale_factor=2.0, mode=nearest)
      (12): Concat()
      (13): C3(
        (cv1): Conv(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (14): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (15): Upsample(scale_factor=2.0, mode=nearest)
      (16): Concat()
      (17): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (18): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (19): Concat()
      (20): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (21): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU()
      )
      (22): Concat()
      (23): C3(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU()
            )
          )
        )
      )
      (24): Detect(
        (m): ModuleList(
          (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
          (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
) , <class 'models.experimental.Ensemble'> 
val_model : DetectionModel(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
      (act): SiLU()
    )
    (1): Conv(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU()
    )
    (2): C3(
      (cv1): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU()
    )
    (4): C3(
      (cv1): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (5): Conv(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU()
    )
    (6): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (7): Conv(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU()
    )
    (8): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (9): SPPF(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    )
    (10): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (11): Upsample(scale_factor=2.0, mode=nearest)
    (12): Concat()
    (13): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (14): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (15): Upsample(scale_factor=2.0, mode=nearest)
    (16): Concat()
    (17): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (18): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU()
    )
    (19): Concat()
    (20): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (21): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU()
    )
    (22): Concat()
    (23): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU()
          )
        )
      )
    )
    (24): Detect(
      (m): ModuleList(
        (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
) , <class 'models.yolo.DetectionModel'> 
inplace : True
val_model : Sequential(
  (0): Conv(
    (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
    (act): SiLU()
  )
  (1): Conv(
    (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (act): SiLU()
  )
  (2): C3(
    (cv1): Conv(
      (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (3): Conv(
    (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (act): SiLU()
  )
  (4): C3(
    (cv1): Conv(
      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (5): Conv(
    (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (act): SiLU()
  )
  (6): C3(
    (cv1): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (7): Conv(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (act): SiLU()
  )
  (8): C3(
    (cv1): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (9): SPPF(
    (cv1): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
  )
  (10): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (11): Upsample(scale_factor=2.0, mode=nearest)
  (12): Concat()
  (13): C3(
    (cv1): Conv(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (14): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (15): Upsample(scale_factor=2.0, mode=nearest)
  (16): Concat()
  (17): C3(
    (cv1): Conv(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (18): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (act): SiLU()
  )
  (19): Concat()
  (20): C3(
    (cv1): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (21): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (act): SiLU()
  )
  (22): Concat()
  (23): C3(
    (cv1): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): SiLU()
        )
      )
    )
  )
  (24): Detect(
    (m): ModuleList(
      (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Conv(
  (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
    (1): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
  (1): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
    (1): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
    (2): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
  (1): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
  (2): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : SPPF(
  (cv1): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
) , <class 'models.common.SPPF'> 
val_model : Conv(
  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False) , <class 'torch.nn.modules.pooling.MaxPool2d'> 
val_model : Conv(
  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Upsample(scale_factor=2.0, mode=nearest) , <class 'torch.nn.modules.upsampling.Upsample'> 
val_model : Concat() , <class 'models.common.Concat'> 
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Upsample(scale_factor=2.0, mode=nearest) , <class 'torch.nn.modules.upsampling.Upsample'> 
val_model : Concat() , <class 'models.common.Concat'> 
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Concat() , <class 'models.common.Concat'> 
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Concat() , <class 'models.common.Concat'> 
val_model : C3(
  (cv1): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv3): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU()
      )
    )
  )
) , <class 'models.common.C3'> 
val_model : Conv(
  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Sequential(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU()
    )
  )
) , <class 'torch.nn.modules.container.Sequential'> 
val_model : Bottleneck(
  (cv1): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (act): SiLU()
  )
) , <class 'models.common.Bottleneck'> 
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Conv(
  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act): SiLU()
) , <class 'models.common.Conv'> 
val_model : Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : SiLU() , <class 'torch.nn.modules.activation.SiLU'> 
inplace : False
val_model : Detect(
  (m): ModuleList(
    (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
    (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
    (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
  )
) , <class 'models.yolo.Detect'> 
inplace : True
detect : tensor([[[[[[ 10.,  13.]]],


          [[[ 16.,  30.]]],


          [[[ 33.,  23.]]]]],




        [[[[[ 30.,  61.]]],


          [[[ 62.,  45.]]],


          [[[ 59., 119.]]]]],




        [[[[[116.,  90.]]],


          [[[156., 198.]]],


          [[[373., 326.]]]]]])   , 3
val_model : ModuleList(
  (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
  (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
  (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
) , <class 'torch.nn.modules.container.ModuleList'> 
val_model : Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 
val_model : Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1)) , <class 'torch.nn.modules.conv.Conv2d'> 











OrderedDict([('0', Conv(
  (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
  (act): SiLU(inplace=True)
)), ('1', Conv(
  (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU(inplace=True)
)), ('2', C3(
  (cv1): Conv(
    (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('3', Conv(
  (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU(inplace=True)
)), ('4', C3(
  (cv1): Conv(
    (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
    (1): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('5', Conv(
  (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU(inplace=True)
)), ('6', C3(
  (cv1): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
    (1): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
    (2): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('7', Conv(
  (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU(inplace=True)
)), ('8', C3(
  (cv1): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('9', SPPF(
  (cv1): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
)), ('10', Conv(
  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU(inplace=True)
)), ('11', Upsample(scale_factor=2.0, mode=nearest)), ('12', Concat()), ('13', C3(
  (cv1): Conv(
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('14', Conv(
  (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (act): SiLU(inplace=True)
)), ('15', Upsample(scale_factor=2.0, mode=nearest)), ('16', Concat()), ('17', C3(
  (cv1): Conv(
    (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('18', Conv(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU(inplace=True)
)), ('19', Concat()), ('20', C3(
  (cv1): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('21', Conv(
  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (act): SiLU(inplace=True)
)), ('22', Concat()), ('23', C3(
  (cv1): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (cv3): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    (act): SiLU(inplace=True)
  )
  (m): Sequential(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
    )
  )
)), ('24', Detect(
  (m): ModuleList(
    (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
    (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
    (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
  )
))])


self.save
[4, 6, 10, 14, 17, 20, 23]

m.f 最后一层[17, 20, 23]

0   tensor 1 H:336
1   tensor 1 H:168
2   tensor 1 H:168
3   tensor 1 H:84
4   tensor 1 H:84
5   tensor 1 H:42
6   tensor 1 H:42
7   tensor 1 H:21
8   tensor 1 H:21
9   tensor 1 H:21
10   tensor 1 H:21
11   tensor 1 H:42
12   tensor 1 H:42
13   tensor 1 H:42
14   tensor 1 H:42
15   tensor 1 H:84
16   tensor 1 H:84
17   tensor 1 H:84
18   tensor 1 H:42
19   tensor 1 H:42
20   tensor 1 H:42
21   tensor 1 H:21
22   tensor 1 H:21
23   tensor 1 H:21










