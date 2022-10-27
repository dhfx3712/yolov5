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






