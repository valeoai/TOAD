model.model._backbone:  HydraBackbone(
  (image_encoder): VoVNet(
    (stem): Sequential(
      (stem_1/conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (stem_1/norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (stem_1/relu): ReLU(inplace=True)
      (stem_2/conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (stem_2/norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (stem_2/relu): ReLU(inplace=True)
      (stem_3/conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (stem_3/norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (stem_3/relu): ReLU(inplace=True)
    )
    (stage2): _OSA_stage(
      (OSA2_1): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA2_1_0/conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA2_1_0/norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA2_1_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA2_1_1/conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA2_1_1/norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA2_1_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA2_1_2/conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA2_1_2/norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA2_1_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA2_1_3/conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA2_1_3/norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA2_1_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA2_1_4/conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA2_1_4/norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA2_1_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA2_1_concat/conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA2_1_concat/norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA2_1_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
    )
    (stage3): _OSA_stage(
      (Pooling): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (OSA3_1): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA3_1_0/conv): Conv2d(256, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_1_0/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_1_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA3_1_1/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_1_1/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_1_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA3_1_2/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_1_2/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_1_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA3_1_3/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_1_3/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_1_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA3_1_4/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_1_4/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_1_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA3_1_concat/conv): Conv2d(1056, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA3_1_concat/norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA3_1_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA3_2): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA3_2_0/conv): Conv2d(512, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_2_0/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_2_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA3_2_1/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_2_1/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_2_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA3_2_2/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_2_2/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_2_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA3_2_3/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_2_3/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_2_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA3_2_4/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_2_4/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_2_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA3_2_concat/conv): Conv2d(1312, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA3_2_concat/norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA3_2_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA3_3): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA3_3_0/conv): Conv2d(512, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_3_0/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_3_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA3_3_1/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_3_1/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_3_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA3_3_2/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_3_2/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_3_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA3_3_3/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_3_3/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_3_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA3_3_4/conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA3_3_4/norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA3_3_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA3_3_concat/conv): Conv2d(1312, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA3_3_concat/norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA3_3_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
    )
    (stage4): _OSA_stage(
      (Pooling): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (OSA4_1): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_1_0/conv): Conv2d(512, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_1_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_1_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_1_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_1_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_1_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_1_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_1_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_1_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_1_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_1_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_1_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_1_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_1_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_1_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_1_concat/conv): Conv2d(1472, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_1_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_1_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_2): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_2_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_2_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_2_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_2_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_2_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_2_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_2_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_2_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_2_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_2_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_2_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_2_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_2_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_2_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_2_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_2_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_2_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_2_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_3): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_3_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_3_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_3_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_3_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_3_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_3_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_3_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_3_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_3_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_3_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_3_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_3_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_3_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_3_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_3_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_3_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_3_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_3_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_4): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_4_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_4_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_4_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_4_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_4_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_4_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_4_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_4_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_4_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_4_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_4_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_4_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_4_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_4_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_4_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_4_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_4_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_4_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_5): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_5_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_5_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_5_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_5_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_5_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_5_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_5_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_5_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_5_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_5_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_5_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_5_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_5_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_5_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_5_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_5_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_5_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_5_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_6): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_6_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_6_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_6_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_6_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_6_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_6_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_6_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_6_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_6_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_6_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_6_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_6_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_6_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_6_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_6_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_6_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_6_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_6_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_7): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_7_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_7_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_7_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_7_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_7_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_7_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_7_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_7_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_7_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_7_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_7_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_7_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_7_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_7_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_7_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_7_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_7_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_7_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_8): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_8_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_8_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_8_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_8_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_8_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_8_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_8_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_8_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_8_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_8_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_8_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_8_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_8_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_8_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_8_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_8_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_8_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_8_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA4_9): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA4_9_0/conv): Conv2d(768, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_9_0/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_9_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA4_9_1/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_9_1/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_9_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA4_9_2/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_9_2/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_9_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA4_9_3/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_9_3/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_9_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA4_9_4/conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA4_9_4/norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA4_9_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA4_9_concat/conv): Conv2d(1728, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA4_9_concat/norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA4_9_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
    )
    (stage5): _OSA_stage(
      (Pooling): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (OSA5_1): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA5_1_0/conv): Conv2d(768, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_1_0/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_1_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA5_1_1/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_1_1/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_1_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA5_1_2/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_1_2/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_1_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA5_1_3/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_1_3/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_1_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA5_1_4/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_1_4/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_1_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA5_1_concat/conv): Conv2d(1888, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA5_1_concat/norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA5_1_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA5_2): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA5_2_0/conv): Conv2d(1024, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_2_0/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_2_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA5_2_1/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_2_1/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_2_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA5_2_2/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_2_2/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_2_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA5_2_3/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_2_3/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_2_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA5_2_4/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_2_4/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_2_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA5_2_concat/conv): Conv2d(2144, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA5_2_concat/norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA5_2_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
      (OSA5_3): _OSA_module(
        (layers): ModuleList(
          (0): Sequential(
            (OSA5_3_0/conv): Conv2d(1024, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_3_0/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_3_0/relu): ReLU(inplace=True)
          )
          (1): Sequential(
            (OSA5_3_1/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_3_1/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_3_1/relu): ReLU(inplace=True)
          )
          (2): Sequential(
            (OSA5_3_2/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_3_2/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_3_2/relu): ReLU(inplace=True)
          )
          (3): Sequential(
            (OSA5_3_3/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_3_3/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_3_3/relu): ReLU(inplace=True)
          )
          (4): Sequential(
            (OSA5_3_4/conv): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (OSA5_3_4/norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (OSA5_3_4/relu): ReLU(inplace=True)
          )
        )
        (concat): Sequential(
          (OSA5_3_concat/conv): Conv2d(2144, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (OSA5_3_concat/norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (OSA5_3_concat/relu): ReLU(inplace=True)
        )
        (ese): eSEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (hsigmoid): Hsigmoid()
        )
      )
    )
  )
  (avgpool_img): AdaptiveAvgPool2d(output_size=(16, 64))
)