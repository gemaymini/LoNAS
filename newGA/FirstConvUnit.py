class FirstConvUnit():
    def __init__(self,unit_id,in_size,in_channel,out_channel):
        """记录首卷积单元的元信息"""
        self.unit_id=unit_id
        self.in_size=in_size
        self.in_channel=in_channel
        self.out_channel=out_channel

    def __str__(self):
        dict={"in_size":self.in_size,"in_channel":self.in_channel,"out_channel":self.out_channel}
        return "fisrt："+str(dict)
"""首卷积单元的结构元定义

说明
- 用于记录网络最前端的输入尺寸/通道与输出通道，实际卷积逻辑在 newNetBlock/FirstConvBlock 中实现
"""
