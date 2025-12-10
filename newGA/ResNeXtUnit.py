class ResNeXtUnit():
    def __init__(self,unit_id,block_amount,in_channel,out_channel,cardinalitys,group_widths,hasSENets,stride):
        """单元初始化

        Args:
            unit_id: 单元编号
            block_amount: Block 数量
            in_channel: 首个 Block 输入通道
            out_channel: 最后一个 Block 输出通道
            cardinalitys: 每个 Block 的 cardinality 列表
            group_widths: 每个 Block 的 group_width 列表
            hasSENets: 每个 Block 是否含 SENet
            stride: 首个 Block 的步长（其余为 1）
        """
        self.unit_id=unit_id #单元的编号
        self.block_amount=block_amount #单元中resnext_block的数量
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.cardinalitys=cardinalitys # 每个Block中的cardinality
        self.group_widths=group_widths # 每个Block中的group_width
        self.hasSENets=hasSENets
        self.stride=stride
    
    def __str__(self):
        dict={"block_amount":self.block_amount,"in_channel":self.in_channel,"out_channel":self.out_channel,"cardinalitys":self.cardinalitys,"group_widths":self.group_widths,"hasSENets":self.hasSENets,"stride":self.stride}
        return "resx："+str(dict)
        
"""ResNeXt 单元：由多个 ResNeXtBlock 组成

说明
- 记录单元的 block 数量、输入输出通道、每个 block 的 `(cardinality, group_width)`、是否包含 SENet、以及 stride
"""
